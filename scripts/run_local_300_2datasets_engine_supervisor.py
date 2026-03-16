#!/usr/bin/env python3
"""Engine supervisor for local_300_2datasets request-mode sweep.

Runs exactly one TEI container per embedding engine and launches a worker pool
that shares that endpoint. The supervisor advances to the next engine only
after the current engine has zero missing run_keys for the request.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

import docker

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.managers.local_docker_tei import LocalDockerTEIManager
from study_query_llm.services.supervisor_mode import create_supervisor_mode
from study_query_llm.services.sweep_request_service import SweepRequestService

from scripts.run_local_300_2datasets_worker import EMBEDDING_ENGINES

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


def _safe_name(s: str) -> str:
    return s.replace("-", "_").replace("/", "_")


def _stop_workers(workers: List[Tuple[str, subprocess.Popen]]) -> None:
    for worker_id, proc in workers:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 20
    while time.time() < deadline:
        if all(proc.poll() is not None for _, proc in workers):
            return
        time.sleep(0.5)
    for worker_id, proc in workers:
        if proc.poll() is None:
            proc.kill()


def _spawn_workers(
    *,
    repo_root: Path,
    request_id: int,
    engine: str,
    tei_endpoint: str,
    workers: int,
    claim_lease_seconds: int,
    provider_label: str,
    idle_exit_seconds: int,
    job_mode: str,
) -> List[Tuple[str, subprocess.Popen]]:
    procs: List[Tuple[str, subprocess.Popen]] = []
    worker_script = repo_root / "scripts" / "run_local_300_2datasets_worker.py"
    for slot in range(workers):
        worker_id = f"supervisor-{_safe_name(engine)}-w{slot + 1}"
        cmd = [
            sys.executable,
            str(worker_script),
            "--request-id",
            str(request_id),
            "--worker-id",
            worker_id,
            "--worker-slot",
            str(slot),
            "--claim-lease-seconds",
            str(claim_lease_seconds),
            "--job-mode",
            job_mode,
            "--embedding-engine",
            engine,
            "--tei-endpoint",
            tei_endpoint,
            "--provider-label",
            provider_label,
            "--idle-exit-seconds",
            str(idle_exit_seconds),
        ]
        proc = subprocess.Popen(cmd, cwd=str(repo_root))
        procs.append((worker_id, proc))
        print(f"  [spawn] {worker_id} pid={proc.pid}")
    return procs


def _backoff_seconds(base_seconds: int, attempt: int, max_seconds: int) -> int:
    return min(max_seconds, int(base_seconds * (2 ** max(0, attempt - 1))))


def _cleanup_stale_supervisor_containers(active_container_name: str) -> None:
    """Best-effort cleanup of stale TEI supervisor containers.

    This enforces the one-container-per-engine intent even if a prior supervisor
    run was interrupted and left containers behind.
    """
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
    except Exception as exc:
        print(f"  [cleanup] warning: cannot list docker containers: {exc}")
        return

    for c in containers:
        # docker-py names are usually like ['/container-name']
        names = [n.lstrip("/") for n in (getattr(c, "attrs", {}).get("Name", ""),) if n]
        if not names:
            # fallback path
            try:
                names = [n.lstrip("/") for n in (c.name,)]
            except Exception:
                names = []
        if not names:
            continue
        name = names[0]
        if not (name.startswith("tei-") and name.endswith("-supervisor")):
            continue
        if name == active_container_name:
            continue
        try:
            if c.status == "running":
                print(f"  [cleanup] stopping stale container: {name}")
                c.stop(timeout=10)
            print(f"  [cleanup] removing stale container: {name}")
            c.remove(force=True)
        except Exception as exc:
            print(f"  [cleanup] warning: failed to remove {name}: {exc}")


def _tei_health_check(endpoint: str, timeout_seconds: int = 3) -> bool:
    # LocalDockerTEIManager returns endpoint like http://localhost:8080/v1
    # while TEI health is served at /health (not /v1/health).
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    health_url = base + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="One-container-per-engine sweep supervisor")
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--claim-lease-seconds", type=int, default=3600)
    parser.add_argument("--idle-exit-seconds", type=int, default=90)
    parser.add_argument("--progress-poll-seconds", type=int, default=10)
    parser.add_argument("--provider-label", type=str, default="local_docker_tei_shared")
    parser.add_argument("--engine-allowlist", type=str, default=None)
    parser.add_argument("--tei-port", type=int, default=8080)
    parser.add_argument("--max-worker-restarts", type=int, default=2)
    parser.add_argument("--worker-restart-backoff-seconds", type=int, default=5)
    parser.add_argument("--max-worker-restart-backoff-seconds", type=int, default=30)
    parser.add_argument("--max-tei-restarts", type=int, default=2)
    parser.add_argument("--tei-restart-backoff-seconds", type=int, default=10)
    parser.add_argument("--max-tei-restart-backoff-seconds", type=int, default=60)
    parser.add_argument(
        "--job-mode",
        type=str,
        default="standalone",
        choices=["standalone", "sharded"],
        help="Execution mode for workers and progress tracking.",
    )
    args = parser.parse_args()

    db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
    db.init_db()
    repo_root = Path(__file__).parent.parent

    engines = EMBEDDING_ENGINES
    if args.engine_allowlist:
        allowed = {e.strip() for e in args.engine_allowlist.split(",") if e.strip()}
        engines = [e for e in EMBEDDING_ENGINES if e in allowed]

    print(f"[supervisor] request_id={args.request_id}")
    print(f"[supervisor] workers={args.workers}")
    print(f"[supervisor] engines={len(engines)}")

    mode = create_supervisor_mode(args.job_mode)

    for engine in engines:
        engine_missing, total_missing = mode.engine_work_remaining(db, args.request_id, engine)
        if engine_missing == 0:
            print(f"\n[engine] {engine} -> already exhausted, skipping")
            continue

        print(f"\n[engine] {engine}")
        print(f"  missing_for_engine={engine_missing}, missing_total={total_missing}")

        model_safe = engine.replace("/", "-").replace(".", "-").lower()
        manager = LocalDockerTEIManager(
            model_id=engine,
            use_gpu=True,
            port=args.tei_port,
            container_name=f"tei-{model_safe}-supervisor",
        )
        _cleanup_stale_supervisor_containers(manager.container_name)

        restart_counts = {}
        tei_restart_count = 0
        workers: List[Tuple[str, subprocess.Popen]] = []
        try:
            manager.start()
            endpoint = manager.endpoint_url
            if not endpoint:
                raise RuntimeError("TEI endpoint was not set after manager.start()")
            if not _tei_health_check(endpoint):
                raise RuntimeError(f"TEI failed health check at startup: {endpoint}")
            print(f"  [tei] ready at {endpoint}")

            workers = _spawn_workers(
                repo_root=repo_root,
                request_id=args.request_id,
                engine=engine,
                tei_endpoint=endpoint,
                workers=args.workers,
                claim_lease_seconds=args.claim_lease_seconds,
                provider_label=args.provider_label,
                idle_exit_seconds=args.idle_exit_seconds,
                job_mode=args.job_mode,
            )

            while True:
                time.sleep(args.progress_poll_seconds)
                mode.before_progress_poll(db, args.request_id)
                engine_missing, total_missing = mode.engine_work_remaining(db, args.request_id, engine)
                print(
                    f"  [progress] engine_missing={engine_missing}, total_missing={total_missing}"
                )
                if engine_missing == 0:
                    print("  [engine] exhausted; advancing")
                    break

                if not _tei_health_check(endpoint):
                    tei_restart_count += 1
                    if tei_restart_count > args.max_tei_restarts:
                        raise RuntimeError(
                            f"TEI unhealthy for {engine}; exceeded max restarts "
                            f"({args.max_tei_restarts})"
                        )
                    backoff = _backoff_seconds(
                        args.tei_restart_backoff_seconds,
                        tei_restart_count,
                        args.max_tei_restart_backoff_seconds,
                    )
                    print(
                        f"  [tei] health check failed; restarting container "
                        f"attempt={tei_restart_count} backoff={backoff}s"
                    )
                    _stop_workers(workers)
                    workers = []
                    manager.stop()
                    time.sleep(backoff)
                    manager.start()
                    endpoint = manager.endpoint_url
                    if not endpoint or not _tei_health_check(endpoint):
                        raise RuntimeError("TEI did not recover after restart")
                    workers = _spawn_workers(
                        repo_root=repo_root,
                        request_id=args.request_id,
                        engine=engine,
                        tei_endpoint=endpoint,
                        workers=args.workers,
                        claim_lease_seconds=args.claim_lease_seconds,
                        provider_label=args.provider_label,
                        idle_exit_seconds=args.idle_exit_seconds,
                        job_mode=args.job_mode,
                    )
                    continue

                # Restart workers that crashed unexpectedly.
                for idx, (worker_id, proc) in enumerate(list(workers)):
                    code = proc.poll()
                    if code is None:
                        continue
                    if code == 0:
                        # Idle exit while work remains can happen due to claim contention.
                        # Restart in bounded fashion.
                        pass
                    count = restart_counts.get(worker_id, 0)
                    if count >= args.max_worker_restarts:
                        continue
                    restart_counts[worker_id] = count + 1
                    backoff = _backoff_seconds(
                        args.worker_restart_backoff_seconds,
                        restart_counts[worker_id],
                        args.max_worker_restart_backoff_seconds,
                    )
                    print(
                        f"  [worker] restarting {worker_id} "
                        f"attempt={restart_counts[worker_id]} backoff={backoff}s"
                    )
                    time.sleep(backoff)
                    cmd = [
                        sys.executable,
                        str(repo_root / "scripts" / "run_local_300_2datasets_worker.py"),
                        "--request-id",
                        str(args.request_id),
                        "--worker-id",
                        worker_id,
                        "--worker-slot",
                        str(idx),
                        "--claim-lease-seconds",
                        str(args.claim_lease_seconds),
                        "--job-mode",
                        args.job_mode,
                        "--embedding-engine",
                        engine,
                        "--tei-endpoint",
                        endpoint,
                        "--provider-label",
                        args.provider_label,
                        "--idle-exit-seconds",
                        str(args.idle_exit_seconds),
                    ]
                    new_proc = subprocess.Popen(cmd, cwd=str(repo_root))
                    workers[idx] = (worker_id, new_proc)
                    print(
                        f"  [restart] {worker_id} pid={new_proc.pid} "
                        f"attempt={restart_counts[worker_id]}"
                    )

                # Guardrail: if all workers exhausted restart budget and exited while work remains.
                if workers and all(proc.poll() is not None for _, proc in workers):
                    if all(restart_counts.get(worker_id, 0) >= args.max_worker_restarts for worker_id, _ in workers):
                        raise RuntimeError(
                            f"All workers exited and restart budget exhausted for engine {engine}"
                        )
        finally:
            _stop_workers(workers)
            manager.stop()
            print("  [tei] stopped")

    # final summary
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        progress = svc.compute_progress(args.request_id)
    print(
        "\n[supervisor] done: "
        f"expected={progress.get('expected_count')} "
        f"completed={progress.get('completed_count')} "
        f"missing={progress.get('missing_count')}"
    )


if __name__ == "__main__":
    main()

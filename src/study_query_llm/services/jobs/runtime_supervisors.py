"""Runtime entrypoints for cached-job supervisor and engine supervisor."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Set, Tuple

import docker

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.experiments.sweep_worker_main import EMBEDDING_ENGINES, worker_main_queued
from study_query_llm.providers.managers.local_docker_tei import LocalDockerTEIManager
from study_query_llm.services.jobs.job_reducer_service import JobReducerService
from study_query_llm.services.supervisor_mode import create_supervisor_mode
from study_query_llm.services.sweep_request_service import SweepRequestService


def _run_cached_job_supervisor(
    *,
    request_id: int,
    worker_count: int,
    engine: str,
    provider_label: str,
    embedding_provider_name: Optional[str],
    tei_endpoint: Optional[str],
    lease_seconds: int,
    idle_exit_seconds: int,
    bundle_size: int,
    repo_root: Path,
    database_url: str,
) -> None:
    db = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=default_write_intent_for_connection(database_url),
    )
    db.init_db()

    manager = multiprocessing.Manager()
    job_queue = manager.Queue()
    result_queue = manager.Queue()

    workers: List[multiprocessing.Process] = []
    for slot in range(worker_count):
        p = multiprocessing.Process(
            target=worker_main_queued,
            args=(
                request_id,
                slot,
                job_queue,
                result_queue,
                tei_endpoint,
                provider_label,
                embedding_provider_name,
                repo_root,
                idle_exit_seconds,
                database_url,
            ),
        )
        p.start()
        workers.append(p)

    in_flight_job_ids: Set[int] = set()
    pending_batch: List[Tuple[int, Optional[str]]] = []

    while True:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            snapshots = repo.claim_orchestration_job_batch(
                request_group_id=request_id,
                job_types=["run_k_try"],
                claim_owner="cached-supervisor",
                lease_seconds=lease_seconds,
                limit=bundle_size,
                filter_payload={"embedding_engine": engine},
            )

        if not snapshots and not in_flight_job_ids and not pending_batch:
            break

        for s in snapshots:
            jid = int(s["id"])
            in_flight_job_ids.add(jid)
            job_queue.put(s)

        drained_any = False
        while True:
            try:
                item = result_queue.get_nowait()
            except Exception:
                break
            drained_any = True
            job_id, result_ref, error = item
            in_flight_job_ids.discard(job_id)
            if error is None:
                pending_batch.append((job_id, result_ref))
            else:
                with db.session_scope() as session:
                    repo = RawCallRepository(session)
                    repo.fail_orchestration_job(job_id, error_json={"error": error})

        if pending_batch and (
            len(pending_batch) >= bundle_size
            or (not snapshots and not in_flight_job_ids)
        ):
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                repo.complete_orchestration_jobs_batch(pending_batch)
            pending_batch = []

        if in_flight_job_ids and not drained_any and not snapshots:
            time.sleep(0.5)

    reducer = JobReducerService(db)
    while True:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            jobs = repo.list_orchestration_jobs(
                request_group_id=request_id,
                status="ready",
            )
            ready_reduce_finalize = [
                j for j in jobs if j.job_type in ("reduce_k", "finalize_run")
            ]
        if not ready_reduce_finalize:
            break
        for job in ready_reduce_finalize:
            if job.job_type == "reduce_k":
                reducer.reduce_k_job(job.id)
            else:
                reducer.finalize_run_job(job.id)

    for _ in range(worker_count):
        job_queue.put(None)
    join_timeout = 30
    deadline = time.time() + join_timeout
    for p in workers:
        remaining = max(1, int(deadline - time.time()))
        p.join(timeout=remaining)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()


def build_cached_supervisor_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cached-job supervisor: single DB client, queue workers, in-process reduce/finalize"
    )
    parser.add_argument("--request-id", type=int, required=True, help="Sweep request id")
    parser.add_argument(
        "--worker-count",
        type=int,
        default=32,
        help="Number of worker processes (cap e.g. 100)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Embedding engine name (e.g. Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--provider-label",
        type=str,
        default="local_docker_tei_shared",
        help="Provider label for metadata",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=None,
        help="Remote embedding provider (e.g. azure, openai) when not using TEI",
    )
    parser.add_argument(
        "--tei-endpoint",
        type=str,
        default=None,
        help="Shared TEI endpoint URL (e.g. http://localhost:8080/v1)",
    )
    parser.add_argument("--lease-seconds", type=int, default=3600)
    parser.add_argument("--idle-exit-seconds", type=int, default=90)
    parser.add_argument(
        "--bundle-size",
        type=int,
        default=50,
        help="Max jobs to claim per batch",
    )
    return parser


def main_cached_job_supervisor(argv: Optional[list[str]] = None) -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    parser = build_cached_supervisor_arg_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[4]
    _run_cached_job_supervisor(
        request_id=args.request_id,
        worker_count=min(args.worker_count, 100),
        engine=args.engine,
        provider_label=args.provider_label,
        embedding_provider_name=args.embedding_provider,
        tei_endpoint=args.tei_endpoint,
        lease_seconds=args.lease_seconds,
        idle_exit_seconds=args.idle_exit_seconds,
        bundle_size=args.bundle_size,
        repo_root=repo_root,
        database_url=database_url,
    )
    return 0


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


def _spawn_engine_workers(
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
    try:
        client = docker.from_env()
        containers = client.containers.list(all=True)
    except Exception as exc:
        print(f"  [cleanup] warning: cannot list docker containers: {exc}")
        return

    for c in containers:
        names = [n.lstrip("/") for n in (getattr(c, "attrs", {}).get("Name", ""),) if n]
        if not names:
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
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    health_url = base + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_seconds) as resp:
            return 200 <= int(resp.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def build_engine_supervisor_arg_parser() -> argparse.ArgumentParser:
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
    return parser


def main_engine_supervisor(argv: Optional[list[str]] = None) -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required")

    parser = build_engine_supervisor_arg_parser()
    args = parser.parse_args(argv)

    db = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=default_write_intent_for_connection(database_url),
    )
    db.init_db()
    repo_root = Path(__file__).resolve().parents[4]

    engines = list(EMBEDDING_ENGINES)
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

            workers = _spawn_engine_workers(
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
                engine_missing, total_missing = mode.engine_work_remaining(
                    db, args.request_id, engine
                )
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
                    workers = _spawn_engine_workers(
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

                for idx, (worker_id, proc) in enumerate(list(workers)):
                    code = proc.poll()
                    if code is None:
                        continue
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

                if workers and all(proc.poll() is not None for _, proc in workers):
                    if all(
                        restart_counts.get(wid, 0) >= args.max_worker_restarts
                        for wid, _ in workers
                    ):
                        raise RuntimeError(
                            f"All workers exited and restart budget exhausted for engine {engine}"
                        )
        finally:
            _stop_workers(workers)
            manager.stop()
            print("  [tei] stopped")

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
    return 0

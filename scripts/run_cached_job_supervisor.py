#!/usr/bin/env python3
"""Cached-job supervisor: single DB client, batch claim/complete, queue-based workers.

For one (request_id, engine), fetches bundles of run_k_try jobs, distributes via
multiprocessing queues to N workers, batch-completes results (repository performs
promote), then runs reduce_k and finalize_run in-process until request fulfilled.
Workers use DB only for read-only L3/L2 embedding cache.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.jobs import JobReducerService

from scripts.run_local_300_2datasets_worker import worker_main_queued

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")


def _run_cached_supervisor(
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
    db = DatabaseConnectionV2(database_url, enable_pgvector=True)
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

    # Run_k_try phase
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

        # Drain result_queue (non-blocking)
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

        # Batch complete when we have enough or queue is empty and no new claims
        if pending_batch and (
            len(pending_batch) >= bundle_size
            or (not snapshots and not in_flight_job_ids)
        ):
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                repo.complete_orchestration_jobs_batch(pending_batch)
            pending_batch = []

        # Avoid busy-wait when workers are still processing
        if in_flight_job_ids and not drained_any and not snapshots:
            time.sleep(0.5)

    # Reduce / finalize phase
    reducer = JobReducerService(db)
    while True:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            jobs = repo.list_orchestration_jobs(
                request_group_id=request_id,
                status="ready",
            )
            ready_reduce_finalize = [
                j for j in jobs
                if j.job_type in ("reduce_k", "finalize_run")
            ]
        if not ready_reduce_finalize:
            break
        for job in ready_reduce_finalize:
            if job.job_type == "reduce_k":
                reducer.reduce_k_job(job.id)
            else:
                reducer.finalize_run_job(job.id)

    # Shutdown: poison pills
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


def main() -> None:
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
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    _run_cached_supervisor(
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
        database_url=DATABASE_URL,
    )


if __name__ == "__main__":
    main()

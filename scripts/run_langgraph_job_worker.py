#!/usr/bin/env python3
"""Worker for langgraph_run orchestration jobs.

Claims langgraph_run jobs from the DB, runs them via LangGraphJobRunner,
and updates job status (complete/fail). Uses same claim/lease semantics
as sweep workers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.jobs import JobRunContext, create_job_runner
from study_query_llm.services.langgraph_provenance import record_langgraph_job_outcome
from study_query_llm.services.method_service import MethodService

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

db = DatabaseConnectionV2(DATABASE_URL, enable_pgvector=True)
db.init_db()


def _claim_next_langgraph_job(
    *,
    request_group_id: int,
    worker_id: str,
    lease_seconds: int,
):
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        job = repo.claim_next_orchestration_job(
            worker_id=worker_id,
            lease_seconds=lease_seconds,
            request_group_id=request_group_id,
            job_types=["langgraph_run"],
        )
        if not job:
            return None
        return {
            "id": int(job.id),
            "job_type": str(job.job_type),
            "payload_json": dict(job.payload_json or {}),
            "job_key": str(job.job_key),
            "base_run_key": job.base_run_key,
            "seed_value": job.seed_value,
            "request_group_id": int(job.request_group_id),
            "attempt_count": int(job.attempt_count or 0),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker for langgraph_run jobs")
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--worker-id", type=str, default="langgraph-worker-1")
    parser.add_argument("--claim-lease-seconds", type=int, default=3600)
    parser.add_argument("--idle-exit-seconds", type=int, default=60)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    worker_id = args.worker_id
    completed = 0
    started_at = time.time()
    last_work_at = started_at

    while True:
        if args.max_runs is not None and completed >= args.max_runs:
            break

        job = _claim_next_langgraph_job(
            request_group_id=args.request_id,
            worker_id=worker_id,
            lease_seconds=args.claim_lease_seconds,
        )
        if not job:
            idle_for = int(time.time() - last_work_at)
            if idle_for >= args.idle_exit_seconds:
                print(f"[{worker_id}] Idle exit after {idle_for}s")
                break
            time.sleep(5)
            continue

        job_id = int(job["id"])
        job_key = str(job.get("job_key"))
        try:
            runner = create_job_runner("langgraph_run")
            context = JobRunContext(
                datasets={},
                provider_cache={},
                manager_cache={},
                tei_endpoint=None,
                provider_label="langgraph",
                embedding_provider_name=None,
                worker_slot=0,
                repo_root=repo_root,
                claim_wait_seconds=0.0,
                reducer=None,
                db=db,
            )
            outcome = runner.run(job, context)
            if outcome.error is not None:
                with db.session_scope() as session:
                    repo = RawCallRepository(session)
                    repo.fail_orchestration_job(
                        outcome.job_id, error_json={"error": outcome.error}
                    )
                    method_svc = MethodService(repo)
                    record_langgraph_job_outcome(
                        method_svc=method_svc,
                        request_group_id=job["request_group_id"],
                        job_id=outcome.job_id,
                        job_key=job_key,
                        payload_json=job["payload_json"],
                        status="failed",
                        error=outcome.error,
                    )
                print(f"[{worker_id}] JOB ERROR {job_key}: {outcome.error}")
            else:
                with db.session_scope() as session:
                    repo = RawCallRepository(session)
                    repo.complete_orchestration_job(
                        outcome.job_id, result_ref=outcome.result_ref
                    )
                    method_svc = MethodService(repo)
                    record_langgraph_job_outcome(
                        method_svc=method_svc,
                        request_group_id=job["request_group_id"],
                        job_id=outcome.job_id,
                        job_key=job_key,
                        payload_json=job["payload_json"],
                        status="completed",
                        result_ref=outcome.result_ref,
                        checkpoint_refs=(
                            outcome.metadata.get("checkpoint_refs")
                            if outcome.metadata
                            else None
                        ),
                    )
                completed += 1
                print(f"[{worker_id}] DONE job {job_key} -> {outcome.result_ref}")
            last_work_at = time.time()
        except Exception as exc:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                repo.fail_orchestration_job(
                    job_id, error_json={"error": str(exc)[:1000]}
                )
                method_svc = MethodService(repo)
                record_langgraph_job_outcome(
                    method_svc=method_svc,
                    request_group_id=job["request_group_id"],
                    job_id=job_id,
                    job_key=job_key,
                    payload_json=job["payload_json"],
                    status="failed",
                    error=str(exc)[:1000],
                )
            print(f"[{worker_id}] JOB ERROR {job_key}: {exc}")

    print(
        f"[{worker_id}] Finished. completed={completed}, "
        f"elapsed={int(time.time() - started_at)}s"
    )


if __name__ == "__main__":
    main()

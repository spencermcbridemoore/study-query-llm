#!/usr/bin/env python3
"""Reconcile and finalize one partial sweep request.

Workflow:
1) Resolve target request (explicit --request-id or newest matching heuristics)
2) Backfill missing request->run contains links for already completed run_keys
3) Optionally ingest PKL artifacts before re-checking progress
4) Finalize request if fulfilled
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import text as sa_text

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.sweep_query_service import SweepQueryService
from study_query_llm.services.sweep_request_service import SweepRequestService


def _select_candidate_request(
    sweep_query: SweepQueryService,
    expected_min: int,
    completed_min: int,
    completed_max: int,
) -> Optional[int]:
    requests = sweep_query.list_clustering_sweep_requests(include_fulfilled=False)
    candidates = []
    for req in requests:
        rid = req["id"]
        summary = sweep_query.get_request_progress_summary(rid)
        if not summary:
            continue
        expected = int(summary.get("expected_count", 0))
        completed = int(summary.get("completed_count", 0))
        if expected >= expected_min and completed_min <= completed <= completed_max:
            candidates.append((req.get("created_at"), rid))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return int(candidates[0][1])


def _run_id_by_run_key(session, run_key: str) -> Optional[int]:
    run = (
        session.query(Group)
        .filter(
            Group.group_type == "clustering_run",
            sa_text("metadata_json->>'run_key' = :rk"),
        )
        .params(rk=run_key)
        .first()
    )
    return int(run.id) if run else None


def _optional_ingest_artifacts(data_dir: Path) -> None:
    """Run existing ingestion script to import pkl artifacts (broad ingestion)."""
    cmd = [sys.executable, "scripts/ingest_sweep_to_db.py", "--data-dir", str(data_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("[INGEST] scripts/ingest_sweep_to_db.py")
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        raise RuntimeError("Artifact ingestion failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile one partial sweep request and finalize if fulfilled",
    )
    parser.add_argument("--request-id", type=int, default=None)
    parser.add_argument("--expected-min", type=int, default=100)
    parser.add_argument("--completed-min", type=int, default=1)
    parser.add_argument("--completed-max", type=int, default=25)
    parser.add_argument(
        "--ingest-artifacts",
        action="store_true",
        help="Run scripts/ingest_sweep_to_db.py before final progress check",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="experimental_results",
        help="Directory passed to ingest_sweep_to_db.py when --ingest-artifacts is set",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        sweep_query = SweepQueryService(repo)
        sweep_req = SweepRequestService(repo)

        request_id = args.request_id
        if request_id is None:
            request_id = _select_candidate_request(
                sweep_query,
                expected_min=args.expected_min,
                completed_min=args.completed_min,
                completed_max=args.completed_max,
            )
        if request_id is None:
            print("No matching partial request found.")
            return

        request = sweep_req.get_request(request_id)
        if not request:
            print(f"Request not found: {request_id}")
            return

        print(f"Target request: id={request_id} name={request.get('name', '?')}")
        progress = sweep_req.compute_progress(request_id)
        print(
            "Before reconciliation: "
            f"expected={progress['expected_count']} "
            f"completed={progress['completed_count']} "
            f"missing={progress['missing_count']}"
        )

        # Backfill request->run links for already-completed run_keys.
        backfilled = 0
        for run_key in progress.get("completed_run_keys", []):
            run_id = _run_id_by_run_key(session, run_key)
            if run_id is None:
                continue
            if not args.dry_run:
                sweep_req.record_delivery(request_id, run_id, run_key)
            backfilled += 1
        print(f"Backfilled/confirmed request->run links: {backfilled}")

    if args.ingest_artifacts and not args.dry_run:
        _optional_ingest_artifacts(Path(args.artifacts_dir))

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        sweep_req = SweepRequestService(repo)
        progress = sweep_req.compute_progress(request_id)
        print(
            "After reconciliation: "
            f"expected={progress['expected_count']} "
            f"completed={progress['completed_count']} "
            f"missing={progress['missing_count']}"
        )

        if args.dry_run:
            print("[DRY RUN] Skipping finalize_if_fulfilled")
            return

        sweep_id = sweep_req.finalize_if_fulfilled(
            request_id,
            sweep_name=f"{(sweep_req.get_request(request_id) or {}).get('name', 'request')}_sweep",
        )
        if sweep_id:
            print(f"Finalized request {request_id} -> clustering_sweep {sweep_id}")
        else:
            print(f"Request {request_id} not fulfilled yet; finalize skipped.")


if __name__ == "__main__":
    main()

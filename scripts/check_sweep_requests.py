#!/usr/bin/env python3
"""Check sweep request status, pending deliveries, and migration candidate.

Prints request id/name/status, expected/completed/missing counts,
and first N missing run keys.

Usage:
  python scripts/check_sweep_requests.py
  python scripts/check_sweep_requests.py --request-id 123
  python scripts/check_sweep_requests.py --pending-only
  python scripts/check_sweep_requests.py --select-last-partial
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.services.sweep_query_service import SweepQueryService


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Check sweep request status")
    parser.add_argument(
        "--request-id",
        type=int,
        default=None,
        help="Show progress for a specific request ID",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="List only non-fulfilled requests",
    )
    parser.add_argument(
        "--missing-limit",
        type=int,
        default=10,
        help="Max missing run keys to print per request (default 10)",
    )
    parser.add_argument(
        "--select-last-partial",
        action="store_true",
        help=(
            "Select the newest request matching expected_count >= expected_min "
            "and completed_count in [completed_min, completed_max]"
        ),
    )
    parser.add_argument(
        "--expected-min",
        type=int,
        default=100,
        help="Minimum expected_count for --select-last-partial (default 100)",
    )
    parser.add_argument(
        "--completed-min",
        type=int,
        default=1,
        help="Minimum completed_count for --select-last-partial (default 1)",
    )
    parser.add_argument(
        "--completed-max",
        type=int,
        default=25,
        help="Maximum completed_count for --select-last-partial (default 25)",
    )
    args = parser.parse_args()

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepQueryService(repo)

        if args.request_id is not None:
            summary = svc.get_request_progress_summary(args.request_id)
            if not summary:
                print(f"Request {args.request_id} not found.")
                sys.exit(1)
            print(f"Request: {summary.get('request_name', '?')} (id={args.request_id})")
            print(f"  Status: {summary.get('request_status', '?')}")
            print(f"  Expected: {summary.get('expected_count', 0)}")
            print(f"  Completed: {summary.get('completed_count', 0)}")
            print(f"  Missing: {summary.get('missing_count', 0)}")
            missing = summary.get("missing_run_keys", [])
            preview = summary.get("missing_run_keys_preview", missing[: args.missing_limit])
            if preview:
                print(f"  First {len(preview)} missing run_keys:")
                for rk in preview:
                    print(f"    - {rk}")
            elif missing:
                print(f"  Missing run_keys (first {args.missing_limit}):")
                for rk in missing[: args.missing_limit]:
                    print(f"    - {rk}")
            return

        if args.select_last_partial:
            candidates = []
            requests = svc.list_clustering_sweep_requests(include_fulfilled=False)
            for r in requests:
                rid = r["id"]
                progress = svc.get_request_progress_summary(rid)
                if not progress:
                    continue
                expected = progress.get("expected_count", 0)
                completed = progress.get("completed_count", 0)
                if (
                    expected >= args.expected_min
                    and args.completed_min <= completed <= args.completed_max
                ):
                    candidates.append(
                        {
                            "id": rid,
                            "name": r.get("name", "?"),
                            "created_at": r.get("created_at"),
                            "expected": expected,
                            "completed": completed,
                            "missing": progress.get("missing_count", 0),
                        }
                    )

            if not candidates:
                print("No matching partial request found.")
                return

            candidates.sort(key=lambda x: x["created_at"], reverse=True)
            chosen = candidates[0]
            print("Selected request candidate:")
            print(f"  id={chosen['id']}")
            print(f"  name={chosen['name']}")
            print(f"  expected={chosen['expected']}")
            print(f"  completed={chosen['completed']}")
            print(f"  missing={chosen['missing']}")
            if len(candidates) > 1:
                print(f"  (Matched {len(candidates)} candidates; selected newest by created_at)")
            return

        requests = svc.list_clustering_sweep_requests(include_fulfilled=not args.pending_only)
        if not requests:
            print("No sweep requests found.")
            return

        print(f"Found {len(requests)} sweep request(s):\n")
        for r in requests:
            rid = r["id"]
            progress = svc.get_request_progress_summary(rid)
            if not progress:
                continue
            status = progress.get("request_status", "?")
            exp = progress.get("expected_count", 0)
            done = progress.get("completed_count", 0)
            miss = progress.get("missing_count", 0)
            print(f"  [{rid}] {r.get('name', '?')}")
            print(f"      status={status}  expected={exp}  completed={done}  missing={miss}")
            if miss > 0:
                missing = progress.get("missing_run_keys", [])[: args.missing_limit]
                for rk in missing:
                    print(f"        - {rk}")
                if miss > args.missing_limit:
                    print(f"        ... and {miss - args.missing_limit} more")
            print()


if __name__ == "__main__":
    main()

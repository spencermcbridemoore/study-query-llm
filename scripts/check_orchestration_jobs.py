#!/usr/bin/env python3
"""Summarize orchestration jobs for one sweep request group."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(description="Check orchestration job status for a request group")
    parser.add_argument(
        "--request-id",
        type=int,
        required=True,
        help="Request group id (groups.id for the sweep request)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent jobs to print (default: 20)",
    )
    args = parser.parse_args()

    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        rid = int(args.request_id)
        summary_rows = session.execute(
            text(
                """
                SELECT job_type, status, COUNT(*)::bigint AS n
                FROM orchestration_jobs
                WHERE request_group_id = :rid
                GROUP BY job_type, status
                ORDER BY job_type, status
                """
            ),
            {"rid": rid},
        ).fetchall()

        if not summary_rows:
            print(f"No orchestration jobs found for request_id={rid}.")
            return 0

        total = sum(int(r[2]) for r in summary_rows)
        print(f"Request {rid}: total orchestration_jobs={total}")
        print("\nBy job_type/status:")
        for job_type, status, count in summary_rows:
            print(f"  {job_type:18s} {status:10s} {int(count)}")

        recent_rows = session.execute(
            text(
                """
                SELECT id, job_type, status, attempt_count, max_attempts, claimed_by
                FROM orchestration_jobs
                WHERE request_group_id = :rid
                ORDER BY id DESC
                LIMIT :lim
                """
            ),
            {"rid": rid, "lim": int(args.limit)},
        ).fetchall()

        print(f"\nRecent {len(recent_rows)} jobs:")
        for row in recent_rows:
            jid, job_type, status, attempts, max_attempts, claimed_by = row
            claimer = claimed_by or "-"
            print(
                f"  id={int(jid):6d} type={job_type:18s} status={status:10s} "
                f"attempts={int(attempts)}/{int(max_attempts)} claimed_by={claimer}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

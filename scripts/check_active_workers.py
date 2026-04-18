#!/usr/bin/env python3
"""Fail fast when orchestration jobs are still active."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled"}


def list_active_jobs(session: Session) -> list[dict[str, Any]]:
    from study_query_llm.db.models_v2 import OrchestrationJob

    rows = (
        session.query(OrchestrationJob)
        .filter(~OrchestrationJob.status.in_(TERMINAL_JOB_STATUSES))
        .order_by(OrchestrationJob.id.asc())
        .all()
    )
    return [
        {
            "id": int(row.id),
            "request_group_id": int(row.request_group_id),
            "job_type": str(row.job_type),
            "job_key": str(row.job_key),
            "status": str(row.status),
            "claimed_by": str(row.claimed_by or ""),
        }
        for row in rows
    ]


def _resolve_database_url(cli_value: str | None) -> str:
    if cli_value and cli_value.strip():
        return cli_value.strip()
    return (
        os.environ.get("LOCAL_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or ""
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check orchestration_jobs for non-terminal worker activity.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Postgres URL (default: LOCAL_DATABASE_URL, then DATABASE_URL from .env)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Max active job rows to print (default: 20).",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    db_url = _resolve_database_url(args.database_url)
    if not db_url:
        print(
            "ERROR: Set LOCAL_DATABASE_URL or DATABASE_URL in .env, or pass --database-url.",
            file=sys.stderr,
        )
        return 1

    engine = create_engine(db_url, pool_pre_ping=True)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    try:
        active_jobs = list_active_jobs(session)
    finally:
        session.close()
        engine.dispose()

    if not active_jobs:
        print("No active orchestration jobs found.")
        return 0

    total = len(active_jobs)
    print(f"Active orchestration jobs detected: {total}")
    sample = active_jobs[: max(0, int(args.sample_limit))]
    for row in sample:
        claimed_by = row["claimed_by"] or "-"
        print(
            "  id={id} request_group_id={request_group_id} job_type={job_type} "
            "job_key={job_key} status={status} claimed_by={claimed_by}".format(
                id=row["id"],
                request_group_id=row["request_group_id"],
                job_type=row["job_type"],
                job_key=row["job_key"],
                status=row["status"],
                claimed_by=claimed_by,
            )
        )
    if total > len(sample):
        print(f"  ... {total - len(sample)} additional active jobs not shown")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

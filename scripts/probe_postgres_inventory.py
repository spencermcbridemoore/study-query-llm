#!/usr/bin/env python3
"""Print public-schema inventory for a Postgres URL (no secrets in output).

Usage (repo root):
  python scripts/probe_postgres_inventory.py
  python scripts/probe_postgres_inventory.py --env-var NEON_DATABASE_URL
  python scripts/probe_postgres_inventory.py --database-url postgresql://...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        try:
            rest = url.split("@", 1)[-1]
            scheme = url.split("://", 1)[0]
            return f"{scheme}://***@{rest[:160]}"
        except Exception:
            return "***"
    return url[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Postgres public schema inventory.")
    parser.add_argument(
        "--env-var",
        default="JETSTREAM_DATABASE_URL",
        help="Environment variable holding the URL (default: JETSTREAM_DATABASE_URL)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override URL (instead of --env-var)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="TCP connect timeout seconds",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    url = (args.database_url or "").strip() or (os.environ.get(args.env_var) or "").strip()
    if not url:
        print(f"ERROR: {args.env_var} not set and --database-url not passed.", file=sys.stderr)
        return 1

    print("url_redacted:", _redact(url))
    eng = create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": int(args.connect_timeout)},
    )
    try:
        with eng.connect() as c:
            dbsize = c.execute(
                text("SELECT pg_size_pretty(pg_database_size(current_database()))")
            ).scalar()
            print("pg_database_size:", dbsize)

            tabs = c.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                    "ORDER BY 1"
                )
            ).fetchall()
            names = [r[0] for r in tabs]
            print("public_table_count:", len(names))
            print("public_tables:", ", ".join(names))

            if "groups" in names:
                rows = c.execute(
                    text(
                        "SELECT group_type, COUNT(*) AS n FROM groups "
                        "GROUP BY 1 ORDER BY n DESC"
                    )
                ).fetchall()
                print("groups_by_type:", rows)
            else:
                print("groups_by_type: (no groups table)")

            if "raw_calls" in names:
                n = c.execute(text("SELECT COUNT(*) FROM raw_calls")).scalar()
                print("raw_calls_count:", n)
            else:
                print("raw_calls_count: (no raw_calls table)")

            if "group_members" in names and "groups" in names:
                n = c.execute(
                    text(
                        "SELECT COUNT(*) FROM group_members gm "
                        "JOIN groups g ON g.id = gm.group_id "
                        "WHERE g.group_type = 'mcq_run'"
                    )
                ).scalar()
                print("group_members_on_mcq_run_groups:", n)
    finally:
        eng.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

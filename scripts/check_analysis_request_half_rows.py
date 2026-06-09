#!/usr/bin/env python3
"""Read-only probe: flag ``analysis_request`` groups with a PARTIAL identity.

A "half row" has some but not all of ``(method_name, input_id, run_key)`` present.
The two legitimate shapes are identity-bearing (all three present) and container
(all three absent); a half row should never exist -- no writer produces one -- and
would silently escape the NULL-distinct unique index
``uq_groups_analysis_request_identity``. Fork B (plan 2.2): detect now, enforce
only if one ever appears.

Read-only: issues a single SELECT built from the shared identity contract
(``db/analysis_request_identity.build_half_row_probe_sql``); no writes. Exits 1 if
any half row is found, 0 if clean. CI cannot reach the canonical DB, so this is an
*operator* probe (run it against canonical over the tunnel); the detector logic is
covered by ``tests/test_db/test_analysis_request_identity.py``.

Usage:
  python scripts/check_analysis_request_half_rows.py                      # JETSTREAM_DATABASE_URL
  python scripts/check_analysis_request_half_rows.py --env-var LOCAL_DATABASE_URL
  python scripts/check_analysis_request_half_rows.py --database-url postgresql://...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from study_query_llm.db.analysis_request_identity import (  # noqa: E402
    build_half_row_probe_sql,
)


def _redact(url: str) -> str:
    if "@" in url and "://" in url:
        try:
            scheme = url.split("://", 1)[0]
            rest = url.split("@", 1)[-1]
            return f"{scheme}://***@{rest[:160]}"
        except Exception:
            return "***"
    return url[:80]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flag analysis_request half-identity rows on a Postgres/SQLite URL (read-only)."
    )
    parser.add_argument(
        "--env-var",
        default="JETSTREAM_DATABASE_URL",
        help="Environment variable holding the target URL (default: JETSTREAM_DATABASE_URL).",
    )
    parser.add_argument("--database-url", default=None, help="Override URL (instead of --env-var).")
    parser.add_argument("--connect-timeout", type=int, default=15, help="TCP connect timeout seconds.")
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    url = (args.database_url or "").strip() or (os.environ.get(args.env_var) or "").strip()
    if not url:
        print(f"ERROR: {args.env_var} not set and --database-url not passed.", file=sys.stderr)
        return 2

    print("url_redacted:", _redact(url))
    engine = create_engine(
        url, pool_pre_ping=True, connect_args={"connect_timeout": int(args.connect_timeout)}
    )
    try:
        with engine.connect() as conn:
            dialect = str(conn.engine.dialect.name)
            rows = conn.execute(text(build_half_row_probe_sql(dialect))).mappings().all()
    finally:
        engine.dispose()

    print("half_identity_rows:", len(rows))
    for row in rows[:20]:
        print("  ", dict(row))
    if rows:
        print(
            "FAIL: analysis_request half-identity rows found (expected 0). Each has some "
            "but not all of (method_name, input_id, run_key) and escapes the unique index.",
            file=sys.stderr,
        )
        return 1
    print("OK: no analysis_request half-identity rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

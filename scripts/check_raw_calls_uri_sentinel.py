#!/usr/bin/env python3
"""Inspect non-blob URI anomalies in raw_calls.response_json['uri']."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_NAME = "idx_raw_calls_uri_non_blob_sentinel"
_BLOB_URI_REGEX = r"^https://[^/]+\.blob\.core\.windows\.net/.+"


def _validate_index_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", candidate):
        raise ValueError(f"Invalid index name: {name!r}")
    return candidate


def _resolve_database_url(*, explicit_url: str | None, env_var: str) -> str:
    if explicit_url and explicit_url.strip():
        return explicit_url.strip()
    for key in (env_var, "DATABASE_URL", "CANONICAL_DATABASE_URL", "JETSTREAM_DATABASE_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    raise ValueError(
        f"Database URL not set. Provide --database-url or set {env_var}/DATABASE_URL."
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Check sentinel index and non-blob raw_calls.response_json uri inventory.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional explicit SQLAlchemy URL.",
    )
    parser.add_argument(
        "--env-var",
        type=str,
        default="CANONICAL_DATABASE_URL",
        help="Primary env var used for URL resolution.",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Sentinel index name to inspect.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Maximum anomalous rows to print.",
    )
    parser.add_argument(
        "--require-zero",
        action="store_true",
        help="Return non-zero if anomalous rows are present.",
    )
    args = parser.parse_args()

    try:
        url = _resolve_database_url(explicit_url=args.database_url, env_var=args.env_var)
        index_name = _validate_index_name(args.index_name)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        dialect = str(conn.dialect.name).lower()
        print(f"dialect={dialect}")
        if dialect != "postgresql":
            print("ERROR: script is PostgreSQL-only.", file=sys.stderr)
            return 2

        index_exists = conn.execute(
            text(
                """
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'raw_calls'
                  AND indexname = :index_name
                LIMIT 1
                """
            ),
            {"index_name": index_name},
        ).scalar() is not None
        print(f"index_exists={str(index_exists).lower()}")

        count = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM raw_calls
                WHERE (response_json::jsonb ? 'uri')
                  AND COALESCE(response_json::jsonb ->> 'uri', '') <> ''
                  AND (response_json::jsonb ->> 'uri') !~* :blob_regex
                """
            ),
            {"blob_regex": _BLOB_URI_REGEX},
        ).scalar()
        anomaly_count = int(count or 0)
        print(f"non_blob_uri_rows={anomaly_count}")

        rows = conn.execute(
            text(
                """
                SELECT id, response_json::jsonb ->> 'uri' AS uri
                FROM raw_calls
                WHERE (response_json::jsonb ? 'uri')
                  AND COALESCE(response_json::jsonb ->> 'uri', '') <> ''
                  AND (response_json::jsonb ->> 'uri') !~* :blob_regex
                ORDER BY id
                LIMIT :limit
                """
            ),
            {"blob_regex": _BLOB_URI_REGEX, "limit": int(args.sample_limit)},
        ).fetchall()
        for row in rows:
            print(f"sample id={int(row[0])} uri={str(row[1])[:180]!r}")

    if args.require_zero and anomaly_count > 0:
        print("ERROR: non-blob raw_calls URI anomalies detected.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

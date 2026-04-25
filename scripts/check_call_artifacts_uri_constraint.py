#!/usr/bin/env python3
"""Inspect and probe call_artifacts URI CHECK constraint state."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _resolve_database_url(explicit_url: str | None, env_var: str) -> str:
    if explicit_url:
        return explicit_url.strip()
    from_env = (os.environ.get(env_var) or "").strip()
    if from_env:
        return from_env
    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    if database_url:
        return database_url
    fallback = (os.environ.get("JETSTREAM_DATABASE_URL") or "").strip()
    if fallback:
        return fallback
    raise ValueError(
        f"Database URL not provided. Set --database-url, {env_var}, or JETSTREAM_DATABASE_URL."
    )


def _constraint_row(conn, constraint_name: str):
    query = text(
        """
        SELECT c.conname, c.convalidated, pg_get_constraintdef(c.oid) AS definition
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'call_artifacts'
          AND c.conname = :constraint_name
        LIMIT 1
        """
    )
    return conn.execute(query, {"constraint_name": constraint_name}).mappings().first()


def _probe_new_insert_rejection(conn) -> tuple[bool, str]:
    conn.rollback()
    call_row = conn.execute(
        text("SELECT call_id FROM call_artifacts ORDER BY id DESC LIMIT 1")
    ).fetchone()
    if call_row is None:
        return (False, "probe skipped: no existing call_id to reference")

    try:
        conn.execute(
            text(
                """
                INSERT INTO call_artifacts
                    (call_id, artifact_type, uri, content_type, byte_size, metadata_json)
                VALUES
                    (:call_id, 'constraint_probe', 'C:/tmp/local.txt', 'text/plain', 1, '{}'::jsonb)
                """
            ),
            {"call_id": int(call_row[0])},
        )
        conn.rollback()
        return (False, "probe insert unexpectedly succeeded")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        return (True, str(exc).replace("\n", " ")[:280])


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Check call_artifacts URI CHECK constraint state."
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Explicit database URL (optional).",
    )
    parser.add_argument(
        "--env-var",
        type=str,
        default="CANONICAL_DATABASE_URL",
        help="Primary env var used when --database-url is omitted.",
    )
    parser.add_argument(
        "--constraint-name",
        type=str,
        default="call_artifacts_uri_must_be_blob",
        help="Constraint name to inspect.",
    )
    parser.add_argument(
        "--probe-insert",
        action="store_true",
        help="Attempt a local-path URI insert in a rollback-only transaction.",
    )
    args = parser.parse_args()

    try:
        url = _resolve_database_url(args.database_url, args.env_var)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        dialect = str(conn.dialect.name).lower()
        print(f"dialect={dialect}")
        if dialect != "postgresql":
            print("constraint_check_skipped=non_postgres")
            return 0

        row = _constraint_row(conn, args.constraint_name)
        if row is None:
            print("constraint_exists=false")
            return 1

        print("constraint_exists=true")
        print(f"constraint_name={row['conname']}")
        print(f"constraint_convalidated={bool(row['convalidated'])}")
        print(f"constraint_definition={row['definition']}")

        if args.probe_insert:
            rejected, detail = _probe_new_insert_rejection(conn)
            print(f"probe_insert_rejected={str(rejected).lower()}")
            print(f"probe_detail={detail}")
            if not rejected:
                return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

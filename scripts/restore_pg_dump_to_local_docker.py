#!/usr/bin/env python3
"""
Restore a pg_dump custom-format (.dump) file into the local Docker Postgres from docker-compose.yml.

Typical use: after `python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream`,
restore Jetstream's data into `study_query_local` so local matches Jetstream.

Prerequisites:
  - PostgreSQL client tools on PATH: pg_restore, dropdb, createdb (same major as server, ideally 17).
  - Local `db` container running: docker compose --profile postgres up -d db
  - Close other connections to the target DB (Panel, IDEs, other scripts).

Usage:
  python scripts/restore_pg_dump_to_local_docker.py pg_migration_dumps/jetstream_for_local_....dump
  python scripts/restore_pg_dump_to_local_docker.py path/to/file.dump --dry-run

Uses LOCAL_DATABASE_URL from .env by default (database name, user, password, host, port).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

from db_target_guardrails import is_loopback_target, parse_postgres_target, redact_database_url


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _pg_env(url: str) -> dict[str, str]:
    """Build env with PGPASSWORD for subprocesses."""
    p = urlparse(url)
    pwd = unquote(p.password) if p.password else ""
    env = os.environ.copy()
    if pwd:
        env["PGPASSWORD"] = pwd
    return env


def _connection_uri_for_cli(url: str) -> str:
    """URI string suitable for pg_restore -d."""
    return url.strip()


def main() -> int:
    load_dotenv(_repo_root() / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Restore custom-format pg_dump into local Docker Postgres (LOCAL_DATABASE_URL).",
    )
    parser.add_argument(
        "dump_path",
        type=Path,
        help="Path to .dump file (custom format from pg_dump -Fc)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Target Postgres URL (default: LOCAL_DATABASE_URL from .env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    parser.add_argument(
        "--skip-recreate",
        action="store_true",
        help="Skip dropdb/createdb; only run pg_restore --clean --if-exists (DB must exist)",
    )
    parser.add_argument(
        "--allow-remote-target",
        action="store_true",
        help="Allow restore to non-loopback host (requires --confirm-target-db).",
    )
    parser.add_argument(
        "--confirm-target-db",
        default=None,
        help="Exact target DB name confirmation when using --allow-remote-target.",
    )
    args = parser.parse_args()

    target = args.database_url or os.environ.get("LOCAL_DATABASE_URL") or ""
    if not target.strip():
        print("ERROR: Set LOCAL_DATABASE_URL in .env or pass --database-url.", file=sys.stderr)
        return 1

    dump_path = args.dump_path.resolve()
    if not dump_path.is_file():
        print(f"ERROR: Dump file not found: {dump_path}", file=sys.stderr)
        return 1

    try:
        target_info = parse_postgres_target(target)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    dbname = target_info.dbname
    user = target_info.username
    host = target_info.host or "localhost"
    port = str(target_info.port)
    if not user:
        print("ERROR: Database URL must include a username.", file=sys.stderr)
        return 1

    if not is_loopback_target(target):
        if not args.allow_remote_target:
            print(
                "ERROR: Target URL resolves to a non-loopback host. "
                "Refusing destructive restore without --allow-remote-target.",
                file=sys.stderr,
            )
            return 1
        expected = (args.confirm_target_db or "").strip()
        if expected != dbname:
            print(
                "ERROR: Remote restore requires explicit DB confirmation. "
                f"Re-run with --confirm-target-db {dbname!r}.",
                file=sys.stderr,
            )
            return 1

    env = _pg_env(target)
    uri = _connection_uri_for_cli(target)

    print(f"Target DB: {redact_database_url(target)}", flush=True)
    print(f"Dump file: {dump_path}", flush=True)

    drop_cmd = ["dropdb", "--if-exists", "-h", host, "-p", port, "-U", user, dbname]
    create_cmd = ["createdb", "-h", host, "-p", port, "-U", user, dbname]
    # After drop+create, DB is empty — no --clean. With --skip-recreate, overwrite in place.
    if args.skip_recreate:
        restore_cmd = [
            "pg_restore",
            "-d",
            uri,
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-acl",
            "--verbose",
            str(dump_path),
        ]
    else:
        restore_cmd = [
            "pg_restore",
            "-d",
            uri,
            "--no-owner",
            "--no-acl",
            "--verbose",
            str(dump_path),
        ]

    if args.dry_run:
        if not args.skip_recreate:
            print("DRY RUN dropdb:", subprocess.list2cmdline(drop_cmd), flush=True)
            print("DRY RUN createdb:", subprocess.list2cmdline(create_cmd), flush=True)
        safe_restore = [
            redact_database_url(x) if x.startswith("postgresql:") else x
            for x in restore_cmd
        ]
        print("DRY RUN pg_restore:", subprocess.list2cmdline(safe_restore), flush=True)
        return 0

    for tool in ("pg_restore", "dropdb", "createdb"):
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except FileNotFoundError:
            print(
                f"ERROR: {tool} not found. Install PostgreSQL client tools and add to PATH.",
                file=sys.stderr,
            )
            return 1

    if not args.skip_recreate:
        print("Dropping database (if exists)...", flush=True)
        try:
            subprocess.run(drop_cmd, check=True, env=env)
        except subprocess.CalledProcessError:
            print(
                "ERROR: dropdb failed. Close other connections to this database "
                "(stop Panel, disconnect IDE) and retry.",
                file=sys.stderr,
            )
            return 1
        print("Creating empty database...", flush=True)
        try:
            subprocess.run(create_cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: createdb failed with exit {e.returncode}.", file=sys.stderr)
            return 1

    print("Running pg_restore (warnings about missing extensions are often OK)...", flush=True)
    try:
        r = subprocess.run(restore_cmd, env=env)
        if r.returncode not in (0, 1):
            # pg_restore often returns 1 for non-fatal warnings
            print(f"pg_restore exited with code {r.returncode}.", file=sys.stderr)
            return r.returncode
        if r.returncode == 1:
            print(
                "NOTE: pg_restore returned 1 — often acceptable (e.g. missing Neon-only extensions). "
                "Verify data with scripts/sanity_check_database_url.py.",
                flush=True,
            )
    except FileNotFoundError:
        print("ERROR: pg_restore not found.", file=sys.stderr)
        return 1

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

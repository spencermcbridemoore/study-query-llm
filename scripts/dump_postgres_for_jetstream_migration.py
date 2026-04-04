#!/usr/bin/env python3
"""
Dump a PostgreSQL database (e.g. Neon) to a custom-format file for Jetstream restore.

Requires PostgreSQL client tools on PATH: pg_dump (same major version as server is ideal).

Usage:
  Set SOURCE_DATABASE_URL or DATABASE_URL in .env, then:
    python scripts/dump_postgres_for_jetstream_migration.py

  Or pass the URL explicitly:
    python scripts/dump_postgres_for_jetstream_migration.py \\
        --source-url "postgresql://user:pass@host/db?sslmode=require"

  Local backup before cloning Jetstream → local:
    python scripts/dump_postgres_for_jetstream_migration.py --from-local

  Dump Jetstream (SSH tunnel must be up):
    python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream

Output defaults to pg_migration_dumps/neon_for_jetstream_<timestamp>.dump (gitignored).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv


def _redact_database_url(url: str) -> str:
    """Return URL safe to print (password replaced)."""
    try:
        p = urlparse(url)
        if p.password is None:
            return url
        netloc = p.hostname or ""
        if p.port:
            netloc = f"{netloc}:{p.port}"
        if p.username:
            netloc = f"{p.username}:***@{netloc}"
        else:
            netloc = f"***@{netloc}"
        return urlunparse(
            (p.scheme, netloc, p.path, p.params, p.query, p.fragment)
        )
    except Exception:
        return "***"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(_repo_root() / ".env")
    parser = argparse.ArgumentParser(
        description="pg_dump (custom format) for Neon → Jetstream migration",
    )
    parser.add_argument(
        "--source-url",
        default=None,
        help="PostgreSQL URL to dump (default: SOURCE_DATABASE_URL, else DATABASE_URL)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .dump file path (default: pg_migration_dumps/neon_for_jetstream_<ts>.dump)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pg_dump command without running",
    )
    parser.add_argument(
        "--from-local",
        action="store_true",
        help="Use LOCAL_DATABASE_URL from .env; default output local_pre_jetstream_clone_<ts>.dump",
    )
    parser.add_argument(
        "--from-jetstream",
        action="store_true",
        help="Use JETSTREAM_DATABASE_URL from .env (tunnel required); default output jetstream_for_local_<ts>.dump",
    )
    args = parser.parse_args()

    if args.from_local and args.from_jetstream:
        print("ERROR: Use only one of --from-local or --from-jetstream.", file=sys.stderr)
        return 1

    if args.source_url:
        source = args.source_url
    elif args.from_local:
        source = os.environ.get("LOCAL_DATABASE_URL") or ""
    elif args.from_jetstream:
        source = os.environ.get("JETSTREAM_DATABASE_URL") or ""
    else:
        source = (
            os.environ.get("SOURCE_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
            or ""
        )
    if not source or not str(source).strip():
        print(
            "ERROR: Set SOURCE_DATABASE_URL or DATABASE_URL in .env, pass --source-url, "
            "or use --from-local / --from-jetstream with the matching URL in .env.",
            file=sys.stderr,
        )
        return 1

    out_dir = _repo_root() / "pg_migration_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    if args.output:
        out_path = Path(args.output).resolve()
    elif args.from_local:
        out_path = out_dir / f"local_pre_jetstream_clone_{ts}.dump"
    elif args.from_jetstream:
        out_path = out_dir / f"jetstream_for_local_{ts}.dump"
    else:
        out_path = out_dir / f"neon_for_jetstream_{ts}.dump"

    cmd = [
        "pg_dump",
        "--format=custom",
        "--no-owner",
        "--no-acl",
        "--verbose",
        "--file",
        str(out_path),
        source,
    ]

    print(f"Source: {_redact_database_url(source)}", flush=True)
    print(f"Output: {out_path}", flush=True)
    if args.dry_run:
        safe_cmd = cmd[:-1] + [_redact_database_url(source)]
        print("DRY RUN:", subprocess.list2cmdline(safe_cmd), flush=True)
        return 0

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print(
            "ERROR: pg_dump not found. Install PostgreSQL client tools and ensure pg_dump is on PATH.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(f"ERROR: pg_dump failed with exit code {e.returncode}.", file=sys.stderr)
        return e.returncode or 1

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.2f} MiB", flush=True)
    if args.from_jetstream:
        print(
            "Next: restore into local Docker Postgres — see docs/LOCAL_DB_CLONE_FROM_JETSTREAM.md "
            "and scripts/restore_pg_dump_to_local_docker.py",
            flush=True,
        )
    else:
        print(
            "Next: copy the .dump to the Jetstream VM and run "
            "deploy/jetstream/restore_pg_dump_to_compose_db.sh (see MIGRATION_FROM_NEON.md).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

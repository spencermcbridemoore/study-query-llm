#!/usr/bin/env python3
"""
Upload a Jetstream-sourced pg_dump (-Fc) file to Azure Blob container ``db-backups``.

Uses the same storage account as artifacts (``AZURE_STORAGE_CONNECTION_STRING``).
Writes a companion manifest under ``backup_pg_dumps/<backup_id>.manifest.json`` for
``scripts/verify_db_backup_inventory.py`` (matches on ``blob_uri`` basename).

Typical flow (tunnel up, dump already created):
  python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream
  python scripts/upload_jetstream_pg_dump_to_blob.py

Or one step from an existing .dump:
  python scripts/upload_jetstream_pg_dump_to_blob.py --dump-path pg_migration_dumps/jetstream_for_local_20260414_003727Z.dump
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent
DUMP_DIR = REPO / "pg_migration_dumps"
MANIFEST_DIR = REPO / "backup_pg_dumps"
CONTAINER = "db-backups"


def _latest_jetstream_dump() -> Path | None:
    if not DUMP_DIR.is_dir():
        return None
    candidates = sorted(
        DUMP_DIR.glob("jetstream_for_local_*.dump"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _table_counts(url: str) -> dict[str, int] | None:
    if not url or not str(url).strip():
        return None
    try:
        eng = create_engine(url, pool_pre_ping=True)
        with eng.connect() as conn:
            rows = conn.execute(
                text(
                    """
                SELECT 'raw_calls' AS t, COUNT(*)::bigint FROM raw_calls
                UNION ALL SELECT 'groups', COUNT(*) FROM groups
                UNION ALL SELECT 'group_members', COUNT(*) FROM group_members
                UNION ALL SELECT 'group_links', COUNT(*) FROM group_links
                UNION ALL SELECT 'embedding_vectors', COUNT(*) FROM embedding_vectors
                UNION ALL SELECT 'call_artifacts', COUNT(*) FROM call_artifacts
                """
                )
            ).fetchall()
        return {str(t): int(n) for t, n in rows}
    except Exception:
        return None


def main() -> int:
    load_dotenv(REPO / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Upload Jetstream pg_dump (-Fc) to Azure db-backups container + manifest",
    )
    parser.add_argument(
        "--dump-path",
        type=Path,
        default=None,
        help="Path to .dump file (default: newest pg_migration_dumps/jetstream_for_local_*.dump)",
    )
    parser.add_argument(
        "--blob-name",
        default=None,
        help="Blob name inside db-backups (default: dump file basename — flat name for verify_db_backup_inventory)",
    )
    parser.add_argument(
        "--skip-table-counts",
        action="store_true",
        help="Do not query JETSTREAM_DATABASE_URL for table_counts in manifest",
    )
    args = parser.parse_args()

    dump_path = args.dump_path
    if dump_path is None:
        found = _latest_jetstream_dump()
        if not found:
            print(
                "ERROR: No --dump-path and no pg_migration_dumps/jetstream_for_local_*.dump found. "
                "Run: python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream",
                file=sys.stderr,
            )
            return 1
        dump_path = found
    else:
        dump_path = dump_path.resolve()
    if not dump_path.is_file():
        print(f"ERROR: Dump file not found: {dump_path}", file=sys.stderr)
        return 1

    conn_str = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    if not conn_str:
        print(
            "ERROR: AZURE_STORAGE_CONNECTION_STRING is not set (required for db-backups upload).",
            file=sys.stderr,
        )
        return 1

    blob_name = (args.blob_name or dump_path.name).strip().replace("\\", "/")
    if "/" in blob_name or ".." in blob_name:
        print(
            "ERROR: Use a flat blob name (no path segments) so verify_db_backup_inventory "
            "manifest basename matches list_blobs names.",
            file=sys.stderr,
        )
        return 1

    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient

    data = dump_path.read_bytes()
    svc = BlobServiceClient.from_connection_string(conn_str)
    cc = svc.get_container_client(CONTAINER)
    try:
        cc.create_container()
    except ResourceExistsError:
        pass

    blob = cc.get_blob_client(blob_name)
    blob.upload_blob(data, overwrite=True)
    blob_uri = blob.url

    jet_url = (os.environ.get("JETSTREAM_DATABASE_URL") or "").strip()
    table_counts: dict[str, int] | None = None
    if not args.skip_table_counts and jet_url:
        table_counts = _table_counts(jet_url)

    backup_id = dump_path.stem
    manifest = {
        "backup_id": backup_id,
        "source": "jetstream",
        "blob_uri": blob_uri,
        "dump_path": str(dump_path.relative_to(REPO)) if dump_path.is_relative_to(REPO) else str(dump_path),
        "byte_size": len(data),
        "uploaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if table_counts is not None:
        manifest["table_counts"] = table_counts

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{backup_id}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Uploaded {len(data)} bytes to {CONTAINER}/{blob_name}")
    print(f"     blob_uri={blob_uri}")
    print(f"     manifest={manifest_path}")
    print("Next: python scripts/verify_db_backup_inventory.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

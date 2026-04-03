"""
Compare local vs Jetstream Postgres row counts, list repo backup manifests, and list Azure `db-backups` blobs.

Usage (repo root):  python scripts/verify_db_backup_inventory.py

Requires `.env` with `JETSTREAM_DATABASE_URL`, `LOCAL_DATABASE_URL`, and optionally
`AZURE_STORAGE_CONNECTION_STRING`. If Jetstream and local both use `127.0.0.1` and the
same port (often 5433), only one service can bind that port — start the SSH tunnel first
or use a different `JETSTREAM_SSH_LOCAL_PORT` and matching URL for Jetstream.

Manifest JSON files live under `backup_pg_dumps/*.manifest.json` (gitignored dumps optional).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO = Path(__file__).resolve().parent.parent
MANIFEST_DIR = REPO / "backup_pg_dumps"


def _norm_host(h: str | None) -> str:
    if not h:
        return ""
    h = h.lower()
    if h == "localhost":
        return "127.0.0.1"
    return h


def _same_host_port(a: str, b: str) -> bool:
    try:
        pa, pb = urlparse(a), urlparse(b)
        return _norm_host(pa.hostname) == _norm_host(pb.hostname) and (pa.port or 5432) == (
            pb.port or 5432
        )
    except Exception:
        return False


def _table_counts(url: str, label: str) -> dict[str, int] | None:
    if not url or not str(url).strip():
        print(f"\n=== {label} ===\n  (DATABASE URL not set — skip)")
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
        out = {str(t): int(n) for t, n in rows}
        print(f"\n=== {label} ===")
        for k in sorted(out.keys()):
            print(f"  {k}: {out[k]}")
        return out
    except Exception as e:
        print(f"\n=== {label} ===\n  FAIL: {type(e).__name__}: {e}")
        return None


def _load_manifests() -> list[dict]:
    out: list[dict] = []
    if not MANIFEST_DIR.is_dir():
        return out
    for p in sorted(MANIFEST_DIR.glob("*.manifest.json")):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _list_db_backup_blobs(conn_str: str) -> list[tuple[str, int]]:
    from azure.storage.blob import BlobServiceClient

    svc = BlobServiceClient.from_connection_string(conn_str)
    cc = svc.get_container_client("db-backups")
    return [(b.name, b.size or 0) for b in cc.list_blobs()]


def main() -> int:
    load_dotenv(REPO / ".env")

    jet_url = os.environ.get("JETSTREAM_DATABASE_URL") or ""
    loc_url = os.environ.get("LOCAL_DATABASE_URL") or ""
    if jet_url and loc_url and _same_host_port(jet_url, loc_url):
        print(
            "NOTE: JETSTREAM_DATABASE_URL and LOCAL_DATABASE_URL use the same host:port.\n"
            "  Only one listener can use that port. Start the SSH tunnel before running this,\n"
            "  or move Jetstream to another local port (see deploy/jetstream/LOCAL_DEV_TUNNEL.md).\n"
        )

    jet = _table_counts(jet_url, "Jetstream (JETSTREAM_DATABASE_URL)")
    loc = _table_counts(loc_url, "Local (LOCAL_DATABASE_URL)")

    print("\n=== Repo manifests (backup_pg_dumps/*.manifest.json) ===")
    manifests = _load_manifests()
    if not manifests:
        print("  (no manifest files)")
    for m in manifests:
        bid = m.get("backup_id", "?")
        src = m.get("source", "?")
        tc = m.get("table_counts") or {}
        print(f"  {bid}  source={src}")
        print(f"    table_counts: {tc}")

    if jet and loc:
        same = jet == loc
        print("\n=== Jetstream vs Local row counts ===")
        print("  Exact match:", same)
        if not same:
            keys = sorted(set(jet) | set(loc))
            for k in keys:
                if jet.get(k) != loc.get(k):
                    print(f"    {k}: jetstream={jet.get(k)} local={loc.get(k)}")

    conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    print("\n=== Azure container 'db-backups' (same account as artifacts) ===")
    if not conn:
        print("  AZURE_STORAGE_CONNECTION_STRING not set — skip blob listing")
        return 0
    try:
        blobs = _list_db_backup_blobs(conn)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return 0

    blob_names = {b[0] for b in blobs}
    print(f"  Blobs in container: {len(blobs)}")
    for name, size in sorted(blobs, key=lambda x: x[0]):
        print(f"    {name}  {size} bytes")

    print("\n=== Manifest vs blob presence ===")
    for m in manifests:
        uri = m.get("blob_uri") or ""
        fname = uri.split("/")[-1] if uri else ""
        if fname and fname in blob_names:
            print(f"  OK manifest {m.get('backup_id')} -> blob {fname} exists")
        elif fname:
            print(f"  MISSING on blob: {fname} (manifest {m.get('backup_id')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

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

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
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


def _load_manifests() -> tuple[list[dict[str, Any]], list[str]]:
    out: list[dict[str, Any]] = []
    errors: list[str] = []
    if not MANIFEST_DIR.is_dir():
        return out, errors
    for p in sorted(MANIFEST_DIR.glob("*.manifest.json")):
        try:
            with open(p, encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                raise ValueError("manifest root must be a JSON object")
            out.append(manifest)
        except Exception as exc:
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
    return out, errors


def _list_db_backup_blobs(conn_str: str) -> list[tuple[str, int]]:
    from azure.storage.blob import BlobServiceClient

    svc = BlobServiceClient.from_connection_string(conn_str)
    cc = svc.get_container_client("db-backups")
    return [(b.name, b.size or 0) for b in cc.list_blobs()]


def _manifest_blob_name(manifest: dict[str, Any]) -> str:
    uri = str(manifest.get("blob_uri") or "").strip()
    return uri.split("/")[-1] if uri else ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DB backup inventory consistency.")
    parser.add_argument(
        "--strict-azure",
        action="store_true",
        help="Treat missing AZURE_STORAGE_CONNECTION_STRING as a verification failure.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv(REPO / ".env")
    failures: list[str] = []

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
    if jet_url and jet is None:
        failures.append("Jetstream table counts could not be loaded.")
    if loc_url and loc is None:
        failures.append("Local table counts could not be loaded.")

    print("\n=== Repo manifests (backup_pg_dumps/*.manifest.json) ===")
    manifests, manifest_errors = _load_manifests()
    if manifest_errors:
        for err in manifest_errors:
            print(f"  MALFORMED manifest: {err}")
        failures.extend([f"Malformed manifest: {e}" for e in manifest_errors])
    if not manifests:
        print("  (no manifest files)")
    for m in manifests:
        bid = m.get("backup_id", "?")
        src = m.get("source", "?")
        tc = m.get("table_counts") or {}
        print(f"  {bid}  source={src}")
        print(f"    table_counts: {tc}")
        if not isinstance(tc, dict):
            failures.append(
                f"Malformed manifest {bid!r}: table_counts must be a JSON object."
            )

    if jet and loc:
        same = jet == loc
        print("\n=== Jetstream vs Local row counts ===")
        print("  Exact match:", same)
        if not same:
            failures.append("Jetstream vs Local row counts mismatch.")
            keys = sorted(set(jet) | set(loc))
            for k in keys:
                if jet.get(k) != loc.get(k):
                    print(f"    {k}: jetstream={jet.get(k)} local={loc.get(k)}")

    conn = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    print("\n=== Azure container 'db-backups' (same account as artifacts) ===")
    if not conn:
        print("  AZURE_STORAGE_CONNECTION_STRING not set — skip blob listing")
        if args.strict_azure:
            failures.append(
                "AZURE_STORAGE_CONNECTION_STRING missing while --strict-azure is set."
            )
    else:
        try:
            blobs = _list_db_backup_blobs(conn)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            failures.append(f"Azure blob listing failed: {type(e).__name__}: {e}")
        else:
            blob_names = {b[0] for b in blobs}
            print(f"  Blobs in container: {len(blobs)}")
            for name, size in sorted(blobs, key=lambda x: x[0]):
                print(f"    {name}  {size} bytes")

            print("\n=== Manifest vs blob presence ===")
            missing_blobs: list[str] = []
            for m in manifests:
                fname = _manifest_blob_name(m)
                if fname and fname in blob_names:
                    print(f"  OK manifest {m.get('backup_id')} -> blob {fname} exists")
                elif fname:
                    print(f"  MISSING on blob: {fname} (manifest {m.get('backup_id')})")
                    missing_blobs.append(fname)
            if missing_blobs:
                failures.append(
                    f"Manifest/blob mismatch: {len(missing_blobs)} manifest blob(s) missing."
                )

    if failures:
        print("\n=== Verification result ===")
        print("  FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\n=== Verification result ===")
    print("  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
Verify DATABASE_URL from .env: connect, count raw_calls, confirm pgvector.

Usage (repo root):  python scripts/sanity_check_database_url.py

On your PC, DATABASE_URL must use a host your machine can resolve (e.g. 127.0.0.1
with an SSH tunnel), not the Compose-only hostname "db".
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def main() -> int:
    load_dotenv()
    url = os.environ.get("DATABASE_URL")
    if not url or not str(url).strip():
        print("FAIL: DATABASE_URL not set")
        return 1

    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
    except Exception:
        host = ""

    if host in ("db", ""):
        print(
            "FAIL: DATABASE_URL host is %r - that only works inside Jetstream Docker.\n"
            "  On your PC use an SSH tunnel, e.g. ssh -L 5433:127.0.0.1:5432 user@vm,\n"
            "  then DATABASE_URL=postgresql://USER:PASS@127.0.0.1:5433/DB?sslmode=prefer"
            % (host or "(missing)")
        )
        return 1

    host_hint = url.split("@")[-1].split("?")[0] if "@" in url else url
    print("Connecting to:", host_hint)
    try:
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
            assert one == 1
            n = conn.execute(text("SELECT COUNT(*) FROM raw_calls")).scalar()
            ext = conn.execute(
                text("SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
        print("OK: SELECT 1 succeeded")
        print("OK: raw_calls count =", n)
        print("OK: pgvector extension present =", ext, "(1 means installed)")
    except Exception as e:
        print("FAIL:", type(e).__name__, ":", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

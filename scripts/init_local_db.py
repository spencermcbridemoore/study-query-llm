#!/usr/bin/env python3
"""
Initialize the local backup PostgreSQL database with the v2 schema.

Run this once after starting the local Docker container for the first time,
or any time you need to recreate the schema.

Prerequisites:
    docker compose --profile postgres up -d db

Usage:
    python scripts/init_local_db.py
    python scripts/init_local_db.py --db-url postgresql://study:study@localhost:5433/study_query_local
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from project root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Initialize local backup PostgreSQL database")
    parser.add_argument(
        "--db-url",
        default=None,
        help="Local DB connection string (default: LOCAL_DATABASE_URL from .env)",
    )
    args = parser.parse_args()

    local_url = args.db_url or os.environ.get("LOCAL_DATABASE_URL")
    if not local_url:
        print("ERROR: LOCAL_DATABASE_URL not set. Add it to .env or pass --db-url.")
        sys.exit(1)

    # Mask password for display
    display_url = local_url.split("@")[-1] if "@" in local_url else local_url
    print(f"Connecting to local DB: ...@{display_url}")

    from study_query_llm.db.connection_v2 import DatabaseConnectionV2

    # pgvector not needed for local archive — store vectors as JSON
    db = DatabaseConnectionV2(local_url, enable_pgvector=False)

    try:
        with db.engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
        print(f"Connected: {version}")
    except Exception as e:
        print(f"ERROR: Could not connect to local DB: {e}")
        print("Is the Docker container running? Try: docker compose --profile postgres up -d db")
        sys.exit(1)

    print("Initializing v2 schema...")
    db.init_db()
    print("Schema initialized successfully.")

    # Report tables created
    with db.engine.connect() as conn:
        result = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ))
        tables = [row[0] for row in result.fetchall()]

    print(f"Tables present: {', '.join(tables)}")
    print("\nLocal DB is ready. You can now run:")
    print("  python scripts/sync_from_online.py      # pull all records from Neon")
    print("  python scripts/archive_defective_data.py # move defective records here")


if __name__ == "__main__":
    main()

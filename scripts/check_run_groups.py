#!/usr/bin/env python3
"""Check run groups and run_key data quality in the database."""

import argparse
import os
import sys

from sqlalchemy import func, text as sa_text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import Group

def main():
    parser = argparse.ArgumentParser(description="Inspect clustering_run groups")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate run_key quality checks",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    
    with db.session_scope() as session:
        repo = RawCallRepository(session)

        # Find all run groups
        run_groups = session.query(Group).filter(
            Group.group_type == "clustering_run"
        ).all()

        total_runs = len(run_groups)
        missing_run_key = (
            session.query(func.count(Group.id))
            .filter(
                Group.group_type == "clustering_run",
                sa_text("(metadata_json->>'run_key') IS NULL"),
            )
            .scalar()
        ) or 0

        duplicate_rows = (
            session.query(
                sa_text("metadata_json->>'run_key' as run_key"),
                func.count(Group.id).label("n"),
            )
            .filter(
                Group.group_type == "clustering_run",
                sa_text("(metadata_json->>'run_key') IS NOT NULL"),
            )
            .group_by(sa_text("metadata_json->>'run_key'"))
            .having(func.count(Group.id) > 1)
            .order_by(func.count(Group.id).desc())
            .all()
        )

        print("Run key quality summary:")
        print(f"  Total clustering_run groups: {total_runs}")
        print(f"  Missing run_key: {missing_run_key}")
        print(f"  Duplicate run_key values: {len(duplicate_rows)}")
        if duplicate_rows:
            print("  Top duplicate run_keys:")
            for row in duplicate_rows[:10]:
                print(f"    - {row.run_key}: {row.n}")

        if args.summary_only:
            return

        print(f"Found {len(run_groups)} run group(s):")
        for group in run_groups:
            print(f"\nGroup ID {group.id}:")
            print(f"  Name: {group.name}")
            print(f"  Type: {group.group_type}")
            print(f"  Created: {group.created_at}")
            if group.metadata_json:
                print(f"  Metadata: {group.metadata_json}")
            
            # Count calls in this group
            calls = repo.get_calls_in_group(group.id)
            print(f"  Calls in group: {len(calls)}")

if __name__ == "__main__":
    main()

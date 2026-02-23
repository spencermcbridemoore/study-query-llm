#!/usr/bin/env python3
"""Check what run groups exist in the database."""

import os
import sys
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import Group

def main():
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

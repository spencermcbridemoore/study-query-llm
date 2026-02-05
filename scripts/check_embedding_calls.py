#!/usr/bin/env python3
"""Check embedding calls in the database."""

import os
import sys
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Get all embedding calls
        embedding_calls = repo.query_raw_calls(modality="embedding", limit=1000)
        
        print(f"Found {len(embedding_calls)} embedding call(s)")
        
        if embedding_calls:
            print("\nFirst 10 embedding calls:")
            for call in embedding_calls[:10]:
                print(f"  Call ID {call.id}: model={call.model}, created={call.created_at}")
                # Check if in any groups
                groups = repo.get_groups_for_call(call.id)
                if groups:
                    print(f"    In groups: {[g.name for g in groups]}")
        
        # Check all groups
        from study_query_llm.db.models_v2 import Group
        all_groups = session.query(Group).all()
        print(f"\nTotal groups in database: {len(all_groups)}")
        for group in all_groups:
            calls = repo.get_calls_in_group(group.id)
            print(f"  Group {group.id} ({group.group_type}): {group.name} - {len(calls)} calls")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check all data in the database."""

import os
import sys
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import RawCall, Group, GroupMember

def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    print(f"Database URL: {db_url.split('@')[-1] if '@' in db_url else 'hidden'}")
    
    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Check total counts
        total_calls = repo.get_total_count()
        print(f"\nTotal RawCalls: {total_calls}")
        
        # Check by modality
        from sqlalchemy import func
        modality_counts = session.query(
            RawCall.modality, 
            func.count(RawCall.id)
        ).group_by(RawCall.modality).all()
        
        print("\nCalls by modality:")
        for modality, count in modality_counts:
            print(f"  {modality}: {count}")
        
        # Check groups
        total_groups = session.query(func.count(Group.id)).scalar()
        print(f"\nTotal Groups: {total_groups}")
        
        if total_groups > 0:
            group_types = session.query(
                Group.group_type,
                func.count(Group.id)
            ).group_by(Group.group_type).all()
            
            print("\nGroups by type:")
            for group_type, count in group_types:
                print(f"  {group_type}: {count}")
        
        # Check group members
        total_members = session.query(func.count(GroupMember.id)).scalar()
        print(f"\nTotal GroupMembers: {total_members}")
        
        # Sample recent calls
        if total_calls > 0:
            recent_calls = session.query(RawCall).order_by(
                RawCall.created_at.desc()
            ).limit(5).all()
            
            print("\nMost recent 5 calls:")
            for call in recent_calls:
                print(f"  Call {call.id}: {call.modality}, {call.model}, created={call.created_at}")

if __name__ == "__main__":
    main()

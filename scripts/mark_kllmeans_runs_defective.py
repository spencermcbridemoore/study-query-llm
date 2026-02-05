#!/usr/bin/env python3
"""
Mark PCA KLLMeans sweep runs as defective.

This script finds all run groups for "pca_kllmeans_sweep" algorithm and marks
all associated RawCalls as defective.
"""

import os
import sys
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory
except ImportError:
    pass  # python-dotenv not installed, rely on system environment

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import Group, GroupMember

def main():
    """Mark all PCA KLLMeans run calls as defective."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        print("Please set it in your .env file or environment variables.")
        sys.exit(1)
    
    # Validate it's not a placeholder
    if "username:password@host:port" in db_url:
        print("ERROR: DATABASE_URL appears to be a placeholder value")
        print("Please set a real database connection string in your .env file.")
        sys.exit(1)
    
    print(f"Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url[:50]}...")
    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        
        # Find all run groups for pca_kllmeans_sweep
        run_groups = session.query(Group).filter(
            Group.group_type == "run"
        ).all()
        
        # Filter for pca_kllmeans_sweep runs
        kllmeans_runs = []
        for group in run_groups:
            if (group.metadata_json and 
                isinstance(group.metadata_json, dict) and
                group.metadata_json.get("algorithm") == "pca_kllmeans_sweep"):
                kllmeans_runs.append(group)
        
        if not kllmeans_runs:
            print("INFO: No PCA KLLMeans run groups found")
            return
        
        print(f"Found {len(kllmeans_runs)} PCA KLLMeans run group(s):")
        for run in kllmeans_runs:
            print(f"   - Group ID {run.id}: {run.name} (created: {run.created_at})")
        
        # Get all calls in these run groups
        all_call_ids = set()
        for run in kllmeans_runs:
            calls = repo.get_calls_in_group(run.id)
            call_ids = [c.id for c in calls]
            all_call_ids.update(call_ids)
            print(f"   - Group {run.id}: {len(call_ids)} calls")
        
        if not all_call_ids:
            print("INFO: No RawCalls found in these run groups")
            return
        
        print(f"\nTotal unique calls to mark: {len(all_call_ids)}")
        
        # Get or create defective group
        defective_group_id = repo.get_or_create_defective_group()
        print(f"Using defective group ID: {defective_group_id}")
        
        # Mark all calls as defective
        marked_count = 0
        already_marked = 0
        for call_id in all_call_ids:
            # Check if already marked
            if repo.is_call_defective(call_id):
                already_marked += 1
                continue
            
            # Mark as defective
            repo.add_call_to_group(
                defective_group_id, 
                call_id, 
                role="bogus_pca_kllmeans_run"
            )
            marked_count += 1
        
        print(f"\nMarked {marked_count} calls as defective")
        if already_marked > 0:
            print(f"INFO: {already_marked} calls were already marked as defective")
        
        print(f"\nDone! All PCA KLLMeans run calls are now marked as defective.")

if __name__ == "__main__":
    main()

"""
Verification script for v1 → v2 migration.

Compares data between v1 SQLite and v2 Postgres databases to ensure
migration was successful.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.models import InferenceRun
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def verify_migration():
    """Verify v1 → v2 migration."""
    logger.info("=" * 60)
    logger.info("Migration Verification")
    logger.info("=" * 60)
    
    # Get connection strings
    legacy_db_url = os.environ.get("LEGACY_DATABASE_URL", "sqlite:///study_query_llm.db")
    v2_db_url = os.environ.get("DATABASE_URL")
    
    if not v2_db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Connect to both databases
    v1_db = DatabaseConnection(legacy_db_url)
    v2_db = DatabaseConnectionV2(v2_db_url)
    
    # Get counts and extract sample data while sessions are open
    v1_data = []
    v1_count = 0
    with v1_db.session_scope() as v1_session:
        v1_count = v1_session.query(InferenceRun).count()
        logger.info(f"V1 database: {v1_count} inference runs")
        
        # Get sample records and extract data immediately
        v1_samples = v1_session.query(InferenceRun).order_by(InferenceRun.id).limit(3).all()
        for v1 in v1_samples:
            v1_data.append({
                'id': v1.id,
                'provider': v1.provider,
                'prompt': v1.prompt,
                'response': v1.response,
                'created_at': v1.created_at,
            })
    
    v2_data = []
    v2_count = 0
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        v2_count = v2_repo.get_total_count()
        logger.info(f"V2 database: {v2_count} raw calls")
        
        # Get sample records and extract data immediately
        from study_query_llm.db.models_v2 import RawCall
        v2_samples = v2_session.query(RawCall).order_by(RawCall.id).limit(3).all()
        for v2 in v2_samples:
            v2_data.append({
                'id': v2.id,
                'provider': v2.provider,
                'request_json': v2.request_json,
                'response_json': v2.response_json,
                'created_at': v2.created_at,
            })
    
    # Compare counts
    logger.info("")
    logger.info("Count Comparison:")
    logger.info(f"  V1 records: {v1_count}")
    logger.info(f"  V2 records: {v2_count}")
    
    if v1_count == v2_count:
        logger.info("  [OK] Counts match!")
    else:
        logger.warning(f"  [WARNING] Count mismatch: {v1_count} vs {v2_count}")
    
    # Compare sample data
    logger.info("")
    logger.info("Sample Data Comparison:")
    
    if v1_data and v2_data:
        min_samples = min(len(v1_data), len(v2_data))
        for i in range(min_samples):
            v1 = v1_data[i]
            v2 = v2_data[i]
            
            logger.info(f"\n  Sample {i+1}:")
            logger.info(f"    V1 ID: {v1['id']}, Provider: {v1['provider']}")
            logger.info(f"    V2 ID: {v2['id']}, Provider: {v2['provider']}")
            
            # Check provider match
            if v1['provider'] == v2['provider']:
                logger.info("    [OK] Provider matches")
            else:
                logger.warning(f"    [WARNING] Provider mismatch: {v1['provider']} vs {v2['provider']}")
            
            # Check prompt/request
            v1_prompt = v1['prompt'][:50] + "..." if len(v1['prompt']) > 50 else v1['prompt']
            v2_prompt = v2['request_json'].get("prompt", "")
            v2_prompt_display = v2_prompt[:50] + "..." if len(str(v2_prompt)) > 50 else v2_prompt
            
            if v1['prompt'] == v2['request_json'].get("prompt"):
                logger.info("    [OK] Prompt matches")
            else:
                logger.warning("    [WARNING] Prompt mismatch")
                logger.info(f"      V1: {v1_prompt}")
                logger.info(f"      V2: {v2_prompt_display}")
            
            # Check response
            if v1['response'] and v2['response_json']:
                v1_response = v1['response'][:50] + "..." if len(v1['response']) > 50 else v1['response']
                v2_response = v2['response_json'].get("text", "")
                v2_response_display = v2_response[:50] + "..." if len(str(v2_response)) > 50 else v2_response
                
                if v1['response'] == v2['response_json'].get("text"):
                    logger.info("    [OK] Response matches")
                else:
                    logger.warning("    [WARNING] Response mismatch")
                    logger.info(f"      V1: {v1_response}")
                    logger.info(f"      V2: {v2_response_display}")
            
            # Check timestamps
            if v1['created_at'] and v2['created_at']:
                logger.info(f"    V1 created_at: {v1['created_at']}")
                logger.info(f"    V2 created_at: {v2['created_at']}")
    
    # Check for batches and verify batch sizes
    logger.info("")
    logger.info("Batch/Group Information:")
    
    v1_batch_sizes = {}
    v1_batch_timestamps = {}
    with v1_db.session_scope() as v1_session:
        from study_query_llm.db.inference_repository import InferenceRepository
        from sqlalchemy import func
        v1_repo = InferenceRepository(v1_session)
        batch_ids = v1_session.query(InferenceRun.batch_id).distinct().all()
        batch_ids = [b[0] for b in batch_ids if b[0] is not None]
        logger.info(f"  V1 batches: {len(batch_ids)}")
        
        # Get batch sizes and timestamp ranges
        for batch_id in batch_ids:
            runs = v1_session.query(InferenceRun).filter(InferenceRun.batch_id == batch_id).all()
            v1_batch_sizes[batch_id] = len(runs)
            if runs:
                timestamps = [r.created_at for r in runs if r.created_at]
                if timestamps:
                    v1_batch_timestamps[batch_id] = (min(timestamps), max(timestamps))
    
    v2_group_sizes = {}
    v2_group_timestamps = {}
    with v2_db.session_scope() as v2_session:
        from study_query_llm.db.models_v2 import Group, GroupMember
        groups = v2_session.query(Group).all()
        logger.info(f"  V2 groups: {len(groups)}")
        
        # Get group sizes and timestamp ranges
        for group in groups:
            members = v2_session.query(GroupMember).filter(GroupMember.group_id == group.id).all()
            v2_group_sizes[group.id] = len(members)
            
            # Get timestamps from raw calls in this group
            if members:
                call_ids = [m.call_id for m in members]
                raw_calls = v2_session.query(RawCall).filter(RawCall.id.in_(call_ids)).all()
                timestamps = [c.created_at for c in raw_calls if c.created_at]
                if timestamps:
                    v2_group_timestamps[group.id] = (min(timestamps), max(timestamps))
    
    # Verify batch sizes match
    logger.info("")
    logger.info("Batch Size Verification:")
    batch_size_mismatches = 0
    for batch_id, v1_size in v1_batch_sizes.items():
        # Find corresponding v2 group by batch_id in metadata
        matching_group = None
        with v2_db.session_scope() as v2_session:
            from study_query_llm.db.models_v2 import Group
            for group in v2_session.query(Group).all():
                if (group.metadata_json and 
                    isinstance(group.metadata_json, dict) and
                    group.metadata_json.get('batch_id') == batch_id):
                    matching_group = group
                    break
        
        if matching_group:
            v2_size = v2_group_sizes.get(matching_group.id, 0)
            if v1_size == v2_size:
                logger.info(f"  [OK] Batch {batch_id}: {v1_size} members")
            else:
                logger.warning(f"  [WARNING] Batch {batch_id}: V1={v1_size}, V2={v2_size}")
                batch_size_mismatches += 1
        else:
            logger.warning(f"  [WARNING] Batch {batch_id}: No matching group in V2")
            batch_size_mismatches += 1
    
    if batch_size_mismatches == 0:
        logger.info("  [OK] All batch sizes match!")
    
    # Verify timestamp ranges
    logger.info("")
    logger.info("Timestamp Range Verification:")
    from datetime import timedelta
    timestamp_tolerance = timedelta(seconds=1)  # Allow 1 second difference for migration timing
    
    timestamp_mismatches = 0
    for batch_id, (v1_min, v1_max) in v1_batch_timestamps.items():
        # Find corresponding v2 group
        matching_group = None
        with v2_db.session_scope() as v2_session:
            from study_query_llm.db.models_v2 import Group
            for group in v2_session.query(Group).all():
                if (group.metadata_json and 
                    isinstance(group.metadata_json, dict) and
                    group.metadata_json.get('batch_id') == batch_id):
                    matching_group = group
                    break
        
        if matching_group and matching_group.id in v2_group_timestamps:
            v2_min, v2_max = v2_group_timestamps[matching_group.id]
            
            # Check if timestamps are within tolerance
            v1_min_diff = abs((v1_min - v2_min).total_seconds()) if v1_min and v2_min else None
            v1_max_diff = abs((v1_max - v2_max).total_seconds()) if v1_max and v2_max else None
            
            if (v1_min_diff is None or v1_min_diff <= timestamp_tolerance.total_seconds()) and \
               (v1_max_diff is None or v1_max_diff <= timestamp_tolerance.total_seconds()):
                logger.info(f"  [OK] Batch {batch_id}: Timestamps within range")
            else:
                logger.warning(f"  [WARNING] Batch {batch_id}: Timestamp mismatch")
                logger.info(f"    V1 range: {v1_min} to {v1_max}")
                logger.info(f"    V2 range: {v2_min} to {v2_max}")
                timestamp_mismatches += 1
    
    if timestamp_mismatches == 0:
        logger.info("  [OK] All timestamp ranges match!")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Verification Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    verify_migration()

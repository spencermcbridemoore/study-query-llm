"""
Migration script: V1 SQLite → V2 Postgres.

Reads from legacy v1 database (SQLite) and migrates data to v2 database (Postgres).
Maps InferenceRun records to RawCall, and converts batch_id to Group + GroupMember.

Usage:
    LEGACY_DATABASE_URL=sqlite:///study_query_llm.db \
    DATABASE_URL=postgresql://user:pass@localhost:5432/study_query_llm_v2 \
    python scripts/migrate_v1_to_v2.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.db.connection import DatabaseConnection
from study_query_llm.db.models import InferenceRun
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def migrate_inference_run_to_raw_call(inference_run: InferenceRun) -> Dict:
    """
    Convert v1 InferenceRun to v2 RawCall data structure.
    
    Args:
        inference_run: V1 InferenceRun instance
    
    Returns:
        Dict with RawCall fields
    """
    # Extract model from provider if possible (e.g., "azure_openai_gpt-4" -> "gpt-4")
    model = None
    if inference_run.provider:
        parts = inference_run.provider.split("_")
        if len(parts) > 2:
            model = "_".join(parts[2:])  # Everything after provider prefix
    
    # Build request JSON
    request_json = {
        "prompt": inference_run.prompt,
    }
    
    # Build response JSON
    response_json = {
        "text": inference_run.response,
    } if inference_run.response else None
    
    # Build tokens JSON
    tokens_json = None
    if inference_run.tokens is not None:
        tokens_json = {
            "total": inference_run.tokens,
        }
        # Try to extract breakdown from metadata if available
        if inference_run.metadata_json:
            if "prompt_tokens" in inference_run.metadata_json:
                tokens_json["prompt"] = inference_run.metadata_json["prompt_tokens"]
            if "completion_tokens" in inference_run.metadata_json:
                tokens_json["completion"] = inference_run.metadata_json["completion_tokens"]
    
    # Status is always "success" for v1 data (v1 only stored successful calls)
    status = "success"
    
    return {
        "provider": inference_run.provider,
        "model": model,
        "modality": "text",  # V1 only had text completions
        "status": status,
        "request_json": request_json,
        "response_json": response_json,
        "error_json": None,
        "latency_ms": inference_run.latency_ms,
        "tokens_json": tokens_json,
        "metadata_json": inference_run.metadata_json or {},
    }


def migrate_batches(
    v1_db: DatabaseConnection,
    v2_repo: RawCallRepository,
    call_id_mapping: Dict[int, int]
) -> None:
    """
    Migrate batch_id groupings from v1 to v2 Group + GroupMember tables.
    
    Args:
        v1_db: V1 database connection
        v2_repo: V2 repository instance
        call_id_mapping: Dict mapping v1 inference_run.id -> v2 raw_call.id
    """
    logger.info("Migrating batch groupings...")
    
    with v1_db.session_scope() as v1_session:
        from study_query_llm.db.inference_repository import InferenceRepository
        v1_repo = InferenceRepository(v1_session)
        
        # Get all unique batch_ids
        batch_ids = v1_session.query(InferenceRun.batch_id).distinct().all()
        batch_ids = [b[0] for b in batch_ids if b[0] is not None]
        
        logger.info(f"Found {len(batch_ids)} unique batch IDs to migrate")
        
        for batch_id in batch_ids:
            # Get all inference runs in this batch
            runs = v1_repo.get_inferences_by_batch_id(batch_id)
            
            if not runs:
                continue
            
            # Create Group in v2
            group_id = v2_repo.create_group(
                group_type="batch",
                name=f"batch_{batch_id}",
                description=f"Migrated batch from v1 (original batch_id: {batch_id})",
                metadata_json={"original_batch_id": batch_id},
            )
            
            # Add each call to the group
            for idx, run in enumerate(runs):
                v2_call_id = call_id_mapping.get(run.id)
                if v2_call_id:
                    v2_repo.add_call_to_group(
                        group_id=group_id,
                        call_id=v2_call_id,
                        position=idx,
                        role=None,
                    )
            
            logger.debug(f"Migrated batch {batch_id} -> group {group_id} ({len(runs)} calls)")


def main() -> None:
    """Main migration function."""
    logger.info("=" * 60)
    logger.info("V1 → V2 Database Migration")
    logger.info("=" * 60)
    
    # Get connection strings from environment
    legacy_db_url = os.environ.get("LEGACY_DATABASE_URL")
    v2_db_url = os.environ.get("DATABASE_URL")
    
    if not legacy_db_url:
        logger.error("LEGACY_DATABASE_URL environment variable not set")
        sys.exit(1)
    
    if not v2_db_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)
    
    logger.info(f"Legacy DB (v1): {legacy_db_url}")
    logger.info(f"Target DB (v2): {v2_db_url.split('@')[-1] if '@' in v2_db_url else v2_db_url}")
    
    # Connect to both databases
    logger.info("Connecting to databases...")
    v1_db = DatabaseConnection(legacy_db_url)
    v2_db = DatabaseConnectionV2(v2_db_url)
    
    # Initialize v2 database
    logger.info("Initializing v2 database schema...")
    v2_db.init_db()
    
    # Get counts
    with v1_db.session_scope() as v1_session:
        v1_count = v1_session.query(InferenceRun).count()
        logger.info(f"V1 database has {v1_count} inference runs")
    
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        v2_count = v2_repo.get_total_count()
        logger.info(f"V2 database currently has {v2_count} raw calls")
    
    if v1_count == 0:
        logger.warning("No data to migrate from v1 database")
        return
    
    # Migrate raw calls
    logger.info("Migrating inference runs to raw calls...")
    call_id_mapping: Dict[int, int] = {}
    batch_size = 100
    
    with v1_db.session_scope() as v1_session:
        with v2_db.session_scope() as v2_session:
            v2_repo = RawCallRepository(v2_session)
            
            # Process in batches
            offset = 0
            migrated = 0
            
            while True:
                # Fetch batch from v1
                runs = v1_session.query(InferenceRun).order_by(
                    InferenceRun.id
                ).limit(batch_size).offset(offset).all()
                
                if not runs:
                    break
                
                # Convert to v2 format
                raw_calls_data = []
                for run in runs:
                    raw_call_data = migrate_inference_run_to_raw_call(run)
                    raw_calls_data.append(raw_call_data)
                
                # Insert into v2
                v2_call_ids = v2_repo.batch_insert_raw_calls(raw_calls_data)
                
                # Build mapping
                for run, v2_id in zip(runs, v2_call_ids):
                    call_id_mapping[run.id] = v2_id
                
                migrated += len(runs)
                offset += batch_size
                
                logger.info(f"Migrated {migrated}/{v1_count} inference runs...")
    
    logger.info(f"Successfully migrated {migrated} inference runs to raw calls")
    
    # Migrate batches
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        migrate_batches(v1_db, v2_repo, call_id_mapping)
    
    # Final counts
    with v2_db.session_scope() as v2_session:
        v2_repo = RawCallRepository(v2_session)
        final_count = v2_repo.get_total_count()
        logger.info(f"V2 database now has {final_count} raw calls")
    
    logger.info("=" * 60)
    logger.info("Migration completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

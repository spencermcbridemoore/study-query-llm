"""
Migration script to add indexes for sweep request and run_key lookups.

Adds non-unique expression indexes on groups.metadata_json for:
- run_key lookup (clustering_run)
- request_status lookup (clustering_sweep_request)

For uniqueness/idempotency safety (unique run_key, unique group links, worker
claim table), run:
    python -m study_query_llm.db.migrations.add_sweep_worker_safety

Usage:
    python -m study_query_llm.db.migrations.add_sweep_request_indexes
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sqlalchemy import text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def add_indexes():
    """Add indexes for sweep request progress and run_key lookups."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding sweep request / run_key indexes")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(database_url, enable_pgvector=False)

    with db.engine.connect() as conn:
        # Index for clustering_run run_key lookup (used by compute_progress)
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_groups_clustering_run_run_key
                ON groups ((metadata_json->>'run_key'))
                WHERE group_type = 'clustering_run'
            """))
            conn.commit()
            logger.info("Created idx_groups_clustering_run_run_key")
        except Exception as e:
            conn.rollback()
            logger.warning("Index idx_groups_clustering_run_run_key: %s", e)

        # Index for clustering_sweep_request status filter (used by list_requests)
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_groups_sweep_request_status
                ON groups ((metadata_json->>'request_status'))
                WHERE group_type = 'clustering_sweep_request'
            """))
            conn.commit()
            logger.info("Created idx_groups_sweep_request_status")
        except Exception as e:
            conn.rollback()
            logger.warning("Index idx_groups_sweep_request_status: %s", e)

    logger.info("=" * 60)
    logger.info("Migration completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    add_indexes()

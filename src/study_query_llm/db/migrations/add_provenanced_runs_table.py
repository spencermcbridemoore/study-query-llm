"""
Migration script to add ProvenancedRun table to v2 schema.

Usage:
    python -m study_query_llm.db.migrations.add_provenanced_runs_table
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import ProvenancedRun
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def add_provenanced_runs_table() -> None:
    """Create provenanced_runs table (idempotent)."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding ProvenancedRun table to v2 schema")
    logger.info("=" * 60)
    db = DatabaseConnectionV2(
        database_url,
        write_intent=WriteIntent.CANONICAL,
    )
    try:
        ProvenancedRun.__table__.create(db.engine, checkfirst=True)
        logger.info("provenanced_runs table created successfully")
        logger.info("=" * 60)
        logger.info("Migration completed successfully")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    add_provenanced_runs_table()

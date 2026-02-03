"""
Migration script to add GroupLink table to v2 schema.

This script adds the group_links table for modeling relationships between groups.

Usage:
    python -m study_query_llm.db.migrations.add_group_links
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import BaseV2, GroupLink
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def add_group_links_table():
    """Add GroupLink table to the database."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding GroupLink table to v2 schema")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(database_url)
    
    try:
        # Create the table
        logger.info("Creating group_links table...")
        GroupLink.__table__.create(db.engine, checkfirst=True)
        logger.info("✅ GroupLink table created successfully")
        
        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    add_group_links_table()

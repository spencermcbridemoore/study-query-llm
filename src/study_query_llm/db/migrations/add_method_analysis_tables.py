"""
Migration script to add MethodDefinition and AnalysisResult tables to v2 schema.

This script adds the method_definitions and analysis_results tables for
structured provenance tracking of analysis methods and their versioned results.

Usage:
    python -m study_query_llm.db.migrations.add_method_analysis_tables
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import MethodDefinition, AnalysisResult
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def add_method_analysis_tables():
    """Add MethodDefinition and AnalysisResult tables to the database."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding MethodDefinition and AnalysisResult tables to v2 schema")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(
        database_url,
        write_intent=WriteIntent.CANONICAL,
    )

    try:
        logger.info("Creating method_definitions table...")
        MethodDefinition.__table__.create(db.engine, checkfirst=True)
        logger.info("method_definitions table created successfully")

        logger.info("Creating analysis_results table...")
        AnalysisResult.__table__.create(db.engine, checkfirst=True)
        logger.info("analysis_results table created successfully")

        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    add_method_analysis_tables()

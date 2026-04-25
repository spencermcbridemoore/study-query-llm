"""
Migration script to add fingerprint_json and fingerprint_hash columns
to the provenanced_runs table.

These columns store a canonical run fingerprint for semantic comparability
independent of orchestration granularity.

Usage:
    python -m study_query_llm.db.migrations.add_fingerprint_columns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sqlalchemy import text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def _column_exists(engine, table: str, column: str) -> bool:
    """Check whether a column already exists (Postgres and SQLite)."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect == "postgresql":
            result = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = :table AND column_name = :col"
                ),
                {"table": table, "col": column},
            )
            return result.fetchone() is not None
        # SQLite fallback
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return any(row[1] == column for row in rows)


def add_fingerprint_columns():
    """Add fingerprint_json and fingerprint_hash to provenanced_runs."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding fingerprint columns to provenanced_runs")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(
        database_url,
        write_intent=WriteIntent.CANONICAL,
    )
    engine = db.engine

    try:
        if not _column_exists(engine, "provenanced_runs", "fingerprint_json"):
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE provenanced_runs ADD COLUMN fingerprint_json JSON")
                )
            logger.info("Added fingerprint_json column")
        else:
            logger.info("fingerprint_json column already exists -- skipping")

        if not _column_exists(engine, "provenanced_runs", "fingerprint_hash"):
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE provenanced_runs "
                        "ADD COLUMN fingerprint_hash VARCHAR(64)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS idx_provenanced_run_fingerprint_hash "
                        "ON provenanced_runs (fingerprint_hash)"
                    )
                )
            logger.info("Added fingerprint_hash column with index")
        else:
            logger.info("fingerprint_hash column already exists -- skipping")

        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    add_fingerprint_columns()

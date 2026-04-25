"""
Migration script to drop the legacy embedding_vectors table.

This migration is intentionally unconditional once the table exists:
it logs the pre-drop row count for auditability, then removes the table.

Usage:
    python -m study_query_llm.db.migrations.drop_embedding_vectors
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import inspect, text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _table_exists(db: DatabaseConnectionV2, table_name: str) -> bool:
    inspector = inspect(db.engine)
    return bool(inspector.has_table(table_name))


def _count_embedding_vectors_rows(db: DatabaseConnectionV2) -> int:
    with db.engine.connect() as conn:
        value = conn.execute(text("SELECT COUNT(*) FROM embedding_vectors")).scalar()
    return int(value or 0)


def drop_embedding_vectors_table(database_url: str | None = None) -> int:
    """
    Drop embedding_vectors table and return the pre-drop row count.
    """
    resolved_url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not resolved_url:
        logger.error("DATABASE_URL environment variable not set")
        raise ValueError("DATABASE_URL is required")

    logger.info("=" * 60)
    logger.info("Dropping legacy embedding_vectors table")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(resolved_url),
    )
    if not _table_exists(db, "embedding_vectors"):
        logger.info("embedding_vectors table not found; migration is already applied.")
        return 0

    row_count = _count_embedding_vectors_rows(db)
    logger.info(
        "embedding_vectors row count before drop (informational): %s",
        row_count,
    )

    with db.engine.begin() as conn:
        if db.engine.dialect.name == "sqlite":
            conn.execute(text("DROP TABLE IF EXISTS embedding_vectors"))
        else:
            conn.execute(text("DROP TABLE IF EXISTS embedding_vectors CASCADE"))

    logger.info("Dropped embedding_vectors table successfully.")
    logger.info("=" * 60)
    return row_count


if __name__ == "__main__":
    try:
        drop_embedding_vectors_table()
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        sys.exit(1)

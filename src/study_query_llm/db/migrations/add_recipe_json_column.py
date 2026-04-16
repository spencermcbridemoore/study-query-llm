"""
Migration script to add the ``recipe_json`` column to ``method_definitions``.

This column stores an optional structured recipe describing a composite
pipeline as an ordered list of component MethodDefinition references. When
populated, callers are expected to include ``canonical_recipe_hash(recipe)``
in run ``config_json`` so the canonical run fingerprint absorbs recipe
identity via the existing ``canonical_config_hash`` path -- no change to
the fingerprint tuple shape is required.

See docs/living/METHOD_RECIPES.md for the v0 recipe contract.

Usage:
    python -m study_query_llm.db.migrations.add_recipe_json_column
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sqlalchemy import text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def _table_exists(engine, table: str) -> bool:
    """Check whether a table already exists (Postgres and SQLite)."""
    dialect = engine.dialect.name
    with engine.connect() as conn:
        if dialect == "postgresql":
            result = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = :table"
                ),
                {"table": table},
            )
            return result.fetchone() is not None
        rows = conn.execute(
            text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type='table' AND name = :table"
            ),
            {"table": table},
        ).fetchall()
        return len(rows) > 0


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
        rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
        return any(row[1] == column for row in rows)


def add_recipe_json_column():
    """Add recipe_json to method_definitions if missing.

    Runs ``init_db()`` first so that deployments which have never created
    ``method_definitions`` (e.g. partial clones) get the table with
    ``recipe_json`` present from the start. For deployments where the table
    already exists without the new column, the subsequent ``ALTER TABLE`` is
    a safe, idempotent no-op guarded by ``_column_exists``.
    """
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding recipe_json column to method_definitions")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(database_url)

    # Create any missing v2 tables using the current ORM models. SQLAlchemy's
    # create_all is a no-op on existing tables, so this is safe on a live DB.
    # Newly created tables will already carry recipe_json.
    db.init_db()

    engine = db.engine

    try:
        if not _table_exists(engine, "method_definitions"):
            # Highly unusual: init_db() should have created the table. Report
            # and stop rather than silently skipping the column add.
            logger.error(
                "method_definitions table does not exist after init_db(); "
                "check model imports and DATABASE_URL target."
            )
            sys.exit(1)

        if not _column_exists(engine, "method_definitions", "recipe_json"):
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE method_definitions "
                        "ADD COLUMN recipe_json JSON"
                    )
                )
            logger.info("Added recipe_json column")
        else:
            logger.info("recipe_json column already exists -- skipping")

        logger.info("=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    add_recipe_json_column()

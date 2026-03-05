"""
Migration: add sweep worker safety constraints and claim table.

Adds:
- unique run_key index for clustering_run groups
- unique group_links triplet index for (parent_group_id, child_group_id, link_type)
- sweep_run_claims table for worker claim/lease ownership

This migration is additive and fails fast if duplicate data must be cleaned first.

Usage:
    python -m study_query_llm.db.migrations.add_sweep_worker_safety
"""

import os
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import SweepRunClaim
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _require_no_duplicates(conn) -> None:
    """Fail fast if uniqueness constraints would be violated."""
    dup_run_keys = conn.execute(
        text(
            """
            SELECT metadata_json->>'run_key' AS run_key, COUNT(*) AS n
            FROM groups
            WHERE group_type = 'clustering_run'
              AND (metadata_json->>'run_key') IS NOT NULL
            GROUP BY metadata_json->>'run_key'
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """
        )
    ).fetchall()

    if dup_run_keys:
        preview = ", ".join(f"{r.run_key}:{r.n}" for r in dup_run_keys)
        raise RuntimeError(
            "Cannot create unique clustering_run run_key index; duplicate run_key values found "
            f"(top 10): {preview}"
        )

    dup_links = conn.execute(
        text(
            """
            SELECT parent_group_id, child_group_id, link_type, COUNT(*) AS n
            FROM group_links
            GROUP BY parent_group_id, child_group_id, link_type
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC
            LIMIT 10
            """
        )
    ).fetchall()

    if dup_links:
        preview = ", ".join(
            f"({r.parent_group_id},{r.child_group_id},{r.link_type}):{r.n}"
            for r in dup_links
        )
        raise RuntimeError(
            "Cannot create unique group_links triplet index; duplicate tuples found "
            f"(top 10): {preview}"
        )


def run_migration() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Adding sweep worker safety constraints/table")
    logger.info("=" * 60)

    db = DatabaseConnectionV2(database_url, enable_pgvector=False)

    with db.engine.connect() as conn:
        try:
            _require_no_duplicates(conn)

            # Unique run_key for clustering_run groups only.
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_groups_clustering_run_run_key
                    ON groups ((metadata_json->>'run_key'))
                    WHERE group_type = 'clustering_run'
                      AND (metadata_json->>'run_key') IS NOT NULL
                    """
                )
            )

            # Unique parent-child-type tuple for group links.
            conn.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_group_links_parent_child_type
                    ON group_links (parent_group_id, child_group_id, link_type)
                    """
                )
            )

            # Claim table for additive worker rollout.
            SweepRunClaim.__table__.create(db.engine, checkfirst=True)

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    logger.info("Migration completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_migration()

"""Add the partial UNIQUE index enforcing analysis_request identity uniqueness.

Identity is ``(method_name, input_id, run_key)`` extracted from the JSON
``groups.metadata_json`` column, scoped to ``group_type='analysis_request'``.
The index closes a get-or-create race in which concurrent ``analyze()`` callers
could each create a duplicate ``analysis_request`` Group row.

This migration is Postgres-only: the SQLite test path and any fresh Postgres
``init_db()`` build the same index automatically from the model's
``after_create`` events (see ``db/models_v2.py``). It exists to add the index to
the already-provisioned canonical Postgres database, which create_all() does not
revisit.

HARD PRECONDITION: existing duplicates must be remediated first, otherwise the
unique index cannot be created. Run:
    python scripts/remediate_analysis_request_duplicates.py --apply
then this migration. The migration refuses to run while duplicates remain.

Usage:
    python -m study_query_llm.db.migrations.add_analysis_request_unique_index
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import (
    ANALYSIS_REQUEST_UNIQUE_INDEX_NAME,
    ANALYSIS_REQUEST_UNIQUE_INDEX_SQL_POSTGRESQL,
)
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Mirrors scripts/remediate_analysis_request_duplicates.find_duplicate_identities.
# The partial unique index is NULL-distinct, so analysis_request rows with any
# NULL identity field (empty-metadata container groups) never collide and must
# be excluded here -- otherwise SQL GROUP BY would fold all such rows into one
# bucket and falsely report a duplicate that the index would happily allow.
_DUPLICATE_PROBE_SQL = """
    SELECT method_name, input_id, run_key, COUNT(*) AS n, MIN(id) AS keeper_id
    FROM (
        SELECT id,
               metadata_json ->> 'method_name' AS method_name,
               metadata_json ->> 'input_id'    AS input_id,
               metadata_json ->> 'run_key'     AS run_key
        FROM groups
        WHERE group_type = 'analysis_request'
    ) AS extracted
    WHERE method_name IS NOT NULL
      AND input_id IS NOT NULL
      AND run_key IS NOT NULL
    GROUP BY method_name, input_id, run_key
    HAVING COUNT(*) > 1
    ORDER BY n DESC, method_name, input_id, run_key
"""


def _find_duplicate_identities(conn) -> list[dict]:
    rows = conn.execute(text(_DUPLICATE_PROBE_SQL)).mappings().all()
    return [dict(r) for r in rows]


def _index_exists(conn, *, index_name: str) -> bool:
    query = text(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'groups'
          AND indexname = :index_name
        LIMIT 1
        """
    )
    return conn.execute(query, {"index_name": index_name}).scalar() is not None


def _resolve_target_url() -> str:
    """Resolve the target DB URL from canonical-aware env vars (tunnel must be up)."""
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL", "JETSTREAM_DATABASE_URL"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def add_analysis_request_unique_index(database_url: str | None = None) -> int:
    """Add the unique index when missing; return 1 if added, 0 if no-op.

    Raises RuntimeError if duplicate analysis_request identities still exist
    (remediation must run first).
    """
    resolved_url = (database_url or _resolve_target_url()).strip()
    if not resolved_url:
        raise ValueError(
            "No target DB URL. Set CANONICAL_DATABASE_URL, DATABASE_URL, or "
            "JETSTREAM_DATABASE_URL (or pass database_url)."
        )

    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(resolved_url),
    )

    dialect = str(db.engine.dialect.name).lower()
    if dialect != "postgresql":
        logger.info("Skipping migration on dialect=%s (Postgres only).", dialect)
        return 0

    with db.engine.begin() as conn:
        table_exists = conn.execute(text("SELECT to_regclass('public.groups')")).scalar()
        if table_exists is None:
            raise RuntimeError("groups table not found on target database.")

        if _index_exists(conn, index_name=ANALYSIS_REQUEST_UNIQUE_INDEX_NAME):
            logger.info(
                "Index %s already exists; no-op.", ANALYSIS_REQUEST_UNIQUE_INDEX_NAME
            )
            return 0

        duplicates = _find_duplicate_identities(conn)
        if duplicates:
            sample = duplicates[:5]
            raise RuntimeError(
                f"Refusing to create {ANALYSIS_REQUEST_UNIQUE_INDEX_NAME}: "
                f"{len(duplicates)} duplicated analysis_request identities remain. "
                "Run scripts/remediate_analysis_request_duplicates.py --apply first. "
                f"Sample (up to 5): {sample}"
            )

        conn.execute(text(ANALYSIS_REQUEST_UNIQUE_INDEX_SQL_POSTGRESQL))
        logger.info("Created index %s.", ANALYSIS_REQUEST_UNIQUE_INDEX_NAME)
        return 1


if __name__ == "__main__":
    try:
        changed = add_analysis_request_unique_index()
        if changed:
            logger.info("Migration applied.")
        else:
            logger.info("Migration already applied or not applicable.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        sys.exit(1)

"""Add a partial sentinel index for non-blob raw_calls.response_json['uri']."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

DEFAULT_INDEX_NAME = "idx_raw_calls_uri_non_blob_sentinel"
_BLOB_URI_REGEX = r"^https://[^/]+\.blob\.core\.windows\.net/.+"


def _validate_index_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", candidate):
        raise ValueError(f"Invalid index name: {name!r}")
    return candidate


def _render_create_index_sql(*, index_name: str) -> str:
    return f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON raw_calls (((response_json::jsonb ->> 'uri')))
        WHERE (response_json::jsonb ? 'uri')
          AND COALESCE(response_json::jsonb ->> 'uri', '') <> ''
          AND (response_json::jsonb ->> 'uri') !~* '{_BLOB_URI_REGEX}'
    """


def add_raw_calls_uri_sentinel_index(
    database_url: str | None = None,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
) -> int:
    """Add the sentinel index when missing; return 1 if added, 0 if no-op."""
    resolved_url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not resolved_url:
        raise ValueError("DATABASE_URL is required.")

    validated_index_name = _validate_index_name(index_name)
    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(resolved_url),
    )
    dialect = str(db.engine.dialect.name).lower()
    if dialect != "postgresql":
        logger.info("Skipping sentinel index migration on dialect=%s.", dialect)
        return 0

    with db.engine.begin() as conn:
        table_exists = conn.execute(text("SELECT to_regclass('public.raw_calls')")).scalar()
        if table_exists is None:
            raise RuntimeError("raw_calls table not found on target database.")
        existing = conn.execute(
            text(
                """
                SELECT 1
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename = 'raw_calls'
                  AND indexname = :index_name
                LIMIT 1
                """
            ),
            {"index_name": validated_index_name},
        ).scalar()
        if existing is not None:
            logger.info("Sentinel index %s already exists; no-op.", validated_index_name)
            return 0

        conn.execute(text(_render_create_index_sql(index_name=validated_index_name)))
        logger.info("Created sentinel index %s.", validated_index_name)
        return 1


if __name__ == "__main__":
    try:
        changed = add_raw_calls_uri_sentinel_index()
        if changed:
            logger.info("Migration applied.")
        else:
            logger.info("Migration already applied or not applicable.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        sys.exit(1)

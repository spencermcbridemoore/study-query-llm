"""Add a NOT VALID blob-URI CHECK constraint on call_artifacts.uri.

Usage:
    python -m study_query_llm.db.migrations.add_call_artifacts_blob_uri_check
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.lane import Lane
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

DEFAULT_CONSTRAINT_NAME = "call_artifacts_uri_must_be_blob"
_BLOB_URI_REGEX = r"^https://[^/]+\.blob\.core\.windows\.net/.+"


def _validate_constraint_name(name: str) -> str:
    candidate = str(name or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", candidate):
        raise ValueError(f"Invalid constraint name: {name!r}")
    return candidate


def _constraint_exists(conn, *, constraint_name: str) -> bool:
    query = text(
        """
        SELECT 1
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'call_artifacts'
          AND c.conname = :constraint_name
        LIMIT 1
        """
    )
    return conn.execute(query, {"constraint_name": constraint_name}).scalar() is not None


def _constraint_validated(conn, *, constraint_name: str) -> bool | None:
    query = text(
        """
        SELECT c.convalidated
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relname = 'call_artifacts'
          AND c.conname = :constraint_name
        LIMIT 1
        """
    )
    value = conn.execute(query, {"constraint_name": constraint_name}).scalar()
    if value is None:
        return None
    return bool(value)


def _render_add_constraint_sql(*, constraint_name: str) -> str:
    return f"""
        ALTER TABLE call_artifacts
        ADD CONSTRAINT {constraint_name}
        CHECK (uri ~* '{_BLOB_URI_REGEX}')
        NOT VALID
    """


def add_call_artifacts_blob_uri_check(
    database_url: str | None = None,
    *,
    constraint_name: str = DEFAULT_CONSTRAINT_NAME,
) -> int:
    """Add CHECK constraint when missing; return 1 if added, 0 if no-op."""
    resolved_url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not resolved_url:
        raise ValueError("DATABASE_URL is required.")

    validated_constraint_name = _validate_constraint_name(constraint_name)
    db = DatabaseConnectionV2(
        resolved_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(resolved_url),
    )

    dialect = str(db.engine.dialect.name).lower()
    if dialect != "postgresql":
        logger.info("Skipping migration on dialect=%s (Postgres only).", dialect)
        return 0

    if db.lane is not Lane.CANONICAL:
        raise RuntimeError(
            "Refusing to add canonical call_artifacts URI constraint on non-canonical lane "
            f"{db.lane.name}."
        )

    with db.engine.begin() as conn:
        table_exists = conn.execute(
            text("SELECT to_regclass('public.call_artifacts')")
        ).scalar()
        if table_exists is None:
            raise RuntimeError("call_artifacts table not found on target database.")

        if _constraint_exists(conn, constraint_name=validated_constraint_name):
            logger.info(
                "Constraint %s already exists; no-op.",
                validated_constraint_name,
            )
            return 0

        conn.execute(text(_render_add_constraint_sql(constraint_name=validated_constraint_name)))
        validated = _constraint_validated(
            conn,
            constraint_name=validated_constraint_name,
        )
        logger.info(
            "Added constraint %s (convalidated=%s).",
            validated_constraint_name,
            validated,
        )
        return 1


if __name__ == "__main__":
    try:
        changed = add_call_artifacts_blob_uri_check()
        if changed:
            logger.info("Migration applied.")
        else:
            logger.info("Migration already applied or not applicable.")
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        sys.exit(1)

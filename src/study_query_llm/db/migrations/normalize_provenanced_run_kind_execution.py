"""
Migration script to normalize provenanced run_kind values to ``execution``.

Usage:
    python -m study_query_llm.db.migrations.normalize_provenanced_run_kind_execution
    python -m study_query_llm.db.migrations.normalize_provenanced_run_kind_execution --non-strict
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from sqlalchemy import text as sa_text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import ProvenancedRun
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def normalize_provenanced_run_kinds(*, strict_constraint: bool = True) -> int:
    """Normalize legacy run_kind values and optionally tighten check constraint."""
    import os

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable not set")
        return 1

    db = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    logger.info("=" * 72)
    logger.info("Normalizing provenanced_runs.run_kind => execution")
    logger.info("=" * 72)

    updated = 0
    with db.session_scope() as session:
        rows = (
            session.query(ProvenancedRun)
            .filter(ProvenancedRun.run_kind.in_(["method_execution", "analysis_execution"]))
            .all()
        )
        for row in rows:
            old_kind = str(row.run_kind or "").strip().lower()
            meta = dict(row.metadata_json or {})
            if old_kind and not meta.get("execution_role"):
                meta["execution_role"] = old_kind
            row.run_kind = "execution"
            row.metadata_json = meta
            row.updated_at = datetime.now(timezone.utc)
            updated += 1
        session.flush()

        dialect_name = session.get_bind().dialect.name
        if strict_constraint and dialect_name == "postgresql":
            logger.info("Applying strict check constraint (run_kind IN ('execution'))")
            session.execute(
                sa_text(
                    "ALTER TABLE provenanced_runs "
                    "DROP CONSTRAINT IF EXISTS check_provenanced_run_kind"
                )
            )
            session.execute(
                sa_text(
                    "ALTER TABLE provenanced_runs "
                    "ADD CONSTRAINT check_provenanced_run_kind "
                    "CHECK (run_kind IN ('execution'))"
                )
            )
        elif strict_constraint:
            logger.info(
                "Strict constraint update skipped for dialect=%s (no safe generic ALTER path).",
                dialect_name,
            )

    logger.info("Updated rows: %s", updated)
    logger.info("Migration completed successfully")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize provenanced_runs.run_kind to execution."
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Do not tighten DB check constraint to execution-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    exit_code = normalize_provenanced_run_kinds(strict_constraint=not args.non_strict)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


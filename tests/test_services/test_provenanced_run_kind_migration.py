"""Tests for provenanced run_kind normalization migration."""

from __future__ import annotations

import os
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.migrations.normalize_provenanced_run_kind_execution import (
    normalize_provenanced_run_kinds,
)


def test_normalize_provenanced_run_kind_backfill_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "provenanced_run_kind.sqlite3"
    database_url = f"sqlite:///{db_path.as_posix()}"
    old_database_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        db = DatabaseConnectionV2(database_url, enable_pgvector=False)
        db.init_db()
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            req_id = repo.create_group(
                group_type="clustering_sweep_request",
                name="migration_req",
                metadata_json={},
            )
            src_id = repo.create_group(
                group_type="clustering_run",
                name="migration_src",
                metadata_json={"run_key": "rk1"},
            )
            repo.create_provenanced_run(
                run_kind="method_execution",
                request_group_id=int(req_id),
                source_group_id=int(src_id),
                run_key="rk1",
                run_status="completed",
                metadata_json={},
            )
            repo.create_provenanced_run(
                run_kind="analysis_execution",
                request_group_id=int(req_id),
                source_group_id=int(src_id),
                run_key="analysis:rk1",
                run_status="failed",
                metadata_json={},
            )

        assert normalize_provenanced_run_kinds(strict_constraint=False) == 0
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            rows = repo.list_provenanced_runs(request_group_id=int(req_id))
            assert len(rows) == 2
            assert {r.run_kind for r in rows} == {"execution"}
            assert {
                str((r.metadata_json or {}).get("execution_role") or "")
                for r in rows
            } == {"method_execution", "analysis_execution"}

        # Idempotency: second run should keep normalized rows unchanged.
        assert normalize_provenanced_run_kinds(strict_constraint=False) == 0
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            rows = repo.list_provenanced_runs(request_group_id=int(req_id))
            assert len(rows) == 2
            assert {r.run_kind for r in rows} == {"execution"}
    finally:
        if old_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = old_database_url


"""Tests for supervisor mode factory and strategy classes."""

from __future__ import annotations

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.supervisor_mode import (
    ShardedSupervisorMode,
    StandaloneSupervisorMode,
    create_supervisor_mode,
)
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_supervisor_mode_standalone():
    mode = create_supervisor_mode("standalone")
    assert isinstance(mode, StandaloneSupervisorMode)


def test_create_supervisor_mode_sharded():
    mode = create_supervisor_mode("sharded")
    assert isinstance(mode, ShardedSupervisorMode)


def test_create_supervisor_mode_unsupported():
    with pytest.raises(ValueError, match="Unsupported job_mode"):
        create_supervisor_mode("invalid")


def test_standalone_engine_work_remaining():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a", "engine/b"],
                "summarizers": ["None"],
            },
            entry_max=10,
            execution_mode="standalone",
        )

    mode = StandaloneSupervisorMode()
    engine_a_missing, total = mode.engine_work_remaining(db, req_id, "engine/a")
    engine_b_missing, _ = mode.engine_work_remaining(db, req_id, "engine/b")

    assert total >= 0
    assert engine_a_missing >= 0
    assert engine_b_missing >= 0


def test_standalone_before_progress_poll_no_op():
    """Standalone mode before_progress_poll is a no-op (does not raise)."""
    db = _db()
    mode = StandaloneSupervisorMode()
    mode.before_progress_poll(db, 1)  # no-op, no exception

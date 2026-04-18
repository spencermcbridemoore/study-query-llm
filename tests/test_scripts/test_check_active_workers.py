"""Unit tests for scripts/check_active_workers.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_active_workers.py"


@pytest.fixture(scope="module")
def workers_mod():
    spec = importlib.util.spec_from_file_location("check_active_workers", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_list_active_jobs_empty(workers_mod) -> None:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        rows = workers_mod.list_active_jobs(session)
    assert rows == []


def test_list_active_jobs_returns_non_terminal_only(workers_mod) -> None:
    from study_query_llm.db.models_v2 import Group, OrchestrationJob

    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        group = Group(group_type="mcq_sweep_request", name="req", metadata_json={})
        session.add(group)
        session.flush()
        session.add(
            OrchestrationJob(
                request_group_id=group.id,
                job_type="mcq_run",
                job_key="mcq_run_active",
                status="claimed",
                payload_json={},
                claimed_by="worker-1",
            )
        )
        session.add(
            OrchestrationJob(
                request_group_id=group.id,
                job_type="analysis_run",
                job_key="analysis_done",
                status="completed",
                payload_json={},
            )
        )
        session.flush()

    with db.session_scope() as session:
        rows = workers_mod.list_active_jobs(session)
    assert len(rows) == 1
    assert rows[0]["job_key"] == "mcq_run_active"
    assert rows[0]["status"] == "claimed"

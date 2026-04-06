"""Tests for one-container-per-engine local_300_2datasets scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.sweep_request_service import SweepRequestService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _create_request_with_two_engines(database_url: str) -> int:
    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        return svc.create_request(
            request_name="pytest_local_300_request",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3},
            parameter_axes={
                "datasets": ["dbpedia", "estela"],
                "embedding_engines": ["engine/a", "engine/b"],
                "summarizers": ["None"],
            },
            entry_max=300,
        )


def test_worker_help_includes_shared_endpoint_flags():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_300_2datasets_worker.py"),
            "--help",
        ],
        env={**os.environ, "DATABASE_URL": "sqlite:///:memory:"},
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "--embedding-engine" in out
    assert "--tei-endpoint" in out
    assert "--provider-label" in out
    assert "--idle-exit-seconds" in out
    assert "--job-mode" in out


def test_supervisor_help_includes_guardrail_flags():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_local_300_2datasets_engine_supervisor.py"),
            "--help",
        ],
        env={**os.environ, "DATABASE_URL": "sqlite:///:memory:"},
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "--max-worker-restarts" in out
    assert "--max-tei-restarts" in out
    assert "--engine-allowlist" in out
    assert "--job-mode" in out


def test_supervisor_engine_missing_count_filters_engine(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        database_url = f"sqlite:///{db_path}"
        request_id = _create_request_with_two_engines(database_url)
        db = DatabaseConnectionV2(database_url, enable_pgvector=False)
        db.init_db()

        monkeypatch.setenv("DATABASE_URL", database_url)
        from study_query_llm.services.supervisor_mode import (
            StandaloneSupervisorMode,
        )

        mode = StandaloneSupervisorMode()
        engine_a_missing, total_missing = mode.engine_work_remaining(
            db, request_id, "engine/a"
        )
        engine_b_missing, _ = mode.engine_work_remaining(
            db, request_id, "engine/b"
        )

        assert total_missing == 4
        assert engine_a_missing == 2
        assert engine_b_missing == 2
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_supervisor_backoff_is_exponential_and_capped(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    from study_query_llm.services.jobs.runtime_supervisors import _backoff_seconds

    assert _backoff_seconds(5, 1, 60) == 5
    assert _backoff_seconds(5, 2, 60) == 10
    assert _backoff_seconds(5, 3, 60) == 20
    assert _backoff_seconds(5, 10, 60) == 60

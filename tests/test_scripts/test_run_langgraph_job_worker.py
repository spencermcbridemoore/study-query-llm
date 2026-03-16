"""Tests for run_langgraph_job_worker script."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_langgraph_worker_help():
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_langgraph_job_worker.py"),
            "--help",
        ],
        env={**os.environ, "DATABASE_URL": "sqlite:///:memory:"},
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--request-id" in result.stdout
    assert "--worker-id" in result.stdout


def test_langgraph_worker_claim_complete_smoke(monkeypatch):
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        database_url = f"sqlite:///{db_path}"
        monkeypatch.setenv("DATABASE_URL", database_url)
        db = DatabaseConnectionV2(database_url, enable_pgvector=False)
        db.init_db()

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            req_id = repo.create_group(
                group_type="clustering_sweep_request",
                name="langgraph_smoke",
                metadata_json={},
            )
            repo.enqueue_orchestration_job(
                request_group_id=req_id,
                job_type="langgraph_run",
                job_key="lg_smoke_1",
                payload_json={"prompt": "smoke"},
            )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_langgraph_job_worker.py"),
                "--request-id",
                str(req_id),
                "--worker-id",
                "smoke-worker",
                "--idle-exit-seconds",
                "3",
                "--max-runs",
                "1",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        assert result.returncode == 0
        assert "DONE" in result.stdout or "JOB ERROR" in result.stderr

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            jobs = repo.list_orchestration_jobs(request_group_id=req_id)
            assert len(jobs) == 1
            assert jobs[0].status == "completed"

            # Provenance: analysis_results row created with parameters
            method_svc = MethodService(repo)
            results = method_svc.query_results(source_group_id=req_id, result_key="job_outcome")
            assert len(results) >= 1
            ar = results[0]
            assert ar.result_json is not None
            assert "parameters" in ar.result_json
            assert ar.result_json["parameters"].get("prompt") == "smoke"
            assert ar.result_json.get("status") == "completed"
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_langgraph_worker_failure_records_provenance(monkeypatch):
    """Failure path: invalid payload triggers validation error, job fails, provenance recorded."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        database_url = f"sqlite:///{db_path}"
        monkeypatch.setenv("DATABASE_URL", database_url)
        db = DatabaseConnectionV2(database_url, enable_pgvector=False)
        db.init_db()

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            req_id = repo.create_group(
                group_type="clustering_sweep_request",
                name="langgraph_fail",
                metadata_json={},
            )
            repo.enqueue_orchestration_job(
                request_group_id=req_id,
                job_type="langgraph_run",
                job_key="lg_fail_1",
                payload_json={"prompt": "ok", "config": "not_a_dict"},
            )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_langgraph_job_worker.py"),
                "--request-id",
                str(req_id),
                "--worker-id",
                "fail-worker",
                "--idle-exit-seconds",
                "3",
                "--max-runs",
                "1",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        assert result.returncode == 0
        assert "JOB ERROR" in result.stdout or "JOB ERROR" in result.stderr

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            jobs = repo.list_orchestration_jobs(request_group_id=req_id)
            assert len(jobs) == 1
            assert jobs[0].status == "failed"

            method_svc = MethodService(repo)
            results = method_svc.query_results(source_group_id=req_id, result_key="job_outcome")
            assert len(results) >= 1
            ar = results[0]
            assert ar.result_json is not None
            assert ar.result_json.get("status") == "failed"
            assert "error" in ar.result_json
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_langgraph_worker_redacts_sensitive_parameters(monkeypatch):
    """Sensitive keys in payload are redacted in persisted result_json.parameters."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        database_url = f"sqlite:///{db_path}"
        monkeypatch.setenv("DATABASE_URL", database_url)
        db = DatabaseConnectionV2(database_url, enable_pgvector=False)
        db.init_db()

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            req_id = repo.create_group(
                group_type="clustering_sweep_request",
                name="langgraph_redact",
                metadata_json={},
            )
            repo.enqueue_orchestration_job(
                request_group_id=req_id,
                job_type="langgraph_run",
                job_key="lg_redact_1",
                payload_json={
                    "prompt": "hello",
                    "api_key": "secret-123",
                    "config": {"token": "xyz"},
                },
            )

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_langgraph_job_worker.py"),
                "--request-id",
                str(req_id),
                "--worker-id",
                "redact-worker",
                "--idle-exit-seconds",
                "3",
                "--max-runs",
                "1",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        assert result.returncode == 0
        assert "DONE" in result.stdout

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            method_svc = MethodService(repo)
            results = method_svc.query_results(source_group_id=req_id, result_key="job_outcome")
            assert len(results) >= 1
            ar = results[0]
            params = ar.result_json.get("parameters", {})
            assert params.get("api_key") == "***REDACTED***"
            assert params.get("prompt") == "hello"
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass

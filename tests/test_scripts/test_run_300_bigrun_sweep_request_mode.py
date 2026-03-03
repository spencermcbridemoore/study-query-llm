"""
Tests for run_300_bigrun_sweep request-driven mode.

Verifies --create-request and --request-id CLI paths work.
Legacy mode (no request flags) is preserved; full integration requires DATABASE_URL.
"""

import os
import tempfile
import subprocess
import sys
from pathlib import Path

import pytest

# Project root for script invocation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture
def temp_db_path():
    """Temporary SQLite database path for subprocess tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


def test_create_request_cli_exits_success(temp_db_path):
    """--create-request creates a sweep request and exits 0."""
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{temp_db_path}"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_300_bigrun_sweep.py"),
            "--create-request",
            "--request-name", "pytest_request",
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    # Verify request was created in DB
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository
    from study_query_llm.services.provenance_service import GROUP_TYPE_CLUSTERING_SWEEP_REQUEST

    db = DatabaseConnectionV2(f"sqlite:///{temp_db_path}", enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        from study_query_llm.db.models_v2 import Group

        requests = session.query(Group).filter(
            Group.group_type == GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
        ).all()
        assert len(requests) >= 1
        names = [r.name for r in requests]
        assert "pytest_request" in names


def test_script_has_request_flags():
    """Script defines --request-id, --create-request, --request-name."""
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_300_bigrun_sweep.py"),
            "--help",
        ],
        env={**os.environ, "DATABASE_URL": "sqlite:///:memory:"},
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "--request-id" in out
    assert "--create-request" in out
    assert "--request-name" in out

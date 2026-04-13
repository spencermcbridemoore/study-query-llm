"""Guardrail coverage for purge_dataset_acquisition.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "purge_dataset_acquisition.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_execute_requires_explicit_database_url() -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5433/study_query_local"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--group-name",
            "acquire_ausem",
            "--execute",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env=env,
    )
    assert r.returncode == 1
    assert "requires explicit --database-url" in (r.stderr + r.stdout).lower()


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_execute_refuses_remote_target_without_override() -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--group-name",
            "acquire_ausem",
            "--execute",
            "--database-url",
            "postgresql://study:pw@example.com:5432/study_query_remote",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "--allow-remote-target" in (r.stderr + r.stdout)

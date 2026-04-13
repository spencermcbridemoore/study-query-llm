"""CLI guardrails for check_orchestration_jobs.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_orchestration_jobs.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_requires_database_url() -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = "postgresql://invalid:invalid@invalid-host:5432/invalid_db"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--request-id", "1"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
        env=env,
    )
    assert result.returncode != 0

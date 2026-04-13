"""Guardrail coverage for sync_from_online.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "sync_from_online.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_refuses_same_source_and_target() -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--online-url",
            "postgresql://study:pw@localhost:5433/study_query_local",
            "--local-url",
            "postgresql://study:pw@127.0.0.1:5433/study_query_local",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "same db target" in (r.stderr + r.stdout).lower()


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_refuses_remote_target_without_override() -> None:
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--online-url",
            "postgresql://study:pw@localhost:5432/source_db",
            "--local-url",
            "postgresql://study:pw@example.com:5432/target_db",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "--allow-remote-target" in (r.stderr + r.stdout)

"""CLI guardrails for dump_postgres_for_jetstream_migration.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "dump_postgres_for_jetstream_migration.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_from_local_and_from_jetstream_are_mutually_exclusive() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--from-local", "--from-jetstream"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "only one" in (r.stderr + r.stdout).lower()

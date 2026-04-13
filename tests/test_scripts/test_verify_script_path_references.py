"""CLI smoke test for verify_script_path_references.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "verify_script_path_references.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_verify_script_path_references_active_scope_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Missing references: 0" in result.stdout

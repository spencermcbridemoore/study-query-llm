"""Guardrail coverage for restore_pg_dump_to_local_docker.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "restore_pg_dump_to_local_docker.py"


def _make_dummy_dump(tmp_path: Path) -> Path:
    dump_path = tmp_path / "sample.dump"
    dump_path.write_bytes(b"dummy")
    return dump_path


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_refuses_remote_target_without_override(tmp_path: Path) -> None:
    dump_path = _make_dummy_dump(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dump_path),
            "--database-url",
            "postgresql://study:pw@example.com:5432/study_query_remote",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "--allow-remote-target" in (r.stderr + r.stdout)


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_remote_target_requires_db_confirmation(tmp_path: Path) -> None:
    dump_path = _make_dummy_dump(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dump_path),
            "--database-url",
            "postgresql://study:pw@example.com:5432/study_query_remote",
            "--allow-remote-target",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "--confirm-target-db" in (r.stderr + r.stdout)


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_remote_target_dry_run_allows_with_confirmation(tmp_path: Path) -> None:
    dump_path = _make_dummy_dump(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dump_path),
            "--database-url",
            "postgresql://study:pw@example.com:5432/study_query_remote",
            "--allow-remote-target",
            "--confirm-target-db",
            "study_query_remote",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert r.returncode == 0
    assert "DRY RUN pg_restore" in r.stdout

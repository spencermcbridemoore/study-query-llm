"""Guardrail coverage for record_dataset_download.py --persist-db path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "record_dataset_download.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_persist_db_refuses_local_target_without_override() -> None:
    env = os.environ.copy()
    env["ARTIFACT_STORAGE_BACKEND"] = "azure_blob"
    env["DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5433/study_query_local"
    env["LOCAL_DATABASE_URL"] = "postgresql://study:pw@localhost:5433/study_query_local"
    env["JETSTREAM_DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5434/study_query_jetstream"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "ausem",
            "--persist-db",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env=env,
    )
    assert r.returncode != 0
    assert "matches local_database_url" in (r.stderr + r.stdout).lower()


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_persist_db_refuses_non_jetstream_target_without_override() -> None:
    env = os.environ.copy()
    env["ARTIFACT_STORAGE_BACKEND"] = "azure_blob"
    env["DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5435/custom_target"
    env["LOCAL_DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5433/study_query_local"
    env["JETSTREAM_DATABASE_URL"] = "postgresql://study:pw@127.0.0.1:5434/study_query_jetstream"
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "ausem",
            "--persist-db",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env=env,
    )
    assert r.returncode != 0
    assert "differs from jetstream_database_url" in (r.stderr + r.stdout).lower()

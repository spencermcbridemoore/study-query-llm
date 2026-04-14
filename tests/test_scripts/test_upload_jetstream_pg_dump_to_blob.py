"""Smoke checks for upload_jetstream_pg_dump_to_blob.py CLI guardrails."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "upload_jetstream_pg_dump_to_blob.py"


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_upload_fails_without_azure_connection_string(tmp_path: Path) -> None:
    dump = tmp_path / "jetstream_for_local_test.dump"
    dump.write_bytes(b"x" * 100)

    # Script calls load_dotenv(); unset allows .env to populate the var. Force empty
    # so the connection string guard runs (dotenv does not override existing keys).
    env = os.environ.copy()
    env["AZURE_STORAGE_CONNECTION_STRING"] = ""
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dump-path", str(dump)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env=env,
    )
    assert r.returncode != 0
    assert "AZURE_STORAGE_CONNECTION_STRING" in (r.stderr + r.stdout)

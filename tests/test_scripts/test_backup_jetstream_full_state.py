"""Unit tests for scripts/backup_jetstream_full_state.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "backup_jetstream_full_state.py"


@pytest.fixture
def backup_mod():
    spec = importlib.util.spec_from_file_location("backup_jetstream_full_state", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_artifact_source_container_prefers_lane_override(
    backup_mod,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_RUNTIME_ENV", "dev")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "artifacts")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_DEV", "my-dev-container")

    resolved = backup_mod._resolve_artifact_source_container_from_env()
    assert resolved == "my-dev-container"


def test_build_destination_blob_name_normalizes_prefix_and_blob_name(backup_mod) -> None:
    destination = backup_mod._build_destination_blob_name(
        backup_prefix="/jetstream-full-state/20260426_000000Z/",
        source_container="artifacts-dev",
        source_blob_name="/dev/group/item.json",
    )
    assert destination == "jetstream-full-state/20260426_000000Z/artifacts-dev/dev/group/item.json"


def test_resolve_destination_connection_string_falls_back_to_source(
    backup_mod,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AZURE_BACKUP_STORAGE_CONNECTION_STRING", raising=False)
    resolved = backup_mod._resolve_destination_connection_string(
        source_connection_string="source-conn",
        destination_env_var="AZURE_BACKUP_STORAGE_CONNECTION_STRING",
    )
    assert resolved == "source-conn"


def test_main_dry_run_skip_steps_does_not_write_receipt(
    backup_mod,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backup_mod, "load_dotenv", lambda *_a, **_k: None)

    def _fail_write(*_a, **_k):
        raise AssertionError("dry-run should not write receipt")

    monkeypatch.setattr(backup_mod, "_write_receipt", _fail_write)
    rc = backup_mod.main(
        ["--dry-run", "--skip-db-backup", "--skip-artifact-backup"]
    )
    assert rc == 0


def test_main_non_dry_run_writes_receipt_when_steps_skipped(
    backup_mod,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backup_mod, "load_dotenv", lambda *_a, **_k: None)
    written: dict[str, object] = {}

    def _capture_write(path, payload):
        written["path"] = path
        written["payload"] = payload

    monkeypatch.setattr(backup_mod, "_write_receipt", _capture_write)
    rc = backup_mod.main(["--skip-db-backup", "--skip-artifact-backup"])
    assert rc == 0
    assert "path" in written
    payload = written["payload"]
    assert isinstance(payload, dict)
    assert payload["status"] == "ok"
    assert payload["db_backup"]["status"] == "skipped"
    assert payload["artifact_backup"]["status"] == "skipped"

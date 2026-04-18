"""Unit tests for scripts/archive_mcq_artifact_blobs.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "archive_mcq_artifact_blobs.py"


@pytest.fixture(scope="module")
def archive_mod():
    spec = importlib.util.spec_from_file_location("archive_mcq_artifact_blobs", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_backup_json(path: Path, uris: list[str]) -> None:
    payload = {
        "call_artifacts": [{"id": idx + 1, "uri": uri} for idx, uri in enumerate(uris)]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_archive_from_backup_copies_local_blobs_and_writes_receipts(
    archive_mod,
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    source_a = artifact_root / "legacy" / "100" / "a.json"
    source_b = artifact_root / "legacy" / "101" / "b.json"
    source_a.parent.mkdir(parents=True, exist_ok=True)
    source_b.parent.mkdir(parents=True, exist_ok=True)
    source_a.write_bytes(b'{"a":1}')
    source_b.write_bytes(b'{"b":2}')

    backup_json = tmp_path / "mcq_export.json"
    _write_backup_json(backup_json, [str(source_a.resolve()), str(source_b.resolve())])
    archive_prefix = "mcq-archive/20260417"

    result = archive_mod.archive_from_backup(
        backup_json_path=backup_json,
        archive_prefix=archive_prefix,
        artifact_root=artifact_root,
        destination_container=None,
        connection_string=None,
        dry_run=False,
    )

    assert result["errors"] == []
    assert result["copied_count"] == 2
    destination_a = Path(result["uri_remap"][str(source_a.resolve())])
    destination_b = Path(result["uri_remap"][str(source_b.resolve())])
    assert destination_a.is_file()
    assert destination_b.is_file()
    assert destination_a.read_bytes() == b'{"a":1}'
    assert destination_b.read_bytes() == b'{"b":2}'
    assert "mcq-archive" in str(destination_a)

    receipts = archive_mod._write_uri_remap(
        uri_remap=result["uri_remap"],
        archive_prefix=archive_prefix,
        artifact_root=artifact_root,
        backup_json_path=backup_json,
        dry_run=False,
    )
    assert Path(receipts["local_receipt"]).is_file()
    assert Path(receipts["archive_receipt"]).is_file()


def test_archive_from_backup_reports_missing_sources(archive_mod, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    backup_json = tmp_path / "mcq_export.json"
    _write_backup_json(backup_json, [str((tmp_path / "missing.bin").resolve())])

    result = archive_mod.archive_from_backup(
        backup_json_path=backup_json,
        archive_prefix="mcq-archive/20260417",
        artifact_root=artifact_root,
        destination_container=None,
        connection_string=None,
        dry_run=False,
    )
    assert result["copied_count"] == 0
    assert len(result["errors"]) == 1


def test_normalize_archive_prefix_replaces_date_token(archive_mod) -> None:
    normalized = archive_mod._normalize_archive_prefix("mcq-archive/<YYYYMMDD>/")
    assert normalized.startswith("mcq-archive/")
    assert len(normalized.split("/")[-1]) == 8

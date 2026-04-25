"""Unit tests for scripts/remediate_call_artifacts_to_blob.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "remediate_call_artifacts_to_blob.py"


def _mod():
    spec = importlib.util.spec_from_file_location("remediate_call_artifacts_to_blob", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_is_azure_blob_uri_detection() -> None:
    mod = _mod()
    assert mod._is_azure_blob_uri("https://acct.blob.core.windows.net/container/item.json")
    assert not mod._is_azure_blob_uri("C:\\tmp\\item.json")


def test_local_path_from_uri_handles_file_and_plain_paths() -> None:
    mod = _mod()
    path_from_file = mod._local_path_from_uri("file:///tmp/example.json")
    assert path_from_file is not None
    assert path_from_file.name == "example.json"
    path_from_plain = mod._local_path_from_uri("C:\\temp\\example.json")
    assert path_from_plain is not None
    assert path_from_plain.name == "example.json"


def test_derive_logical_path_prefers_group_step_and_filename() -> None:
    mod = _mod()
    artifact = SimpleNamespace(
        id=21,
        artifact_type="dataset_acquisition_file",
        metadata_json={
            "group_id": 15,
            "step_name": "acquisition",
            "logical_filename": "data_train.parquet",
        },
    )
    logical = mod._derive_logical_path(artifact, Path("C:/tmp/data_train.parquet"))
    assert logical == "15/acquisition/data_train.parquet"


def test_derive_logical_path_falls_back_to_legacy_prefix() -> None:
    mod = _mod()
    artifact = SimpleNamespace(
        id=99,
        artifact_type="unknown_artifact",
        metadata_json={},
    )
    logical = mod._derive_logical_path(artifact, Path("C:/tmp/payload.bin"))
    assert logical.startswith("legacy_remediation/99_")
    assert logical.endswith(".bin")


def test_replace_uri_values_rewrites_nested_mirror_payloads() -> None:
    mod = _mod()
    payload = {
        "uri": "C:/tmp/local.txt",
        "nested": {"items": ["keep", "C:/tmp/local.txt"]},
    }
    updated, changed = mod._replace_uri_values(
        payload,
        old_uri="C:/tmp/local.txt",
        new_uri="https://acct.blob.core.windows.net/container/path.txt",
    )
    assert changed is True
    assert updated["uri"].startswith("https://")
    assert updated["nested"]["items"][1].startswith("https://")

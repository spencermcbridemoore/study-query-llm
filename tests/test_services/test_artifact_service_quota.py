"""Tests for artifact storage quota hard-fail behavior."""

from __future__ import annotations

import pytest

from study_query_llm.services.artifact_service import ArtifactService


class _FakeAzureStorage:
    backend_type = "azure_blob"

    def __init__(self, existing_bytes: int):
        self._existing_bytes = int(existing_bytes)

    def get_total_bytes(self) -> int:
        return int(self._existing_bytes)

    def write(self, logical_path: str, data: bytes, content_type=None) -> str:
        return f"azure://{logical_path}"

    def read(self, logical_path: str) -> bytes:
        raise NotImplementedError

    def read_from_uri(self, uri: str) -> bytes:
        raise NotImplementedError

    def exists(self, logical_path: str) -> bool:
        return False

    def exists_from_uri(self, uri: str) -> bool:
        return False

    def delete(self, logical_path: str) -> None:
        return None

    def get_uri(self, logical_path: str) -> str:
        return f"azure://{logical_path}"


class _FakeLocalStorage:
    backend_type = "local"

    def write(self, logical_path: str, data: bytes, content_type=None) -> str:
        return f"file://{logical_path}"

    def read(self, logical_path: str) -> bytes:
        raise NotImplementedError

    def read_from_uri(self, uri: str) -> bytes:
        raise NotImplementedError

    def exists(self, logical_path: str) -> bool:
        return False

    def exists_from_uri(self, uri: str) -> bool:
        return False

    def delete(self, logical_path: str) -> None:
        return None

    def get_uri(self, logical_path: str) -> str:
        return f"file://{logical_path}"


def test_azure_quota_guard_blocks_write_when_over_limit(monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_BLOB_MAX_BYTES", "100")
    svc = ArtifactService(
        repository=None,
        storage_backend=_FakeAzureStorage(existing_bytes=95),
    )
    with pytest.raises(RuntimeError, match="Artifact quota exceeded"):
        svc.store_sweep_results(
            run_id=1,
            sweep_results={"payload": "x" * 32},
            step_name="quota_test",
        )


def test_non_azure_backend_skips_quota_guard(monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_BLOB_MAX_BYTES", "1")
    svc = ArtifactService(
        repository=None,
        storage_backend=_FakeLocalStorage(),
    )
    artifact_id = svc.store_sweep_results(
        run_id=1,
        sweep_results={"payload": "x" * 32},
        step_name="quota_test_local",
    )
    assert artifact_id == 0

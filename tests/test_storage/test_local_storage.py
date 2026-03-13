"""Tests for LocalStorageBackend."""

import pytest

from study_query_llm.storage import StorageBackend
from study_query_llm.storage.local import LocalStorageBackend


def test_write_read_roundtrip(tmp_path):
    """write() then read() returns the same data."""
    backend = LocalStorageBackend(base_dir=tmp_path)
    logical_path = "run_42/step/sweep_results.json"
    data = b'{"by_k": {"5": {"labels": [0, 1, 0]}}}'

    backend.write(logical_path, data)
    result = backend.read(logical_path)

    assert result == data


def test_exists(tmp_path):
    """exists() returns True for written path, False for missing."""
    backend = LocalStorageBackend(base_dir=tmp_path)

    assert backend.exists("missing/file.json") is False

    backend.write("run_1/artifacts/data.json", b"{}")
    assert backend.exists("run_1/artifacts/data.json") is True


def test_delete(tmp_path):
    """delete() removes the file."""
    backend = LocalStorageBackend(base_dir=tmp_path)
    logical_path = "run_1/data.json"
    backend.write(logical_path, b"x")

    assert backend.exists(logical_path) is True

    backend.delete(logical_path)
    assert backend.exists(logical_path) is False


def test_get_uri_returns_absolute_path(tmp_path):
    """get_uri() returns absolute path."""
    backend = LocalStorageBackend(base_dir=tmp_path)
    logical_path = "run_1/data.json"
    backend.write(logical_path, b"x")

    uri = backend.get_uri(logical_path)

    assert str(tmp_path.resolve()) in uri
    assert "run_1" in uri
    assert "data.json" in uri
    assert not uri.startswith(".")  # Should be absolute


def test_backend_type():
    """backend_type is 'local'."""
    backend = LocalStorageBackend(base_dir="artifacts")
    assert backend.backend_type == "local"


def test_implements_storage_backend_protocol():
    """LocalStorageBackend satisfies StorageBackend protocol."""
    backend = LocalStorageBackend(base_dir="artifacts")
    assert isinstance(backend, StorageBackend)


def test_write_creates_parent_dirs(tmp_path):
    """write() creates parent directories."""
    backend = LocalStorageBackend(base_dir=tmp_path)
    logical_path = "deep/nested/path/file.json"

    backend.write(logical_path, b"data")

    full_path = tmp_path / "deep" / "nested" / "path" / "file.json"
    assert full_path.exists()
    assert full_path.read_bytes() == b"data"

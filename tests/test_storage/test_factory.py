"""Tests for StorageBackendFactory."""

from unittest.mock import MagicMock, patch

import pytest

from study_query_llm.storage import StorageBackend, StorageBackendFactory
from study_query_llm.storage.local import LocalStorageBackend


def test_create_local_returns_local_storage_backend():
    """StorageBackendFactory.create('local') returns LocalStorageBackend."""
    backend = StorageBackendFactory.create("local", base_dir="artifacts")
    assert isinstance(backend, LocalStorageBackend)
    assert isinstance(backend, StorageBackend)
    assert backend.backend_type == "local"


def test_create_azure_blob_returns_azure_backend():
    """StorageBackendFactory.create('azure_blob') returns AzureBlobStorageBackend."""
    with patch(
        "study_query_llm.storage.azure_blob.BlobServiceClient"
    ) as mock_client:
        mock_client.from_connection_string.return_value = MagicMock()
        backend = StorageBackendFactory.create(
            "azure_blob",
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;...",
        )
        assert backend.backend_type == "azure_blob"
        assert isinstance(backend, StorageBackend)


def test_create_unknown_raises():
    """StorageBackendFactory.create with unknown type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown storage backend"):
        StorageBackendFactory.create("unknown")


def test_available_backends():
    """available_backends returns local and azure_blob."""
    backends = StorageBackendFactory.available_backends()
    assert "local" in backends
    assert "azure_blob" in backends

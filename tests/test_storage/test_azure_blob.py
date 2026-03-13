"""Tests for AzureBlobStorageBackend (mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from study_query_llm.storage import StorageBackend
from study_query_llm.storage.azure_blob import AzureBlobStorageBackend


@pytest.fixture
def mock_blob_service():
    """Create mocked BlobServiceClient and container/blob clients."""
    mock_container = MagicMock()
    mock_blob = MagicMock()
    mock_blob.url = "https://account.blob.core.windows.net/artifacts/path/to/blob.json"
    mock_container.get_blob_client.return_value = mock_blob

    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container

    with patch(
        "study_query_llm.storage.azure_blob.BlobServiceClient"
    ) as mock_client_class:
        mock_client_class.from_connection_string.return_value = mock_service
        yield mock_service, mock_container, mock_blob


def test_write_calls_upload_blob(mock_blob_service):
    """write() calls upload_blob on the blob client."""
    mock_service, mock_container, mock_blob = mock_blob_service

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    uri = backend.write("run_42/sweep_results.json", b'{"data": 1}')

    mock_blob.upload_blob.assert_called_once()
    call_args = mock_blob.upload_blob.call_args
    assert call_args[0][0] == b'{"data": 1}'
    assert call_args[1].get("overwrite") is True
    assert "https://" in uri


def test_read_calls_download_blob(mock_blob_service):
    """read() calls download_blob and returns readall()."""
    mock_service, mock_container, mock_blob = mock_blob_service
    mock_stream = MagicMock()
    mock_stream.readall.return_value = b"binary data"
    mock_blob.download_blob.return_value = mock_stream

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    result = backend.read("path/to/blob.bin")

    mock_blob.download_blob.assert_called_once()
    assert result == b"binary data"


def test_exists_returns_true_when_blob_exists(mock_blob_service):
    """exists() returns True when get_blob_properties succeeds."""
    mock_service, mock_container, mock_blob = mock_blob_service

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    result = backend.exists("path/to/blob.json")

    mock_blob.get_blob_properties.assert_called_once()
    assert result is True


def test_exists_returns_false_when_blob_missing(mock_blob_service):
    """exists() returns False when get_blob_properties raises."""
    mock_service, mock_container, mock_blob = mock_blob_service
    mock_blob.get_blob_properties.side_effect = Exception("BlobNotFound")

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    result = backend.exists("path/to/missing.json")

    assert result is False


def test_get_uri_format(mock_blob_service):
    """get_uri() returns URL from blob client."""
    mock_service, mock_container, mock_blob = mock_blob_service
    mock_blob.url = "https://myaccount.blob.core.windows.net/artifacts/sweeps/run1.json"

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    uri = backend.get_uri("sweeps/run1.json")

    assert uri == "https://myaccount.blob.core.windows.net/artifacts/sweeps/run1.json"
    assert "blob.core.windows.net" in uri


def test_delete_calls_delete_blob(mock_blob_service):
    """delete() calls delete_blob on the blob client."""
    mock_service, mock_container, mock_blob = mock_blob_service

    backend = AzureBlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
    )

    backend.delete("path/to/old.json")

    mock_blob.delete_blob.assert_called_once()


def test_backend_type():
    """backend_type is 'azure_blob'."""
    with patch(
        "study_query_llm.storage.azure_blob.BlobServiceClient"
    ) as mock_client_class:
        mock_client_class.from_connection_string.return_value = MagicMock()
        backend = AzureBlobStorageBackend(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
        )
        assert backend.backend_type == "azure_blob"


def test_implements_storage_backend_protocol():
    """AzureBlobStorageBackend satisfies StorageBackend protocol."""
    with patch(
        "study_query_llm.storage.azure_blob.BlobServiceClient"
    ) as mock_client_class:
        mock_client_class.from_connection_string.return_value = MagicMock()
        backend = AzureBlobStorageBackend(
            connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
        )
        assert isinstance(backend, StorageBackend)


def test_requires_azure_storage_blob():
    """AzureBlobStorageBackend raises ImportError when azure-storage-blob not installed."""
    with patch("study_query_llm.storage.azure_blob.BlobServiceClient", None):
        with pytest.raises(ImportError, match="azure-storage-blob"):
            AzureBlobStorageBackend(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test;..."
            )

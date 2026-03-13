"""
AzureBlobStorageBackend - store artifacts in Azure Blob Storage.
"""

from __future__ import annotations

import os
from typing import Optional

from .protocol import StorageBackend

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except ImportError:
    BlobServiceClient = None  # type: ignore
    ContentSettings = None  # type: ignore


class AzureBlobStorageBackend:
    """
    Storage backend that writes artifacts to Azure Blob Storage.

    Uses connection_string or account_url+credential. logical_path becomes
    the blob name (object key). get_uri() returns the blob URL.
    """

    backend_type: str = "azure_blob"

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_url: Optional[str] = None,
        credential: Optional[object] = None,
        container_name: str = "artifacts",
    ) -> None:
        """
        Initialize the Azure Blob storage backend.

        Args:
            connection_string: Azure Storage connection string (from env if not set)
            account_url: Alternative: account URL (e.g. https://account.blob.core.windows.net)
            credential: Alternative: credential object (e.g. DefaultAzureCredential)
            container_name: Blob container name (default: "artifacts")
        """
        if BlobServiceClient is None:
            raise ImportError(
                "azure-storage-blob is required for AzureBlobStorageBackend. "
                "Install with: pip install azure-storage-blob"
            )

        conn_str = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = container_name or os.environ.get(
            "AZURE_STORAGE_CONTAINER", "artifacts"
        )

        if conn_str:
            self._client = BlobServiceClient.from_connection_string(conn_str)
        elif account_url and credential:
            self._client = BlobServiceClient(account_url=account_url, credential=credential)
        else:
            raise ValueError(
                "Provide connection_string or (account_url + credential). "
                "Or set AZURE_STORAGE_CONNECTION_STRING."
            )

        self._container_client = self._client.get_container_client(self.container_name)

    def write(
        self,
        logical_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Upload data to blob at logical_path.

        Args:
            logical_path: Blob name (object key)
            data: Raw bytes to upload
            content_type: Optional MIME type for Content-Type header

        Returns:
            The blob URI (https://...).
        """
        blob_client = self._container_client.get_blob_client(logical_path)
        content_settings = ContentSettings(content_type=content_type) if content_type else None
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=content_settings,
        )
        return self.get_uri(logical_path)

    def read(self, logical_path: str) -> bytes:
        """
        Download blob at logical_path.

        Args:
            logical_path: Blob name (object key)

        Returns:
            Raw bytes of the blob.

        Raises:
            Exception: If blob does not exist.
        """
        blob_client = self._container_client.get_blob_client(logical_path)
        stream = blob_client.download_blob()
        return stream.readall()

    def exists(self, logical_path: str) -> bool:
        """
        Check if blob exists.

        Args:
            logical_path: Blob name (object key)

        Returns:
            True if the blob exists.
        """
        blob_client = self._container_client.get_blob_client(logical_path)
        try:
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False

    def delete(self, logical_path: str) -> None:
        """
        Delete blob at logical_path.

        Args:
            logical_path: Blob name (object key)
        """
        blob_client = self._container_client.get_blob_client(logical_path)
        try:
            blob_client.delete_blob()
        except Exception:
            pass

    def get_uri(self, logical_path: str) -> str:
        """
        Return the blob URL for logical_path.

        Args:
            logical_path: Blob name (object key)

        Returns:
            Full blob URL (https://account.blob.core.windows.net/container/path).
        """
        blob_client = self._container_client.get_blob_client(logical_path)
        return blob_client.url

    def read_from_uri(self, uri: str) -> bytes:
        """
        Read data from a URI previously returned by get_uri().

        Args:
            uri: Blob URL (from get_uri)

        Returns:
            Raw bytes of the blob.
        """
        blob_path = self._blob_path_from_uri(uri)
        return self.read(blob_path)

    def exists_from_uri(self, uri: str) -> bool:
        """
        Check if blob exists at the given URI.

        Args:
            uri: Blob URL (from get_uri)

        Returns:
            True if the blob exists.
        """
        blob_path = self._blob_path_from_uri(uri)
        return self.exists(blob_path)

    def _blob_path_from_uri(self, uri: str) -> str:
        """Extract blob path from blob URL."""
        from urllib.parse import urlparse, unquote

        parsed = urlparse(uri)
        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) < 2:
            raise ValueError(f"Cannot parse blob path from URI: {uri}")
        return unquote(path_parts[1])

"""
StorageBackendFactory - create storage backends by type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import StorageBackend


class StorageBackendFactory:
    """Factory for creating storage backends."""

    @staticmethod
    def create(backend_type: str = "local", **kwargs) -> "StorageBackend":
        """
        Create a storage backend by type.

        Args:
            backend_type: "local" or "azure_blob"
            **kwargs: Passed to the backend constructor

        Returns:
            StorageBackend instance

        Raises:
            ValueError: If backend_type is unknown
        """
        if backend_type == "local":
            from .local import LocalStorageBackend

            return LocalStorageBackend(**kwargs)
        if backend_type == "azure_blob":
            from .azure_blob import AzureBlobStorageBackend

            return AzureBlobStorageBackend(**kwargs)
        raise ValueError(f"Unknown storage backend: {backend_type}")

    @staticmethod
    def available_backends() -> list[str]:
        """Return list of available backend type names."""
        return ["local", "azure_blob"]

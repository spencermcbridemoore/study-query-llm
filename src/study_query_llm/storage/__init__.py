"""
Storage backends for artifact persistence (local filesystem, Azure Blob, etc.).

Provides the StorageBackend protocol and factory for creating backend instances.
"""

from .factory import StorageBackendFactory
from .local import LocalStorageBackend
from .protocol import StorageBackend

__all__ = [
    "StorageBackend",
    "StorageBackendFactory",
    "LocalStorageBackend",
]

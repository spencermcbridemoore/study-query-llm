"""
StorageBackend Protocol -- shared interface for artifact storage.

logical_path is the canonical identity (e.g. "artifacts/42/embedding_matrix/embedding_matrix.npy").
get_uri() returns the physical URI (local path or blob URL).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Structural interface for storage backends."""

    backend_type: str

    def write(
        self,
        logical_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str: ...
    def read(self, logical_path: str) -> bytes: ...
    def read_from_uri(self, uri: str) -> bytes: ...
    def exists(self, logical_path: str) -> bool: ...
    def exists_from_uri(self, uri: str) -> bool: ...
    def delete(self, logical_path: str) -> None: ...
    def get_uri(self, logical_path: str) -> str: ...

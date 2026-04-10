"""
LocalStorageBackend - store artifacts on the local filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class LocalStorageBackend:
    """
    Storage backend that writes artifacts to the local filesystem.

    logical_path is relative to base_dir. get_uri() returns the absolute path.
    """

    backend_type: str = "local"

    def __init__(self, base_dir: str | Path = "artifacts") -> None:
        """
        Initialize the local storage backend.

        Args:
            base_dir: Base directory for storing artifacts (default: "artifacts")
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _full_path(self, logical_path: str) -> Path:
        """Resolve logical_path to full filesystem path."""
        # Normalize path to avoid traversal
        parts = Path(logical_path).parts
        if not parts:
            raise ValueError("logical_path cannot be empty")
        return self.base_dir.joinpath(*parts)

    def write(
        self,
        logical_path: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """
        Write data to logical_path.

        Args:
            logical_path: Path relative to base_dir (e.g. "42/sweep_complete/sweep_results.json")
            data: Raw bytes to write
            content_type: Optional MIME type (ignored for local storage)

        Returns:
            The URI (absolute path) of the written artifact.
        """
        path = self._full_path(logical_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.get_uri(logical_path)

    def read(self, logical_path: str) -> bytes:
        """
        Read data from logical_path.

        Args:
            logical_path: Path relative to base_dir

        Returns:
            Raw bytes of the artifact.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = self._full_path(logical_path)
        return path.read_bytes()

    def exists(self, logical_path: str) -> bool:
        """
        Check if logical_path exists.

        Args:
            logical_path: Path relative to base_dir

        Returns:
            True if the artifact exists.
        """
        path = self._full_path(logical_path)
        return path.exists()

    def delete(self, logical_path: str) -> None:
        """
        Delete the artifact at logical_path.

        Args:
            logical_path: Path relative to base_dir
        """
        path = self._full_path(logical_path)
        if path.exists():
            path.unlink()

    def get_uri(self, logical_path: str) -> str:
        """
        Return the physical URI (absolute path) for logical_path.

        Args:
            logical_path: Path relative to base_dir

        Returns:
            Absolute filesystem path as string.
        """
        path = self._full_path(logical_path)
        return str(path.resolve())

    def get_total_bytes(self) -> int:
        """Return total bytes currently stored under base_dir."""
        total = 0
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                total += int(path.stat().st_size)
        return total

    def read_from_uri(self, uri: str) -> bytes:
        """
        Read data from a URI previously returned by get_uri().

        Args:
            uri: Absolute filesystem path (from get_uri)

        Returns:
            Raw bytes of the artifact.
        """
        path = Path(uri)
        return path.read_bytes()

    def exists_from_uri(self, uri: str) -> bool:
        """
        Check if artifact exists at the given URI.

        Args:
            uri: Absolute filesystem path (from get_uri)

        Returns:
            True if the artifact exists.
        """
        return Path(uri).exists()

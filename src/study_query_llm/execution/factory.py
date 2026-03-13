"""
ExecutionBackendFactory - creates execution backend instances by type.
"""

from __future__ import annotations

from .protocol import ExecutionBackend


class ExecutionBackendFactory:
    """Factory for creating execution backend instances."""

    @staticmethod
    def create(backend_type: str, **kwargs) -> ExecutionBackend:
        """
        Create an execution backend by type.

        Args:
            backend_type: One of "local_docker", "ssh_docker", "vastai".
            **kwargs: Backend-specific constructor arguments.

        Returns:
            An ExecutionBackend instance.

        Raises:
            ValueError: If backend_type is unknown.
        """
        backend_type = backend_type.lower()
        if backend_type == "local_docker":
            from .local_docker import LocalDockerExecution

            return LocalDockerExecution(**kwargs)
        if backend_type == "ssh_docker":
            from .ssh_docker import SSHDockerExecution

            return SSHDockerExecution(**kwargs)
        if backend_type == "vastai":
            from .vastai import VastAIExecution

            return VastAIExecution(**kwargs)
        raise ValueError(f"Unknown backend: {backend_type}")

    @staticmethod
    def available_backends() -> list[str]:
        """Return list of supported backend type names."""
        return ["local_docker", "ssh_docker", "vastai"]

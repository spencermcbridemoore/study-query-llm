"""Tests for ExecutionBackendFactory."""

import pytest

from study_query_llm.execution import ExecutionBackend, ExecutionBackendFactory


def test_create_local_docker_returns_backend():
    """create('local_docker') returns a LocalDockerExecution instance."""
    backend = ExecutionBackendFactory.create("local_docker")
    assert backend is not None
    assert isinstance(backend, ExecutionBackend)
    assert backend.backend_type == "local_docker"


def test_create_unknown_raises():
    """create('unknown') raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        ExecutionBackendFactory.create("unknown")


def test_available_backends_includes_local_docker():
    """available_backends() includes 'local_docker'."""
    backends = ExecutionBackendFactory.available_backends()
    assert "local_docker" in backends

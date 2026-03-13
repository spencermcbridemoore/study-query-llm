"""Tests for LocalDockerExecution backend."""

from unittest.mock import MagicMock, patch

import pytest

from study_query_llm.execution import (
    ExecutionBackend,
    JobSpec,
    JobState,
    ResourceSpec,
)
from study_query_llm.execution.local_docker import LocalDockerExecution


def _make_docker_client(container=None):
    """Return a mock docker client."""
    client = MagicMock()
    mock_container = container or MagicMock()
    mock_container.id = "abc123def456"
    mock_container.attrs = {
        "State": {
            "Status": "running",
            "ExitCode": None,
            "StartedAt": "2024-01-01T00:00:00Z",
            "FinishedAt": None,
        }
    }
    client.containers.run.return_value = mock_container
    client.containers.get.return_value = mock_container
    return client, mock_container


def _patch_docker(client):
    return patch(
        "study_query_llm.execution.local_docker.docker.from_env",
        return_value=client,
    )


def test_submit_returns_container_id():
    """submit() returns the container ID as job_ref."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client):
        job_ref = backend.submit(
            JobSpec(image="alpine:latest", command=["echo", "hello"])
        )

    assert job_ref == "abc123def456"
    docker_client.containers.run.assert_called_once()
    call_args = docker_client.containers.run.call_args
    # containers.run(image, command, **kwargs) - image and command can be positional
    assert call_args[0][0] == "alpine:latest"
    assert call_args[0][1] == ["echo", "hello"]
    assert call_args[1].get("detach") is True


def test_poll_running():
    """poll() returns RUNNING when container is running."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()
    mock_container.attrs = {
        "State": {
            "Status": "running",
            "ExitCode": None,
            "StartedAt": "2024-01-01T00:00:00Z",
            "FinishedAt": None,
        }
    }

    with _patch_docker(docker_client):
        status = backend.poll("abc123def456")

    assert status.state == JobState.RUNNING
    assert status.exit_code is None


def test_poll_succeeded():
    """poll() returns SUCCEEDED when container exited with 0."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()
    mock_container.attrs = {
        "State": {
            "Status": "exited",
            "ExitCode": 0,
            "StartedAt": "2024-01-01T00:00:00Z",
            "FinishedAt": "2024-01-01T00:01:00Z",
        }
    }

    with _patch_docker(docker_client):
        status = backend.poll("abc123def456")

    assert status.state == JobState.SUCCEEDED
    assert status.exit_code == 0


def test_poll_failed():
    """poll() returns FAILED when container exited with non-zero."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()
    mock_container.attrs = {
        "State": {
            "Status": "exited",
            "ExitCode": 1,
            "StartedAt": "2024-01-01T00:00:00Z",
            "FinishedAt": "2024-01-01T00:01:00Z",
        }
    }

    with _patch_docker(docker_client):
        status = backend.poll("abc123def456")

    assert status.state == JobState.FAILED
    assert status.exit_code == 1


def test_cancel_stops_container():
    """cancel() stops and removes the container."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client):
        backend.cancel("abc123def456")

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_logs_returns_output():
    """logs() returns container output as string."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()
    mock_container.logs.return_value = b"hello world\nline two"

    with _patch_docker(docker_client):
        output = backend.logs("abc123def456", tail=50)

    assert output == "hello world\nline two"
    mock_container.logs.assert_called_once_with(tail=50)


def test_gpu_device_requests():
    """submit() passes device_requests when gpu_count > 0."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client):
        backend.submit(
            JobSpec(
                image="nvidia/cuda:12.0",
                command=["nvidia-smi"],
                resources=ResourceSpec(gpu_count=1),
            )
        )

    call_args = docker_client.containers.run.call_args
    call_kw = call_args[1] if len(call_args) > 1 else {}
    device_requests = call_kw.get("device_requests", [])
    assert len(device_requests) == 1
    assert device_requests[0].capabilities == [["gpu"]]
    assert device_requests[0].count == 1


def test_no_gpu_when_gpu_count_zero():
    """submit() passes no device_requests when gpu_count is 0."""
    backend = LocalDockerExecution()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client):
        backend.submit(
            JobSpec(
                image="alpine:latest",
                command=["echo", "hi"],
                resources=ResourceSpec(gpu_count=0),
            )
        )

    call_args = docker_client.containers.run.call_args
    call_kw = call_args[1] if len(call_args) > 1 else {}
    device_requests = call_kw.get("device_requests", [])
    assert device_requests == []


def test_backend_type():
    """backend_type is 'local_docker'."""
    backend = LocalDockerExecution()
    assert backend.backend_type == "local_docker"


def test_implements_execution_backend_protocol():
    """LocalDockerExecution satisfies ExecutionBackend protocol."""
    backend = LocalDockerExecution()
    assert isinstance(backend, ExecutionBackend)


def test_poll_container_not_found():
    """poll() returns FAILED when container does not exist."""
    backend = LocalDockerExecution()
    docker_client, _ = _make_docker_client()
    docker_client.containers.get.side_effect = Exception("not found")

    with _patch_docker(docker_client):
        status = backend.poll("nonexistent")

    assert status.state == JobState.FAILED
    assert "not found" in (status.error_message or "").lower()

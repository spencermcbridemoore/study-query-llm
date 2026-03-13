"""Tests for SSHDockerExecution (mocked subprocess)."""

from unittest.mock import patch

import pytest

from study_query_llm.execution import (
    ExecutionBackend,
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)
from study_query_llm.execution.ssh_docker import SSHDockerExecution


@pytest.fixture
def backend():
    """Create SSHDockerExecution with test config."""
    return SSHDockerExecution(
        host="remote.example.com",
        user="ubuntu",
        ssh_key_path="/path/to/key",
    )


def test_submit_builds_correct_ssh_command(backend):
    """submit() builds correct SSH docker run command."""
    spec = JobSpec(
        image="alpine:latest",
        command=["echo", "hello"],
        name="myjob",
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "abc123\n", "stderr": ""})()
        job_ref = backend.submit(spec)

    assert job_ref == "abc123"
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert "ssh" in call_args
    assert "-i" in call_args
    assert "/path/to/key" in call_args
    assert "ubuntu@remote.example.com" in call_args
    cmd_str = " ".join(call_args)
    assert "docker" in cmd_str
    assert "run" in cmd_str
    assert "-d" in cmd_str
    assert "myjob" in cmd_str
    assert "alpine:latest" in cmd_str


def test_poll_parses_docker_inspect_output(backend):
    """poll() parses docker inspect output and maps to JobState."""
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = "running"
        status = backend.poll("abc123")

    assert status.state == JobState.RUNNING
    mock_ssh.assert_called_once()
    call_cmd = mock_ssh.call_args[0][0]
    assert "inspect" in call_cmd
    assert "abc123" in call_cmd


def test_poll_exited_returns_succeeded(backend):
    """poll() maps 'exited' to SUCCEEDED."""
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = "exited"
        status = backend.poll("abc123")
    assert status.state == JobState.SUCCEEDED


def test_cancel_sends_stop_and_rm(backend):
    """cancel() sends docker stop and rm via SSH."""
    with patch.object(backend, "_ssh_run") as mock_ssh:
        backend.cancel("abc123")
    mock_ssh.assert_called_once()
    call_cmd = mock_ssh.call_args[0][0]
    assert "stop" in call_cmd
    assert "rm" in call_cmd
    assert "abc123" in call_cmd


def test_logs_returns_output(backend):
    """logs() returns output from docker logs."""
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.return_value = "line1\nline2\nline3"
        result = backend.logs("abc123", tail=50)
    assert result == "line1\nline2\nline3"
    mock_ssh.assert_called_once()
    call_cmd = mock_ssh.call_args[0][0]
    assert "logs" in call_cmd
    assert "50" in call_cmd
    assert "abc123" in call_cmd


def test_ssh_errors_raise_runtime_error(backend):
    """SSH command failures raise RuntimeError."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R", (), {"returncode": 1, "stdout": "", "stderr": "Permission denied"}
        )()
        with pytest.raises(RuntimeError, match="SSH command failed"):
            backend._ssh_run("docker ps")


def test_poll_ssh_error_returns_failed_status(backend):
    """poll() returns FAILED JobStatus when SSH raises."""
    with patch.object(backend, "_ssh_run") as mock_ssh:
        mock_ssh.side_effect = RuntimeError("Connection refused")
        status = backend.poll("abc123")
    assert status.state == JobState.FAILED
    assert "Connection refused" in (status.error_message or "")


def test_backend_type():
    """backend_type is 'ssh_docker'."""
    backend = SSHDockerExecution(host="h", user="u")
    assert backend.backend_type == "ssh_docker"


def test_implements_execution_backend_protocol():
    """SSHDockerExecution satisfies ExecutionBackend protocol."""
    backend = SSHDockerExecution(host="h", user="u")
    assert isinstance(backend, ExecutionBackend)


def test_gpu_flag_all(backend):
    """submit() passes --gpus all when gpu_count < 0."""
    spec = JobSpec(
        image="nvidia/cuda:latest",
        command=["nvidia-smi"],
        resources=ResourceSpec(gpu_count=-1),
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "cid\n", "stderr": ""})()
        backend.submit(spec)
    call_args = mock_run.call_args[0][0]
    cmd_str = " ".join(call_args)
    assert "--gpus all" in cmd_str

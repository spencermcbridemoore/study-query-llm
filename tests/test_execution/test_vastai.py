"""Tests for VastAIExecution (mocked subprocess)."""

from unittest.mock import patch

import pytest

from study_query_llm.execution import (
    ExecutionBackend,
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)
from study_query_llm.execution.vastai import VastAIExecution


@pytest.fixture
def backend():
    """Create VastAIExecution with test api_key."""
    return VastAIExecution(api_key="test-api-key-123")


def test_submit_searches_offers_then_creates_instance(backend):
    """submit() searches offers then creates instance."""
    with patch.object(backend, "_vastai_cmd") as mock_cmd:
        mock_cmd.side_effect = [
            '[{"id": 12345, "dph": 0.5}]',
            '{"success": true, "new_contract": 99999}',
        ]
        job_ref = backend.submit(
            JobSpec(image="alpine:latest", command=["echo", "hello"])
        )

    assert job_ref == "99999"
    assert mock_cmd.call_count == 2
    first_call = mock_cmd.call_args_list[0][0][0]
    assert "search" in first_call
    assert "offers" in first_call
    second_call = mock_cmd.call_args_list[1][0][0]
    assert "create" in second_call
    assert "instance" in second_call
    assert "alpine:latest" in second_call


def test_poll_parses_instance_status(backend):
    """poll() parses vastai show instance output and maps to JobState."""
    with patch.object(backend, "_vastai_cmd") as mock_cmd:
        mock_cmd.return_value = '{"status": "running"}'
        status = backend.poll("99999")

    assert status.state == JobState.RUNNING
    mock_cmd.assert_called_once()
    call_args = mock_cmd.call_args[0][0]
    assert "show" in call_args
    assert "instance" in call_args
    assert "99999" in call_args


def test_poll_exited_returns_succeeded(backend):
    """poll() maps 'exited' to SUCCEEDED."""
    with patch.object(backend, "_vastai_cmd") as mock_cmd:
        mock_cmd.return_value = '{"status": "exited"}'
        status = backend.poll("99999")
    assert status.state == JobState.SUCCEEDED


def test_cancel_destroys_instance(backend):
    """cancel() calls vastai destroy instance."""
    with patch.object(backend, "_vastai_cmd") as mock_cmd:
        backend.cancel("99999")
    mock_cmd.assert_called_once()
    call_args = mock_cmd.call_args[0][0]
    assert "destroy" in call_args
    assert "instance" in call_args
    assert "99999" in call_args


def test_poll_cli_error_returns_failed_status(backend):
    """poll() returns FAILED JobStatus when CLI raises."""
    with patch.object(backend, "_vastai_cmd") as mock_cmd:
        mock_cmd.side_effect = RuntimeError("Instance not found")
        status = backend.poll("99999")
    assert status.state == JobState.FAILED
    assert "Instance not found" in (status.error_message or "")


def test_backend_type():
    """backend_type is 'vastai'."""
    with patch.dict("os.environ", {"VASTAI_API_KEY": "x"}):
        backend = VastAIExecution()
        assert backend.backend_type == "vastai"


def test_implements_execution_backend_protocol():
    """VastAIExecution satisfies ExecutionBackend protocol."""
    backend = VastAIExecution(api_key="test")
    assert isinstance(backend, ExecutionBackend)


def test_requires_api_key():
    """VastAIExecution raises ValueError when no api_key and no env."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="api_key or VASTAI_API_KEY"):
            VastAIExecution()


def test_uses_env_api_key():
    """VastAIExecution uses VASTAI_API_KEY when api_key not passed."""
    with patch.dict("os.environ", {"VASTAI_API_KEY": "env-key"}):
        backend = VastAIExecution()
        assert backend.api_key == "env-key"

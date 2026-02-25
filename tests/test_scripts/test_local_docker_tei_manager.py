"""Tests for scripts.common.local_docker_tei_manager.LocalDockerTEIManager."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from scripts.common.local_docker_tei_manager import LocalDockerTEIManager, _TEI_GPU_IMAGE, _TEI_CPU_IMAGE


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_manager(**kwargs) -> LocalDockerTEIManager:
    defaults = dict(
        model_id="BAAI/bge-m3",
        port=8080,
        idle_timeout_seconds=300,
        hf_cache_dir="/tmp/hf_cache",
        health_check_timeout=30,
        health_check_interval=1,
    )
    defaults.update(kwargs)
    return LocalDockerTEIManager(**defaults)


def _make_docker_client(container=None):
    """Return a mock docker client."""
    client = MagicMock()
    mock_container = container or MagicMock()
    client.containers.run.return_value = mock_container
    client.containers.get.side_effect = __import__("docker").errors.NotFound("not found")
    return client, mock_container


def _patch_docker(client):
    return patch("scripts.common.local_docker_tei_manager.docker.from_env", return_value=client)


def _patch_health():
    return patch.object(LocalDockerTEIManager, "_wait_for_healthy", return_value=None)


# ---------------------------------------------------------------------------
# provider_label
# ---------------------------------------------------------------------------

def test_provider_label():
    """provider_label is 'local_docker_tei'."""
    manager = _make_manager()
    assert manager.provider_label == "local_docker_tei"


# ---------------------------------------------------------------------------
# container_name
# ---------------------------------------------------------------------------

def test_default_container_name():
    """Default container name is derived from model_id."""
    manager = _make_manager(model_id="BAAI/bge-m3")
    assert manager.container_name == "tei-baai-bge-m3"


def test_explicit_container_name():
    """Explicit container_name overrides the default."""
    manager = _make_manager(container_name="my-tei")
    assert manager.container_name == "my-tei"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

def test_start_sets_endpoint_url():
    """start() sets endpoint_url to http://localhost:{port}/v1."""
    manager = _make_manager(port=8080)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        url = manager.start()

    assert url == "http://localhost:8080/v1"
    assert manager.endpoint_url == "http://localhost:8080/v1"


def test_start_runs_gpu_image_by_default():
    """start() uses the GPU image when use_gpu=True."""
    manager = _make_manager(use_gpu=True)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    image_arg = docker_client.containers.run.call_args.kwargs["image"]
    assert image_arg == _TEI_GPU_IMAGE


def test_start_runs_cpu_image_when_gpu_disabled():
    """start() uses the CPU image when use_gpu=False."""
    manager = _make_manager(use_gpu=False)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    image_arg = docker_client.containers.run.call_args.kwargs["image"]
    assert image_arg == _TEI_CPU_IMAGE


def test_start_passes_model_id_in_command():
    """start() passes --model-id to the container command."""
    manager = _make_manager(model_id="intfloat/e5-large-v2")
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    cmd = docker_client.containers.run.call_args[1].get("command") or \
          docker_client.containers.run.call_args[0][1]
    assert "intfloat/e5-large-v2" in cmd


def test_start_mounts_hf_cache():
    """start() mounts the hf_cache_dir volume (resolved to absolute path)."""
    import tempfile, os
    # Use a real temp dir so Path.resolve() is deterministic on Windows
    with tempfile.TemporaryDirectory() as tmp:
        manager = _make_manager(hf_cache_dir=tmp)
        docker_client, _ = _make_docker_client()

        with _patch_docker(docker_client), _patch_health():
            manager.start()

        volumes = docker_client.containers.run.call_args[1]["volumes"]
        resolved = str(Path(tmp).expanduser().resolve())
        assert resolved in volumes
        assert volumes[resolved]["bind"] == "/root/.cache/huggingface"


def test_start_binds_port():
    """start() binds the host port."""
    manager = _make_manager(port=9090)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    ports = docker_client.containers.run.call_args[1]["ports"]
    assert "80/tcp" in ports
    assert ports["80/tcp"] == 9090


def test_start_is_idempotent():
    """Calling start() twice returns cached endpoint without re-creating."""
    manager = _make_manager()
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.start()

    assert docker_client.containers.run.call_count == 1


def test_start_starts_idle_timer():
    """start() starts the idle timer."""
    manager = _make_manager(idle_timeout_seconds=9999)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    assert manager._idle_timer is not None
    manager._idle_timer.cancel()


def test_start_removes_stale_container():
    """start() removes a stale container with the same name before creating."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()
    stale = MagicMock()
    docker_client.containers.get.side_effect = None
    docker_client.containers.get.return_value = stale

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    stale.stop.assert_called_once()
    stale.remove.assert_called_once()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

def test_stop_calls_container_stop_and_remove():
    """stop() stops and removes the Docker container."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.stop()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_stop_clears_endpoint_url():
    """stop() sets endpoint_url to None."""
    manager = _make_manager()
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.stop()

    assert manager.endpoint_url is None


def test_stop_cancels_idle_timer():
    """stop() cancels the idle timer."""
    manager = _make_manager(idle_timeout_seconds=9999)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        timer = manager._idle_timer
        manager.stop()

    assert timer.finished.is_set()
    assert manager._idle_timer is None


def test_stop_is_idempotent():
    """Calling stop() multiple times is safe."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.stop()
        manager.stop()

    assert mock_container.stop.call_count == 1


def test_stop_handles_exception_gracefully():
    """stop() logs and swallows exceptions from Docker."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()
    mock_container.stop.side_effect = RuntimeError("gone")

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.stop()  # should not raise


# ---------------------------------------------------------------------------
# ping()
# ---------------------------------------------------------------------------

def test_ping_resets_idle_timer():
    """ping() cancels the existing timer and starts a new one."""
    manager = _make_manager(idle_timeout_seconds=9999)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        first_timer = manager._idle_timer
        manager.ping()
        second_timer = manager._idle_timer

    assert first_timer.finished.is_set()
    assert second_timer is not None
    assert second_timer is not first_timer
    second_timer.cancel()


def test_ping_noop_after_stop():
    """ping() does nothing after the container has been stopped."""
    manager = _make_manager(idle_timeout_seconds=9999)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        manager.stop()
        manager.ping()

    assert manager._idle_timer is None


# ---------------------------------------------------------------------------
# _idle_shutdown()
# ---------------------------------------------------------------------------

def test_idle_shutdown_calls_stop():
    """_idle_shutdown() calls stop() when idle timeout expires."""
    manager = _make_manager(idle_timeout_seconds=0.01)
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()
        time.sleep(0.2)

    assert mock_container.stop.call_count == 1


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_starts_and_stops():
    """__enter__ calls start(), __exit__ calls stop()."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        with manager:
            assert manager.endpoint_url is not None

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_context_manager_stops_on_exception():
    """__exit__ still calls stop() even if an exception is raised."""
    manager = _make_manager()
    docker_client, mock_container = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        with pytest.raises(RuntimeError):
            with manager:
                raise RuntimeError("sweep failed")

    mock_container.stop.assert_called_once()


# ---------------------------------------------------------------------------
# GPU device requests
# ---------------------------------------------------------------------------

def test_gpu_device_requests_included_when_use_gpu():
    """Device requests include GPU capabilities when use_gpu=True."""
    import docker.types

    manager = _make_manager(use_gpu=True, gpu_device="all")
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    device_requests = docker_client.containers.run.call_args[1]["device_requests"]
    assert len(device_requests) == 1
    assert device_requests[0].capabilities == [["gpu"]]
    assert device_requests[0].count == -1


def test_no_device_requests_when_cpu_only():
    """No device requests are passed when use_gpu=False."""
    manager = _make_manager(use_gpu=False)
    docker_client, _ = _make_docker_client()

    with _patch_docker(docker_client), _patch_health():
        manager.start()

    device_requests = docker_client.containers.run.call_args[1]["device_requests"]
    assert device_requests == []


# ---------------------------------------------------------------------------
# HF_CACHE_DIR env var fallback
# ---------------------------------------------------------------------------

def test_hf_cache_dir_from_env(monkeypatch):
    """hf_cache_dir defaults to HF_CACHE_DIR env var if not passed."""
    monkeypatch.setenv("HF_CACHE_DIR", "/env/hf_cache")
    manager = LocalDockerTEIManager(model_id="BAAI/bge-m3")
    assert manager.hf_cache_dir == "/env/hf_cache"


def test_hf_cache_dir_default(monkeypatch):
    """hf_cache_dir falls back to ~/.cache/huggingface if env var not set."""
    monkeypatch.delenv("HF_CACHE_DIR", raising=False)
    manager = LocalDockerTEIManager(model_id="BAAI/bge-m3")
    assert manager.hf_cache_dir == str(Path.home() / ".cache" / "huggingface")

"""Fully-mocked unit tests for the vLLM local-Docker backend.

Mirrors the docker-mocking style of
``tests/test_scripts/test_local_docker_tei_manager.py``: a mock docker client is
injected (here through the ``docker_client_factory`` seam rather than by patching
``docker.from_env``), and the VRAM probe + HF snapshot download are replaced with
fakes.  NOTHING in this file touches the network, the docker daemon, a GPU, or
HuggingFace -- every side-effecting dependency of
:class:`study_query_llm.vllm_serving.backends.LocalDockerBackend` is supplied as
an injected stub.

Covered surface:

* ``containers.run`` kwargs: image, host->internal port map, command flags,
  GPU ``device_requests``, stale-container removal.
* The TLS-interception offline path: ``/model`` read-only mount + offline env.
* The display-safety guard: a tiny-free-VRAM probe raises ``VRAMSafetyError``
  *before* ``containers.run`` is ever called.
* ``stop()`` idempotency + error swallowing, and the returned ``/v1`` URL.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import docker  # installed in this env; used only for docker.errors.NotFound
import pytest

from study_query_llm.vllm_serving.backends import LocalDockerBackend
from study_query_llm.vllm_serving.config import (
    INTERNAL_PORT,
    VLLM_IMAGE,
    VLLMServeConfig,
)
from study_query_llm.vllm_serving.vram import VRAMStatus, VRAMSafetyError


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _ample_vram(index: int = 0) -> VRAMStatus:
    """A VRAMStatus with plenty of free memory -- the guard always passes.

    24 GB total, 23 GB free: a 0.4 cap (~9.8 GB) is well under the default
    16 GB ceiling and well under free+headroom, so launch is permitted.
    """
    return VRAMStatus(
        index=index,
        name="NVIDIA GeForce RTX 4090",
        total_mb=24564,
        free_mb=23000,
        used_mb=1564,
    )


def _tiny_vram(index: int = 0) -> VRAMStatus:
    """A VRAMStatus with almost no free memory -- the guard refuses the launch.

    24 GB total (so the 0.4 cap is ~9.8 GB, under the ceiling) but only 100 MB
    free, which is far below cap+headroom -> ``VRAMSafetyError`` (free check).
    """
    return VRAMStatus(
        index=index,
        name="NVIDIA GeForce RTX 4090",
        total_mb=24564,
        free_mb=100,
        used_mb=24464,
    )


def _make_docker_client(container=None):
    """Return ``(client, container)`` mocks mirroring the TEI test helper.

    ``containers.get`` raises ``docker.errors.NotFound`` by default (the common
    "no stale container" case); ``containers.run`` returns the mock container.
    """
    client = MagicMock()
    mock_container = container or MagicMock()
    client.containers.run.return_value = mock_container
    client.containers.get.side_effect = docker.errors.NotFound("not found")
    return client, mock_container


def _make_backend(docker_client, *, vram_probe=None, **kwargs):
    """Construct a LocalDockerBackend with all side-effecting seams injected.

    Defaults: ample VRAM, a fake hf_download that returns a sentinel host dir
    (so the offline path never touches HuggingFace).
    """
    defaults = dict(
        port=8000,
        gpu_device="all",
        vram_probe=vram_probe if vram_probe is not None else _ample_vram,
        hf_download=lambda repo, dest: dest,
        docker_client_factory=lambda: docker_client,
    )
    defaults.update(kwargs)
    return LocalDockerBackend(**defaults)


def _config(**kwargs) -> VLLMServeConfig:
    defaults = dict(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        served_model_name="qwen2.5-7b-instruct-awq",
        quantization="awq",
    )
    defaults.update(kwargs)
    return VLLMServeConfig(**defaults)


def _run_kwargs(docker_client):
    return docker_client.containers.run.call_args.kwargs


# --------------------------------------------------------------------------- #
# name
# --------------------------------------------------------------------------- #
def test_backend_name():
    """The backend advertises name 'local_docker' and is not remote."""
    backend = LocalDockerBackend()
    assert backend.name == "local_docker"
    assert backend.is_remote() is False


# --------------------------------------------------------------------------- #
# containers.run kwargs
# --------------------------------------------------------------------------- #
def test_start_uses_vllm_image():
    """start() launches the canonical vLLM image."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.start(_config())

    assert _run_kwargs(client)["image"] == VLLM_IMAGE


def test_start_maps_host_port_to_internal_8000():
    """Host port maps onto the internal vLLM port (8000/tcp -> host port)."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, port=8123, offline=True)

    backend.start(_config())

    ports = _run_kwargs(client)["ports"]
    assert ports == {f"{INTERNAL_PORT}/tcp": 8123}
    assert ports == {"8000/tcp": 8123}


def test_start_command_contains_serve_flags():
    """The container command carries served-model-name + display-safety flags."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, offline=True)
    cfg = _config(served_model_name="qwen2.5-7b-instruct-awq")

    backend.start(cfg)

    command = _run_kwargs(client)["command"]
    # served-model-name pair
    assert "--served-model-name" in command
    assert "qwen2.5-7b-instruct-awq" in command
    # gpu-memory-utilization renders the default 0.4 exactly
    idx = command.index("--gpu-memory-utilization")
    assert command[idx + 1] == "0.4"
    # display-safety: --enforce-eager on by default
    assert "--enforce-eager" in command


def test_start_command_is_to_serve_args_for_model_ref():
    """The command equals config.to_serve_args(model_ref) for the offline /model ref."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, offline=True)
    cfg = _config()

    backend.start(cfg)

    command = _run_kwargs(client)["command"]
    # Offline path resolves model_ref to the mounted "/model".
    assert command == cfg.to_serve_args("/model")
    assert command[:2] == ["--model", "/model"]


def test_start_runs_detached_without_autoremove():
    """Container runs detached and is NOT auto-removed (manager owns teardown)."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.start(_config())

    kwargs = _run_kwargs(client)
    assert kwargs["detach"] is True
    assert kwargs["remove"] is False


# --------------------------------------------------------------------------- #
# GPU device requests
# --------------------------------------------------------------------------- #
def test_start_includes_gpu_device_requests():
    """device_requests pass GPU capabilities with count=-1 for 'all'."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, gpu_device="all", offline=True)

    backend.start(_config())

    device_requests = _run_kwargs(client)["device_requests"]
    assert len(device_requests) == 1
    assert device_requests[0].capabilities == [["gpu"]]
    assert device_requests[0].count == -1


def test_start_gpu_device_count_for_digit():
    """A digit gpu_device passes that many GPUs as the device-request count."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, gpu_device="2", offline=True)

    backend.start(_config())

    device_requests = _run_kwargs(client)["device_requests"]
    assert device_requests[0].count == 2


# --------------------------------------------------------------------------- #
# container name + stale-container removal
# --------------------------------------------------------------------------- #
def test_start_derives_container_name_from_served_name():
    """Default container name is 'vllm-<sanitized served name>'.

    The sanitizer lowercases and replaces only Docker-illegal chars ('/' here)
    with '-'; '.' is a legal container-name char and is preserved.
    """
    client, _ = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.start(_config(served_model_name="Qwen2.5/7B.Instruct"))

    assert _run_kwargs(client)["name"] == "vllm-qwen2.5-7b.instruct"


def test_explicit_container_name_overrides_default():
    """An explicit container_name is used verbatim instead of the derived one."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, container_name="my-vllm", offline=True)

    backend.start(_config())

    assert _run_kwargs(client)["name"] == "my-vllm"


def test_start_removes_stale_container():
    """A pre-existing container of the same name is stopped + removed first."""
    client, _ = _make_docker_client()
    stale = MagicMock()
    # Override the default NotFound so a stale container is "found".
    client.containers.get.side_effect = None
    client.containers.get.return_value = stale
    backend = _make_backend(client, offline=True)

    backend.start(_config())

    stale.stop.assert_called_once()
    stale.remove.assert_called_once()


def test_start_ignores_missing_stale_container():
    """No stale container (docker NotFound) is the normal case -> run proceeds."""
    client, _ = _make_docker_client()  # get() raises NotFound by default
    backend = _make_backend(client, offline=True)

    backend.start(_config())

    client.containers.run.assert_called_once()


# --------------------------------------------------------------------------- #
# offline / TLS-interception path: /model mount + offline env
# --------------------------------------------------------------------------- #
def test_offline_path_mounts_model_readonly_with_offline_env():
    """Offline mode mounts the host snapshot at /model:ro and sets offline env."""
    captured = {}

    def fake_dl(repo, dest):
        captured["repo"] = repo
        captured["dest"] = dest
        return dest

    client, _ = _make_docker_client()
    backend = _make_backend(
        client,
        offline=True,
        models_dir="/host/models",
        hf_download=fake_dl,
    )

    backend.start(_config(model="Qwen/Qwen2.5-7B-Instruct-AWQ"))

    kwargs = _run_kwargs(client)
    volumes = kwargs["volumes"]
    # exactly one host dir mounted read-only at /model
    assert len(volumes) == 1
    (host_dir, mount), = volumes.items()
    assert mount == {"bind": "/model", "mode": "ro"}
    assert host_dir == captured["dest"]
    # model_ref points at the mount
    assert kwargs["command"][:2] == ["--model", "/model"]
    # offline env (TLS-interception workaround)
    assert kwargs["environment"] == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # the snapshot was requested for the configured repo
    assert captured["repo"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_local_dir_path_mounts_and_serves_offline():
    """When config.model is an existing dir it is mounted at /model:ro offline."""
    client, _ = _make_docker_client()
    download_called = {"n": 0}

    def fake_dl(repo, dest):  # must NOT be called for a local dir
        download_called["n"] += 1
        return dest

    with tempfile.TemporaryDirectory() as tmp:
        backend = _make_backend(client, hf_download=fake_dl)
        backend.start(_config(model=tmp))

        kwargs = _run_kwargs(client)
        volumes = kwargs["volumes"]
        host_dir = str(next(iter(volumes)))
        assert volumes[host_dir] == {"bind": "/model", "mode": "ro"}
        assert os.path.samefile(host_dir, tmp)
        assert kwargs["command"][:2] == ["--model", "/model"]
        assert kwargs["environment"] == {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    # a real local dir bypasses the HF download entirely
    assert download_called["n"] == 0


def test_online_path_mounts_hf_cache_no_offline_env():
    """offline=False (in-container download) mounts the HF cache rw, no offline env."""
    client, _ = _make_docker_client()
    backend = _make_backend(
        client,
        offline=False,
        hf_cache_dir="/host/hf",
        hf_download=lambda repo, dest: pytest.fail("hf_download must not run online"),
    )

    backend.start(_config(model="Qwen/Qwen2.5-7B-Instruct-AWQ"))

    kwargs = _run_kwargs(client)
    # model_ref is the raw repo id (vLLM downloads in-container)
    assert kwargs["command"][:2] == ["--model", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    # HF cache mounted read-write at the container cache path
    (host_dir, mount), = kwargs["volumes"].items()
    assert mount == {"bind": "/root/.cache/huggingface", "mode": "rw"}
    # no offline env on the online path
    assert kwargs["environment"] == {}


# --------------------------------------------------------------------------- #
# returned URL
# --------------------------------------------------------------------------- #
def test_start_returns_v1_url():
    """start() returns http://localhost:<host port>/v1."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, port=8000, offline=True)

    url = backend.start(_config())

    assert url == "http://localhost:8000/v1"


def test_start_returns_v1_url_for_custom_port():
    """The returned URL reflects the configured HOST port, not the internal port."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, port=9001, offline=True)

    url = backend.start(_config())

    assert url == "http://localhost:9001/v1"


# --------------------------------------------------------------------------- #
# VRAM display-safety guard: refuse BEFORE docker run
# --------------------------------------------------------------------------- #
def test_vram_refusal_aborts_before_docker_run():
    """A tiny-free-VRAM probe raises VRAMSafetyError and never calls containers.run."""
    client, _ = _make_docker_client()
    backend = _make_backend(client, vram_probe=_tiny_vram, offline=True)

    with pytest.raises(VRAMSafetyError):
        backend.start(_config())

    client.containers.run.assert_not_called()
    client.containers.get.assert_not_called()


def test_vram_refusal_when_cap_exceeds_ceiling():
    """A cap larger than the ceiling is refused before any docker call."""
    client, _ = _make_docker_client()
    # ample free, but a tiny ceiling makes the 0.4 cap (~9.8 GB) too large.
    backend = _make_backend(
        client, vram_probe=_ample_vram, vram_ceiling_mb=1000, offline=True
    )

    with pytest.raises(VRAMSafetyError):
        backend.start(_config())

    client.containers.run.assert_not_called()


def test_vram_probe_uses_configured_gpu_index():
    """The VRAM probe is called with the backend's gpu_index."""
    client, _ = _make_docker_client()
    probe = MagicMock(return_value=_ample_vram())
    backend = _make_backend(client, vram_probe=probe, gpu_index=3, offline=True)

    backend.start(_config())

    probe.assert_called_once_with(3)


def test_vram_check_skipped_when_disabled():
    """enable_vram_check=False bypasses the probe entirely and launches."""
    client, _ = _make_docker_client()
    probe = MagicMock(side_effect=AssertionError("probe must not be called"))
    backend = _make_backend(
        client, vram_probe=probe, enable_vram_check=False, offline=True
    )

    backend.start(_config())

    probe.assert_not_called()
    client.containers.run.assert_called_once()


# --------------------------------------------------------------------------- #
# stop(): idempotent + swallows
# --------------------------------------------------------------------------- #
def test_stop_stops_and_removes_container():
    """stop() stops and removes the launched container."""
    client, container = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.start(_config())
    backend.stop()

    container.stop.assert_called_once()
    container.remove.assert_called_once()


def test_stop_before_start_is_noop():
    """stop() with nothing started does nothing and does not raise."""
    client, container = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.stop()  # never started

    container.stop.assert_not_called()


def test_stop_is_idempotent():
    """Calling stop() twice stops/removes the container only once."""
    client, container = _make_docker_client()
    backend = _make_backend(client, offline=True)

    backend.start(_config())
    backend.stop()
    backend.stop()

    assert container.stop.call_count == 1
    assert container.remove.call_count == 1


def test_stop_swallows_docker_errors():
    """stop() logs and swallows exceptions raised during teardown."""
    client, container = _make_docker_client()
    container.stop.side_effect = RuntimeError("daemon gone")
    backend = _make_backend(client, offline=True)

    backend.start(_config())
    backend.stop()  # must not raise

    container.stop.assert_called_once()

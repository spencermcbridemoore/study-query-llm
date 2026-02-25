"""
Local Docker TEI Manager - lifecycle manager for a GPU-accelerated
HuggingFace Text Embeddings Inference container running on the local machine.

The container exposes an OpenAI-compatible /v1/embeddings endpoint on
localhost, making it a zero-cost, zero-auth alternative to the ACI manager
for development and local sweeps.

Typical usage:

    from scripts.common.local_docker_tei_manager import LocalDockerTEIManager
    from study_query_llm.providers import ManagedTEIEmbeddingProvider

    models = ["BAAI/bge-m3", "BAAI/bge-large-en-v1.5", "nomic-ai/nomic-embed-text-v1.5"]

    for model_id in models:
        with LocalDockerTEIManager(model_id=model_id) as manager:
            async with ManagedTEIEmbeddingProvider(manager) as provider:
                service = EmbeddingService(repository=repo, provider=provider)
                # ... run sweep for this model ...

Prerequisites:
- Docker Desktop running
- NVIDIA Container Toolkit installed (verified: RTX 4090, CUDA 12.6)
- HF model weights are cached in hf_cache_dir (downloaded automatically on first run)
"""

import os
import time
import threading
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import docker
import docker.types

logger = logging.getLogger(__name__)

# :89-1.9 targets CUDA compute capability 8.9 (Ada Lovelace — RTX 4090, RTX 4000 series).
# It enables Ada-specific Flash Attention 2 kernels for materially faster throughput
# vs. the generic :1.9 image.  Swap to :86-1.9 for Ampere 86 (A10/A40) or
# :1.9 for Ampere 80 (A100).
_TEI_GPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.9"
_TEI_CPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9"

# Default HuggingFace model cache directory on the host.
# Shared with the conda environment and other tools so models are not
# re-downloaded across different runners.
_DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")


class LocalDockerTEIManager:
    """
    Manages a local Docker container running HuggingFace TEI with GPU support.

    Exposes the same duck-typed interface as ``ACITEIManager`` so that
    ``ManagedTEIEmbeddingProvider`` can use either interchangeably:

        endpoint_url  -- set after start(), None before and after stop()
        model_id      -- the HuggingFace model ID passed at construction
        provider_label -- "local_docker_tei"
        ping()        -- reset idle timer

    Model weights are mounted from the host's HuggingFace cache directory, so
    the first run for a given model downloads the weights once and subsequent
    container starts are fast (seconds, not minutes).

    The idle timer stops the container after a configurable period of
    inactivity. This frees GPU VRAM, which matters when sweeping multiple models
    sequentially.
    """

    def __init__(
        self,
        model_id: str,
        port: int = 8080,
        idle_timeout_seconds: int = 1800,
        hf_cache_dir: Optional[str] = None,
        gpu_device: str = "all",
        use_gpu: bool = True,
        health_check_timeout: int = 600,
        health_check_interval: int = 5,
        container_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_id: HuggingFace model ID, e.g. ``"BAAI/bge-m3"``.
                This is a per-instance argument — to sweep multiple models
                create one ``LocalDockerTEIManager`` per model.
            port: Host port to bind. The TEI container always listens on port
                80 internally; this is the port exposed on localhost.
            idle_timeout_seconds: Seconds of inactivity before the container
                is automatically stopped. Defaults to 1800 (30 min). Call
                ``ping()`` on each embedding request to reset the timer.
            hf_cache_dir: Path on the host to mount as the HuggingFace model
                cache inside the container. Defaults to the ``HF_CACHE_DIR``
                environment variable, or ``~/.cache/huggingface`` if unset.
            gpu_device: Docker DeviceRequest count string. ``"all"`` passes all
                GPUs; use ``"1"`` to pass a single GPU (useful if you have
                multiple GPUs and want to reserve some).
            use_gpu: Set to ``False`` to run the CPU-only TEI image (for
                testing without GPU access).
            health_check_timeout: Seconds to wait for TEI to load the model
                before giving up.
            health_check_interval: Seconds between health-check polls.
            container_name: Optional explicit Docker container name. Defaults
                to ``"tei-<sanitised-model-id>"``.
        """
        self.model_id = model_id
        self.port = port
        self.idle_timeout_seconds = idle_timeout_seconds
        self.hf_cache_dir = (
            hf_cache_dir
            or os.environ.get("HF_CACHE_DIR", "")
            or _DEFAULT_HF_CACHE
        )
        self.gpu_device = gpu_device
        self.use_gpu = use_gpu
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.container_name = container_name or (
            "tei-" + model_id.replace("/", "-").replace(".", "-").lower()
        )

        # Duck-typed interface expected by ManagedTEIEmbeddingProvider
        self.endpoint_url: Optional[str] = None
        self.provider_label: str = "local_docker_tei"

        self._container = None
        self._idle_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._stopped = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_docker_client(self):
        """Return a Docker client connected to the local daemon."""
        return docker.from_env()

    def _wait_for_healthy(self) -> None:
        """Poll GET /health until TEI returns 200 (model loaded)."""
        health_url = f"{self.endpoint_url}/health"
        deadline = time.time() + self.health_check_timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(
                            "[Docker TEI] Healthy: %s  (model=%s)",
                            self.endpoint_url,
                            self.model_id,
                        )
                        return
            except (urllib.error.URLError, OSError):
                pass
            logger.debug(
                "[Docker TEI] Waiting for health check at %s ...", health_url
            )
            time.sleep(self.health_check_interval)
        raise TimeoutError(
            f"TEI endpoint at {health_url} did not become healthy within "
            f"{self.health_check_timeout}s. The model may be too large, or the "
            f"container may have crashed. Check: docker logs {self.container_name}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> str:
        """
        Pull the TEI image (if needed), start the container with GPU access,
        wait for the model to load, and start the idle timer.

        Returns:
            The base endpoint URL, e.g. ``"http://localhost:8080/v1"``.

        Raises:
            TimeoutError: If TEI does not pass its health check within
                ``health_check_timeout`` seconds.
            docker.errors.DockerException: If the Docker daemon is not
                reachable or the ``--gpus all`` flag is unavailable.
        """
        if self.endpoint_url is not None:
            logger.warning(
                "[Docker TEI] Container '%s' is already running at %s",
                self.container_name,
                self.endpoint_url,
            )
            return self.endpoint_url

        self._stopped = False
        client = self._get_docker_client()

        # Remove any stale container with the same name
        try:
            old = client.containers.get(self.container_name)
            logger.info(
                "[Docker TEI] Removing stale container '%s' ...", self.container_name
            )
            old.stop()
            old.remove()
        except docker.errors.NotFound:
            pass

        image = _TEI_GPU_IMAGE if self.use_gpu else _TEI_CPU_IMAGE

        device_requests = []
        if self.use_gpu:
            device_requests = [
                docker.types.DeviceRequest(
                    count=-1 if self.gpu_device == "all" else int(self.gpu_device),
                    capabilities=[["gpu"]],
                )
            ]

        hf_cache = str(Path(self.hf_cache_dir).expanduser().resolve())
        logger.info(
            "[Docker TEI] Starting container '%s' (model=%s, port=%s, gpu=%s) ...",
            self.container_name,
            self.model_id,
            self.port,
            self.gpu_device if self.use_gpu else "none",
        )

        self._container = client.containers.run(
            image=image,
            command=["--model-id", self.model_id, "--port", "80"],
            name=self.container_name,
            detach=True,
            ports={"80/tcp": self.port},
            volumes={hf_cache: {"bind": "/root/.cache/huggingface", "mode": "rw"}},
            device_requests=device_requests,
            remove=False,
        )

        self.endpoint_url = f"http://localhost:{self.port}/v1"
        logger.info(
            "[Docker TEI] Container started. Waiting for model load at %s ...",
            self.endpoint_url,
        )

        self._wait_for_healthy()
        self._reset_idle_timer()

        logger.info(
            "[Docker TEI] Ready: %s  (model=%s)", self.endpoint_url, self.model_id
        )
        return self.endpoint_url

    def stop(self) -> None:
        """
        Stop and remove the Docker container, cancel the idle timer, and
        free GPU VRAM.

        Safe to call multiple times (subsequent calls are no-ops).
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

        try:
            if self._container is not None:
                logger.info(
                    "[Docker TEI] Stopping container '%s' ...", self.container_name
                )
                self._container.stop()
                self._container.remove()
                logger.info("[Docker TEI] Stopped '%s'.", self.container_name)
        except Exception as exc:
            logger.warning(
                "[Docker TEI] Stop/remove failed for '%s' (may already be gone): %s",
                self.container_name,
                exc,
            )
        finally:
            self._container = None
            self.endpoint_url = None

    def ping(self) -> None:
        """
        Reset the idle shutdown timer.

        Called automatically by ``ManagedTEIEmbeddingProvider`` on every
        embedding request to keep the container alive during active sweeps.
        """
        if not self._stopped:
            self._reset_idle_timer()

    def _reset_idle_timer(self) -> None:
        """Cancel any existing timer and start a fresh countdown."""
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(
                self.idle_timeout_seconds, self._idle_shutdown
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_shutdown(self) -> None:
        """Called by the timer thread when idle timeout expires."""
        logger.warning(
            "[Docker TEI] Idle timeout (%ss) reached — stopping '%s'",
            self.idle_timeout_seconds,
            self.container_name,
        )
        self.stop()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "LocalDockerTEIManager":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

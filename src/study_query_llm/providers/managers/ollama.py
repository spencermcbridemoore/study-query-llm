"""
Ollama Model Manager - lifecycle manager for Ollama LLM models.

Loads / unloads models from GPU VRAM via the Ollama REST API so that
VRAM is deterministically reclaimed between sweep iterations or after
tests.  Satisfies the ``ModelManager`` protocol alongside
``LocalDockerTEIManager`` and ``ACITEIManager``.

Typical usage:

    from study_query_llm.providers.managers import OllamaModelManager

    with OllamaModelManager("qwen2.5:32b") as mgr:
        provider = OpenAICompatibleChatProvider(
            base_url=mgr.endpoint_url, model=mgr.model_id,
        )
        result = await provider.complete("Summarise this text.")
    # model unloaded, VRAM freed

Prerequisites:
- Ollama installed and running (``ollama serve`` or system service)
- Model already pulled (``ollama pull <tag>``)
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://localhost:11434"


class OllamaModelManager:
    """Manages an Ollama model's lifecycle in GPU VRAM.

    ``start()`` warm-loads the model so it is ready for inference.
    ``stop()`` unloads it via ``keep_alive=0`` to free VRAM immediately.

    The idle timer mirrors ``LocalDockerTEIManager``: after
    ``idle_timeout_seconds`` without a ``ping()``, the model is
    automatically unloaded.
    """

    def __init__(
        self,
        model_id: str,
        endpoint: str = _DEFAULT_ENDPOINT,
        idle_timeout_seconds: int = 1800,
        provider_label: str = "ollama",
    ) -> None:
        """
        Args:
            model_id: Ollama model tag, e.g. ``"qwen2.5:32b"``.
            endpoint: Ollama API root (no ``/v1`` suffix).
            idle_timeout_seconds: Seconds of inactivity before the model
                is automatically unloaded. Call ``ping()`` to reset.
            provider_label: Human-readable label used by the Protocol.
        """
        self.model_id = model_id
        self.endpoint_url: Optional[str] = None
        self.provider_label = provider_label
        self.idle_timeout_seconds = idle_timeout_seconds

        self._endpoint = endpoint
        self._idle_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._stopped = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _api_request(self, path: str, payload: dict, timeout: int = 120) -> bytes:
        """Send a POST request to the Ollama HTTP API and return the body."""
        url = f"{self._endpoint}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _warm_load(self) -> None:
        """Force Ollama to load the model into VRAM.

        Sends a minimal ``/api/chat`` request with ``stream: false``.
        Ollama loads the model on first use; this call ensures it is
        resident before any real inference request arrives.
        """
        logger.info(
            "[Ollama] Warm-loading model '%s' into VRAM ...", self.model_id
        )
        t0 = time.time()
        self._api_request(
            "/api/chat",
            {
                "model": self.model_id,
                "messages": [],
                "stream": False,
            },
        )
        elapsed = time.time() - t0
        logger.info(
            "[Ollama] Model '%s' loaded in %.1fs.", self.model_id, elapsed
        )

    def _unload(self) -> None:
        """Unload the model from VRAM via ``keep_alive: 0``."""
        try:
            self._api_request(
                "/api/chat",
                {
                    "model": self.model_id,
                    "messages": [],
                    "keep_alive": "0",
                    "stream": False,
                },
            )
            logger.info("[Ollama] Unloaded model '%s'.", self.model_id)
        except Exception as exc:
            logger.warning(
                "[Ollama] Failed to unload model '%s': %s", self.model_id, exc
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Warm-load the model and return the OpenAI-compatible endpoint URL.

        Returns:
            ``"http://localhost:11434/v1"`` (or equivalent).

        Raises:
            urllib.error.URLError: If the Ollama server is unreachable.
        """
        if self.endpoint_url is not None:
            logger.warning(
                "[Ollama] Model '%s' is already loaded at %s",
                self.model_id,
                self.endpoint_url,
            )
            return self.endpoint_url

        self._stopped = False
        self._warm_load()
        self.endpoint_url = f"{self._endpoint}/v1"
        self._reset_idle_timer()
        return self.endpoint_url

    def stop(self) -> None:
        """Unload the model from VRAM and free resources.

        Safe to call multiple times (subsequent calls are no-ops).
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

        self._unload()
        self.endpoint_url = None

    def ping(self) -> None:
        """Reset the idle-shutdown timer.

        Call this on every inference request to keep the model loaded
        during active sweeps.
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
            "[Ollama] Idle timeout (%ss) reached — unloading '%s'",
            self.idle_timeout_seconds,
            self.model_id,
        )
        self.stop()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaModelManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

"""
vLLM model lifecycle manager.

:class:`VLLMModelManager` is the backend-agnostic lifecycle wrapper around a
:class:`~study_query_llm.vllm_serving.backends.VLLMBackend` (local Docker or
remote vast.ai).  It owns the *guaranteed teardown* story so that a VRAM-hogging
local vLLM container -- or a paid remote instance that costs money every minute
it stays up -- can never be left running by accident:

* **Context manager** -- ``__enter__`` starts, ``__exit__`` stops even on an
  exception or ``KeyboardInterrupt``.
* **Signal handlers** -- ``SIGINT`` + ``SIGTERM`` are trapped (main thread only),
  the manager is torn down, the *previous* handler is restored, and the signal is
  re-raised so the program still terminates / raises ``KeyboardInterrupt`` as the
  user expects.
* **atexit backstop** -- a final ``atexit`` hook catches any interpreter exit
  path the above missed.
* **Idle timer** -- a daemon ``threading.Timer`` tears the server down after a
  configurable period without a :meth:`ping`, mirroring
  :class:`~study_query_llm.providers.managers.local_docker_tei.LocalDockerTEIManager`.

It conforms *structurally* to the ``ModelManager`` Protocol (no inheritance);
:func:`is_model_manager` is a genuine ``isinstance`` check against that
``@runtime_checkable`` Protocol -- the single use of the only existing-code
import this whole module is allowed to make.

Lane wiring (after ``start()`` returns the URL)::

    export LOCAL_LLM_ENDPOINT=<returned url>
    export LOCAL_LLM_API_KEY=not-needed
    export MCQ_LOGPROB_PROVIDER_PIN=""        # self-hosted -> no OpenRouter routing
    # run the lane with provider=local_llm, model=<config.served_model_name>

Thinking-off note (verified against the live codebase): the production lane
(``mcq_logprob_basic``) has ``MCQ_LOGPROB_MAX_TOKENS`` and
``MCQ_LOGPROB_PROVIDER_PIN`` hooks but **no** ``MCQ_LOGPROB_EXTRA_BODY`` hook, so
a per-request ``extra_body`` never reaches the provider.  For the production lane
thinking-off MUST be configured at the SERVE layer (``reasoning_parser`` /
``chat_template`` / a non-thinking snapshot) via :class:`VLLMServeConfig`; only
the standalone gate-(b) probe can disable thinking per-request.
"""

from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Optional

# The ONLY import from existing study_query_llm code anywhere in this module.
# It is genuinely used by ``is_model_manager`` below.
from study_query_llm.providers.managers.protocol import ModelManager

from . import probe
from .backends import VLLMBackend
from .config import VLLMServeConfig

logger = logging.getLogger(__name__)


class VLLMModelManager:
    """Backend-agnostic lifecycle manager for a single vLLM serve target.

    Satisfies the ``ModelManager`` Protocol structurally (no inheritance):

        model_id       -- the served-model-name clients pass as ``"model"``
        endpoint_url   -- set after ``start()``, ``None`` before / after ``stop()``
        provider_label -- e.g. ``"vllm_local_docker"`` / ``"vllm_vast"``
        start()        -- launch the backend, wait for readiness, arm teardown
        stop()         -- idempotent teardown (lock-guarded); frees the backend
        ping()         -- reset the idle-shutdown timer

    Teardown is guaranteed by four overlapping mechanisms (context manager,
    signal handlers, atexit, idle timer) so a costly/VRAM-hogging server is never
    orphaned.  ``stop()`` is idempotent, so the overlap is safe.
    """

    def __init__(
        self,
        backend: VLLMBackend,
        config: VLLMServeConfig,
        *,
        idle_timeout_seconds: int = 1800,
        health_check_timeout: int = 600,
        health_check_interval: int = 5,
        install_signal_handlers: bool = True,
        install_atexit: bool = True,
    ) -> None:
        """
        Args:
            backend: The :class:`VLLMBackend` that physically launches/stops the
                server (``LocalDockerBackend`` or ``VastAIBackend``).
            config: The :class:`VLLMServeConfig` describing what / how to serve.
            idle_timeout_seconds: Seconds of inactivity (no :meth:`ping`) before
                the server is auto-torn-down to free VRAM / stop billing.
            health_check_timeout: Seconds to wait for ``GET /models`` to return
                200 after the backend starts.
            health_check_interval: Seconds between readiness polls.
            install_signal_handlers: When ``True`` (default) install SIGINT +
                SIGTERM handlers in ``start()`` -- but only if running on the main
                thread (signal handlers can only be installed there).  The
                handler tears down, restores the previous handler, and re-raises
                so default termination / ``KeyboardInterrupt`` still happens.
            install_atexit: When ``True`` (default) register a one-shot
                ``atexit`` backstop in ``start()``.
        """
        self.backend = backend
        self.config = config
        self.idle_timeout_seconds = idle_timeout_seconds
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval
        self.install_signal_handlers = install_signal_handlers
        self.install_atexit = install_atexit

        # ModelManager Protocol surface.
        self.model_id: str = config.served_model_name
        self.endpoint_url: Optional[str] = None
        self.provider_label: str = f"vllm_{backend.name}"

        self._lock = threading.Lock()
        # Mirror OllamaModelManager: start "stopped" so a stop() before start()
        # (or a double stop()) is a clean no-op.
        self._stopped = True
        self._idle_timer: Optional[threading.Timer] = None

        # Saved previous signal handlers, keyed by signal number, so the handler
        # can restore them on teardown and the program keeps its default exit
        # behaviour.  Empty when no handlers are installed (e.g. non-main thread).
        self._prev_signal_handlers: dict[int, object] = {}
        self._atexit_registered = False

    # ------------------------------------------------------------------
    # Public API (ModelManager Protocol)
    # ------------------------------------------------------------------

    def start(self) -> str:
        """Launch the backend, wait for readiness, arm all teardown guards.

        Returns:
            The network-reachable base URL ending in ``/v1`` (the same value the
            backend returned).

        Raises:
            TimeoutError: If ``GET /models`` does not return 200 within
                ``health_check_timeout`` seconds.  The backend is torn down
                before re-raising so a half-up server is not orphaned.
            Exception: Any error raised by the backend's ``start`` (e.g.
                :class:`~study_query_llm.vllm_serving.vram.VRAMSafetyError`,
                Docker errors).  Propagated unchanged; nothing is left running.
        """
        if self.endpoint_url is not None:
            logger.warning(
                "[vLLM:%s] Already running at %s (model=%s)",
                self.backend.name,
                self.endpoint_url,
                self.model_id,
            )
            return self.endpoint_url

        self._stopped = False

        logger.info(
            "[vLLM:%s] Starting (model=%s, served_model_name=%s) ...",
            self.backend.name,
            self.config.model,
            self.model_id,
        )

        # Arm the teardown guards BEFORE provisioning. The startup window
        # (backend.start() + the readiness wait) is the longest and most
        # dangerous one for a PAID vast instance: create_instance() may already
        # have brought a billing instance up when a later step (wait_until_running,
        # SSH tunnel) fails or the operator hits Ctrl-C. Installing the signal
        # handlers now -- and catching BaseException below -- guarantees that any
        # failure OR interrupt during startup tears the backend down. stop() is
        # idempotent and safe to call when nothing (or only part) was created.
        if self.install_signal_handlers:
            self._install_signal_handlers()
        if self.install_atexit:
            self._register_atexit()

        try:
            base = self.backend.start(self.config)
            # Wait for the OpenAI-compatible server to come up.
            probe.wait_for_models_ready(
                base,
                timeout=self.health_check_timeout,
                interval=self.health_check_interval,
            )
        except BaseException:
            # Covers backend errors (VRAMSafetyError / Docker / vast), the
            # readiness TimeoutError, AND KeyboardInterrupt / SystemExit raised
            # mid-startup -- never leak a half-started container or a paid
            # instance that create_instance() already provisioned.
            logger.error(
                "[vLLM:%s] Startup failed or was interrupted; tearing down to "
                "avoid an orphaned server.",
                self.backend.name,
            )
            self.stop()
            raise

        self.endpoint_url = base
        self._reset_idle_timer()

        logger.info(
            "[vLLM:%s] Ready: %s (model=%s, provider_label=%s)",
            self.backend.name,
            self.endpoint_url,
            self.model_id,
            self.provider_label,
        )
        return self.endpoint_url

    def stop(self) -> None:
        """Tear the server down and free all resources.  Idempotent.

        Lock-guarded like
        :meth:`LocalDockerTEIManager.stop`: the first call sets ``_stopped``,
        cancels the idle timer, restores any installed signal handlers and then
        calls the backend's (swallowing) ``stop``.  Subsequent calls return
        immediately, so the overlapping teardown guards are safe to all fire.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            self._restore_signal_handlers()

        # Outside the lock: the backend's stop swallows + logs its own errors,
        # and we never want a slow backend teardown to hold our lock.
        try:
            self.backend.stop()
        except Exception as exc:  # noqa: BLE001 - backend.stop should swallow, but be defensive
            logger.warning(
                "[vLLM:%s] backend.stop() raised (continuing teardown): %s",
                self.backend.name,
                exc,
            )
        finally:
            self.endpoint_url = None
            logger.info("[vLLM:%s] Stopped (model=%s).", self.backend.name, self.model_id)

    def ping(self) -> None:
        """Reset the idle-shutdown timer.  No-op once stopped."""
        if not self._stopped:
            self._reset_idle_timer()

    # ------------------------------------------------------------------
    # Idle timer (mirrors LocalDockerTEIManager)
    # ------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        """Cancel any existing idle timer and start a fresh daemon countdown."""
        with self._lock:
            if self._stopped:
                return
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(
                self.idle_timeout_seconds, self._idle_shutdown
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_shutdown(self) -> None:
        """Called by the timer thread when the idle timeout expires."""
        logger.warning(
            "[vLLM:%s] Idle timeout (%ss) reached -- stopping (model=%s)",
            self.backend.name,
            self.idle_timeout_seconds,
            self.model_id,
        )
        self.stop()

    # ------------------------------------------------------------------
    # Signal handlers (main thread only; restore + re-raise)
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Trap SIGINT + SIGTERM so a Ctrl-C / kill tears the server down.

        Signal handlers can only be installed from the *main* thread; if we are
        not on it we log and skip (the context manager + atexit backstops still
        guarantee teardown).
        """
        if threading.current_thread() is not threading.main_thread():
            logger.info(
                "[vLLM:%s] Not on the main thread; skipping signal handlers "
                "(context manager + atexit still guarantee teardown).",
                self.backend.name,
            )
            return

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
                self._prev_signal_handlers[signum] = previous
            except (ValueError, OSError, RuntimeError) as exc:
                # ValueError: not main thread (already guarded) / unsupported
                # signal on this platform (e.g. SIGTERM semantics on Windows).
                logger.info(
                    "[vLLM:%s] Could not install handler for signal %s "
                    "(continuing without it): %s",
                    self.backend.name,
                    signum,
                    exc,
                )

    def _restore_signal_handlers(self) -> None:
        """Restore the signal handlers saved by :meth:`_install_signal_handlers`.

        Called under ``self._lock`` from :meth:`stop`.  Best-effort: a failure to
        restore one handler must not block the rest of teardown.
        """
        for signum, previous in list(self._prev_signal_handlers.items()):
            try:
                signal.signal(signum, previous)  # type: ignore[arg-type]
            except (ValueError, OSError, RuntimeError, TypeError) as exc:
                logger.debug(
                    "[vLLM:%s] Could not restore handler for signal %s: %s",
                    self.backend.name,
                    signum,
                    exc,
                )
        self._prev_signal_handlers.clear()

    def _handle_signal(self, signum: int, frame: object) -> None:
        """SIGINT/SIGTERM handler: tear down, restore, then re-deliver.

        ``stop()`` restores the previous handler (under the lock), so after
        teardown we re-raise the signal to ourselves and the *original* handler
        (default behaviour: ``KeyboardInterrupt`` for SIGINT, termination for
        SIGTERM) runs.  This keeps the user's expected Ctrl-C / kill semantics
        intact while guaranteeing the server is gone first.
        """
        logger.warning(
            "[vLLM:%s] Caught signal %s -- tearing down before re-raising.",
            self.backend.name,
            signum,
        )
        # stop() restores the previous handlers for both signals, so re-raising
        # below dispatches to whatever was installed before us.
        self.stop()

        try:
            signal.raise_signal(signum)
        except Exception as exc:  # noqa: BLE001 - extremely defensive
            logger.debug(
                "[vLLM:%s] raise_signal(%s) failed: %s",
                self.backend.name,
                signum,
                exc,
            )
            # Fall back to a KeyboardInterrupt for SIGINT so the program still
            # unwinds, rather than swallowing the interrupt entirely.
            if signum == signal.SIGINT:
                raise KeyboardInterrupt from None

    # ------------------------------------------------------------------
    # atexit backstop
    # ------------------------------------------------------------------

    def _register_atexit(self) -> None:
        """Register ``stop`` as an ``atexit`` backstop, at most once."""
        if self._atexit_registered:
            return
        atexit.register(self.stop)
        self._atexit_registered = True

    # ------------------------------------------------------------------
    # Context manager (tears down even on exception / Ctrl-C)
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLLMModelManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()


def is_model_manager(obj: object) -> bool:
    """Return ``True`` iff *obj* structurally satisfies the ``ModelManager`` Protocol.

    A genuine ``isinstance`` check against the ``@runtime_checkable`` Protocol --
    the single use of this module's one allowed existing-code import.  A
    :class:`VLLMModelManager` (and any other conforming manager) passes.
    """
    return isinstance(obj, ModelManager)

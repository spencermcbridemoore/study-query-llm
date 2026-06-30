"""Tests for study_query_llm.vllm_serving.manager.VLLMModelManager.

Fully mocked: NO network, NO docker daemon, NO GPU, NO vast/ssh.  The backend is
a tiny in-memory fake (``_FakeBackend``) that records ``start``/``stop`` calls and
hands back a canned ``/v1`` URL, and the manager's one external dependency --
``probe.wait_for_models_ready`` -- is patched to a no-op so no HTTP ever happens.

Mirrors the mocking style of ``test_local_docker_tei_manager.py`` (patch the side
-effecting seam, assert on the recorded calls) and the ``@runtime_checkable``
isinstance pattern of ``test_model_manager_protocol.py``.

Construction note: every manager here is built with
``install_signal_handlers=False, install_atexit=False`` so the tests never mutate
global ``signal``/``atexit`` state -- EXCEPT the one focused signal-handler test,
which monkeypatches ``signal.signal`` + ``threading.current_thread`` so it can
assert SIGINT/SIGTERM handlers are installed in ``start()`` and restored in
``stop()`` without touching the real process handlers.
"""

import signal
import threading
import time
from unittest.mock import patch

import pytest

from study_query_llm.vllm_serving.backends import VLLMBackend
from study_query_llm.vllm_serving.config import VLLMServeConfig
from study_query_llm.vllm_serving.manager import (
    VLLMModelManager,
    is_model_manager,
)
from study_query_llm.providers.managers.protocol import ModelManager


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class _FakeBackend(VLLMBackend):
    """In-memory ``VLLMBackend`` that records lifecycle calls (no docker/GPU/net).

    ``start`` returns a canned ``/v1`` URL and bumps ``start_calls``; ``stop``
    bumps ``stop_calls`` and is idempotent + non-raising like a real backend.  An
    optional ``stop_error`` lets a test verify the manager survives a backend whose
    ``stop`` raises despite the contract (the manager is defensive about it).
    """

    def __init__(
        self,
        name="fake",
        url="http://localhost:8000/v1",
        stop_error=None,
        start_error=None,
    ):
        self.name = name
        self._url = url
        self._stop_error = stop_error
        # ``start_error`` simulates a backend whose start() raises AFTER it has
        # already provisioned a (paid) resource -- e.g. vast create_instance()
        # succeeded but wait_until_running/tunnel failed. The manager must still
        # call stop() so the resource is torn down.
        self._start_error = start_error
        self.start_calls = 0
        self.stop_calls = 0
        self.last_config = None

    def start(self, config: VLLMServeConfig) -> str:
        self.start_calls += 1
        self.last_config = config
        if self._start_error is not None:
            raise self._start_error
        return self._url

    def stop(self) -> None:
        self.stop_calls += 1
        if self._stop_error is not None:
            raise self._stop_error


def _make_config(**kwargs) -> VLLMServeConfig:
    defaults = dict(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        served_model_name="qwen2.5-7b-instruct-awq",
    )
    defaults.update(kwargs)
    return VLLMServeConfig(**defaults)


def _make_manager(backend=None, **kwargs) -> VLLMModelManager:
    """Build a manager with global-state mutation disabled by default."""
    defaults = dict(
        idle_timeout_seconds=9999,
        health_check_timeout=30,
        health_check_interval=1,
        install_signal_handlers=False,
        install_atexit=False,
    )
    defaults.update(kwargs)
    return VLLMModelManager(backend or _FakeBackend(), _make_config(), **defaults)


def _patch_ready():
    """Patch the manager's readiness probe so ``start()`` never hits the network."""
    return patch(
        "study_query_llm.vllm_serving.manager.probe.wait_for_models_ready",
        return_value=None,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_model_manager_protocol():
    """VLLMModelManager structurally satisfies the runtime-checkable Protocol."""
    mgr = _make_manager()
    assert isinstance(mgr, ModelManager)


def test_is_model_manager_helper_true():
    """is_model_manager() (the module's one allowed import) returns True."""
    mgr = _make_manager()
    assert is_model_manager(mgr) is True


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def test_start_sets_endpoint_url_and_provider_label():
    """start() returns/sets endpoint_url and provider_label == vllm_<backend.name>."""
    backend = _FakeBackend(name="fake", url="http://localhost:8000/v1")
    mgr = _make_manager(backend)

    with _patch_ready():
        url = mgr.start()

    assert url == "http://localhost:8000/v1"
    assert mgr.endpoint_url == "http://localhost:8000/v1"
    assert mgr.provider_label == "vllm_fake"
    assert backend.start_calls == 1


def test_provider_label_folds_backend_name():
    """provider_label uses the concrete backend's name verbatim."""
    mgr = _make_manager(_FakeBackend(name="local_docker"))
    assert mgr.provider_label == "vllm_local_docker"


def test_start_waits_for_models_ready():
    """start() drives probe.wait_for_models_ready against the backend URL."""
    backend = _FakeBackend(url="http://localhost:8123/v1")
    mgr = _make_manager(backend, health_check_timeout=42, health_check_interval=3)

    with _patch_ready() as mock_wait:
        mgr.start()

    mock_wait.assert_called_once()
    assert mock_wait.call_args.args[0] == "http://localhost:8123/v1"
    assert mock_wait.call_args.kwargs["timeout"] == 42
    assert mock_wait.call_args.kwargs["interval"] == 3


def test_start_is_idempotent():
    """A second start() returns the cached URL without re-launching the backend."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with _patch_ready():
        mgr.start()
        mgr.start()

    assert backend.start_calls == 1


def test_start_tears_down_when_readiness_fails():
    """If readiness times out, the backend is stopped (no orphaned server)."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with patch(
        "study_query_llm.vllm_serving.manager.probe.wait_for_models_ready",
        side_effect=TimeoutError("not ready"),
    ):
        with pytest.raises(TimeoutError):
            mgr.start()

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


def test_start_tears_down_when_backend_start_fails():
    """If backend.start() raises (e.g. vast create_instance() already provisioned
    a PAID instance, then wait/tunnel failed), the manager still calls
    backend.stop() so the paid resource is never orphaned.
    """
    backend = _FakeBackend(start_error=RuntimeError("provisioned then wait timed out"))
    mgr = _make_manager(backend)

    with _patch_ready():
        with pytest.raises(RuntimeError):
            mgr.start()

    assert backend.start_calls == 1
    assert backend.stop_calls == 1  # teardown fired despite the start-phase failure
    assert mgr.endpoint_url is None


def test_start_tears_down_on_keyboardinterrupt_during_startup():
    """Ctrl-C during the (long, possibly PAID) startup window tears down. The
    interrupt is a BaseException, not Exception, so this guards the exact case a
    narrow ``except Exception`` would leak.
    """
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with patch(
        "study_query_llm.vllm_serving.manager.probe.wait_for_models_ready",
        side_effect=KeyboardInterrupt(),
    ):
        with pytest.raises(KeyboardInterrupt):
            mgr.start()

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_calls_backend_stop_and_clears_endpoint():
    """stop() calls backend.stop() and resets endpoint_url to None."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with _patch_ready():
        mgr.start()
        assert mgr.endpoint_url is not None
        mgr.stop()

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


def test_stop_is_idempotent():
    """Multiple stop() calls hit the backend exactly once."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with _patch_ready():
        mgr.start()
        mgr.stop()
        mgr.stop()
        mgr.stop()

    assert backend.stop_calls == 1


def test_stop_swallows_backend_error():
    """A raising backend.stop() does not propagate; teardown still completes."""
    backend = _FakeBackend(stop_error=RuntimeError("boom"))
    mgr = _make_manager(backend)

    with _patch_ready():
        mgr.start()
        mgr.stop()  # must not raise

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


def test_stop_cancels_idle_timer():
    """stop() cancels and clears the idle timer."""
    backend = _FakeBackend()
    mgr = _make_manager(backend, idle_timeout_seconds=9999)

    with _patch_ready():
        mgr.start()
        timer = mgr._idle_timer
        mgr.stop()

    assert timer.finished.is_set()
    assert mgr._idle_timer is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager_starts_and_stops():
    """__enter__ starts (URL set) and __exit__ stops (URL cleared)."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with _patch_ready():
        with mgr:
            assert mgr.endpoint_url is not None
        assert mgr.endpoint_url is None

    assert backend.start_calls == 1
    assert backend.stop_calls == 1


def test_exit_tears_down_on_exception():
    """__exit__ stops the backend even when the with-body raises."""
    backend = _FakeBackend()
    mgr = _make_manager(backend)

    with _patch_ready():
        with pytest.raises(RuntimeError):
            with mgr:
                raise RuntimeError("body blew up")

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


# ---------------------------------------------------------------------------
# ping() / idle timer
# ---------------------------------------------------------------------------


def test_ping_resets_idle_timer():
    """ping() cancels the existing timer and starts a fresh one."""
    backend = _FakeBackend()
    mgr = _make_manager(backend, idle_timeout_seconds=9999)

    with _patch_ready():
        mgr.start()
        first = mgr._idle_timer
        mgr.ping()
        second = mgr._idle_timer

    assert first.finished.is_set()  # old timer cancelled
    assert second is not None
    assert second is not first
    second.cancel()


def test_ping_noop_after_stop():
    """ping() does nothing once the manager is stopped."""
    backend = _FakeBackend()
    mgr = _make_manager(backend, idle_timeout_seconds=9999)

    with _patch_ready():
        mgr.start()
        mgr.stop()
        mgr.ping()

    assert mgr._idle_timer is None


def test_idle_timer_fires_stop():
    """When the idle timeout expires the manager tears the backend down."""
    backend = _FakeBackend()
    # Tiny timeout so the daemon timer fires almost immediately.
    mgr = _make_manager(backend, idle_timeout_seconds=0.01)

    with _patch_ready():
        mgr.start()
        # Poll briefly for the timer thread to call stop().
        deadline = time.monotonic() + 2.0
        while backend.stop_calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    assert backend.stop_calls == 1
    assert mgr.endpoint_url is None


# ---------------------------------------------------------------------------
# Signal handlers (focused; monkeypatched so real process handlers are untouched)
# ---------------------------------------------------------------------------


def test_signal_handlers_installed_in_start_and_restored_in_stop(monkeypatch):
    """With install_signal_handlers=True, start() installs SIGINT+SIGTERM handlers
    and stop() restores the originals -- asserted via a fake signal.signal so the
    real process handlers are never mutated.
    """
    installed: dict[int, object] = {}
    sentinel_prev = {signal.SIGINT: "orig_int", signal.SIGTERM: "orig_term"}

    def fake_getsignal(signum):
        return sentinel_prev[signum]

    def fake_signal(signum, handler):
        previous = installed.get(signum, sentinel_prev[signum])
        installed[signum] = handler
        return previous

    monkeypatch.setattr(signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(signal, "signal", fake_signal)
    # Force the "main thread" branch regardless of where pytest runs the test.
    monkeypatch.setattr(
        threading, "current_thread", lambda: threading.main_thread()
    )

    backend = _FakeBackend()
    mgr = VLLMModelManager(
        backend,
        _make_config(),
        idle_timeout_seconds=9999,
        install_signal_handlers=True,
        install_atexit=False,
    )

    with _patch_ready():
        mgr.start()

        # Both signals now point at the manager's handler.
        assert installed[signal.SIGINT] == mgr._handle_signal
        assert installed[signal.SIGTERM] == mgr._handle_signal
        # The manager saved the originals so it can restore them.
        assert mgr._prev_signal_handlers[signal.SIGINT] == "orig_int"
        assert mgr._prev_signal_handlers[signal.SIGTERM] == "orig_term"

        mgr.stop()

    # stop() restored the originals and forgot its saved handlers.
    assert installed[signal.SIGINT] == "orig_int"
    assert installed[signal.SIGTERM] == "orig_term"
    assert mgr._prev_signal_handlers == {}


def test_signal_handlers_skipped_off_main_thread(monkeypatch):
    """Off the main thread, start() installs no signal handlers (and never calls
    signal.signal), per the main-thread-only guard.
    """
    calls = []
    monkeypatch.setattr(signal, "signal", lambda *a, **k: calls.append(a))

    # Pretend we are NOT on the main thread.  Return a real, non-main Thread
    # object (it has a ``.name``, which the logging machinery reads) so only the
    # main-thread *identity* check trips, not unrelated logging internals.
    not_main = threading.Thread(target=lambda: None, name="fake-worker")
    monkeypatch.setattr(threading, "current_thread", lambda: not_main)

    mgr = VLLMModelManager(
        _FakeBackend(),
        _make_config(),
        idle_timeout_seconds=9999,
        install_signal_handlers=True,
        install_atexit=False,
    )

    with _patch_ready():
        mgr.start()
        mgr.stop()

    assert calls == []
    assert mgr._prev_signal_handlers == {}

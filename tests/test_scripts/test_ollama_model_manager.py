"""Tests for scripts.common.ollama_model_manager.OllamaModelManager.

All Ollama API calls are mocked -- no live server required.
"""

import time
from unittest.mock import MagicMock, patch, call

import pytest

from study_query_llm.providers.managers.ollama import OllamaModelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(**kwargs) -> OllamaModelManager:
    defaults = dict(
        model_id="llama3.1:8b",
        endpoint="http://localhost:11434",
        idle_timeout_seconds=300,
    )
    defaults.update(kwargs)
    return OllamaModelManager(**defaults)


def _patch_api():
    """Patch _api_request so no real HTTP calls are made."""
    return patch.object(OllamaModelManager, "_api_request", return_value=b"{}")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_init_sets_model_id():
    mgr = _make_manager(model_id="qwen2.5:32b")
    assert mgr.model_id == "qwen2.5:32b"


def test_init_endpoint_url_is_none():
    mgr = _make_manager()
    assert mgr.endpoint_url is None


def test_init_provider_label_default():
    mgr = _make_manager()
    assert mgr.provider_label == "ollama"


def test_init_custom_provider_label():
    mgr = _make_manager(provider_label="my_ollama")
    assert mgr.provider_label == "my_ollama"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

def test_start_sets_endpoint_url():
    mgr = _make_manager()
    with _patch_api():
        url = mgr.start()
    assert url == "http://localhost:11434/v1"
    assert mgr.endpoint_url == "http://localhost:11434/v1"
    mgr.stop()


def test_start_calls_warm_load():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        mgr.start()
    mock_api.assert_called_once()
    args = mock_api.call_args
    assert args[0][0] == "/api/chat"
    assert args[0][1]["model"] == "llama3.1:8b"
    assert args[0][1]["stream"] is False
    mgr.stop()


def test_start_is_idempotent():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        mgr.start()
        mgr.start()
    assert mock_api.call_count == 1
    mgr.stop()


def test_start_with_custom_endpoint():
    mgr = _make_manager(endpoint="http://192.168.1.100:11434")
    with _patch_api():
        url = mgr.start()
    assert url == "http://192.168.1.100:11434/v1"
    mgr.stop()


def test_start_starts_idle_timer():
    mgr = _make_manager(idle_timeout_seconds=9999)
    with _patch_api():
        mgr.start()
    assert mgr._idle_timer is not None
    mgr._idle_timer.cancel()
    mgr.stop()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------

def test_stop_sends_unload_request():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        mgr.start()
        mock_api.reset_mock()
        mgr.stop()
    mock_api.assert_called_once()
    args = mock_api.call_args
    assert args[0][1]["keep_alive"] == "0"


def test_stop_clears_endpoint_url():
    mgr = _make_manager()
    with _patch_api():
        mgr.start()
        mgr.stop()
    assert mgr.endpoint_url is None


def test_stop_cancels_idle_timer():
    mgr = _make_manager(idle_timeout_seconds=9999)
    with _patch_api():
        mgr.start()
        timer = mgr._idle_timer
        mgr.stop()
    assert timer.finished.is_set()
    assert mgr._idle_timer is None


def test_stop_is_idempotent():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        mgr.start()
        mock_api.reset_mock()
        mgr.stop()
        mgr.stop()
    assert mock_api.call_count == 1


def test_stop_handles_exception_gracefully():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        mgr.start()
        mock_api.side_effect = RuntimeError("connection refused")
        mgr.stop()  # should not raise
    assert mgr.endpoint_url is None


# ---------------------------------------------------------------------------
# ping()
# ---------------------------------------------------------------------------

def test_ping_resets_idle_timer():
    mgr = _make_manager(idle_timeout_seconds=9999)
    with _patch_api():
        mgr.start()
        first_timer = mgr._idle_timer
        mgr.ping()
        second_timer = mgr._idle_timer

    assert first_timer.finished.is_set()
    assert second_timer is not None
    assert second_timer is not first_timer
    second_timer.cancel()
    mgr.stop()


def test_ping_noop_after_stop():
    mgr = _make_manager(idle_timeout_seconds=9999)
    with _patch_api():
        mgr.start()
        mgr.stop()
        mgr.ping()
    assert mgr._idle_timer is None


# ---------------------------------------------------------------------------
# _idle_shutdown()
# ---------------------------------------------------------------------------

def test_idle_shutdown_calls_stop():
    mgr = _make_manager(idle_timeout_seconds=0.05)
    with _patch_api():
        mgr.start()
        time.sleep(0.3)
    assert mgr.endpoint_url is None


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_starts_and_stops():
    mgr = _make_manager()
    with _patch_api() as mock_api:
        with mgr:
            assert mgr.endpoint_url is not None
        assert mgr.endpoint_url is None


def test_context_manager_stops_on_exception():
    mgr = _make_manager()
    with _patch_api():
        with pytest.raises(RuntimeError):
            with mgr:
                raise RuntimeError("sweep failed")
    assert mgr.endpoint_url is None

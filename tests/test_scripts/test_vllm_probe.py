"""Fully-mocked unit tests for ``study_query_llm.vllm_serving.probe``.

These tests exercise the gate-(a)/gate-(b) probe utilities and the
``wait_for_models_ready`` health poll **without any network, docker daemon,
GPU, or vast/ssh access**.  All HTTP is fed in through the injectable
``http_get`` / ``http_post`` seams as canned PLAIN-JSON responses (``.json()``
returns a dict, never an ``openai`` SDK object), mirroring the mocking style in
``test_local_docker_tei_manager.py`` and ``test_model_manager_protocol.py``.

Covered:

* ``_normalize_letter_token`` -- the locally reimplemented token->letter
  normalizer (strips ``▁``/``Ġ`` and whitespace, upper-cases).
* ``probe_gate_a`` -- passes on a non-empty ``content[0].top_logprobs``; fails
  when that list is missing / empty.
* ``probe_gate_b`` -- passes when an A-E token (including a ``▁B``/``ĠC``-style
  prefixed token) appears; fails when the first-token top list is reasoning /
  prose (``<think>`` / words); ``predicted_letter`` is the highest-logprob A-E
  token; ``extra_body`` is merged into the request body.
* ``wait_for_models_ready`` -- returns once the getter yields HTTP 200; raises
  ``TimeoutError`` when it never does (tiny timeout + fake getter, no real
  sleeping of any consequence).
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from study_query_llm.vllm_serving.probe import (
    DEFAULT_PROBE_MCQ,
    DEFAULT_PROBE_SYSTEM,
    GateResult,
    OPTION_LABELS,
    _normalize_letter_token,
    probe_gate_a,
    probe_gate_b,
    wait_for_models_ready,
)


BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen2.5-7B-Instruct-AWQ"


# --------------------------------------------------------------------------- #
# Canned-response helpers (PLAIN JSON; no openai SDK objects, no network)
# --------------------------------------------------------------------------- #


class _FakeResponse:
    """Minimal stand-in for a ``requests`` response.

    Exposes only what the probe touches: ``.status_code`` and ``.json()``.
    """

    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload


def _chat_payload(top_logprobs: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    """Build an OpenAI-schema chat-completions JSON dict.

    Passing ``top_logprobs=None`` omits the ``top_logprobs`` key entirely so we
    can exercise the "missing" branch as distinct from an empty list.
    """
    first_token: dict[str, Any] = {"token": "X", "logprob": -0.1}
    if top_logprobs is not None:
        first_token["top_logprobs"] = top_logprobs
    return {
        "choices": [
            {
                "logprobs": {"content": [first_token]},
                "message": {"role": "assistant", "content": "A"},
            }
        ]
    }


def _make_poster(payload: Any, *, status_code: int = 200):
    """Return ``(poster, calls)`` where ``poster`` records each invocation."""
    calls: list[dict[str, Any]] = []

    def poster(url: str, json: Any = None, timeout: Any = None):  # noqa: A002
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse(payload, status_code=status_code)

    return poster, calls


# --------------------------------------------------------------------------- #
# _normalize_letter_token
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        (" b ", "B"),
        ("▁B", "B"),  # SentencePiece "▁B"
        ("ĠC", "C"),  # GPT-2 byte-level BPE "ĠC"
        ("▁Ġd", "D"),  # both markers stacked
        ("\te\n", "E"),
        ("<think>", "<THINK>"),  # not a letter; stays non-A-E
        ("the", "THE"),
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_letter_token(raw, expected):
    """Marker glyphs (▁/Ġ) and whitespace are stripped, then upper-cased."""
    assert _normalize_letter_token(raw) == expected


def test_normalize_letter_token_prefixed_round_trips_into_option_labels():
    """A ▁/Ġ-prefixed letter normalizes into the recognized OPTION_LABELS set."""
    assert _normalize_letter_token("▁B") in OPTION_LABELS
    assert _normalize_letter_token("ĠC") in OPTION_LABELS
    assert _normalize_letter_token("<think>") not in OPTION_LABELS


# --------------------------------------------------------------------------- #
# probe_gate_a
# --------------------------------------------------------------------------- #


def test_gate_a_passes_when_top_logprobs_present():
    """Gate (a) PASSES when content[0].top_logprobs is a non-empty list."""
    payload = _chat_payload(
        [
            {"token": "A", "logprob": -0.05},
            {"token": "B", "logprob": -1.2},
        ]
    )
    poster, calls = _make_poster(payload)

    result = probe_gate_a(BASE_URL, MODEL, http_post=poster)

    assert isinstance(result, GateResult)
    assert result.gate == "a"
    assert result.passed is True
    # Request was shaped for logprob extraction and hit /chat/completions.
    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    body = calls[0]["json"]
    assert body["model"] == MODEL
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["temperature"] == 0


def test_gate_a_fails_when_top_logprobs_empty():
    """Gate (a) FAILS when content[0].top_logprobs is an empty list."""
    poster, _ = _make_poster(_chat_payload([]))

    result = probe_gate_a(BASE_URL, MODEL, http_post=poster)

    assert result.gate == "a"
    assert result.passed is False


def test_gate_a_fails_when_top_logprobs_missing():
    """Gate (a) FAILS when the top_logprobs key is absent entirely."""
    poster, _ = _make_poster(_chat_payload(None))

    result = probe_gate_a(BASE_URL, MODEL, http_post=poster)

    assert result.gate == "a"
    assert result.passed is False


def test_gate_a_fails_on_empty_choices():
    """Gate (a) FAILS defensively when choices is empty (no exception)."""
    poster, _ = _make_poster({"choices": []})

    result = probe_gate_a(BASE_URL, MODEL, http_post=poster)

    assert result.passed is False


# --------------------------------------------------------------------------- #
# probe_gate_b
# --------------------------------------------------------------------------- #


def test_gate_b_passes_with_plain_letter_tokens():
    """Gate (b) PASSES when an A-E token appears in the first-token top list."""
    payload = _chat_payload(
        [
            {"token": "A", "logprob": -0.3},
            {"token": "B", "logprob": -0.1},  # highest -> predicted
            {"token": "D", "logprob": -1.0},
            {"token": "C", "logprob": -2.0},
            {"token": "E", "logprob": -3.0},
        ]
    )
    poster, _ = _make_poster(payload)

    result = probe_gate_b(BASE_URL, MODEL, http_post=poster)

    assert result.gate == "b"
    assert result.passed is True
    assert result.predicted_letter == "B"
    # top_tokens carries the full (token, logprob) list.
    assert ("A", -0.3) in result.top_tokens
    assert len(result.top_tokens) == 5


def test_gate_b_passes_with_prefixed_letter_tokens():
    """Gate (b) PASSES on ▁B / ĠC-prefixed tokens (marker-stripped to A-E)."""
    payload = _chat_payload(
        [
            {"token": "▁B", "logprob": -0.4},  # "▁B" -> B
            {"token": "ĠC", "logprob": -0.2},  # "ĠC" -> C (highest)
            {"token": "the", "logprob": -1.5},
        ]
    )
    poster, _ = _make_poster(payload)

    result = probe_gate_b(BASE_URL, MODEL, http_post=poster)

    assert result.passed is True
    # Highest-logprob normalized A-E token wins.
    assert result.predicted_letter == "C"


def test_gate_b_predicted_letter_is_highest_logprob():
    """predicted_letter is the highest-logprob A-E token, not first-seen."""
    payload = _chat_payload(
        [
            {"token": "A", "logprob": -2.0},
            {"token": "E", "logprob": -0.05},  # highest A-E
            {"token": "C", "logprob": -1.0},
        ]
    )
    poster, _ = _make_poster(payload)

    result = probe_gate_b(BASE_URL, MODEL, http_post=poster)

    assert result.passed is True
    assert result.predicted_letter == "E"


def test_gate_b_fails_on_thinking_token():
    """Gate (b) FAILS when the first-token top list is <think> / prose words."""
    payload = _chat_payload(
        [
            {"token": "<think>", "logprob": -0.05},
            {"token": "Okay", "logprob": -0.5},
            {"token": "The", "logprob": -1.0},
            {"token": "Let", "logprob": -2.0},
        ]
    )
    poster, _ = _make_poster(payload)

    result = probe_gate_b(BASE_URL, MODEL, http_post=poster)

    assert result.gate == "b"
    assert result.passed is False
    assert result.predicted_letter is None
    # The observed tokens are still surfaced for diagnostics.
    assert len(result.top_tokens) == 4


def test_gate_b_fails_when_top_logprobs_empty():
    """Gate (b) FAILS (gate-a prerequisite) when first-token top list is empty."""
    poster, _ = _make_poster(_chat_payload([]))

    result = probe_gate_b(BASE_URL, MODEL, http_post=poster)

    assert result.passed is False
    assert result.predicted_letter is None


def test_gate_b_merges_extra_body_into_request():
    """extra_body (e.g. thinking-off) is merged into the JSON request body."""
    payload = _chat_payload([{"token": "A", "logprob": -0.1}])
    poster, calls = _make_poster(payload)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}

    result = probe_gate_b(BASE_URL, MODEL, extra_body=extra, http_post=poster)

    assert result.passed is True
    body = calls[0]["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    # Protocol fields are still present alongside the merged extras.
    assert body["model"] == MODEL
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["temperature"] == 0
    # System / user messages follow the verified gate-(b) protocol.
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == DEFAULT_PROBE_SYSTEM
    assert body["messages"][1]["content"] == DEFAULT_PROBE_MCQ


def test_gate_b_honours_custom_top_logprobs_count():
    """top_logprobs kwarg flows through to the request body."""
    payload = _chat_payload([{"token": "A", "logprob": -0.1}])
    poster, calls = _make_poster(payload)

    probe_gate_b(BASE_URL, MODEL, top_logprobs=10, http_post=poster)

    assert calls[0]["json"]["top_logprobs"] == 10


# --------------------------------------------------------------------------- #
# wait_for_models_ready
# --------------------------------------------------------------------------- #


def test_wait_for_models_ready_returns_on_200():
    """Returns (no raise) the moment the getter yields HTTP 200."""
    calls: list[dict[str, Any]] = []

    def getter(url: str, timeout: Any = None):
        calls.append({"url": url, "timeout": timeout})
        return _FakeResponse(payload={"data": []}, status_code=200)

    # Should return immediately; assert it hit GET {base}/models.
    wait_for_models_ready(BASE_URL, timeout=5, interval=0.01, http_get=getter)

    assert len(calls) == 1
    assert calls[0]["url"] == "http://localhost:8000/v1/models"


def test_wait_for_models_ready_times_out_when_never_200():
    """Raises TimeoutError when the getter never yields 200 (tiny timeout)."""
    calls: list[int] = []

    def getter(url: str, timeout: Any = None):
        calls.append(1)
        return _FakeResponse(payload={"error": "loading"}, status_code=503)

    with pytest.raises(TimeoutError):
        # Tiny timeout + tiny interval -> loop gives up almost immediately.
        wait_for_models_ready(BASE_URL, timeout=0.01, interval=0.001, http_get=getter)

    # At least one attempt was made before timing out.
    assert calls


def test_wait_for_models_ready_times_out_on_connection_errors():
    """Connection errors are swallowed during warm-up; still times out cleanly."""

    def getter(url: str, timeout: Any = None):
        raise ConnectionError("server not up yet")

    with pytest.raises(TimeoutError):
        wait_for_models_ready(BASE_URL, timeout=0.01, interval=0.001, http_get=getter)

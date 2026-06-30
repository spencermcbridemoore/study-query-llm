"""
vLLM serving health-check and gate probes.

This module provides the small, self-contained utilities used to confirm that a
freshly-launched vLLM OpenAI-compatible server is (1) up and (2) usable for the
first-token logprob MCQ lane:

* :func:`wait_for_models_ready` -- poll ``GET {base_url}/models`` until HTTP 200.
* :func:`probe_gate_a`         -- gate (a): does the server return OpenAI-schema
  ``choices[0].logprobs.content[0].top_logprobs`` at all?  vLLM *guarantees*
  this, so a failure here means the server / request is broken.
* :func:`probe_gate_b`         -- gate (b): does the *specific model* place an
  A-E option letter in the first decoded token's ``top_logprobs``?  This is
  per-model and MUST be re-probed before any bulk run.

Design constraints (deliberate):

* Everything is parsed from **plain JSON dicts** (``response.json()``), never
  from ``openai`` SDK objects.  The production method runner
  (``mcq_logprob_basic``) walks SDK attribute objects; here we walk dicts so the
  probe stays dependency-light and works against any HTTP client.
* The token -> option-letter normalization is **reimplemented locally**
  (:func:`_normalize_letter_token`) rather than imported from
  ``mcq_logprob_basic`` -- this module must not import production lane code.
* All HTTP is routed through injectable ``http_get`` / ``http_post`` seams so
  unit tests never touch the network.

Thinking-off note (important / verified against the live codebase):
    The production lane (``mcq_logprob_basic``) honours ``MCQ_LOGPROB_MAX_TOKENS``
    and ``MCQ_LOGPROB_PROVIDER_PIN`` but has **no** ``MCQ_LOGPROB_EXTRA_BODY``
    hook -- a per-request ``extra_body`` would never reach the provider.  For the
    PRODUCTION lane, thinking-off MUST therefore be applied at the SERVE layer
    (vLLM ``--reasoning-parser`` / a no-think ``--chat-template`` / a
    non-thinking snapshot repo).  This standalone gate-(b) probe, by contrast,
    CAN pass ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``
    directly, which is the recommended way to verify a reasoning model emits a
    letter (not ``<think>``) as its first token.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: The five MCQ option letters this lane recognizes.
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E")

#: System prompt that pins the model to emitting exactly one option letter.
DEFAULT_PROBE_SYSTEM: str = (
    "Reply with exactly one letter (A, B, C, D, or E). Do not explain."
)

#: A trivial 5-option MCQ whose only purpose is to elicit a single-letter
#: first token so gate (b) can inspect its ``top_logprobs``.
DEFAULT_PROBE_MCQ: str = (
    "Which of the following is a primary color?\n"
    "A) Green\n"
    "B) Orange\n"
    "C) Red\n"
    "D) Purple\n"
    "E) Brown\n"
    "Answer:"
)

#: Leading marker glyphs some tokenizers prepend to single-letter tokens:
#: ``▁`` is the SentencePiece "▁" lower-one-eighth block, ``Ġ`` is the
#: GPT-2 byte-level BPE "Ġ" space marker.  Stripped before comparison.
_LEADING_TOKEN_MARKERS: str = "▁Ġ \t\r\n\f\v"


# --------------------------------------------------------------------------- #
# Token -> option-letter normalization (local reimplementation)
# --------------------------------------------------------------------------- #


def _normalize_letter_token(token: Any) -> str:
    """Normalize a raw model token to a bare uppercase option letter.

    Mirrors ``mcq_logprob_basic._token_matches_letter`` but is **reimplemented
    locally** so this module never imports production lane code.  Strips the
    leading SentencePiece / byte-level-BPE whitespace marker glyphs, trims
    surrounding whitespace, and upper-cases.

    A ``None`` token (or anything falsy) normalizes to ``""``.
    """
    return str(token or "").lstrip(_LEADING_TOKEN_MARKERS).strip().upper()


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateResult:
    """Outcome of a single gate probe.

    Attributes:
        gate: Short gate identifier (``"a"`` or ``"b"``).
        passed: Whether the gate's pass condition was met.
        detail: Human-readable explanation (also used for failures / errors).
        top_tokens: For gate (b), the first-token ``(token, logprob)`` pairs as
            returned by the server (logprob may be ``None`` if unparsable).
        predicted_letter: For gate (b), the highest-logprob A-E token (or
            ``None`` if no option letter appeared).
    """

    gate: str
    passed: bool
    detail: str
    top_tokens: tuple[tuple[str, Optional[float]], ...] = ()
    predicted_letter: Optional[str] = None


# --------------------------------------------------------------------------- #
# Defensive JSON walking helpers
# --------------------------------------------------------------------------- #


def _first_choice_content_top_logprobs(payload: Any) -> Optional[list[Any]]:
    """Return ``choices[0].logprobs.content[0].top_logprobs`` from a JSON dict.

    Returns ``None`` (rather than raising) whenever any link in that chain is
    missing or of an unexpected type, so callers can render a clean failure.
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    logprobs_obj = first_choice.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return None
    content = logprobs_obj.get("content")
    if not isinstance(content, list) or not content:
        return None
    first_token = content[0]
    if not isinstance(first_token, dict):
        return None
    top_logprobs = first_token.get("top_logprobs")
    if not isinstance(top_logprobs, list):
        return None
    return top_logprobs


def _coerce_logprob(raw: Any) -> Optional[float]:
    """Best-effort float coercion for a logprob value; ``None`` on failure."""
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_top_token_pairs(
    top_logprobs: list[Any],
) -> tuple[tuple[str, Optional[float]], ...]:
    """Flatten ``top_logprobs`` entries into ``(token, logprob)`` pairs."""
    pairs: list[tuple[str, Optional[float]]] = []
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        pairs.append((str(token if token is not None else ""), _coerce_logprob(entry.get("logprob"))))
    return tuple(pairs)


# --------------------------------------------------------------------------- #
# Request body builders
# --------------------------------------------------------------------------- #


def _chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #


def wait_for_models_ready(
    base_url: str,
    *,
    timeout: float = 600,
    interval: float = 5,
    http_get: Optional[Callable[..., Any]] = None,
) -> None:
    """Poll ``GET {base_url}/models`` until it returns HTTP 200.

    Args:
        base_url: Server base URL ending in ``/v1`` (trailing slash tolerated).
        timeout: Total seconds to keep polling before giving up.
        interval: Seconds to sleep between attempts.
        http_get: Injectable getter ``http_get(url, timeout) -> obj`` where the
            returned object exposes ``.status_code``.  Defaults to
            :func:`requests.get`.  Tests can patch this to return 200 on the
            first call.

    Raises:
        TimeoutError: if no HTTP-200 response is observed within ``timeout``.
    """
    getter = http_get if http_get is not None else requests.get
    url = _models_url(base_url)

    deadline = time.monotonic() + float(timeout)
    last_error: Optional[str] = None
    attempt = 0

    while True:
        attempt += 1
        try:
            response = getter(url, timeout=min(float(interval) + 5.0, 30.0))
            status = getattr(response, "status_code", None)
            if status == 200:
                logger.info("vLLM /models ready at %s after %d attempt(s)", url, attempt)
                return
            last_error = f"HTTP {status}"
        except Exception as exc:  # noqa: BLE001 - connection errors are expected while warming up
            last_error = f"{type(exc).__name__}: {exc}"

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"vLLM server at {url} not ready within {timeout}s "
                f"(last status: {last_error})"
            )

        logger.debug(
            "vLLM /models not ready yet (attempt %d, %s); retrying in %ss",
            attempt,
            last_error,
            interval,
        )
        time.sleep(float(interval))


# --------------------------------------------------------------------------- #
# Gate (a): OpenAI-schema top_logprobs presence
# --------------------------------------------------------------------------- #


def probe_gate_a(
    base_url: str,
    model: str,
    *,
    http_post: Optional[Callable[..., Any]] = None,
) -> GateResult:
    """Gate (a): confirm the server returns OpenAI-schema top_logprobs.

    Sends a minimal chat-completions request with ``logprobs=True`` /
    ``top_logprobs=5`` and PASSES iff
    ``choices[0].logprobs.content[0].top_logprobs`` is a non-empty list.

    vLLM guarantees this gate; a failure indicates a broken server, an
    unsupported request, or a non-vLLM endpoint.

    Args:
        base_url: Server base URL ending in ``/v1``.
        model: The ``model`` field clients pass (== served-model-name).
        http_post: Injectable poster ``http_post(url, json, timeout) -> obj``
            exposing ``.json()`` and ``.status_code``.  Defaults to
            :func:`requests.post`.
    """
    poster = http_post if http_post is not None else requests.post
    url = _chat_completions_url(base_url)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single letter A."}],
        "temperature": 0,
        "max_tokens": 16,
        "logprobs": True,
        "top_logprobs": 5,
    }

    try:
        response = poster(url, json=body, timeout=60)
    except Exception as exc:  # noqa: BLE001 - surface as a clean gate failure
        return GateResult(
            gate="a",
            passed=False,
            detail=f"request error: {type(exc).__name__}: {exc}",
        )

    status = getattr(response, "status_code", None)
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="a",
            passed=False,
            detail=f"non-JSON response (HTTP {status}): {type(exc).__name__}: {exc}",
        )

    top_logprobs = _first_choice_content_top_logprobs(payload)
    if top_logprobs:
        return GateResult(
            gate="a",
            passed=True,
            detail=(
                f"OpenAI-schema top_logprobs present "
                f"({len(top_logprobs)} entries, HTTP {status})"
            ),
        )

    return GateResult(
        gate="a",
        passed=False,
        detail=(
            f"missing/empty choices[0].logprobs.content[0].top_logprobs "
            f"(HTTP {status})"
        ),
    )


# --------------------------------------------------------------------------- #
# Gate (b): per-model first-token option-letter presence
# --------------------------------------------------------------------------- #


def probe_gate_b(
    base_url: str,
    model: str,
    *,
    mcq: str = DEFAULT_PROBE_MCQ,
    system: str = DEFAULT_PROBE_SYSTEM,
    extra_body: Optional[dict[str, Any]] = None,
    top_logprobs: int = 5,
    http_post: Optional[Callable[..., Any]] = None,
) -> GateResult:
    """Gate (b): confirm *this model* emits an A-E letter in its first token.

    Implements the verified gate-(b) probe protocol:

    * ``system`` pins the model to a single option letter.
    * ``user`` is a 5-option (A-E) MCQ.
    * ``temperature=0``, ``max_tokens=16``, ``logprobs=True``,
      ``top_logprobs=<top_logprobs>``.
    * Any ``extra_body`` is merged into the request body (this is where a
      standalone probe can disable reasoning-model "thinking" via
      ``{"chat_template_kwargs": {"enable_thinking": False}}``).

    PASSES iff any token in ``content[0].top_logprobs`` normalizes to an A-E
    letter.  ``predicted_letter`` is set to the highest-logprob A-E token and
    ``top_tokens`` carries the full ``(token, logprob)`` list.

    Args:
        base_url: Server base URL ending in ``/v1``.
        model: The ``model`` field clients pass (== served-model-name).
        mcq: The 5-option MCQ user prompt.
        system: System prompt pinning single-letter output.
        extra_body: Optional dict merged into the JSON body (e.g. thinking-off).
        top_logprobs: Number of top tokens to request (default 5).
        http_post: Injectable poster ``http_post(url, json, timeout) -> obj``
            exposing ``.json()`` and ``.status_code``.  Defaults to
            :func:`requests.post`.
    """
    poster = http_post if http_post is not None else requests.post
    url = _chat_completions_url(base_url)
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": mcq},
        ],
        "temperature": 0,
        "max_tokens": 16,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    if extra_body:
        # Merge probe-only extras (e.g. chat_template_kwargs.enable_thinking).
        body.update(extra_body)

    try:
        response = poster(url, json=body, timeout=60)
    except Exception as exc:  # noqa: BLE001 - surface as a clean gate failure
        return GateResult(
            gate="b",
            passed=False,
            detail=f"request error: {type(exc).__name__}: {exc}",
        )

    status = getattr(response, "status_code", None)
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate="b",
            passed=False,
            detail=f"non-JSON response (HTTP {status}): {type(exc).__name__}: {exc}",
        )

    raw_top_logprobs = _first_choice_content_top_logprobs(payload)
    if not raw_top_logprobs:
        return GateResult(
            gate="b",
            passed=False,
            detail=(
                f"missing/empty first-token top_logprobs (HTTP {status}); "
                f"gate (a) prerequisite not met"
            ),
        )

    top_tokens = _extract_top_token_pairs(raw_top_logprobs)

    # Find the highest-logprob token that normalizes to an option letter.
    best_letter: Optional[str] = None
    best_logprob: Optional[float] = None
    for token, logprob in top_tokens:
        letter = _normalize_letter_token(token)
        if letter not in OPTION_LABELS:
            continue
        if best_letter is None:
            # First option letter seen; entries with a None logprob still count
            # as "an A-E letter is present" but cannot win a numeric comparison.
            best_letter, best_logprob = letter, logprob
            continue
        if logprob is not None and (best_logprob is None or logprob > best_logprob):
            best_letter, best_logprob = letter, logprob

    if best_letter is not None:
        return GateResult(
            gate="b",
            passed=True,
            detail=(
                f"first-token top_logprobs contains option letter "
                f"'{best_letter}' (HTTP {status})"
            ),
            top_tokens=top_tokens,
            predicted_letter=best_letter,
        )

    observed = ", ".join(repr(tok) for tok, _ in top_tokens) or "<none>"
    return GateResult(
        gate="b",
        passed=False,
        detail=(
            f"no A-E option letter in first-token top_logprobs "
            f"(HTTP {status}); observed tokens: {observed}"
        ),
        top_tokens=top_tokens,
        predicted_letter=None,
    )

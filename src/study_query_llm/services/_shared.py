"""Shared utilities for service-layer classes.

Centralises patterns that were previously duplicated across
embedding_service, summarization_service, and inference_service.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# Union of retryable error patterns used by both EmbeddingService and
# InferenceService.  Order doesn't matter -- we just check substring membership.
RETRYABLE_ERROR_PATTERNS = [
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "internal server",
    "service unavailable",
    "temporary",
    "throttl",
]


def should_retry_exception(exc: BaseException) -> bool:
    """Return ``True`` if *exc* looks like a transient / retryable error.

    Checks ``TimeoutError`` / ``ConnectionError`` by type, then falls back to
    substring matching on the stringified exception message.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    error_msg = str(exc).lower()
    return any(p in error_msg for p in RETRYABLE_ERROR_PATTERNS)


@contextmanager
def deployment_override(env_var: str, value: str):
    """Temporarily set an environment variable, restoring it on exit.

    Usage::

        with deployment_override("AZURE_OPENAI_DEPLOYMENT", deployment):
            config = Config()
            ...
    """
    original = os.environ.get(env_var)
    os.environ[env_var] = value
    try:
        yield
    finally:
        if original is not None:
            os.environ[env_var] = original
        elif env_var in os.environ:
            del os.environ[env_var]


def handle_db_persistence_error(
    svc_logger: logging.Logger,
    error: Exception,
    require_db_persistence: bool,
    context_msg: str,
) -> Optional[int]:
    """Handle a database persistence failure consistently.

    When *require_db_persistence* is ``True``, logs at ERROR level and
    re-raises as ``RuntimeError``.  Otherwise logs a WARNING and returns
    ``None`` so the caller can continue gracefully.
    """
    if require_db_persistence:
        svc_logger.error(
            "DB persistence failed (required) – %s: %s",
            context_msg,
            error,
            exc_info=True,
        )
        raise RuntimeError(
            f"Database persistence failed ({context_msg}). "
            f"Original error: {error}"
        ) from error

    svc_logger.warning(
        "DB persistence failed (non-required) – %s: %s",
        context_msg,
        error,
        exc_info=True,
    )
    return None

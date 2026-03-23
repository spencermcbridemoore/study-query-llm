"""Deterministic hashing for embedding cache keys."""

import hashlib
import re
from typing import Optional

from .constants import CACHE_KEY_VERSION


def normalize_embedding_text(text: str) -> str:
    """
    Normalize text for validation (empty check, token limits).

    Removes null bytes and collapses whitespace.
    """
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_request_hash(
    text: str,
    model: str,
    dimensions: Optional[int],
    encoding_format: str,
    provider: str,
) -> str:
    """Compute deterministic hash for cache lookup (raw text identity)."""
    components = [
        CACHE_KEY_VERSION,
        provider,
        model,
        text,
        str(dimensions) if dimensions else "",
        encoding_format,
    ]
    hash_input = "|".join(components)
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def compute_raw_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

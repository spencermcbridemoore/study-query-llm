"""Embedding service constants: deployment limits and cache key version."""

from typing import Dict

# Known maximum token limits for embedding deployments
DEPLOYMENT_MAX_TOKENS: Dict[str, int] = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
}

DEFAULT_MAX_TOKENS = 8191
CACHE_KEY_VERSION = "raw_v1"

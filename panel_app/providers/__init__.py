"""
LLM Provider Abstraction Layer

This package provides a unified interface for interacting with different
LLM providers (Azure OpenAI, OpenAI, Hyperbolic, etc.).

All providers implement the BaseLLMProvider interface and return
standardized ProviderResponse objects.
"""

from .base import BaseLLMProvider, ProviderResponse

__all__ = [
    "BaseLLMProvider",
    "ProviderResponse",
]

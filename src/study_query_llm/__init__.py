"""
Study Query LLM - Core Package

A framework for running LLM inference experiments across multiple providers
and analyzing results.

This package provides:
- Provider abstraction layer for LLM APIs
- Service layer for business logic
- Database layer for persistence and analytics
"""

__version__ = "0.1.0"

from .providers import BaseLLMProvider, ProviderResponse

__all__ = [
    "BaseLLMProvider",
    "ProviderResponse",
]

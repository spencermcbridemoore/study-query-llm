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

# Import from subpackages to make them discoverable
from .providers import BaseLLMProvider, ProviderResponse

# Explicitly import subpackages to ensure they're discoverable
# This makes study_query_llm.db, study_query_llm.services, etc. available for autocomplete
from . import db
from . import services
from . import utils

__all__ = [
    "BaseLLMProvider",
    "ProviderResponse",
    "db",
    "services",
    "utils",
]

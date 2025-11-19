"""
Business Logic Layer (Services)

This package contains services that orchestrate business logic for LLM operations.
Services sit between the provider layer (low-level API calls) and the GUI/database layers.

Services handle:
- Retry logic with exponential backoff
- Prompt preprocessing and validation
- Response post-processing
- Multi-turn conversations
- Request batching and deduplication
- Integration with database repository (logging)

All services are provider-agnostic and work with any BaseLLMProvider implementation.
"""

from .inference_service import InferenceService
from .preprocessors import PromptPreprocessor
from .study_service import StudyService

__all__ = [
    "InferenceService",
    "PromptPreprocessor",
    "StudyService",
]

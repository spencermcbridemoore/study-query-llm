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
from .model_registry import ModelRegistry
from .preprocessors import PromptPreprocessor
from .study_service import StudyService
from .embedding_service import EmbeddingService, EmbeddingRequest, EmbeddingResponse
from .provenance_service import ProvenanceService

__all__ = [
    "InferenceService",
    "ModelRegistry",
    "PromptPreprocessor",
    "StudyService",
    "EmbeddingService",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ProvenanceService",
]

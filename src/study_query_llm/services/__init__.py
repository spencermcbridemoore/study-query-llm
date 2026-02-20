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
from .embedding_file_cache import (
    load_embedding_cache,
    save_embedding_cache,
    get_embeddings_with_file_cache,
    get_cache_path,
)
from .provenance_service import ProvenanceService
from .summarization_service import (
    SummarizationService,
    SummarizationRequest,
    SummarizationResponse,
)
from .artifact_service import ArtifactService

__all__ = [
    "InferenceService",
    "ModelRegistry",
    "PromptPreprocessor",
    "StudyService",
    "EmbeddingService",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "load_embedding_cache",
    "save_embedding_cache",
    "get_embeddings_with_file_cache",
    "get_cache_path",
    "ProvenanceService",
    "SummarizationService",
    "SummarizationRequest",
    "SummarizationResponse",
    "ArtifactService",
]

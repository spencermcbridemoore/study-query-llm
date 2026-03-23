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
from .embeddings import (
    EmbeddingService,
    EmbeddingRequest,
    EmbeddingResponse,
    fetch_embeddings_async,
    get_cache_path,
    get_embeddings_with_file_cache,
    load_embedding_cache,
    save_embedding_cache,
)
from .provenance_service import ProvenanceService
from .summarization_service import (
    SummarizationService,
    SummarizationRequest,
    SummarizationResponse,
)
from .artifact_service import ArtifactService
from .data_quality_service import DataQualityService
from .sweep_request_service import SweepRequestService

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
    "fetch_embeddings_async",
    "ProvenanceService",
    "SummarizationService",
    "SummarizationRequest",
    "SummarizationResponse",
    "ArtifactService",
    "DataQualityService",
    "SweepRequestService",
]

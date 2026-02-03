"""
Embedding Service - Core business logic for embedding operations.

This service handles embedding generation with deterministic caching, deployment
validation, retry/backoff, and persistence to v2 DB tables.

Usage:
    from study_query_llm.services.embedding_service import EmbeddingService
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository

    db = DatabaseConnectionV2("postgresql://...")
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo)

        # Get single embedding
        result = await service.get_embedding(
            text="Hello world",
            deployment="text-embedding-ada-002"
        )
        print(result.vector)
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    RetryError,
)
from openai import AsyncAzureOpenAI
from openai.types.embedding import Embedding
from openai import InternalServerError, APIConnectionError, RateLimitError

from ..config import Config, ProviderConfig
from ..providers.factory import ProviderFactory
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


@dataclass
class EmbeddingRequest:
    """Request parameters for embedding generation."""

    text: str
    deployment: str
    provider: str = "azure"
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    group_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    vector: List[float]
    model: str
    dimension: int
    request_hash: str
    cached: bool
    raw_call_id: Optional[int] = None
    latency_ms: Optional[float] = None


class EmbeddingService:
    """
    Service for generating embeddings with deterministic caching and persistence.

    Features:
    - Deterministic caching based on request hash
    - Deployment validation with cached results
    - Retry/backoff for transient errors
    - DB persistence to RawCall and EmbeddingVector
    - Failure logging with error details
    """

    def __init__(
        self,
        repository: Optional["RawCallRepository"] = None,
        max_retries: int = 6,
        initial_wait: float = 1.0,
        max_wait: float = 30.0,
    ):
        """
        Initialize the embedding service.

        Args:
            repository: Optional RawCallRepository for DB persistence
            max_retries: Maximum number of retry attempts (default: 6)
            initial_wait: Initial wait time in seconds for exponential backoff (default: 1.0)
            max_wait: Maximum wait time in seconds between retries (default: 30.0)
        """
        self.repository = repository
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait

        # Cache for deployment validation results
        self._deployment_cache: Dict[str, bool] = {}

        # Cache for Azure clients per deployment (to avoid recreating)
        self._client_cache: Dict[str, AsyncAzureOpenAI] = {}

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent hashing.

        Removes null bytes and normalizes whitespace.
        """
        # Remove null bytes
        text = text.replace("\x00", "")
        # Normalize whitespace (collapse multiple spaces/tabs/newlines)
        import re

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _compute_request_hash(
        self,
        text: str,
        model: str,
        dimensions: Optional[int],
        encoding_format: str,
        provider: str,
    ) -> str:
        """
        Compute deterministic hash for cache lookup.

        Args:
            text: Input text (will be normalized)
            model: Model/deployment name
            dimensions: Optional dimension override
            encoding_format: Encoding format (e.g., "float")
            provider: Provider name

        Returns:
            SHA256 hash as hex string
        """
        normalized_text = self._normalize_text(text)

        # Build hash components
        components = [
            provider,
            model,
            normalized_text,
            str(dimensions) if dimensions else "",
            encoding_format,
        ]

        # Compute hash
        hash_input = "|".join(components)
        return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    def _check_cache(
        self, request_hash: str, deployment: str
    ) -> Optional[EmbeddingResponse]:
        """
        Check database cache for existing embedding.

        Args:
            request_hash: Request hash to look up
            deployment: Deployment name for filtering

        Returns:
            EmbeddingResponse if found in cache, None otherwise
        """
        if not self.repository:
            return None

        try:
            from ..db.models_v2 import EmbeddingVector, RawCall
            from sqlalchemy import func

            # Query for embedding with matching hash in metadata
            # Note: JSON filtering is database-specific, so we query and filter in Python
            session = self.repository.session

            # Get all successful embedding calls for this deployment
            raw_calls = (
                session.query(RawCall)
                .filter(
                    RawCall.modality == "embedding",
                    RawCall.model == deployment,
                    RawCall.status == "success",
                )
                .order_by(RawCall.created_at.desc())
                .limit(1000)  # Limit to recent calls for performance
                .all()
            )

            # Check metadata for matching hash
            for raw_call in raw_calls:
                if (
                    raw_call.metadata_json
                    and isinstance(raw_call.metadata_json, dict)
                    and raw_call.metadata_json.get("request_hash") == request_hash
                ):
                    # Found matching embedding
                    embedding_vector = (
                        session.query(EmbeddingVector)
                        .filter(EmbeddingVector.call_id == raw_call.id)
                        .first()
                    )

                    if embedding_vector:
                        return EmbeddingResponse(
                            vector=embedding_vector.vector,
                            model=raw_call.model or deployment,
                            dimension=embedding_vector.dimension,
                            request_hash=request_hash,
                            cached=True,
                            raw_call_id=raw_call.id,
                            latency_ms=raw_call.latency_ms,
                        )

        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)}", exc_info=True)

        return None

    async def _validate_deployment(
        self, deployment: str, provider: str = "azure"
    ) -> bool:
        """
        Validate that a deployment exists and supports embeddings.

        Uses cached results to avoid repeated validation.

        Args:
            deployment: Deployment name to validate
            provider: Provider name (default: "azure")

        Returns:
            True if deployment is valid, False otherwise
        """
        cache_key = f"{provider}:{deployment}"

        # Check cache first
        if cache_key in self._deployment_cache:
            return self._deployment_cache[cache_key]

        # Validate deployment
        try:
            # Create fresh Config to pick up environment changes
            fresh_config = Config()
            provider_config = fresh_config.get_provider_config(provider)

            # Temporarily override deployment
            original_deployment = None
            import os

            if provider == "azure":
                original_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                os.environ["AZURE_OPENAI_DEPLOYMENT"] = deployment

            try:
                # Create client and test with minimal embedding call
                client = AsyncAzureOpenAI(
                    api_key=provider_config.api_key,
                    api_version=provider_config.api_version,
                    azure_endpoint=provider_config.endpoint,
                )

                # Try a minimal embedding call (without dimensions to let model use default)
                # Some models don't support dimensions parameter, and some only accept specific values
                await client.embeddings.create(
                    model=deployment, input=["test"]
                )

                # Cache success
                self._deployment_cache[cache_key] = True
                return True

            finally:
                await client.close()
                if original_deployment:
                    os.environ["AZURE_OPENAI_DEPLOYMENT"] = original_deployment
                elif provider == "azure" and "AZURE_OPENAI_DEPLOYMENT" in os.environ:
                    del os.environ["AZURE_OPENAI_DEPLOYMENT"]

        except Exception as e:
            logger.warning(
                f"Deployment validation failed: {deployment}, error: {str(e)}"
            )
            # Cache failure
            self._deployment_cache[cache_key] = False
            return False

    async def _create_embedding_with_retry(
        self,
        text: str,
        deployment: str,
        provider: str = "azure",
        dimensions: Optional[int] = None,
    ) -> Embedding:
        """
        Create embedding with retry logic.

        Args:
            text: Input text
            deployment: Deployment name
            provider: Provider name
            dimensions: Optional dimension override

        Returns:
            Embedding object from OpenAI SDK

        Raises:
            Exception: If all retries are exhausted or non-retryable error occurs
        """
        # Get or create client
        client_key = f"{provider}:{deployment}"
        if client_key not in self._client_cache:
            # Create fresh Config to pick up environment changes
            fresh_config = Config()
            provider_config = fresh_config.get_provider_config(provider)

            # For Azure, we need to use the deployment name directly in the API call
            # The client doesn't need the deployment in env var for embeddings
            client = AsyncAzureOpenAI(
                api_key=provider_config.api_key,
                api_version=provider_config.api_version,
                azure_endpoint=provider_config.endpoint,
            )
            self._client_cache[client_key] = client

        client = self._client_cache[client_key]

        # Define retry strategy
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.initial_wait, min=self.initial_wait, max=self.max_wait
            ),
            retry=retry_if_exception_type(
                (InternalServerError, APIConnectionError, RateLimitError)
            )
            | retry_if_exception(self._should_retry_exception),
            reraise=True,
        )

        @retry_decorator
        async def _make_call():
            params = {"model": deployment, "input": [text]}
            if dimensions:
                params["dimensions"] = dimensions

            response = await client.embeddings.create(**params)
            return response.data[0]

        return await _make_call()

    def _should_retry_exception(self, exc: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exc: Exception to check

        Returns:
            True if should retry, False otherwise
        """
        error_str = str(exc).lower()

        # Retry on transient errors
        retryable_patterns = [
            "rate limit",
            "429",
            "502",
            "503",
            "504",
            "timeout",
            "connection",
            "internal server",
            "service unavailable",
        ]

        return any(pattern in error_str for pattern in retryable_patterns)

    def _log_failure(
        self,
        request: EmbeddingRequest,
        request_hash: str,
        error: Exception,
        latency_ms: Optional[float] = None,
    ) -> Optional[int]:
        """
        Log failed embedding request to database.

        Args:
            request: Original embedding request
            request_hash: Request hash
            error: Exception that occurred
            latency_ms: Optional latency if call was attempted

        Returns:
            RawCall ID if logged, None otherwise
        """
        if not self.repository:
            return None

        try:
            error_json = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            metadata_json = {
                "request_hash": request_hash,
            }
            if request.group_id:
                metadata_json["group_id"] = request.group_id
            metadata_json.update(request.metadata)

            call_id = self.repository.insert_raw_call(
                provider=f"{request.provider}_openai_{request.deployment}",
                request_json={"input": request.text, "model": request.deployment},
                model=request.deployment,
                modality="embedding",
                status="failed",
                response_json=None,
                error_json=error_json,
                latency_ms=latency_ms,
                tokens_json=None,
                metadata_json=metadata_json,
            )

            logger.warning(
                f"Logged failed embedding: id={call_id}, deployment={request.deployment}, "
                f"error={str(error)}"
            )

            return call_id

        except Exception as db_error:
            logger.error(
                f"Failed to log embedding failure: {str(db_error)}", exc_info=True
            )
            return None

    async def get_embedding(
        self, request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """
        Get a single embedding with caching.

        Args:
            request: EmbeddingRequest with text, deployment, etc.

        Returns:
            EmbeddingResponse with vector and metadata

        Raises:
            Exception: If embedding generation fails after retries
        """
        # Compute request hash
        request_hash = self._compute_request_hash(
            request.text,
            request.deployment,
            request.dimensions,
            request.encoding_format,
            request.provider,
        )

        # Check cache
        cached = self._check_cache(request_hash, request.deployment)
        if cached:
            logger.debug(
                f"Cache hit for embedding: deployment={request.deployment}, "
                f"hash={request_hash[:16]}..."
            )
            return cached

        # Validate deployment
        is_valid = await self._validate_deployment(
            request.deployment, request.provider
        )
        if not is_valid:
            error = ValueError(
                f"Invalid deployment: {request.deployment} (does not exist or does not support embeddings)"
            )
            self._log_failure(request, request_hash, error)
            raise error

        # Generate embedding with retry
        start_time = time.time()
        try:
            embedding_obj = await self._create_embedding_with_retry(
                request.text,
                request.deployment,
                request.provider,
                request.dimensions,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract vector
            vector = embedding_obj.embedding
            dimension = len(vector)

            # Persist to database
            raw_call_id = None
            if self.repository:
                try:
                    # Build request_json
                    request_json = {"input": request.text, "model": request.deployment}
                    if request.dimensions:
                        request_json["dimensions"] = request.dimensions

                    # Build response_json
                    response_json = {
                        "model": request.deployment,
                        "embedding_dim": dimension,
                    }

                    # Build metadata_json
                    metadata_json = {
                        "request_hash": request_hash,
                    }
                    if request.group_id:
                        metadata_json["group_id"] = request.group_id
                    metadata_json.update(request.metadata)

                    # Insert RawCall
                    raw_call_id = self.repository.insert_raw_call(
                        provider=f"{request.provider}_openai_{request.deployment}",
                        request_json=request_json,
                        model=request.deployment,
                        modality="embedding",
                        status="success",
                        response_json=response_json,
                        latency_ms=latency_ms,
                        tokens_json=None,  # Embeddings don't always have token usage
                        metadata_json=metadata_json,
                    )

                    # Insert EmbeddingVector
                    from ..db.models_v2 import EmbeddingVector
                    import numpy as np

                    vector_norm = float(np.linalg.norm(vector))

                    embedding_vector = EmbeddingVector(
                        call_id=raw_call_id,
                        vector=vector,
                        dimension=dimension,
                        norm=vector_norm,
                        metadata_json={"model": request.deployment},
                    )

                    self.repository.session.add(embedding_vector)
                    self.repository.session.flush()

                    logger.info(
                        f"Stored embedding: id={raw_call_id}, deployment={request.deployment}, "
                        f"dimension={dimension}"
                    )

                except Exception as db_error:
                    logger.error(
                        f"Failed to persist embedding: {str(db_error)}", exc_info=True
                    )
                    # Continue even if DB save fails

            return EmbeddingResponse(
                vector=vector,
                model=request.deployment,
                dimension=dimension,
                request_hash=request_hash,
                cached=False,
                raw_call_id=raw_call_id,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._log_failure(request, request_hash, e, latency_ms)
            raise

    async def get_embeddings_batch(
        self, requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResponse]:
        """
        Get embeddings for multiple texts with per-item caching.

        Args:
            requests: List of EmbeddingRequest objects

        Returns:
            List of EmbeddingResponse objects (one per request)

        Note:
            Each request is processed independently with its own cache lookup.
            Failed requests will raise exceptions.
        """
        import asyncio

        # Process all requests concurrently
        tasks = [self.get_embedding(req) for req in requests]
        return await asyncio.gather(*tasks)

    async def filter_valid_deployments(
        self, deployments: List[str], provider: str = "azure"
    ) -> List[str]:
        """
        Pre-validate a list of deployments and return only valid ones.

        Args:
            deployments: List of deployment names to validate
            provider: Provider name (default: "azure")

        Returns:
            List of valid deployment names
        """
        valid = []
        for deployment in deployments:
            if await self._validate_deployment(deployment, provider):
                valid.append(deployment)
            else:
                logger.warning(f"Skipping invalid deployment: {deployment}")

        return valid

    async def close(self):
        """Close all cached clients."""
        for client in self._client_cache.values():
            await client.close()
        self._client_cache.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes clients."""
        await self.close()

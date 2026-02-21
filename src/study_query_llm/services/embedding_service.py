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
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple
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


# Known maximum token limits for embedding deployments
# These are the maximum input tokens supported by each model
DEPLOYMENT_MAX_TOKENS: Dict[str, int] = {
    # OpenAI/Azure OpenAI embedding models
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    # Add more as needed
}


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Uses tiktoken if available (most accurate), otherwise falls back to approximation.
    
    Args:
        text: Input text to estimate tokens for
        model: Optional model name for tiktoken encoding (default: "cl100k_base" for embedding models)
        
    Returns:
        Estimated number of tokens
    """
    # Try to use tiktoken if available (most accurate)
    try:
        import tiktoken
        
        # Use appropriate encoding for embedding models
        if model and model.startswith("text-embedding-3"):
            encoding_name = "cl100k_base"  # Used by text-embedding-3 models
        elif model and "ada-002" in model:
            encoding_name = "cl100k_base"  # Also used by ada-002
        else:
            encoding_name = "cl100k_base"  # Default for most OpenAI models
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to default encoding
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
            
    except ImportError:
        # Fallback: rough approximation (1 token ≈ 4 characters for English text)
        # This is less accurate but works without dependencies
        return len(text) // 4


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
        require_db_persistence: bool = True,
        max_retries: int = 6,
        initial_wait: float = 1.0,
        max_wait: float = 30.0,
    ):
        """
        Initialize the embedding service.

        Args:
            repository: Optional RawCallRepository for DB persistence
            require_db_persistence: If True (default), raise exception if DB save fails when repository is provided.
                                   If False, log warning and continue (graceful degradation).
                                   Ignored if repository is None.
            max_retries: Maximum number of retry attempts (default: 6)
            initial_wait: Initial wait time in seconds for exponential backoff (default: 1.0)
            max_wait: Maximum wait time in seconds between retries (default: 30.0)
        """
        self.repository = repository
        self.require_db_persistence = require_db_persistence
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait

        # Cache for deployment validation results
        self._deployment_cache: Dict[str, bool] = {}

        # Cache for Azure clients per deployment (to avoid recreating)
        self._client_cache: Dict[str, AsyncAzureOpenAI] = {}
        
        # Cache for deployment limits (max tokens)
        self._deployment_limits_cache: Dict[str, Optional[int]] = {}

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

    def get_deployment_max_tokens(
        self, deployment: str, provider: str = "azure"
    ) -> Optional[int]:
        """
        Get the maximum input tokens supported by a deployment.
        
        Checks lookup table first, then attempts to query from API if available.
        Results are cached to avoid repeated lookups.
        
        Args:
            deployment: Deployment name
            provider: Provider name (default: "azure")
            
        Returns:
            Maximum tokens if known, None otherwise
        """
        cache_key = f"{provider}:{deployment}"
        
        # Check cache first
        if cache_key in self._deployment_limits_cache:
            return self._deployment_limits_cache[cache_key]
        
        # Check lookup table
        if deployment in DEPLOYMENT_MAX_TOKENS:
            limit = DEPLOYMENT_MAX_TOKENS[deployment]
            self._deployment_limits_cache[cache_key] = limit
            return limit
        
        # Try to infer from deployment name patterns
        # text-embedding-3-* and text-embedding-ada-002 typically have 8191 token limit
        if "text-embedding-3" in deployment or "text-embedding-ada-002" in deployment:
            limit = 8191
            self._deployment_limits_cache[cache_key] = limit
            logger.info(f"Inferred max tokens for {deployment}: {limit}")
            return limit
        
        # Unknown deployment - cache None to avoid repeated lookups
        self._deployment_limits_cache[cache_key] = None
        logger.warning(
            f"Unknown max tokens for deployment: {deployment}. "
            f"Consider adding to DEPLOYMENT_MAX_TOKENS lookup table."
        )
        return None
    
    def validate_text_length(
        self, text: str, deployment: str, provider: str = "azure"
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        """
        Validate that text length is within deployment limits.
        
        Args:
            text: Input text to validate
            deployment: Deployment name
            provider: Provider name (default: "azure")
            
        Returns:
            Tuple of (is_valid, estimated_tokens, max_tokens)
            - is_valid: True if text is within limits
            - estimated_tokens: Estimated token count (None if estimation failed)
            - max_tokens: Maximum tokens for deployment (None if unknown)
        """
        max_tokens = self.get_deployment_max_tokens(deployment, provider)
        
        if max_tokens is None:
            # Unknown limit - can't validate, but don't block
            logger.debug(f"Unknown max tokens for {deployment}, skipping validation")
            return True, None, None
        
        # Estimate tokens
        try:
            estimated_tokens = estimate_tokens(text, deployment)
        except Exception as e:
            logger.warning(f"Failed to estimate tokens: {e}")
            estimated_tokens = None
        
        if estimated_tokens is None:
            # Estimation failed - can't validate, but don't block
            return True, None, max_tokens
        
        is_valid = estimated_tokens <= max_tokens
        return is_valid, estimated_tokens, max_tokens

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

    async def _create_embedding_batch_with_retry(
        self,
        texts: List[str],
        deployment: str,
        provider: str = "azure",
        dimensions: Optional[int] = None,
    ) -> List[Embedding]:
        """
        Create embeddings for multiple texts in a single API call with retry logic.

        Sends all texts to the API as one request (input: [text1, text2, ...]) and
        returns one Embedding object per text in the same order.  A single API call
        per chunk dramatically reduces RPM and avoids thundering-herd 429s.

        Args:
            texts: List of input texts.
            deployment: Deployment name.
            provider: Provider name (default: 'azure').
            dimensions: Optional dimension override.

        Returns:
            List of Embedding objects in the same order as ``texts``.

        Raises:
            Exception: If all retries are exhausted or a non-retryable error occurs.
        """
        client_key = f"{provider}:{deployment}"
        if client_key not in self._client_cache:
            fresh_config = Config()
            provider_config = fresh_config.get_provider_config(provider)
            client = AsyncAzureOpenAI(
                api_key=provider_config.api_key,
                api_version=provider_config.api_version,
                azure_endpoint=provider_config.endpoint,
            )
            self._client_cache[client_key] = client

        client = self._client_cache[client_key]

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
        async def _make_batch_call():
            params = {"model": deployment, "input": texts}
            if dimensions:
                params["dimensions"] = dimensions
            response = await client.embeddings.create(**params)
            # response.data is sorted by index; return in input order
            sorted_data = sorted(response.data, key=lambda e: e.index)
            return sorted_data

        return await _make_batch_call()

    def _persist_embedding_batch(
        self,
        requests: List["EmbeddingRequest"],
        embeddings: List[Embedding],
        request_hashes: List[str],
        latency_ms: float,
    ) -> List[int]:
        """
        Persist a batch of (request, embedding) pairs to the DB using the same
        schema as single-request persistence: one RawCall + one EmbeddingVector
        per text, with request_hash stored in metadata_json for future lookup.

        Args:
            requests: Original EmbeddingRequest list.
            embeddings: Corresponding Embedding objects from the API.
            request_hashes: Pre-computed hashes, one per request.
            latency_ms: Latency of the whole batch API call.

        Returns:
            List of raw_call_ids in the same order as requests.
        """
        import numpy as np
        from ..db.models_v2 import EmbeddingVector

        if not self.repository:
            return [0] * len(requests)

        raw_call_ids = []
        for req, emb_obj, req_hash in zip(requests, embeddings, request_hashes):
            vector = emb_obj.embedding
            dimension = len(vector)
            vector_norm = float(np.linalg.norm(vector))

            request_json = {"input": req.text, "model": req.deployment}
            if req.dimensions:
                request_json["dimensions"] = req.dimensions

            response_json = {
                "model": req.deployment,
                "embedding_dim": dimension,
            }

            metadata_json = {"request_hash": req_hash}
            if req.group_id:
                metadata_json["group_id"] = req.group_id
            metadata_json.update(req.metadata)

            try:
                raw_call_id = self.repository.insert_raw_call(
                    provider=f"{req.provider}_openai_{req.deployment}",
                    request_json=request_json,
                    model=req.deployment,
                    modality="embedding",
                    status="success",
                    response_json=response_json,
                    latency_ms=latency_ms,
                    tokens_json=None,
                    metadata_json=metadata_json,
                )

                ev = EmbeddingVector(
                    call_id=raw_call_id,
                    vector=vector,
                    dimension=dimension,
                    norm=vector_norm,
                    metadata_json={"model": req.deployment},
                )
                self.repository.session.add(ev)
                self.repository.session.flush()
                raw_call_ids.append(raw_call_id)

            except Exception as e:
                logger.warning("Failed to persist embedding for hash %s: %s", req_hash, e)
                raw_call_ids.append(0)

        return raw_call_ids

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
            ValueError: If text is empty or becomes empty after normalization
            Exception: If embedding generation fails after retries
        """
        # Validate text is not empty after normalization
        normalized_text = self._normalize_text(request.text)
        if not normalized_text:
            error = ValueError(
                "Cannot generate embedding for empty text (after normalization). "
                "Text must contain at least one non-whitespace character."
            )
            # Log failure if repository is available
            if self.repository:
                request_hash = self._compute_request_hash(
                    request.text,
                    request.deployment,
                    request.dimensions,
                    request.encoding_format,
                    request.provider,
                )
                self._log_failure(request, request_hash, error)
            raise error
        
        # Validate text length against deployment limits
        is_valid, estimated_tokens, max_tokens = self.validate_text_length(
            normalized_text, request.deployment, request.provider
        )
        if not is_valid and estimated_tokens is not None and max_tokens is not None:
            error = ValueError(
                f"Text exceeds maximum token limit for deployment '{request.deployment}'. "
                f"Estimated tokens: {estimated_tokens}, Maximum: {max_tokens}. "
                f"Please reduce the text length or split it into smaller chunks."
            )
            # Log failure if repository is available
            if self.repository:
                request_hash = self._compute_request_hash(
                    request.text,
                    request.deployment,
                    request.dimensions,
                    request.encoding_format,
                    request.provider,
                )
                self._log_failure(request, request_hash, error)
            raise error

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
                    if self.require_db_persistence:
                        # Fail-fast: raise the error for experimental data tracking
                        logger.error(
                            f"Failed to persist embedding (require_db_persistence=True): {str(db_error)}",
                            exc_info=True
                        )
                        raise RuntimeError(
                            f"Database persistence failed. This is required for experimental data tracking. "
                            f"Original error: {str(db_error)}"
                        ) from db_error
                    else:
                        # Graceful degradation: log warning and continue
                        logger.warning(
                            f"Failed to persist embedding (require_db_persistence=False): {str(db_error)}",
                            exc_info=True
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
        self,
        requests: List[EmbeddingRequest],
        chunk_size: Optional[int] = None,
    ) -> List[EmbeddingResponse]:
        """
        Get embeddings for multiple texts with per-item caching.

        Args:
            requests: List of EmbeddingRequest objects.
            chunk_size: When provided, use true API batching: process the list in
                sequential chunks of this size, performing a single DB cache lookup
                and a single ``embeddings.create`` call (with multiple inputs) per
                chunk.  Only texts not already in the DB are sent to the API.
                This dramatically reduces RPM and enables resume-without-re-fetching.
                When ``None`` (default), retain the original concurrent single-request
                behaviour.

        Returns:
            List of EmbeddingResponse objects (one per request, in input order).

        Note:
            With ``chunk_size=None`` each request is processed independently with
            its own cache lookup.  With ``chunk_size`` set, cache lookup is batched
            per chunk (scalable to large N), and the API is called once per chunk
            with all uncached texts as a multi-input request.
        """
        import asyncio
        import time

        if chunk_size is None:
            # Original behaviour: concurrent individual requests
            tasks = [self.get_embedding(req) for req in requests]
            return await asyncio.gather(*tasks)

        # --- Chunked batching path ---
        all_responses: List[EmbeddingResponse] = []
        total_chunks = (len(requests) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            chunk = requests[chunk_idx * chunk_size: (chunk_idx + 1) * chunk_size]

            # 1. Compute request hashes for this chunk
            hashes = [
                self._compute_request_hash(
                    req.text,
                    req.deployment,
                    req.dimensions,
                    req.encoding_format,
                    req.provider,
                )
                for req in chunk
            ]

            # 2. Batch cache lookup
            cached_map: Dict[str, tuple] = {}
            if self.repository:
                try:
                    from ..db.raw_call_repository import RawCallRepository
                    cached_map = self.repository.get_embedding_vectors_by_request_hashes(
                        chunk[0].deployment, hashes
                    )
                except Exception as e:
                    logger.warning("Batch cache lookup failed: %s", e)

            # 3. Partition chunk into cached vs uncached
            uncached_indices = [
                i for i, h in enumerate(hashes) if h not in cached_map
            ]

            # 4. Fetch uncached texts from API in one batch call
            fetched_map: Dict[str, EmbeddingResponse] = {}
            if uncached_indices:
                uncached_reqs = [chunk[i] for i in uncached_indices]
                uncached_hashes = [hashes[i] for i in uncached_indices]
                uncached_texts = [req.text for req in uncached_reqs]

                start_time = time.time()
                try:
                    emb_objects = await self._create_embedding_batch_with_retry(
                        uncached_texts,
                        uncached_reqs[0].deployment,
                        uncached_reqs[0].provider,
                        uncached_reqs[0].dimensions,
                    )
                    batch_latency_ms = (time.time() - start_time) * 1000

                    # 5. Persist newly fetched embeddings to DB
                    raw_call_ids = self._persist_embedding_batch(
                        uncached_reqs, emb_objects, uncached_hashes, batch_latency_ms
                    )

                    for req, emb_obj, req_hash, raw_call_id in zip(
                        uncached_reqs, emb_objects, uncached_hashes, raw_call_ids
                    ):
                        vector = emb_obj.embedding
                        fetched_map[req_hash] = EmbeddingResponse(
                            vector=vector,
                            model=req.deployment,
                            dimension=len(vector),
                            request_hash=req_hash,
                            cached=False,
                            raw_call_id=raw_call_id,
                            latency_ms=batch_latency_ms,
                        )

                except Exception as e:
                    logger.error(
                        "Batch embedding call failed for chunk %d/%d: %s",
                        chunk_idx + 1,
                        total_chunks,
                        e,
                    )
                    raise

            # 6. Build response list for this chunk in original order
            for req, req_hash in zip(chunk, hashes):
                if req_hash in cached_map:
                    vector, raw_call_id = cached_map[req_hash]
                    all_responses.append(
                        EmbeddingResponse(
                            vector=vector,
                            model=req.deployment,
                            dimension=len(vector),
                            request_hash=req_hash,
                            cached=True,
                            raw_call_id=raw_call_id,
                        )
                    )
                else:
                    all_responses.append(fetched_map[req_hash])

            logger.debug(
                "Chunk %d/%d: %d cached, %d fetched from API",
                chunk_idx + 1,
                total_chunks,
                len(chunk) - len(uncached_indices),
                len(uncached_indices),
            )

        return all_responses

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

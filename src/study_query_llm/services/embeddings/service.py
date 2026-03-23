"""
Embedding Service - Core business logic for embedding operations.

Usage:
    from study_query_llm.services.embeddings import EmbeddingService, EmbeddingRequest
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository

    db = DatabaseConnectionV2("postgresql://...")
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = EmbeddingService(repository=repo)
        result = await service.get_embedding(
            EmbeddingRequest(text="Hello world", deployment="text-embedding-ada-002")
        )
        print(result.vector)
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from openai import APIConnectionError, InternalServerError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...providers.base_embedding import BaseEmbeddingProvider, EmbeddingResult
from ...utils.logging_config import get_logger
from .._shared import should_retry_exception
from .constants import DEFAULT_MAX_TOKENS, DEPLOYMENT_MAX_TOKENS
from .hashing import (
    compute_raw_text_sha256,
    compute_request_hash,
    normalize_embedding_text,
)
from .models import EmbeddingRequest, EmbeddingResponse
from . import persistence
from .tokens import estimate_tokens

if TYPE_CHECKING:
    from ...db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


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
        provider: Optional[BaseEmbeddingProvider] = None,
        l1_cache_size: int = 2000,
        singleflight_lease_seconds: int = 45,
        singleflight_wait_timeout_seconds: float = 90.0,
        singleflight_poll_seconds: float = 0.1,
    ) -> None:
        self.repository = repository
        self.require_db_persistence = require_db_persistence
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.max_wait = max_wait
        self.l1_cache_size = max(1, int(l1_cache_size))
        self.singleflight_lease_seconds = max(1, int(singleflight_lease_seconds))
        self.singleflight_wait_timeout_seconds = max(
            1.0, float(singleflight_wait_timeout_seconds)
        )
        self.singleflight_poll_seconds = max(0.01, float(singleflight_poll_seconds))

        if provider is not None:
            self._provider = provider
            self._owns_provider = False
        else:
            from ...config import Config
            from ...providers.azure_embedding_provider import AzureEmbeddingProvider

            cfg = Config()
            provider_config = cfg.get_provider_config("azure")
            self._provider = AzureEmbeddingProvider(provider_config)
            self._owns_provider = True

        self._deployment_cache: Dict[str, bool] = {}
        self._deployment_limits_cache: Dict[str, Optional[int]] = {}
        self._l1_embedding_cache: OrderedDict[
            str, Tuple[List[float], Optional[int]]
        ] = OrderedDict()
        self._inflight_by_cache_key: Dict[str, asyncio.Task[EmbeddingResponse]] = {}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for validation (delegates to hashing module)."""
        return normalize_embedding_text(text)

    def _compute_request_hash(
        self,
        text: str,
        model: str,
        dimensions: Optional[int],
        encoding_format: str,
        provider: str,
    ) -> str:
        return compute_request_hash(
            text, model, dimensions, encoding_format, provider
        )

    @staticmethod
    def _compute_raw_text_sha256(text: str) -> str:
        return compute_raw_text_sha256(text)

    def _l1_get(self, cache_key: str) -> Optional[Tuple[List[float], Optional[int]]]:
        hit = self._l1_embedding_cache.get(cache_key)
        if hit is None:
            return None
        self._l1_embedding_cache.move_to_end(cache_key)
        return hit

    def _l1_put(self, cache_key: str, vector: List[float], raw_call_id: Optional[int]) -> None:
        self._l1_embedding_cache[cache_key] = (vector, raw_call_id)
        self._l1_embedding_cache.move_to_end(cache_key)
        while len(self._l1_embedding_cache) > self.l1_cache_size:
            self._l1_embedding_cache.popitem(last=False)

    def _check_cache(
        self, request_hash: str, deployment: str
    ) -> Optional[EmbeddingResponse]:
        if not self.repository:
            return None

        try:
            entry = self.repository.get_embedding_cache_entry(request_hash)
            if entry:
                self.repository.touch_embedding_cache_hit(request_hash)
                vector = list(entry.vector or [])
                raw_call_id = int(entry.source_raw_call_id) if entry.source_raw_call_id else None
                self._l1_put(request_hash, vector, raw_call_id)
                return EmbeddingResponse(
                    vector=vector,
                    model=entry.deployment or deployment,
                    dimension=int(entry.dimension),
                    request_hash=request_hash,
                    cached=True,
                    raw_call_id=raw_call_id,
                    latency_ms=None,
                )

            legacy = self.repository.get_embedding_vectors_by_request_hashes(
                deployment, [request_hash]
            )
            if request_hash in legacy:
                vector, raw_call_id = legacy[request_hash]
                self._l1_put(request_hash, list(vector), raw_call_id)
                return EmbeddingResponse(
                    vector=list(vector),
                    model=deployment,
                    dimension=len(vector),
                    request_hash=request_hash,
                    cached=True,
                    raw_call_id=raw_call_id,
                    latency_ms=None,
                )
        except Exception as e:
            logger.warning("Error checking cache: %s", e, exc_info=True)

        return None

    def get_deployment_max_tokens(
        self, deployment: str, provider: str = "azure"
    ) -> Optional[int]:
        cache_key = f"{provider}:{deployment}"

        if cache_key in self._deployment_limits_cache:
            return self._deployment_limits_cache[cache_key]

        if deployment in DEPLOYMENT_MAX_TOKENS:
            limit = DEPLOYMENT_MAX_TOKENS[deployment]
            self._deployment_limits_cache[cache_key] = limit
            return limit

        if "text-embedding-3" in deployment or "text-embedding-ada-002" in deployment:
            limit = DEFAULT_MAX_TOKENS
            self._deployment_limits_cache[cache_key] = limit
            logger.info("Inferred max tokens for %s: %s", deployment, limit)
            return limit

        self._deployment_limits_cache[cache_key] = None
        logger.warning(
            "Unknown max tokens for deployment: %s. Consider adding to DEPLOYMENT_MAX_TOKENS.",
            deployment,
        )
        return None

    def validate_text_length(
        self, text: str, deployment: str, provider: str = "azure"
    ) -> Tuple[bool, Optional[int], Optional[int]]:
        max_tokens = self.get_deployment_max_tokens(deployment, provider)

        if max_tokens is None:
            logger.debug("Unknown max tokens for %s, skipping validation", deployment)
            return True, None, None

        try:
            estimated_tokens = estimate_tokens(text, deployment)
        except Exception as e:
            logger.warning("Failed to estimate tokens: %s", e)
            estimated_tokens = None

        if estimated_tokens is None:
            return True, None, max_tokens

        is_valid = estimated_tokens <= max_tokens
        return is_valid, estimated_tokens, max_tokens

    async def _validate_deployment(
        self, deployment: str, provider: str = "azure"
    ) -> bool:
        cache_key = f"{provider}:{deployment}"

        if cache_key in self._deployment_cache:
            return self._deployment_cache[cache_key]

        is_valid = await self._provider.validate_model(deployment)
        self._deployment_cache[cache_key] = is_valid

        if not is_valid:
            logger.warning("Deployment validation failed: %s", deployment)

        return is_valid

    async def _create_embedding_with_retry(
        self,
        text: str,
        deployment: str,
        provider: str = "azure",
        dimensions: Optional[int] = None,
    ) -> EmbeddingResult:
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.initial_wait, min=self.initial_wait, max=self.max_wait
            ),
            retry=retry_if_exception_type(
                (InternalServerError, APIConnectionError, RateLimitError)
            )
            | retry_if_exception(should_retry_exception),
            reraise=True,
        )

        @retry_decorator
        async def _make_call() -> EmbeddingResult:
            results = await self._provider.create_embeddings(
                [text], deployment, dimensions
            )
            return results[0]

        return await _make_call()

    async def _create_embedding_batch_with_retry(
        self,
        texts: List[str],
        deployment: str,
        provider: str = "azure",
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.initial_wait, min=self.initial_wait, max=self.max_wait
            ),
            retry=retry_if_exception_type(
                (InternalServerError, APIConnectionError, RateLimitError)
            )
            | retry_if_exception(should_retry_exception),
            reraise=True,
        )

        @retry_decorator
        async def _make_batch_call() -> List[EmbeddingResult]:
            return await self._provider.create_embeddings(
                texts, deployment, dimensions
            )

        return await _make_batch_call()

    def _log_failure(
        self,
        request: EmbeddingRequest,
        request_hash: str,
        error: Exception,
        latency_ms: Optional[float] = None,
    ) -> Optional[int]:
        return persistence.log_embedding_failure(
            self.repository, request, request_hash, error, latency_ms
        )

    async def _resolve_deployment(
        self, deployment: str, provider: str, request: EmbeddingRequest, request_hash: str
    ) -> None:
        is_valid = await self._validate_deployment(deployment, provider)
        if not is_valid:
            error = ValueError(
                f"Invalid deployment: {deployment} (does not exist or does not support embeddings)"
            )
            self._log_failure(request, request_hash, error)
            raise error

    def _persist_embedding(
        self,
        request: EmbeddingRequest,
        vector: List[float],
        dimension: int,
        deployment: str,
        latency_ms: float,
        request_hash: str,
    ) -> Optional[int]:
        return persistence.persist_embedding(
            self.repository,
            self.require_db_persistence,
            request,
            vector,
            dimension,
            deployment,
            latency_ms,
            request_hash,
            l1_put=self._l1_put,
        )

    def _persist_embedding_batch(
        self,
        requests: List[EmbeddingRequest],
        embeddings: List[EmbeddingResult],
        request_hashes: List[str],
        latency_ms: float,
    ) -> List[int]:
        return persistence.persist_embedding_batch(
            self.repository,
            requests,
            embeddings,
            request_hashes,
            latency_ms,
            l1_put=self._l1_put,
        )

    async def _wait_for_cache_or_lease(
        self,
        *,
        cache_key: str,
        owner: str,
    ) -> bool:
        if not self.repository:
            return True
        acquired = self.repository.try_acquire_embedding_cache_lease(
            cache_key=cache_key,
            owner=owner,
            lease_seconds=self.singleflight_lease_seconds,
        )
        if acquired:
            return True

        start = time.time()
        while (time.time() - start) < self.singleflight_wait_timeout_seconds:
            cached = self._check_cache(cache_key, "")
            if cached is not None:
                return False
            acquired = self.repository.try_acquire_embedding_cache_lease(
                cache_key=cache_key,
                owner=owner,
                lease_seconds=self.singleflight_lease_seconds,
            )
            if acquired:
                return True
            await asyncio.sleep(self.singleflight_poll_seconds)
        return True

    async def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        normalized_text = normalize_embedding_text(request.text)
        if not normalized_text:
            error = ValueError(
                "Cannot generate embedding for empty text (after normalization). "
                "Text must contain at least one non-whitespace character."
            )
            if self.repository:
                request_hash = compute_request_hash(
                    request.text,
                    request.deployment,
                    request.dimensions,
                    request.encoding_format,
                    request.provider,
                )
                self._log_failure(request, request_hash, error)
            raise error

        is_valid, estimated_tokens, max_tokens = self.validate_text_length(
            normalized_text, request.deployment, request.provider
        )
        if not is_valid and estimated_tokens is not None and max_tokens is not None:
            error = ValueError(
                f"Text exceeds maximum token limit for deployment '{request.deployment}'. "
                f"Estimated tokens: {estimated_tokens}, Maximum: {max_tokens}. "
                f"Please reduce the text length or split it into smaller chunks."
            )
            if self.repository:
                request_hash = compute_request_hash(
                    request.text,
                    request.deployment,
                    request.dimensions,
                    request.encoding_format,
                    request.provider,
                )
                self._log_failure(request, request_hash, error)
            raise error

        request_hash = compute_request_hash(
            request.text,
            request.deployment,
            request.dimensions,
            request.encoding_format,
            request.provider,
        )

        l1_hit = self._l1_get(request_hash)
        if l1_hit is not None:
            vector, raw_call_id = l1_hit
            return EmbeddingResponse(
                vector=list(vector),
                model=request.deployment,
                dimension=len(vector),
                request_hash=request_hash,
                cached=True,
                raw_call_id=raw_call_id,
                latency_ms=None,
            )

        cached = self._check_cache(request_hash, request.deployment)
        if cached:
            logger.debug(
                "Cache hit for embedding: deployment=%s, hash=%s...",
                request.deployment,
                request_hash[:16],
            )
            return cached

        inflight = self._inflight_by_cache_key.get(request_hash)
        if inflight is not None:
            return await inflight

        async def _compute() -> EmbeddingResponse:
            owner = f"pid{os.getpid()}-{id(asyncio.current_task())}"
            should_compute = await self._wait_for_cache_or_lease(
                cache_key=request_hash,
                owner=owner,
            )
            if not should_compute:
                cached_after_wait = self._check_cache(request_hash, request.deployment)
                if cached_after_wait is not None:
                    return cached_after_wait

            await self._resolve_deployment(
                request.deployment, request.provider, request, request_hash
            )

            start_time = time.time()
            try:
                embedding_obj = await self._create_embedding_with_retry(
                    request.text,
                    request.deployment,
                    request.provider,
                    request.dimensions,
                )

                latency_ms = (time.time() - start_time) * 1000
                vector = embedding_obj.vector
                dimension = len(vector)

                raw_call_id = self._persist_embedding(
                    request, vector, dimension, request.deployment, latency_ms, request_hash
                )
                self._l1_put(request_hash, list(vector), raw_call_id)
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
            finally:
                if self.repository:
                    self.repository.release_embedding_cache_lease(
                        cache_key=request_hash, owner=owner
                    )

        task = asyncio.create_task(_compute())
        self._inflight_by_cache_key[request_hash] = task
        try:
            return await task
        finally:
            self._inflight_by_cache_key.pop(request_hash, None)

    def _batch_cache_lookup(
        self, requests: List[EmbeddingRequest], hashes: List[str]
    ) -> Tuple[Dict[str, tuple], List[int]]:
        cached_map: Dict[str, tuple] = {}
        if self.repository:
            try:
                l2 = self.repository.get_embedding_cache_vectors_by_keys(hashes)
                if l2:
                    cached_map.update(l2)
                else:
                    cached_map = self.repository.get_embedding_vectors_by_request_hashes(
                        requests[0].deployment, hashes
                    )
            except Exception as e:
                logger.warning("Batch cache lookup failed: %s", e)

        uncached_indices = [i for i, h in enumerate(hashes) if h not in cached_map]

        return cached_map, uncached_indices

    async def _batch_api_call(
        self,
        uncached_requests: List[EmbeddingRequest],
        uncached_hashes: List[str],
        deployment: str,
        dimensions: Optional[int],
    ) -> Dict[str, EmbeddingResponse]:
        uncached_texts = [req.text for req in uncached_requests]

        start_time = time.time()
        try:
            emb_objects = await self._create_embedding_batch_with_retry(
                uncached_texts,
                uncached_requests[0].deployment,
                uncached_requests[0].provider,
                uncached_requests[0].dimensions,
            )
            batch_latency_ms = (time.time() - start_time) * 1000

            raw_call_ids = self._persist_embedding_batch(
                uncached_requests, emb_objects, uncached_hashes, batch_latency_ms
            )

            fetched_map: Dict[str, EmbeddingResponse] = {}
            for req, emb_obj, req_hash, raw_call_id in zip(
                uncached_requests, emb_objects, uncached_hashes, raw_call_ids
            ):
                vector = emb_obj.vector
                fetched_map[req_hash] = EmbeddingResponse(
                    vector=vector,
                    model=req.deployment,
                    dimension=len(vector),
                    request_hash=req_hash,
                    cached=False,
                    raw_call_id=raw_call_id,
                    latency_ms=batch_latency_ms,
                )

            return fetched_map

        except Exception as e:
            logger.error("Batch embedding call failed: %s", e)
            raise

    async def get_embeddings_batch(
        self,
        requests: List[EmbeddingRequest],
        chunk_size: Optional[int] = None,
    ) -> List[EmbeddingResponse]:
        if chunk_size is None:
            tasks = [self.get_embedding(req) for req in requests]
            return await asyncio.gather(*tasks)

        all_responses: List[EmbeddingResponse] = []
        total_chunks = (len(requests) + chunk_size - 1) // chunk_size

        for chunk_idx in range(total_chunks):
            chunk = requests[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

            hashes = [
                compute_request_hash(
                    req.text,
                    req.deployment,
                    req.dimensions,
                    req.encoding_format,
                    req.provider,
                )
                for req in chunk
            ]

            cached_map, uncached_indices = self._batch_cache_lookup(chunk, hashes)

            fetched_map: Dict[str, EmbeddingResponse] = {}
            if uncached_indices:
                unique_uncached_indices: List[int] = []
                seen_uncached: set[str] = set()
                for i in uncached_indices:
                    h = hashes[i]
                    if h in seen_uncached:
                        continue
                    seen_uncached.add(h)
                    unique_uncached_indices.append(i)

                uncached_reqs = [chunk[i] for i in unique_uncached_indices]
                uncached_hashes = [hashes[i] for i in unique_uncached_indices]

                try:
                    fetched_map = await self._batch_api_call(
                        uncached_reqs,
                        uncached_hashes,
                        uncached_reqs[0].deployment,
                        uncached_reqs[0].dimensions,
                    )
                except Exception as e:
                    logger.error(
                        "Batch embedding call failed for chunk %d/%d: %s",
                        chunk_idx + 1,
                        total_chunks,
                        e,
                    )
                    raise

            for req, req_hash in zip(chunk, hashes):
                if req_hash in cached_map:
                    vector, raw_call_id = cached_map[req_hash]
                    self._l1_put(req_hash, list(vector), raw_call_id)
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
        valid = []
        for deployment in deployments:
            if await self._validate_deployment(deployment, provider):
                valid.append(deployment)
            else:
                logger.warning("Skipping invalid deployment: %s", deployment)

        return valid

    async def close(self) -> None:
        if self._owns_provider:
            await self._provider.close()

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

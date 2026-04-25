"""Embedding helpers for batch embedding retrieval with provider wiring."""

import asyncio
from typing import List, Optional

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.utils.logging_config import get_logger

from .constants import CACHE_KEY_VERSION
from .models import EmbeddingRequest
from .service import EmbeddingService

logger = get_logger(__name__)


async def fetch_embeddings_async(
    texts_list: List[str],
    deployment: str,
    db: DatabaseConnectionV2,
    timeout: float = 600.0,
    chunk_size: Optional[int] = None,
    chunk_worker_concurrency: int = 1,
    chunk_circuit_breaker_enabled: bool = False,
    chunk_failure_fallback_threshold: int = 2,
    provider_name: str = "azure",
    max_retries: int = 6,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    singleflight_lease_seconds: int = 45,
    singleflight_wait_timeout_seconds: float = 90.0,
    singleflight_poll_seconds: float = 0.1,
    l3_cache_key: Optional[str] = None,
    l3_entry_max: Optional[int] = None,
    l3_snapshot_group_id: Optional[int] = None,
    l3_run_group_id: Optional[int] = None,
) -> np.ndarray:
    """Fetch embeddings asynchronously with optional API batching."""

    async def _fetch() -> np.ndarray:
        factory = ProviderFactory()
        embedding_provider = factory.create_embedding_provider(provider_name)
        service_kwargs = {
            "max_retries": int(max_retries),
            "initial_wait": float(initial_wait),
            "max_wait": float(max_wait),
            "singleflight_lease_seconds": int(singleflight_lease_seconds),
            "singleflight_wait_timeout_seconds": float(
                singleflight_wait_timeout_seconds
            ),
            "singleflight_poll_seconds": float(singleflight_poll_seconds),
        }

        def _load_l3_hit() -> Optional[np.ndarray]:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                artifact_service = ArtifactService(repository=repo)
                l3_hit = artifact_service.find_embedding_matrix_artifact(
                    dataset_key=l3_key,
                    embedding_engine=deployment,
                    provider=provider_name,
                    entry_max=l3_size,
                    key_version=CACHE_KEY_VERSION,
                )
                if not l3_hit:
                    return None
                return np.asarray(
                    artifact_service.load_artifact(l3_hit["uri"], "embedding_matrix"),
                    dtype=np.float64,
                )

        def _store_l3_matrix(matrix: np.ndarray) -> None:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                artifact_service = ArtifactService(repository=repo)
                provenance = ProvenanceService(repo)
                embedding_batch_group_id = provenance.create_embedding_batch_group(
                    deployment=deployment,
                    metadata={
                        "dataset_key": l3_key,
                        "provider": provider_name,
                        "entry_max": l3_size,
                        "key_version": CACHE_KEY_VERSION,
                    },
                )
                artifact_service.store_embedding_matrix(
                    embedding_batch_group_id,
                    matrix,
                    dataset_key=l3_key,
                    embedding_engine=deployment,
                    provider=provider_name,
                    entry_max=l3_size,
                    key_version=CACHE_KEY_VERSION,
                )
                if l3_snapshot_group_id is not None:
                    try:
                        provenance.link_embedding_batch_to_dataset_snapshot(
                            embedding_batch_group_id, int(l3_snapshot_group_id)
                        )
                    except Exception:
                        pass
                if l3_run_group_id is not None:
                    try:
                        provenance.link_run_to_embedding_batch(
                            int(l3_run_group_id), embedding_batch_group_id
                        )
                    except Exception:
                        pass

        async with embedding_provider:
            l3_key = l3_cache_key or f"default:{deployment}:{len(texts_list)}"
            l3_size = int(l3_entry_max if l3_entry_max is not None else len(texts_list))

            cached = _load_l3_hit()
            if cached is not None:
                return cached

            # For chunked mode, run one DB transaction per chunk so progress is
            # committed incrementally and restart/resume can reuse persisted cache rows.
            if chunk_size is not None and int(chunk_size) > 0:
                total = len(texts_list)
                step = int(chunk_size)
                total_chunks = (total + step - 1) // step
                chunk_mats: List[Optional[np.ndarray]] = [None] * total_chunks
                requested_workers = max(1, int(chunk_worker_concurrency))
                worker_count = (
                    requested_workers if provider_name == "openrouter" else 1
                )
                circuit_breaker_enabled = bool(chunk_circuit_breaker_enabled)
                failure_threshold = max(1, int(chunk_failure_fallback_threshold))

                async def _run_chunk(
                    chunk_zero_idx: int,
                    start: int,
                    texts_chunk: List[str],
                ) -> tuple[int, int, int, np.ndarray]:
                    with db.session_scope() as session:
                        repo = RawCallRepository(session)
                        service = EmbeddingService(
                            repository=repo,
                            provider=embedding_provider,
                            **service_kwargs,
                        )
                        requests = [
                            EmbeddingRequest(
                                text=text,
                                deployment=deployment,
                                provider=provider_name,
                            )
                            for text in texts_chunk
                        ]
                        responses = await service.get_embeddings_batch(
                            requests, chunk_size=step
                        )
                    return (
                        chunk_zero_idx,
                        start,
                        len(texts_chunk),
                        np.asarray([resp.vector for resp in responses], dtype=np.float64),
                    )

                chunk_specs = list(enumerate(range(0, total, step), start=0))
                if worker_count <= 1:
                    for chunk_zero_idx, start in chunk_specs:
                        texts_chunk = texts_list[start : start + step]
                        idx, chunk_start, chunk_len, chunk_matrix = await _run_chunk(
                            chunk_zero_idx=chunk_zero_idx,
                            start=start,
                            texts_chunk=texts_chunk,
                        )
                        chunk_mats[idx] = chunk_matrix
                        chunk_idx = idx + 1
                        if (
                            total_chunks <= 20
                            or chunk_idx == total_chunks
                            or chunk_idx % 10 == 0
                        ):
                            logger.info(
                                "Embedding progress: chunk %s/%s (%s/%s rows)",
                                chunk_idx,
                                total_chunks,
                                min(chunk_start + chunk_len, total),
                                total,
                            )
                else:
                    active_worker_limit = worker_count
                    chunk_failures = 0
                    next_spec_idx = 0
                    in_flight: dict[asyncio.Task, tuple[int, int, int]] = {}
                    failed_specs: list[tuple[int, int, int]] = []

                    async def _run_limited(
                        chunk_zero_idx: int,
                        start: int,
                    ) -> tuple[int, int, int, np.ndarray]:
                        texts_chunk = texts_list[start : start + step]
                        return await _run_chunk(
                            chunk_zero_idx=chunk_zero_idx,
                            start=start,
                            texts_chunk=texts_chunk,
                        )

                    while next_spec_idx < len(chunk_specs) or in_flight:
                        while (
                            next_spec_idx < len(chunk_specs)
                            and len(in_flight) < active_worker_limit
                        ):
                            chunk_zero_idx, start = chunk_specs[next_spec_idx]
                            chunk_len = min(step, total - start)
                            task = asyncio.create_task(
                                _run_limited(chunk_zero_idx=chunk_zero_idx, start=start)
                            )
                            in_flight[task] = (chunk_zero_idx, start, chunk_len)
                            next_spec_idx += 1

                        done, _ = await asyncio.wait(
                            in_flight.keys(),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            chunk_zero_idx, chunk_start, chunk_len = in_flight.pop(task)
                            try:
                                idx, row_start, row_len, chunk_matrix = await task
                                chunk_mats[idx] = chunk_matrix
                                chunk_idx = idx + 1
                                if (
                                    total_chunks <= 20
                                    or chunk_idx == total_chunks
                                    or chunk_idx % 10 == 0
                                ):
                                    logger.info(
                                        "Embedding progress: chunk %s/%s (%s/%s rows)",
                                        chunk_idx,
                                        total_chunks,
                                        min(row_start + row_len, total),
                                        total,
                                    )
                            except Exception as exc:
                                # Each exception here is terminal *after* EmbeddingService
                                # retry/backoff handling for that chunk call.
                                chunk_failures += 1
                                failed_specs.append(
                                    (chunk_zero_idx, chunk_start, chunk_len)
                                )
                                logger.error(
                                    "Chunk embedding failed: chunk=%s/%s rows=[%s:%s) "
                                    "model=%s provider=%s error=%s",
                                    chunk_zero_idx + 1,
                                    total_chunks,
                                    chunk_start,
                                    chunk_start + chunk_len,
                                    deployment,
                                    provider_name,
                                    exc,
                                    exc_info=True,
                                )
                                if (
                                    circuit_breaker_enabled
                                    and active_worker_limit > 1
                                    and chunk_failures >= failure_threshold
                                ):
                                    active_worker_limit = 1
                                    logger.warning(
                                        "Chunk circuit breaker triggered for model=%s "
                                        "(provider=%s, failures=%s, threshold=%s). "
                                        "Downgrading remaining undispatched chunks to serial.",
                                        deployment,
                                        provider_name,
                                        chunk_failures,
                                        failure_threshold,
                                    )

                    if failed_specs and circuit_breaker_enabled and active_worker_limit == 1:
                        logger.info(
                            "Retrying %s failed chunks serially after circuit breaker for model=%s",
                            len(failed_specs),
                            deployment,
                        )
                        for chunk_zero_idx, start, chunk_len in failed_specs:
                            texts_chunk = texts_list[start : start + step]
                            try:
                                idx, row_start, row_len, chunk_matrix = await _run_chunk(
                                    chunk_zero_idx=chunk_zero_idx,
                                    start=start,
                                    texts_chunk=texts_chunk,
                                )
                                chunk_mats[idx] = chunk_matrix
                                logger.info(
                                    "Recovered chunk %s/%s (%s/%s rows) in serial fallback",
                                    idx + 1,
                                    total_chunks,
                                    min(row_start + row_len, total),
                                    total,
                                )
                            except Exception as exc:
                                logger.error(
                                    "Serial fallback failed for chunk=%s/%s rows=[%s:%s) "
                                    "model=%s provider=%s error=%s",
                                    chunk_zero_idx + 1,
                                    total_chunks,
                                    start,
                                    start + chunk_len,
                                    deployment,
                                    provider_name,
                                    exc,
                                    exc_info=True,
                                )
                                raise
                    elif failed_specs:
                        first_idx, first_start, first_len = failed_specs[0]
                        raise RuntimeError(
                            "Chunk embedding failed without fallback "
                            f"(chunk={first_idx + 1}/{total_chunks}, "
                            f"rows=[{first_start}:{first_start + first_len}), "
                            f"model={deployment}, provider={provider_name})."
                        )

                if any(mat is None for mat in chunk_mats):
                    raise RuntimeError(
                        "Chunk embedding assembly failed: one or more chunks are missing."
                    )
                matrix = (
                    np.vstack([mat for mat in chunk_mats if mat is not None])
                    if chunk_mats
                    else np.empty((0, 0), dtype=np.float64)
                )
                _store_l3_matrix(matrix)
                return matrix

            # Default (non-chunked) mode keeps previous behavior.
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = EmbeddingService(
                    repository=repo,
                    provider=embedding_provider,
                    **service_kwargs,
                )
                requests = [
                    EmbeddingRequest(
                        text=text,
                        deployment=deployment,
                        provider=provider_name,
                    )
                    for text in texts_list
                ]
                responses = await service.get_embeddings_batch(
                    requests, chunk_size=None
                )
            matrix = np.asarray(
                [resp.vector for resp in responses], dtype=np.float64
            )
            _store_l3_matrix(matrix)
            return matrix

    try:
        return await asyncio.wait_for(_fetch(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Embedding fetch timed out after {timeout}s "
            f"for {len(texts_list)} texts"
        )

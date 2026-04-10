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
    provider_name: str = "azure",
    l3_cache_key: Optional[str] = None,
    l3_entry_max: Optional[int] = None,
    l3_snapshot_group_id: Optional[int] = None,
    l3_run_group_id: Optional[int] = None,
) -> np.ndarray:
    """Fetch embeddings asynchronously with optional API batching."""

    async def _fetch() -> np.ndarray:
        factory = ProviderFactory()
        embedding_provider = factory.create_embedding_provider(provider_name)

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
                chunk_mats: List[np.ndarray] = []
                for chunk_idx, start in enumerate(range(0, total, step), start=1):
                    texts_chunk = texts_list[start : start + step]
                    with db.session_scope() as session:
                        repo = RawCallRepository(session)
                        service = EmbeddingService(
                            repository=repo, provider=embedding_provider
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
                    chunk_mats.append(
                        np.asarray([resp.vector for resp in responses], dtype=np.float64)
                    )
                    if total_chunks <= 20 or chunk_idx == total_chunks or chunk_idx % 10 == 0:
                        logger.info(
                            "Embedding progress: chunk %s/%s (%s/%s rows)",
                            chunk_idx,
                            total_chunks,
                            min(start + len(texts_chunk), total),
                            total,
                        )
                matrix = np.vstack(chunk_mats) if chunk_mats else np.empty((0, 0), dtype=np.float64)
                _store_l3_matrix(matrix)
                return matrix

            # Default (non-chunked) mode keeps previous behavior.
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                service = EmbeddingService(
                    repository=repo, provider=embedding_provider
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

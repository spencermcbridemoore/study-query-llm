"""Embedding helpers for batch embedding retrieval with provider wiring."""

import asyncio
from typing import List, Optional

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import ProvenanceService

from .constants import CACHE_KEY_VERSION
from .models import EmbeddingRequest
from .service import EmbeddingService


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

        async with embedding_provider:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                artifact_service = ArtifactService(repository=repo)
                provenance = ProvenanceService(repo)
                l3_key = l3_cache_key or f"default:{deployment}:{len(texts_list)}"
                l3_size = int(l3_entry_max if l3_entry_max is not None else len(texts_list))

                l3_hit = artifact_service.find_embedding_matrix_artifact(
                    dataset_key=l3_key,
                    embedding_engine=deployment,
                    provider=provider_name,
                    entry_max=l3_size,
                    key_version=CACHE_KEY_VERSION,
                )
                if l3_hit:
                    return np.asarray(
                        artifact_service.load_artifact(l3_hit["uri"], "embedding_matrix"),
                        dtype=np.float64,
                    )

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
                    requests, chunk_size=chunk_size
                )
                matrix = np.asarray(
                    [resp.vector for resp in responses], dtype=np.float64
                )

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

                return matrix

    try:
        return await asyncio.wait_for(_fetch(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Embedding fetch timed out after {timeout}s "
            f"for {len(texts_list)} texts"
        )

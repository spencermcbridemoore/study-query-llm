"""Embedding helpers shared across sweep scripts."""

import asyncio
from typing import List, Optional

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.embedding_service import EmbeddingService, EmbeddingRequest


async def fetch_embeddings_async(
    texts_list: List[str],
    deployment: str,
    db: DatabaseConnectionV2,
    timeout: float = 600.0,
    chunk_size: Optional[int] = None,
    provider_name: str = "azure",
) -> np.ndarray:
    """Fetch embeddings asynchronously with optional API batching.

    Args:
        texts_list: Texts to embed.
        deployment: Embedding deployment / model name.
        db: Database connection (cache lookup and persistence).
        timeout: Wall-clock timeout in seconds (default 600).
        chunk_size: When set, process chunks of this size sequentially via
            a single ``embeddings.create`` call per chunk.  When ``None``
            (default), use concurrent per-text requests.
        provider_name: Embedding provider to use (default ``"azure"``).
            Accepted values: ``"azure"``, ``"openai"``, ``"huggingface"``,
            ``"local"``, ``"ollama"``, or any label supported by
            ``ProviderFactory.create_embedding_provider``.
    """

    async def _fetch() -> np.ndarray:
        factory = ProviderFactory()
        embedding_provider = factory.create_embedding_provider(provider_name)

        async with embedding_provider:
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
                    requests, chunk_size=chunk_size
                )
                return np.asarray(
                    [resp.vector for resp in responses], dtype=np.float64
                )

    try:
        return await asyncio.wait_for(_fetch(), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Embedding fetch timed out after {timeout}s "
            f"for {len(texts_list)} texts"
        )

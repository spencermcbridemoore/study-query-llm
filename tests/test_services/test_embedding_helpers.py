"""Unit tests for embedding helper orchestration paths."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.services.embeddings.helpers import fetch_embeddings_async
from study_query_llm.services.embeddings.models import EmbeddingResponse


def _sqlite_db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "helpers.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


class _FakeProvider:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def close(self) -> None:
        return None

    async def validate_model(self, _model: str) -> bool:
        return True


@pytest.mark.asyncio
async def test_fetch_embeddings_parallel_chunks_preserve_order(tmp_path: Path, monkeypatch):
    """Parallel chunk workers preserve row order in assembled matrix."""
    monkeypatch.chdir(tmp_path)
    db = _sqlite_db(tmp_path)

    class _ParallelService:
        def __init__(self, repository, provider, **kwargs):
            self.repository = repository
            self.provider = provider
            self.kwargs = kwargs

        async def get_embeddings_batch(self, requests, chunk_size=None):
            first_idx = int(requests[0].text.split("-")[1])
            chunk_idx = first_idx // 2
            # Force out-of-order completion by varying sleep by chunk.
            await asyncio.sleep({0: 0.03, 1: 0.01, 2: 0.02, 3: 0.0}.get(chunk_idx, 0.0))
            return [
                EmbeddingResponse(
                    vector=[float(int(req.text.split("-")[1]))],
                    model=req.deployment,
                    dimension=1,
                    request_hash=f"h-{req.text}",
                    cached=False,
                )
                for req in requests
            ]

    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.ProviderFactory.create_embedding_provider",
        lambda _self, _provider_name: _FakeProvider(),
    )
    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.EmbeddingService",
        _ParallelService,
    )

    texts = [f"row-{i}" for i in range(8)]
    matrix = await fetch_embeddings_async(
        texts_list=texts,
        deployment="mock-deployment",
        db=db,
        provider_name="openrouter",
        chunk_size=2,
        chunk_worker_concurrency=4,
        timeout=30.0,
        l3_cache_key="helpers:parallel-order",
        l3_entry_max=len(texts),
    )

    assert matrix.shape == (8, 1)
    assert matrix[:, 0].tolist() == [float(i) for i in range(8)]


@pytest.mark.asyncio
async def test_fetch_embeddings_chunk_circuit_breaker_recovers_failed_chunk(
    tmp_path: Path, monkeypatch
):
    """Chunk circuit breaker retries failed chunks serially and recovers."""
    monkeypatch.chdir(tmp_path)
    db = _sqlite_db(tmp_path)

    call_counts: dict[int, int] = {}

    class _FlakyService:
        def __init__(self, repository, provider, **kwargs):
            self.repository = repository
            self.provider = provider
            self.kwargs = kwargs

        async def get_embeddings_batch(self, requests, chunk_size=None):
            first_idx = int(requests[0].text.split("-")[1])
            chunk_idx = first_idx // 2
            call_counts[chunk_idx] = call_counts.get(chunk_idx, 0) + 1
            if chunk_idx == 1 and call_counts[chunk_idx] == 1:
                raise TimeoutError("simulated chunk failure")
            await asyncio.sleep(0.005)
            return [
                EmbeddingResponse(
                    vector=[float(int(req.text.split("-")[1]))],
                    model=req.deployment,
                    dimension=1,
                    request_hash=f"h-{req.text}",
                    cached=False,
                )
                for req in requests
            ]

    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.ProviderFactory.create_embedding_provider",
        lambda _self, _provider_name: _FakeProvider(),
    )
    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.EmbeddingService",
        _FlakyService,
    )

    texts = [f"row-{i}" for i in range(6)]
    matrix = await fetch_embeddings_async(
        texts_list=texts,
        deployment="mock-deployment",
        db=db,
        provider_name="openrouter",
        chunk_size=2,
        chunk_worker_concurrency=3,
        chunk_circuit_breaker_enabled=True,
        chunk_failure_fallback_threshold=1,
        timeout=30.0,
        l3_cache_key="helpers:circuit-breaker",
        l3_entry_max=len(texts),
    )

    assert matrix.shape == (6, 1)
    assert matrix[:, 0].tolist() == [float(i) for i in range(6)]
    # Chunk index 1 should fail once, then recover in serial fallback.
    assert call_counts.get(1) == 2


@pytest.mark.asyncio
async def test_fetch_embeddings_non_openrouter_ignores_parallel_chunk_workers(
    tmp_path: Path, monkeypatch
):
    """Non-openrouter providers remain serial even if chunk_worker_concurrency>1."""
    monkeypatch.chdir(tmp_path)
    db = _sqlite_db(tmp_path)

    active = {"current": 0, "max": 0}

    class _TrackingService:
        def __init__(self, repository, provider, **kwargs):
            self.repository = repository
            self.provider = provider
            self.kwargs = kwargs

        async def get_embeddings_batch(self, requests, chunk_size=None):
            active["current"] += 1
            active["max"] = max(active["max"], active["current"])
            try:
                await asyncio.sleep(0.01)
                return [
                    EmbeddingResponse(
                        vector=[float(int(req.text.split("-")[1]))],
                        model=req.deployment,
                        dimension=1,
                        request_hash=f"h-{req.text}",
                        cached=False,
                    )
                    for req in requests
                ]
            finally:
                active["current"] -= 1

    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.ProviderFactory.create_embedding_provider",
        lambda _self, _provider_name: _FakeProvider(),
    )
    monkeypatch.setattr(
        "study_query_llm.services.embeddings.helpers.EmbeddingService",
        _TrackingService,
    )

    texts = [f"row-{i}" for i in range(6)]
    matrix = await fetch_embeddings_async(
        texts_list=texts,
        deployment="mock-deployment",
        db=db,
        provider_name="local",
        chunk_size=2,
        chunk_worker_concurrency=4,
        timeout=30.0,
        l3_cache_key="helpers:serial-gate",
        l3_entry_max=len(texts),
    )

    assert matrix.shape == (6, 1)
    assert active["max"] == 1

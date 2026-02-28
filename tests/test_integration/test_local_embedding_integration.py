"""
Tier 2 -- Embedding-only integration tests.

Require a running TEI Docker container with GPU support.
Uses nomic-ai/nomic-embed-text-v1.5 (137M, fastest to load).

Run with:
    pytest tests/test_integration/test_local_embedding_integration.py -m "requires_tei"
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from study_query_llm.providers.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
)

TEI_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_DIM = 768
TEI_PORT = 9900


@pytest.fixture(scope="module")
def tei_endpoint(tei_manager_factory):
    """Use an existing TEI container on the test port, or start a new one."""
    import urllib.request
    url = f"http://localhost:{TEI_PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                yield f"http://localhost:{TEI_PORT}/v1"
                return
    except Exception:
        pass

    mgr = tei_manager_factory(TEI_MODEL, port=TEI_PORT)
    mgr.start()
    yield mgr.endpoint_url
    mgr.stop()


@pytest.mark.requires_tei
@pytest.mark.requires_gpu
@pytest.mark.asyncio
async def test_tei_embedding_roundtrip(tei_endpoint):
    """Embed 3 texts, verify shape and non-zero vectors."""
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=tei_endpoint, provider_label="tei_test"
    )
    async with provider:
        results = await provider.create_embeddings(
            ["What is gravity?", "The sky is blue.", "Hello world."],
            model=TEI_MODEL,
        )

    assert len(results) == 3
    for r in results:
        arr = np.array(r.vector)
        assert arr.shape == (NOMIC_DIM,)
        assert np.linalg.norm(arr) > 0


@pytest.mark.requires_tei
@pytest.mark.requires_gpu
@pytest.mark.asyncio
async def test_tei_embedding_deterministic(tei_endpoint):
    """Same text embedded twice should produce identical vectors."""
    text = "Deterministic embedding test."
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=tei_endpoint, provider_label="tei_test"
    )
    async with provider:
        r1 = await provider.create_embeddings([text], model=TEI_MODEL)
        r2 = await provider.create_embeddings([text], model=TEI_MODEL)

    np.testing.assert_array_almost_equal(r1[0].vector, r2[0].vector, decimal=6)


@pytest.mark.requires_tei
@pytest.mark.requires_gpu
@pytest.mark.asyncio
async def test_tei_embedding_batch(tei_endpoint):
    """Embed 30 texts in one call, verify all return (TEI max_client_batch_size=32)."""
    texts = [f"Sample text number {i} for batch embedding." for i in range(30)]
    provider = OpenAICompatibleEmbeddingProvider(
        base_url=tei_endpoint, provider_label="tei_test"
    )
    async with provider:
        results = await provider.create_embeddings(texts, model=TEI_MODEL)

    assert len(results) == 30
    embeddings = np.array([r.vector for r in results])
    assert embeddings.shape == (30, NOMIC_DIM)


@pytest.mark.requires_tei
@pytest.mark.requires_gpu
@pytest.mark.asyncio
async def test_fetch_embeddings_async_local_provider(tei_endpoint):
    """End-to-end test of fetch_embeddings_async with provider_name='local'."""
    import os

    os.environ["LOCAL_EMBEDDING_ENDPOINT"] = tei_endpoint

    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.services.embedding_helpers import fetch_embeddings_async

    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()

    texts = ["The quick brown fox.", "Jumps over the lazy dog."]
    embeddings = await fetch_embeddings_async(
        texts, TEI_MODEL, db, provider_name="local"
    )

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, NOMIC_DIM)

"""
Tier 4 -- Combined integration tests.

Require BOTH a running TEI Docker container AND Ollama server simultaneously.
This is the real-world smoke test for the full local sweep pipeline.

Uses ``OllamaModelManager`` as a context manager so VRAM is deterministically
reclaimed after the test module finishes.

Run with:
    pytest tests/test_integration/test_combined_sweep_integration.py -m "requires_tei and requires_ollama"
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from study_query_llm.providers.managers import OllamaModelManager
from study_query_llm.providers.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
)
from study_query_llm.providers.openai_compatible_chat_provider import (
    OpenAICompatibleChatProvider,
)

TEI_MODEL = "nomic-ai/nomic-embed-text-v1.5"
CHAT_MODEL = "llama3.1:8b"
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


@pytest.fixture(scope="module")
def ollama_chat_mgr(ollama_available):
    """Module-scoped OllamaModelManager -- model loaded once, unloaded at teardown."""
    with OllamaModelManager(CHAT_MODEL, idle_timeout_seconds=300) as mgr:
        yield mgr


@pytest.mark.requires_tei
@pytest.mark.requires_ollama
@pytest.mark.requires_gpu
@pytest.mark.asyncio
async def test_embedding_and_summarizer_coexist(tei_endpoint, ollama_chat_mgr):
    """TEI and Ollama serving simultaneously -- embed 20 texts, then summarize 3."""
    texts = [f"Sample text number {i} about science and nature." for i in range(20)]

    emb_provider = OpenAICompatibleEmbeddingProvider(
        base_url=tei_endpoint, provider_label="tei_combined"
    )
    async with emb_provider:
        emb_results = await emb_provider.create_embeddings(texts, model=TEI_MODEL)

    assert len(emb_results) == 20
    embeddings = np.array([r.vector for r in emb_results])
    assert embeddings.shape[0] == 20
    assert embeddings.shape[1] > 0

    chat_provider = OpenAICompatibleChatProvider(
        base_url=ollama_chat_mgr.endpoint_url, model=ollama_chat_mgr.model_id,
    )
    summaries = []
    for text in texts[:3]:
        result = await chat_provider.complete(
            f"Summarize in one sentence: {text}",
            temperature=0.1,
            max_tokens=50,
        )
        summaries.append(result.text)
    await chat_provider.close()

    assert len(summaries) == 3
    assert all(len(s.strip()) > 0 for s in summaries)


@pytest.mark.requires_tei
@pytest.mark.requires_ollama
@pytest.mark.requires_gpu
@pytest.mark.slow
@pytest.mark.asyncio
async def test_mini_sweep_local(tei_endpoint, ollama_chat_mgr):
    """Run a small sweep with local embedder + local paraphraser end-to-end."""
    os.environ["LOCAL_EMBEDDING_ENDPOINT"] = tei_endpoint

    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.algorithms.sweep import run_sweep, SweepConfig
    from scripts.common.embedding_utils import fetch_embeddings_async
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()

    texts = [
        "The sun is a star at the center of our solar system.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "Water freezes at zero degrees Celsius under normal pressure.",
        "The mitochondria is the powerhouse of the cell.",
        "Gravity is the force that attracts objects with mass.",
        "DNA contains the genetic instructions for all living organisms.",
        "The speed of light is approximately 300,000 km per second.",
        "Electrons orbit the nucleus of an atom.",
        "Evolution is the process of change in inherited characteristics.",
        "The periodic table organises chemical elements by atomic number.",
        "Tectonic plates move slowly across the Earth's surface.",
        "Neurons transmit electrical signals in the nervous system.",
        "Sound travels faster in water than in air.",
        "The human body contains approximately 37 trillion cells.",
        "Kinetic energy is the energy of motion.",
        "Antibiotics fight bacterial infections but not viruses.",
        "The moon causes tides through gravitational pull.",
        "Carbon dioxide is a greenhouse gas that traps heat.",
        "Magnetic fields are produced by moving electric charges.",
        "Proteins are made of chains of amino acids.",
    ]

    embeddings = await fetch_embeddings_async(
        texts, TEI_MODEL, db, provider_name="local"
    )
    assert embeddings.shape == (20, 768)

    paraphraser = create_paraphraser_for_llm(
        CHAT_MODEL, db, provider="local_llm"
    )

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    async def embedder_func(texts_list):
        return await fetch_embeddings_async(
            texts_list, TEI_MODEL, db, provider_name="local"
        )

    def embedder_sync(texts_list):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(embedder_func(texts_list))
        except RuntimeError:
            pass
        return asyncio.run(embedder_func(texts_list))

    config = SweepConfig(
        skip_pca=True,
        k_min=2,
        k_max=4,
        max_iter=50,
        base_seed=0,
        n_restarts=1,
        compute_stability=False,
        llm_interval=20,
        max_samples=5,
        distance_metric="cosine",
        normalize_vectors=True,
    )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: run_sweep(
                texts,
                embeddings,
                config,
                paraphraser=paraphraser,
                embedder=embedder_sync,
            ),
        )

    assert result is not None
    assert hasattr(result, "by_k") or isinstance(result, dict)
    by_k = result.by_k if hasattr(result, "by_k") else result.get("by_k", result)
    assert len(by_k) > 0

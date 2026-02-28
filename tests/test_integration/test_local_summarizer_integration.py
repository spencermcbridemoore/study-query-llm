"""
Tier 3 -- Summarizer-only integration tests.

Require a running Ollama server at localhost:11434 with llama3.1:8b pulled.

Uses ``OllamaModelManager`` as a context manager so VRAM is deterministically
reclaimed after each test that loads a model.

Run with:
    pytest tests/test_integration/test_local_summarizer_integration.py -m "requires_ollama"
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from study_query_llm.providers.managers import OllamaModelManager
from study_query_llm.providers.openai_compatible_chat_provider import (
    OpenAICompatibleChatProvider,
)
from study_query_llm.providers.base import ProviderResponse
from study_query_llm.services.summarization_service import (
    SummarizationService,
    SummarizationRequest,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository

TEST_MODEL = "llama3.1:8b"


@pytest.fixture(scope="module")
def ollama_mgr(ollama_available):
    """Module-scoped OllamaModelManager -- model loaded once, unloaded at teardown."""
    with OllamaModelManager(TEST_MODEL, idle_timeout_seconds=300) as mgr:
        yield mgr


@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_chat_provider_roundtrip(ollama_mgr):
    """Create a real chat provider, call complete(), verify non-empty reply."""
    provider = OpenAICompatibleChatProvider(
        base_url=ollama_mgr.endpoint_url,
        model=ollama_mgr.model_id,
        provider_label="ollama_test",
    )
    result = await provider.complete(
        "Say hello in exactly three words.", temperature=0.1, max_tokens=20
    )

    assert isinstance(result, ProviderResponse)
    assert len(result.text.strip()) > 0
    assert result.provider == "ollama_test"
    assert result.tokens is not None and result.tokens > 0
    await provider.close()


@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_chat_provider_respects_max_tokens(ollama_mgr):
    """Setting max_tokens=10 should produce a short response."""
    provider = OpenAICompatibleChatProvider(
        base_url=ollama_mgr.endpoint_url, model=ollama_mgr.model_id,
    )
    result = await provider.complete(
        "Write a long essay about the history of mathematics.",
        temperature=0.1,
        max_tokens=10,
    )
    word_count = len(result.text.split())
    assert word_count < 30, f"Expected short response, got {word_count} words"
    await provider.close()


@pytest.mark.requires_ollama
@pytest.mark.asyncio
async def test_summarization_service_local_llm(ollama_mgr):
    """Full SummarizationService.summarize_batch() with provider='local_llm'."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        service = SummarizationService(repository=repo)

        request = SummarizationRequest(
            texts=["The quick brown fox jumps over the lazy dog."],
            llm_deployment=TEST_MODEL,
            provider="local_llm",
            validate_deployment=False,
            temperature=0.1,
            max_tokens=50,
        )

        result = await service.summarize_batch(request)

        assert len(result.summaries) == 1
        assert len(result.summaries[0].strip()) > 0
        assert result.metadata["provider"] == "local_llm"


@pytest.mark.requires_ollama
def test_create_paraphraser_local_llm_sync(ollama_mgr):
    """Test the sync wrapper create_paraphraser_for_llm with local_llm provider.

    This exercises the tricky event-loop threading used inside sweep scripts.
    """
    from scripts.common.sweep_utils import create_paraphraser_for_llm

    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()

    paraphraser = create_paraphraser_for_llm(
        TEST_MODEL, db, provider="local_llm"
    )
    assert paraphraser is not None

    result = paraphraser(["The cat sat on the mat.", "Dogs like to play fetch."])
    assert isinstance(result, str)
    assert len(result.strip()) > 0

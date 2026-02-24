"""Tests for BaseEmbeddingProvider ABC and EmbeddingResult dataclass."""

import pytest
from typing import List, Optional

from study_query_llm.providers.base_embedding import (
    BaseEmbeddingProvider,
    EmbeddingResult,
)


class _ConcreteEmbeddingProvider(BaseEmbeddingProvider):
    """Minimal concrete subclass for testing the ABC."""

    async def create_embeddings(
        self,
        texts: List[str],
        model: str,
        dimensions: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        return [
            EmbeddingResult(vector=[float(i)] * 3, index=i)
            for i in range(len(texts))
        ]

    def get_provider_name(self) -> str:
        return "test"

    async def close(self) -> None:
        pass


def test_embedding_result_construction():
    """EmbeddingResult stores vector and index."""
    result = EmbeddingResult(vector=[0.1, 0.2, 0.3], index=0)
    assert result.vector == [0.1, 0.2, 0.3]
    assert result.index == 0


def test_base_embedding_provider_cannot_be_instantiated():
    """BaseEmbeddingProvider is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseEmbeddingProvider()  # type: ignore[abstract]


@pytest.mark.asyncio
async def test_concrete_subclass_works():
    """A minimal concrete subclass can be instantiated and called."""
    provider = _ConcreteEmbeddingProvider()

    results = await provider.create_embeddings(["a", "b"], "model-x")
    assert len(results) == 2
    assert results[0].index == 0
    assert results[1].index == 1
    assert provider.get_provider_name() == "test"


@pytest.mark.asyncio
async def test_validate_model_defaults_to_true():
    """Default validate_model() returns True for any model."""
    provider = _ConcreteEmbeddingProvider()
    assert await provider.validate_model("anything") is True


@pytest.mark.asyncio
async def test_context_manager_calls_close():
    """async with should call close() on exit."""
    provider = _ConcreteEmbeddingProvider()
    closed = False
    original_close = provider.close

    async def tracking_close():
        nonlocal closed
        closed = True
        await original_close()

    provider.close = tracking_close  # type: ignore[assignment]

    async with provider as p:
        assert p is provider

    assert closed is True

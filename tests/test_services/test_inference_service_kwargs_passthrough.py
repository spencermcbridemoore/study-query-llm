"""Tests for InferenceService kwargs passthrough boundaries."""

from __future__ import annotations

from typing import Any

import pytest

from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.inference_service import InferenceService


class _CapturingProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return ProviderResponse(
            text="ok",
            provider="capturing",
            tokens=3,
            latency_ms=1.0,
            metadata={},
        )

    def get_provider_name(self) -> str:
        return "capturing"


@pytest.mark.asyncio
async def test_run_inference_forwards_provider_kwargs():
    """InferenceService should preserve provider kwargs to complete()."""
    provider = _CapturingProvider()
    service = InferenceService(provider=provider, repository=None)
    out = await service.run_inference(
        prompt="hello",
        temperature=0.1,
        max_tokens=44,
        logprobs=True,
        top_logprobs=4,
    )
    assert out["response"] == "ok"
    assert len(provider.calls) == 1
    kwargs = provider.calls[0]["kwargs"]
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 44
    assert kwargs["logprobs"] is True
    assert kwargs["top_logprobs"] == 4


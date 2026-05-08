"""Tests for stage-1 perturbation-then-inference runner."""

from __future__ import annotations

from typing import Any

import pytest

from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.method_runners.perturb_then_infer_basic import (
    run_perturbation_then_inference_basic,
)
from study_query_llm.services.method_runtime_registry import MethodRunnerContext


class _CapturingProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return ProviderResponse(
            text=f"reply:{prompt}",
            provider="capturing",
            tokens=11,
            latency_ms=1.0,
            metadata={"finish_reason": "stop"},
        )

    def get_provider_name(self) -> str:
        return "capturing"


@pytest.mark.asyncio
async def test_perturb_runner_executes_variants_and_returns_export_role(monkeypatch):
    """Runner should execute each prompt variant with provider-safe controls."""
    provider = _CapturingProvider()

    from study_query_llm.providers.factory import ProviderFactory

    monkeypatch.setattr(
        ProviderFactory,
        "create_chat_provider",
        lambda _self, provider_name, model: provider,
    )
    context = MethodRunnerContext(
        repository=None,  # type: ignore[arg-type]
        request_group_id=7,
        source_group_id=7,
        method_name="perturbation_then_inference.basic",
        method_version="0.1",
        run_key="rk",
    )
    out = await run_perturbation_then_inference_basic(
        {
            "prompt": "base",
            "provider": "local_llm",
            "model": "qwen2.5:14b",
            "variants": ["{prompt} :: v1", "{prompt} :: v2"],
            "include_original": True,
            "temperature": 0.3,
            "max_tokens": 55,
        },
        context,
    )
    assert out.pipeline_stage_role == "export"
    assert out.pipeline_stage_context["variant_count"] == 3
    assert out.output_json["response_count"] == 3
    assert [row["prompt"] for row in provider.calls] == [
        "base",
        "base :: v1",
        "base :: v2",
    ]
    # Stage-1 runner forwards only provider-safe controls.
    for row in provider.calls:
        assert row["kwargs"]["temperature"] == 0.3
        assert row["kwargs"]["max_tokens"] == 55
        assert "logprobs" not in row["kwargs"]


@pytest.mark.asyncio
async def test_perturb_runner_rejects_unknown_parameters(monkeypatch):
    """Boundary validation should reject unsupported runner parameters."""
    provider = _CapturingProvider()
    from study_query_llm.providers.factory import ProviderFactory

    monkeypatch.setattr(
        ProviderFactory,
        "create_chat_provider",
        lambda _self, provider_name, model: provider,
    )
    context = MethodRunnerContext(
        repository=None,  # type: ignore[arg-type]
        request_group_id=9,
        source_group_id=9,
        method_name="perturbation_then_inference.basic",
        method_version="0.1",
        run_key="rk",
    )
    with pytest.raises(ValueError, match="invalid_perturbation_then_inference_parameters"):
        await run_perturbation_then_inference_basic(
            {
                "prompt": "base",
                "provider": "local_llm",
                "model": "qwen2.5:14b",
                "beam_width": 4,
            },
            context,
        )


"""Stage-2 logprobs pressure-test runner."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
)


class LogprobsBasicParams(BaseModel):
    """Boundary-validated parameter payload for stage-2 logprobs runner."""

    prompt: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = 0.0
    max_tokens: int | None = None
    logprobs: bool = True
    top_logprobs: int | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("prompt", "provider", "model")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text


async def run_logprobs_basic(
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Execute one logprobs-oriented inference run."""
    try:
        parsed = LogprobsBasicParams.model_validate(parameters or {})
    except ValidationError as exc:
        raise ValueError(f"invalid_logprobs_basic_parameters: {exc}") from exc

    provider = ProviderFactory().create_chat_provider(parsed.provider, parsed.model)
    service = InferenceService(provider=provider, repository=None)
    try:
        inference_kwargs: Dict[str, Any] = {"logprobs": bool(parsed.logprobs)}
        if parsed.top_logprobs is not None:
            inference_kwargs["top_logprobs"] = int(parsed.top_logprobs)
        result = await service.run_inference(
            prompt=parsed.prompt,
            temperature=parsed.temperature,
            max_tokens=parsed.max_tokens,
            **inference_kwargs,
        )
    finally:
        await service.close()

    return MethodRunnerResult(
        output_json={
            "method_name": context.method_name,
            "method_version": context.method_version,
            "response": str(result.get("response") or ""),
            "metadata": dict(result.get("metadata") or {}),
            "request_controls": inference_kwargs,
        },
        pipeline_stage_role="export",
        pipeline_stage_context={
            "source": "prompt",
            "mode": "logprobs",
        },
        metadata_json={
            "provider": parsed.provider,
            "model": parsed.model,
            "request_group_id": int(context.request_group_id),
        },
    )


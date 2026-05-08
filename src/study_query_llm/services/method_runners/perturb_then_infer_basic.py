"""Stage-1 perturbation-then-inference runner."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
)


class PerturbThenInferParams(BaseModel):
    """Boundary-validated parameter payload for stage-1 perturbation runner."""

    prompt: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    variants: List[str] = Field(default_factory=list)
    include_original: bool = True
    temperature: float = 0.7
    max_tokens: int | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("prompt", "provider", "model")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text

    @field_validator("variants")
    @classmethod
    def _normalize_variants(cls, value: List[str]) -> List[str]:
        out: List[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out


def _build_prompt_variants(params: PerturbThenInferParams) -> List[str]:
    prompts: List[str] = []
    if params.include_original:
        prompts.append(params.prompt)
    for item in params.variants:
        if "{prompt}" in item:
            prompts.append(item.format(prompt=params.prompt))
        else:
            prompts.append(item)
    if not prompts:
        prompts = [params.prompt]
    return prompts


async def run_perturbation_then_inference_basic(
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Execute the stage-1 perturbation/inference method."""
    try:
        parsed = PerturbThenInferParams.model_validate(parameters or {})
    except ValidationError as exc:
        raise ValueError(
            f"invalid_perturbation_then_inference_parameters: {exc}"
        ) from exc

    prompt_variants = _build_prompt_variants(parsed)
    provider = ProviderFactory().create_chat_provider(parsed.provider, parsed.model)
    service = InferenceService(provider=provider, repository=None)
    responses: List[Dict[str, Any]] = []
    try:
        for idx, prompt in enumerate(prompt_variants):
            result = await service.run_inference(
                prompt=prompt,
                temperature=parsed.temperature,
                max_tokens=parsed.max_tokens,
            )
            responses.append(
                {
                    "variant_idx": int(idx),
                    "prompt": prompt,
                    "response": str(result.get("response") or ""),
                    "metadata": dict(result.get("metadata") or {}),
                }
            )
    finally:
        await service.close()

    return MethodRunnerResult(
        output_json={
            "method_name": context.method_name,
            "method_version": context.method_version,
            "response_count": len(responses),
            "responses": responses,
        },
        pipeline_stage_role="export",
        pipeline_stage_context={
            "source": "prompt",
            "variant_count": len(prompt_variants),
            "include_original": bool(parsed.include_original),
        },
        metadata_json={
            "provider": parsed.provider,
            "model": parsed.model,
            "request_group_id": int(context.request_group_id),
        },
    )


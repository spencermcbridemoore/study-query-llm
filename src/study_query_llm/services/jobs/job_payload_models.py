"""Pydantic models for orchestration job snapshots and payloads."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobSnapshot(BaseModel):
    """Envelope for a claimed orchestration job snapshot."""

    id: int
    job_type: str
    payload_json: Dict[str, Any] = Field(default_factory=dict)
    job_key: str = ""
    base_run_key: Optional[str] = None
    seed_value: Optional[int] = None

    model_config = {"extra": "allow"}


class RunKTryPayload(BaseModel):
    """Payload for run_k_try jobs."""

    embedding_engine: str
    dataset: str
    summarizer: str = "None"
    k_min: int = 2
    k_max: int = 20
    try_idx: int = 0
    seed_value: Optional[int] = None

    model_config = {"extra": "allow"}


class ReduceKPayload(BaseModel):
    """Payload for reduce_k jobs."""

    run_key: str
    dataset: str
    embedding_engine: str
    summarizer: str = "None"
    k_min: int = 2
    k_max: int = 20
    tries_per_k: int = 1

    model_config = {"extra": "allow"}


class FinalizeRunPayload(BaseModel):
    """Payload for finalize_run jobs."""

    run_key: str
    dataset: str
    embedding_engine: str
    summarizer: str = "None"
    k_ranges: list[list[int]] = Field(default_factory=list)
    tries_per_k: int = 1

    model_config = {"extra": "allow"}


def parse_job_snapshot(raw: Dict[str, Any]) -> JobSnapshot:
    """Parse and validate job snapshot. Raises ValidationError on invalid data."""
    return JobSnapshot.model_validate(raw)


class LangGraphRunPayload(BaseModel):
    """Payload for langgraph_run jobs."""

    prompt: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class McqRunPayload(BaseModel):
    """Payload for mcq_run jobs."""

    run_key: str
    deployment: str
    level: str
    subject: str
    options_per_question: int = 4
    questions_per_test: int = 20
    label_style: str = "upper"
    spread_correct_answer_uniformly: bool = False
    samples_per_combo: int = 1
    template_version: str = "v1"
    concurrency: int = 8
    temperature: float = 0.7
    max_tokens: int = 900
    progress_every: int = 0
    determinism_class: str = "non_deterministic"

    model_config = {"extra": "allow"}


class AnalysisRunPayload(BaseModel):
    """Payload for analysis_run jobs."""

    request_id: int
    sweep_type: str
    analysis_key: str
    run_key: str = ""
    scope: str = "run"
    method_name: str = ""
    method_version: str = "1.0"
    required: bool = False
    blocking: bool = False
    result_keys: list[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    force: bool = False

    model_config = {"extra": "allow"}


def parse_run_k_try_payload(payload_json: Dict[str, Any]) -> RunKTryPayload:
    """Parse and validate run_k_try payload. Raises ValidationError on invalid data."""
    return RunKTryPayload.model_validate(payload_json or {})


def parse_reduce_k_payload(payload_json: Dict[str, Any]) -> ReduceKPayload:
    """Parse and validate reduce_k payload. Raises ValidationError on invalid data."""
    return ReduceKPayload.model_validate(payload_json or {})


def parse_finalize_run_payload(payload_json: Dict[str, Any]) -> FinalizeRunPayload:
    """Parse and validate finalize_run payload. Raises ValidationError on invalid data."""
    return FinalizeRunPayload.model_validate(payload_json or {})


def parse_langgraph_run_payload(payload_json: Dict[str, Any]) -> LangGraphRunPayload:
    """Parse and validate langgraph_run payload. Raises ValidationError on invalid data."""
    return LangGraphRunPayload.model_validate(payload_json or {})


def parse_mcq_run_payload(payload_json: Dict[str, Any]) -> McqRunPayload:
    """Parse and validate mcq_run payload. Raises ValidationError on invalid data."""
    return McqRunPayload.model_validate(payload_json or {})


def parse_analysis_run_payload(payload_json: Dict[str, Any]) -> AnalysisRunPayload:
    """Parse and validate analysis_run payload. Raises ValidationError on invalid data."""
    return AnalysisRunPayload.model_validate(payload_json or {})

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


def parse_job_snapshot(raw: Dict[str, Any]) -> JobSnapshot:
    """Parse and validate job snapshot. Raises ValidationError on invalid data."""
    return JobSnapshot.model_validate(raw)


class LangGraphRunPayload(BaseModel):
    """Payload for langgraph_run jobs."""

    prompt: str = ""
    config: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


def parse_run_k_try_payload(payload_json: Dict[str, Any]) -> RunKTryPayload:
    """Parse and validate run_k_try payload. Raises ValidationError on invalid data."""
    return RunKTryPayload.model_validate(payload_json or {})


def parse_langgraph_run_payload(payload_json: Dict[str, Any]) -> LangGraphRunPayload:
    """Parse and validate langgraph_run payload. Raises ValidationError on invalid data."""
    return LangGraphRunPayload.model_validate(payload_json or {})

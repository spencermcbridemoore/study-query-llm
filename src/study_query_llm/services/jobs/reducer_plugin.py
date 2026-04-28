"""Typed reducer plugin seam for orchestration reduce/finalize jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class ReducerInput:
    """Input contract for reducer plugin execution."""

    job_snapshot: Dict[str, Any]
    context: Any


@dataclass(frozen=True)
class ReducerOutput:
    """Output contract for reducer plugin execution."""

    job_id: int
    result_ref: Optional[str]
    run_id: Optional[int] = None


class ReducerPlugin(Protocol):
    """Typed reducer/finalizer plugin protocol."""

    def reduce_k(self, reducer_input: ReducerInput) -> ReducerOutput:
        ...

    def finalize_run(self, reducer_input: ReducerInput) -> ReducerOutput:
        ...


class ClusteringReducerPlugin:
    """Default reducer plugin that wraps the existing JobReducerService behavior."""

    def __init__(self, reducer_service: Any) -> None:
        self._reducer_service = reducer_service

    def reduce_k(self, reducer_input: ReducerInput) -> ReducerOutput:
        job_id = int(reducer_input.job_snapshot["id"])
        result_ref = self._reducer_service.reduce_k_job(job_id)
        return ReducerOutput(job_id=job_id, result_ref=result_ref, run_id=None)

    def finalize_run(self, reducer_input: ReducerInput) -> ReducerOutput:
        job_id = int(reducer_input.job_snapshot["id"])
        run_id = self._reducer_service.finalize_run_job(job_id)
        return ReducerOutput(
            job_id=job_id,
            result_ref=str(run_id) if run_id is not None else None,
            run_id=run_id,
        )


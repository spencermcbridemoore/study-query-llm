"""Factory for job runners by job_type."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .job_runners import (
    FinalizeRunRunner,
    JobRunner,
    JobRunContext,
    JobRunOutcome,
    ReduceKRunner,
    RunKTryRunner,
)
from .langgraph_job_runner import LangGraphJobRunner


def create_job_runner(
    job_type: str,
    *,
    run_k_try_fn: Optional[Callable[..., tuple]] = None,
    reducer: Optional[Any] = None,
) -> JobRunner:
    """Create a job runner for the given job_type.

    Args:
        job_type: "run_k_try", "reduce_k", or "finalize_run"
        run_k_try_fn: Callable for run_k_try (required when job_type is run_k_try)
        reducer: JobReducerService instance (required for reduce_k, finalize_run)

    Returns:
        JobRunner instance

    Raises:
        ValueError: If job_type is unsupported or required deps missing
    """
    job_type = str(job_type).strip().lower()
    if job_type == "run_k_try":
        if run_k_try_fn is None:
            raise ValueError("run_k_try_fn required for job_type run_k_try")
        return RunKTryRunner(run_fn=run_k_try_fn)
    if job_type == "reduce_k":
        if reducer is None:
            raise ValueError("reducer required for job_type reduce_k")
        return ReduceKRunner(reducer=reducer)
    if job_type == "finalize_run":
        if reducer is None:
            raise ValueError("reducer required for job_type finalize_run")
        return FinalizeRunRunner(reducer=reducer)
    if job_type == "langgraph_run":
        return LangGraphJobRunner()
    raise ValueError(
        f"Unsupported job_type: {job_type!r}. "
        "Use 'run_k_try', 'reduce_k', 'finalize_run', or 'langgraph_run'."
    )


def get_supported_job_types() -> list[str]:
    """Return list of supported job_type values."""
    return ["run_k_try", "reduce_k", "finalize_run", "langgraph_run"]

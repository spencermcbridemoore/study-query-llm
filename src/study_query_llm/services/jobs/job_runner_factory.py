"""Factory for job runners by job_type."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .job_runners import (
    AnalysisRunRunner,
    FinalizeRunRunner,
    JobRunner,
    McqRunRunner,
    ReduceKRunner,
    RunKTryRunner,
)
from .langgraph_job_runner import LangGraphJobRunner
from .reducer_plugin import ClusteringReducerPlugin, ReducerPlugin

JobRunnerBuilder = Callable[..., JobRunner]


def _build_run_k_try_runner(*, run_k_try_fn: Optional[Callable[..., tuple]], **_: Any) -> JobRunner:
    if run_k_try_fn is None:
        raise ValueError("run_k_try_fn required for job_type run_k_try")
    return RunKTryRunner(run_fn=run_k_try_fn)


def _build_mcq_run_runner(*, mcq_run_fn: Optional[Callable[..., tuple]], **_: Any) -> JobRunner:
    if mcq_run_fn is None:
        raise ValueError("mcq_run_fn required for job_type mcq_run")
    return McqRunRunner(run_fn=mcq_run_fn)


def _build_analysis_run_runner(
    *, analysis_run_fn: Optional[Callable[..., tuple]], **_: Any
) -> JobRunner:
    if analysis_run_fn is None:
        raise ValueError("analysis_run_fn required for job_type analysis_run")
    return AnalysisRunRunner(run_fn=analysis_run_fn)


def _build_reduce_k_runner(
    *,
    reducer_plugin: Optional[ReducerPlugin],
    reducer: Optional[Any],
    **_: Any,
) -> JobRunner:
    plugin = reducer_plugin or (ClusteringReducerPlugin(reducer) if reducer is not None else None)
    if plugin is None:
        raise ValueError("reducer_plugin (or reducer) required for job_type reduce_k")
    return ReduceKRunner(reducer_plugin=plugin)


def _build_finalize_run_runner(
    *,
    reducer_plugin: Optional[ReducerPlugin],
    reducer: Optional[Any],
    **_: Any,
) -> JobRunner:
    plugin = reducer_plugin or (ClusteringReducerPlugin(reducer) if reducer is not None else None)
    if plugin is None:
        raise ValueError("reducer_plugin (or reducer) required for job_type finalize_run")
    return FinalizeRunRunner(reducer_plugin=plugin)


def _build_langgraph_runner(**_: Any) -> JobRunner:
    return LangGraphJobRunner()


_JOB_RUNNER_REGISTRY: Dict[str, JobRunnerBuilder] = {
    "run_k_try": _build_run_k_try_runner,
    "mcq_run": _build_mcq_run_runner,
    "analysis_run": _build_analysis_run_runner,
    "reduce_k": _build_reduce_k_runner,
    "finalize_run": _build_finalize_run_runner,
    "langgraph_run": _build_langgraph_runner,
}


def register_job_runner(job_type: str, builder: JobRunnerBuilder) -> None:
    """Register or override a job runner builder for job_type."""
    key = str(job_type).strip().lower()
    if not key:
        raise ValueError("job_type must be non-empty")
    _JOB_RUNNER_REGISTRY[key] = builder


def create_job_runner(
    job_type: str,
    *,
    run_k_try_fn: Optional[Callable[..., tuple]] = None,
    mcq_run_fn: Optional[Callable[..., tuple]] = None,
    analysis_run_fn: Optional[Callable[..., tuple]] = None,
    reducer_plugin: Optional[ReducerPlugin] = None,
    reducer: Optional[Any] = None,
) -> JobRunner:
    """Create a job runner for the given job_type.

    Args:
        job_type: "run_k_try", "mcq_run", "analysis_run", "reduce_k", "finalize_run", or "langgraph_run"
        run_k_try_fn: Callable for run_k_try (required when job_type is run_k_try)
        mcq_run_fn: Callable for mcq_run (required when job_type is mcq_run)
        analysis_run_fn: Callable for analysis_run (required when job_type is analysis_run)
        reducer_plugin: Typed reducer plugin (preferred for reduce_k/finalize_run)
        reducer: Legacy JobReducerService instance (wrapped for compatibility)

    Returns:
        JobRunner instance

    Raises:
        ValueError: If job_type is unsupported or required deps missing
    """
    job_type = str(job_type).strip().lower()
    builder = _JOB_RUNNER_REGISTRY.get(job_type)
    if builder is None:
        supported = ", ".join(get_supported_job_types())
        raise ValueError(
            f"Unsupported job_type: {job_type!r}. Use one of: {supported}."
        )
    return builder(
        run_k_try_fn=run_k_try_fn,
        mcq_run_fn=mcq_run_fn,
        analysis_run_fn=analysis_run_fn,
        reducer_plugin=reducer_plugin,
        reducer=reducer,
    )


def get_supported_job_types() -> list[str]:
    """Return list of supported job_type values."""
    return [
        "run_k_try",
        "mcq_run",
        "analysis_run",
        "reduce_k",
        "finalize_run",
        "langgraph_run",
    ]

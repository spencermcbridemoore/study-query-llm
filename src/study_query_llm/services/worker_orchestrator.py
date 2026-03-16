"""Worker orchestrator abstraction and factory for standalone vs sharded execution modes."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class WorkerOrchestrator(Protocol):
    """Protocol for worker orchestration strategies."""

    def run(self) -> int:
        """Execute the worker loop. Returns completed count."""
        ...


class StandaloneWorkerOrchestrator:
    """Orchestrator for standalone mode (run-key claims via sweep_run_claims)."""

    def __init__(self, run_fn: Callable[..., int], **kwargs: Any) -> None:
        self._run_fn = run_fn
        self._kwargs = kwargs

    def run(self) -> int:
        return self._run_fn(**self._kwargs)


class ShardedWorkerOrchestrator:
    """Orchestrator for sharded mode (orchestration_jobs path)."""

    def __init__(self, run_fn: Callable[..., int], **kwargs: Any) -> None:
        self._run_fn = run_fn
        self._kwargs = kwargs

    def run(self) -> int:
        return self._run_fn(**self._kwargs)


def create_worker_orchestrator(
    mode: str,
    run_standalone_fn: Callable[..., int],
    run_sharded_fn: Callable[..., int],
    **kwargs: Any,
) -> WorkerOrchestrator:
    """Create a worker orchestrator for the given mode.

    Args:
        mode: "standalone" or "sharded"
        run_standalone_fn: Callable for standalone worker loop
        run_sharded_fn: Callable for sharded worker loop
        **kwargs: Arguments passed to the selected loop

    Returns:
        WorkerOrchestrator instance

    Raises:
        ValueError: If mode is not supported
    """
    if mode == "standalone":
        return StandaloneWorkerOrchestrator(run_fn=run_standalone_fn, **kwargs)
    if mode == "sharded":
        return ShardedWorkerOrchestrator(run_fn=run_sharded_fn, **kwargs)
    raise ValueError(f"Unsupported job_mode: {mode!r}. Use 'standalone' or 'sharded'.")

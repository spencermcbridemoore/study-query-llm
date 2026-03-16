"""Supervisor mode adapters for standalone vs sharded progress tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple

from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.sweep_request_service import SweepRequestService

if TYPE_CHECKING:
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2


def _engine_missing_count(
    db: "DatabaseConnectionV2", request_id: int, engine: str
) -> Tuple[int, int]:
    """Return (engine_missing, request_missing_total) for standalone mode."""
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        progress = svc.compute_progress(request_id)
        missing_run_keys = progress.get("missing_run_keys") or []
        request = svc.get_request(request_id) or {}
        run_key_to_target = request.get("run_key_to_target") or {}
    engine_missing = 0
    for rk in missing_run_keys:
        target = run_key_to_target.get(rk)
        if target and target.get("embedding_engine") == engine:
            engine_missing += 1
    return engine_missing, int(progress.get("missing_count", 0))


def _engine_pending_jobs_count(
    db: "DatabaseConnectionV2", request_id: int, engine: str
) -> Tuple[int, int]:
    """Return (engine_pending, total_pending) for sharded mode."""
    relevant = 0
    total_pending = 0
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        jobs = repo.list_orchestration_jobs(request_group_id=request_id)
        for j in jobs:
            status = j.status
            job_type = j.job_type
            if status in ("pending", "ready", "claimed") and job_type in (
                "run_k_try",
                "reduce_k",
                "finalize_run",
            ):
                total_pending += 1
            if (
                status in ("pending", "ready", "claimed")
                and job_type == "run_k_try"
                and (j.payload_json or {}).get("embedding_engine") == engine
            ):
                relevant += 1
    return relevant, total_pending


class SupervisorMode(Protocol):
    """Protocol for supervisor mode-specific behavior."""

    def engine_work_remaining(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
        engine: str,
    ) -> Tuple[int, int]:
        """Return (engine_missing, total_missing) for progress tracking."""
        ...

    def before_progress_poll(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
    ) -> None:
        """Hook called before each progress poll (e.g. promote-ready for sharded)."""
        ...


class StandaloneSupervisorMode:
    """Standalone mode: uses missing run-key counts."""

    def engine_work_remaining(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
        engine: str,
    ) -> Tuple[int, int]:
        return _engine_missing_count(db, request_id, engine)

    def before_progress_poll(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
    ) -> None:
        pass  # No-op for standalone


class ShardedSupervisorMode:
    """Sharded mode: uses pending job counts and promote-ready pass."""

    def engine_work_remaining(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
        engine: str,
    ) -> Tuple[int, int]:
        return _engine_pending_jobs_count(db, request_id, engine)

    def before_progress_poll(
        self,
        db: "DatabaseConnectionV2",
        request_id: int,
    ) -> None:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            repo.promote_ready_orchestration_jobs(request_group_id=request_id)


def create_supervisor_mode(mode: str) -> SupervisorMode:
    """Create a supervisor mode adapter for the given mode.

    Args:
        mode: "standalone" or "sharded"

    Returns:
        SupervisorMode instance

    Raises:
        ValueError: If mode is not supported
    """
    if mode == "standalone":
        return StandaloneSupervisorMode()
    if mode == "sharded":
        return ShardedSupervisorMode()
    raise ValueError(f"Unsupported job_mode: {mode!r}. Use 'standalone' or 'sharded'.")

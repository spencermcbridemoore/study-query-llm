"""Job runners for orchestration job types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol

from pydantic import ValidationError


@dataclass
class JobRunOutcome:
    """Result of running one orchestration job."""

    job_id: int
    result_ref: Optional[str]
    error: Optional[str]
    db_updated_by_runner: bool
    """If True, runner already called complete/fail; worker must not."""
    metadata: Optional[Dict[str, Any]] = None
    """Optional extra data (e.g., checkpoint_refs for langgraph_run)."""


@dataclass
class JobRunContext:
    """Context passed to job runners."""

    datasets: Dict[str, Any]
    provider_cache: Dict[str, object]
    manager_cache: Dict[str, object]
    tei_endpoint: Optional[str]
    provider_label: str
    embedding_provider_name: Optional[str]
    worker_slot: int
    repo_root: Any  # Path
    claim_wait_seconds: float
    reducer: Any  # JobReducerService
    db: Any  # DatabaseConnectionV2


class JobRunner(Protocol):
    """Protocol for job execution strategies."""

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        """Execute the job. Returns outcome; worker handles complete/fail if not db_updated_by_runner."""
        ...


class RunKTryRunner:
    """Runner for run_k_try jobs. Does not update DB; worker does."""

    def __init__(self, run_fn: Callable[..., tuple]) -> None:
        self._run_fn = run_fn

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        from .job_payload_models import parse_job_snapshot, parse_run_k_try_payload

        try:
            parse_job_snapshot(job_snapshot)
            parse_run_k_try_payload(job_snapshot.get("payload_json") or {})
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )
        job_id_out, result_ref_out, error_out = self._run_fn(
            job_snapshot=job_snapshot,
            datasets=context.datasets,
            provider_cache=context.provider_cache,
            manager_cache=context.manager_cache,
            tei_endpoint=context.tei_endpoint,
            provider_label=context.provider_label,
            embedding_provider_name=context.embedding_provider_name,
            worker_slot=context.worker_slot,
            repo_root=context.repo_root,
            db=context.db,
            claim_wait_seconds=context.claim_wait_seconds,
        )
        return JobRunOutcome(
            job_id=job_id_out,
            result_ref=result_ref_out,
            error=error_out,
            db_updated_by_runner=False,
        )


class ReduceKRunner:
    """Runner for reduce_k jobs. Reducer completes job internally."""

    def __init__(self, reducer: Any) -> None:
        self._reducer = reducer

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        from .job_payload_models import parse_job_snapshot

        try:
            parse_job_snapshot(job_snapshot)
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )
        job_id = int(job_snapshot["id"])
        result_ref = self._reducer.reduce_k_job(job_id)
        return JobRunOutcome(
            job_id=job_id,
            result_ref=result_ref,
            error=None,
            db_updated_by_runner=True,
        )


class FinalizeRunRunner:
    """Runner for finalize_run jobs. Reducer completes job internally."""

    def __init__(self, reducer: Any) -> None:
        self._reducer = reducer

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        from .job_payload_models import parse_job_snapshot

        try:
            parse_job_snapshot(job_snapshot)
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )
        job_id = int(job_snapshot["id"])
        run_id = self._reducer.finalize_run_job(job_id)
        return JobRunOutcome(
            job_id=job_id,
            result_ref=str(run_id) if run_id is not None else None,
            error=None,
            db_updated_by_runner=True,
        )


class McqRunRunner:
    """Runner for mcq_run jobs. Does not update DB; worker does."""

    def __init__(self, run_fn: Callable[..., tuple]) -> None:
        self._run_fn = run_fn

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        from .job_payload_models import (
            parse_job_snapshot,
            parse_mcq_run_payload,
        )

        try:
            parse_job_snapshot(job_snapshot)
            parse_mcq_run_payload(job_snapshot.get("payload_json") or {})
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )
        job_id_out, result_ref_out, error_out = self._run_fn(
            job_snapshot=job_snapshot,
            db=context.db,
            worker_label=f"orchestration-job-{int(job_snapshot.get('id', 0))}",
        )
        return JobRunOutcome(
            job_id=job_id_out,
            result_ref=result_ref_out,
            error=error_out,
            db_updated_by_runner=False,
        )


class AnalysisRunRunner:
    """Runner for analysis_run jobs. Does not update DB; worker does."""

    def __init__(self, run_fn: Callable[..., tuple]) -> None:
        self._run_fn = run_fn

    def run(self, job_snapshot: Dict[str, Any], context: JobRunContext) -> JobRunOutcome:
        from .job_payload_models import (
            parse_analysis_run_payload,
            parse_job_snapshot,
        )

        try:
            parse_job_snapshot(job_snapshot)
            parse_analysis_run_payload(job_snapshot.get("payload_json") or {})
        except ValidationError as e:
            job_id = int(job_snapshot.get("id", 0))
            return JobRunOutcome(
                job_id=job_id,
                result_ref=None,
                error=f"payload_validation_error: {e}",
                db_updated_by_runner=False,
            )
        job_id_out, result_ref_out, error_out = self._run_fn(
            job_snapshot=job_snapshot,
            db=context.db,
            worker_label=f"orchestration-job-{int(job_snapshot.get('id', 0))}",
        )
        return JobRunOutcome(
            job_id=job_id_out,
            result_ref=result_ref_out,
            error=error_out,
            db_updated_by_runner=False,
        )

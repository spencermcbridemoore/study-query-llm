"""
Sweep Request Service - typed request lifecycle and orchestration planning.

Supports clustering and MCQ sweep request types, formal analysis status tracking,
and OrchestrationJob planning where standalone execution is modeled as a
special-case orchestration profile.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.attributes import flag_modified

from study_query_llm.experiments.sweep_request_types import (
    ANALYSIS_STATUS_COMPLETE,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_NOT_REQUIRED,
    ANALYSIS_STATUS_NOT_STARTED,
    ANALYSIS_STATUS_RUNNING,
    REQUEST_SCHEMA_VERSION,
    REQUEST_STATUS_FULFILLED,
    REQUEST_STATUS_REQUESTED,
    REQUEST_STATUS_RUNNING,
    SWEEP_TYPE_CLUSTERING,
    SWEEP_TYPE_MCQ,
    get_sweep_type_adapter,
)
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
    GROUP_TYPE_MCQ_SWEEP,
    GROUP_TYPE_MCQ_SWEEP_REQUEST,
    ProvenanceService,
)
from study_query_llm.services.provenanced_run_service import ProvenancedRunService
from study_query_llm.utils.logging_config import get_logger

if TYPE_CHECKING:
    from study_query_llm.db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)

MCQ_ORCHESTRATION_GRAPH_KIND = "single_job_per_run"
MCQ_HISTORICAL_BACKFILL_POLICY = "compatibility_only"
MCQ_DETERMINISM_CLASS = "non_deterministic"
JOB_TYPE_MCQ_RUN = "mcq_run"
JOB_TYPE_ANALYSIS_RUN = "analysis_run"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


class SweepRequestService:
    """Typed sweep request lifecycle manager."""

    def __init__(self, repository: "RawCallRepository"):
        self.repository = repository
        self.provenance = ProvenanceService(repository)
        self.method_service = MethodService(repository)
        self.provenanced_runs = ProvenancedRunService(repository)
        self.use_derived_analysis_status = _env_flag("SQ_DERIVE_ANALYSIS_STATUS_READ", True)
        self.enable_analysis_jobs = _env_flag("SQ_ENABLE_ANALYSIS_JOBS", True)
        self.record_analysis_parity = _env_flag("SQ_RECORD_ANALYSIS_PARITY", True)
        self.unified_execution_writes = _env_flag("SQ_UNIFIED_EXECUTION_WRITES", True)

    @staticmethod
    def _request_group_types() -> set[str]:
        return {
            GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
            GROUP_TYPE_MCQ_SWEEP_REQUEST,
        }

    @staticmethod
    def _sweep_type_from_group_type(group_type: str) -> Optional[str]:
        if group_type == GROUP_TYPE_CLUSTERING_SWEEP_REQUEST:
            return "clustering"
        if group_type == GROUP_TYPE_MCQ_SWEEP_REQUEST:
            return "mcq"
        return None

    @staticmethod
    def _analysis_catalog_to_dict(
        analysis_catalog: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        by_key: Dict[str, Dict[str, Any]] = {}
        for item in analysis_catalog:
            if not isinstance(item, dict):
                continue
            key = str(item.get("analysis_key") or "")
            if not key:
                continue
            by_key[key] = item
        return by_key

    @staticmethod
    def _recompute_analysis_status(meta: Dict[str, Any]) -> str:
        required = set(str(x) for x in (meta.get("required_analyses") or []))
        completed = set(str(x) for x in (meta.get("completed_analyses") or []))
        failed_items = list(meta.get("failed_analyses") or [])
        failed = {str(x.get("analysis_key")) for x in failed_items if isinstance(x, dict)}
        if not required:
            return ANALYSIS_STATUS_NOT_REQUIRED
        if failed:
            return ANALYSIS_STATUS_FAILED
        if required.issubset(completed):
            return ANALYSIS_STATUS_COMPLETE
        if completed:
            return ANALYSIS_STATUS_RUNNING
        return ANALYSIS_STATUS_NOT_STARTED

    @staticmethod
    def _analysis_keys_from_meta(meta: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for item in list(meta.get("analysis_catalog") or []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("analysis_key") or "").strip()
            if key:
                keys.append(key)
        for key in list(meta.get("required_analyses") or []):
            text = str(key).strip()
            if text:
                keys.append(text)
        # Preserve order while deduplicating.
        deduped = list(dict.fromkeys(keys))
        return deduped

    def _analysis_job_state_by_key(
        self,
        request_id: int,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return (status_by_key, last_error_by_key) from analysis_run jobs."""
        jobs = self.repository.list_orchestration_jobs(
            request_group_id=int(request_id),
            job_type=JOB_TYPE_ANALYSIS_RUN,
        )
        # status precedence: failed > completed > running-ish > unknown
        rank = {
            "failed": 4,
            "completed": 3,
            "claimed": 2,
            "ready": 2,
            "pending": 2,
            "cancelled": 1,
        }
        status_by_key: Dict[str, str] = {}
        error_by_key: Dict[str, str] = {}
        for job in jobs:
            payload = dict(job.payload_json or {})
            key = str(payload.get("analysis_key") or "").strip()
            if not key:
                continue
            status = str(job.status or "").strip().lower()
            cur = status_by_key.get(key)
            if cur is None or rank.get(status, 0) >= rank.get(cur, 0):
                status_by_key[key] = status
            err_obj = dict(job.error_json or {})
            err = str(err_obj.get("error") or "").strip()
            if err:
                error_by_key[key] = err
        return status_by_key, error_by_key

    def _analysis_execution_state_by_key(
        self,
        request_id: int,
    ) -> Dict[str, str]:
        """Return latest completion/failure state by analysis_key from executions."""
        rows = self.repository.list_provenanced_runs(
            request_group_id=int(request_id),
        )
        status_by_key: Dict[str, str] = {}
        rank = {"failed": 2, "completed": 1}
        for row in rows:
            meta = dict(row.metadata_json or {})
            role = str(meta.get("execution_role") or "").strip().lower()
            raw_kind = str(row.run_kind or "").strip().lower()
            if not role and raw_kind == "analysis_execution":
                role = "analysis_execution"
            if role != "analysis_execution":
                continue
            key = str(meta.get("analysis_key") or "").strip()
            if not key:
                run_key = str(row.run_key or "")
                key = run_key.split(":", 1)[0] if run_key else ""
            if not key:
                continue
            status = str(row.run_status or "").strip().lower()
            if status not in ("completed", "failed"):
                continue
            cur = status_by_key.get(key)
            if cur is None or rank.get(status, 0) >= rank.get(cur, 0):
                status_by_key[key] = status
        return status_by_key

    def _derive_analysis_state(self, request_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Compute request-level analysis state from execution/jobs."""
        required = [str(x) for x in (meta.get("required_analyses") or [])]
        keys = self._analysis_keys_from_meta(meta)
        if not keys:
            keys = list(dict.fromkeys(required))
        if not keys:
            return {
                "analysis_status": ANALYSIS_STATUS_NOT_REQUIRED,
                "completed_analyses": [],
                "failed_analyses": [],
            }

        job_state, job_errors = self._analysis_job_state_by_key(int(request_id))
        execution_state = self._analysis_execution_state_by_key(int(request_id))

        legacy_completed = {str(x) for x in (meta.get("completed_analyses") or [])}
        legacy_failed_items = [
            x for x in (meta.get("failed_analyses") or []) if isinstance(x, dict)
        ]
        legacy_failed_map = {
            str(x.get("analysis_key")): dict(x)
            for x in legacy_failed_items
            if str(x.get("analysis_key") or "")
        }

        completed: List[str] = []
        failed_items: List[Dict[str, Any]] = []
        has_running = False

        for key in keys:
            if key in legacy_completed:
                completed.append(key)
                continue
            if key in legacy_failed_map:
                legacy_item = dict(legacy_failed_map[key])
                failed_items.append(
                    {
                        "analysis_key": key,
                        "error": str(legacy_item.get("error") or "analysis execution failed"),
                        "failed_at": legacy_item.get("failed_at"),
                    }
                )
                continue
            e_status = execution_state.get(key)
            j_status = job_state.get(key, "")
            if e_status == "completed" or j_status == "completed":
                completed.append(key)
                continue
            if e_status == "failed" or j_status == "failed":
                failed_items.append(
                    {
                        "analysis_key": key,
                        "error": job_errors.get(key) or "analysis execution failed",
                        "failed_at": None,
                    }
                )
                continue
            if j_status in {"ready", "claimed"}:
                has_running = True

        required_set = set(required)
        completed_set = set(completed)
        failed_keys = {str(x.get("analysis_key")) for x in failed_items}

        if not required_set:
            status = ANALYSIS_STATUS_NOT_REQUIRED
        elif failed_keys:
            status = ANALYSIS_STATUS_FAILED
        elif required_set.issubset(completed_set):
            status = ANALYSIS_STATUS_COMPLETE
        elif has_running or completed_set:
            status = ANALYSIS_STATUS_RUNNING
        else:
            status = ANALYSIS_STATUS_NOT_STARTED

        return {
            "analysis_status": status,
            "completed_analyses": sorted(completed_set),
            "failed_analyses": failed_items,
        }

    def _attach_analysis_parity(
        self,
        *,
        request_id: int,
        legacy_meta: Dict[str, Any],
        derived_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        legacy_status = str(
            legacy_meta.get("analysis_status")
            or (
                ANALYSIS_STATUS_NOT_REQUIRED
                if not list(legacy_meta.get("required_analyses") or [])
                else ANALYSIS_STATUS_NOT_STARTED
            )
        )
        legacy_completed = sorted(str(x) for x in (legacy_meta.get("completed_analyses") or []))
        legacy_failed = sorted(
            str(x.get("analysis_key"))
            for x in (legacy_meta.get("failed_analyses") or [])
            if isinstance(x, dict)
        )
        derived_completed = sorted(str(x) for x in (derived_state.get("completed_analyses") or []))
        derived_failed = sorted(
            str(x.get("analysis_key"))
            for x in (derived_state.get("failed_analyses") or [])
            if isinstance(x, dict)
        )
        mismatch = not (
            legacy_status == str(derived_state.get("analysis_status") or "")
            and legacy_completed == derived_completed
            and legacy_failed == derived_failed
        )
        parity = {
            "legacy": {
                "analysis_status": legacy_status,
                "completed_analyses": legacy_completed,
                "failed_analyses": legacy_failed,
            },
            "derived": {
                "analysis_status": str(derived_state.get("analysis_status") or ""),
                "completed_analyses": derived_completed,
                "failed_analyses": derived_failed,
            },
            "mismatch": mismatch,
        }
        if mismatch:
            logger.warning(
                "Analysis parity mismatch request_id=%s legacy=%s derived=%s",
                request_id,
                parity["legacy"],
                parity["derived"],
            )
        return parity

    def _is_sqlite(self) -> bool:
        try:
            return self.repository.session.get_bind().dialect.name == "sqlite"
        except Exception:
            return False

    def _normalize_k_ranges(
        self,
        *,
        fixed_config: Dict[str, Any],
        shard_config: Dict[str, Any],
    ) -> List[Tuple[int, int]]:
        raw_ranges = list(shard_config.get("k_ranges") or [])
        ranges: List[Tuple[int, int]] = []
        for item in raw_ranges:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            lo = _safe_int(item[0], 2)
            hi = _safe_int(item[1], lo)
            if hi < lo:
                lo, hi = hi, lo
            ranges.append((lo, hi))
        if ranges:
            return ranges
        k_min = _safe_int(fixed_config.get("k_min"), 2)
        k_max = _safe_int(fixed_config.get("k_max"), max(k_min, 20))
        if k_max < k_min:
            k_min, k_max = k_max, k_min
        return [(k_min, k_max)]

    def _enqueue_sharded_jobs(
        self,
        *,
        request_group_id: int,
        run_key_to_target: Dict[str, Dict[str, Any]],
        fixed_config: Dict[str, Any],
        shard_config: Dict[str, Any],
    ) -> int:
        k_ranges = self._normalize_k_ranges(fixed_config=fixed_config, shard_config=shard_config)
        tries_per_k = max(
            1,
            _safe_int(
                shard_config.get("tries_per_k", fixed_config.get("n_restarts", 1)),
                1,
            ),
        )
        max_attempts = max(1, _safe_int(fixed_config.get("max_attempts"), 3))
        base_seed = _safe_int(fixed_config.get("base_seed"), 0)
        planned = 0

        for run_key, target in run_key_to_target.items():
            reduce_job_ids: List[int] = []
            for k_min, k_max in k_ranges:
                leaf_job_ids: List[int] = []
                for try_idx in range(tries_per_k):
                    seed_value = base_seed + try_idx
                    leaf_job_key = f"{run_key}__k{k_min}_{k_max}__try{try_idx}"
                    leaf_payload = {
                        "run_key": run_key,
                        "dataset": target.get("dataset"),
                        "embedding_engine": target.get("embedding_engine"),
                        "summarizer": target.get("summarizer", "None"),
                        "k_min": int(k_min),
                        "k_max": int(k_max),
                        "try_idx": int(try_idx),
                        "tries_per_k": int(tries_per_k),
                    }
                    leaf_id = self.repository.enqueue_orchestration_job(
                        request_group_id=request_group_id,
                        job_type="run_k_try",
                        job_key=leaf_job_key,
                        base_run_key=run_key,
                        payload_json=leaf_payload,
                        seed_value=seed_value,
                        max_attempts=max_attempts,
                    )
                    leaf_job_ids.append(int(leaf_id))
                    planned += 1

                reduce_job_key = f"{run_key}__reduce_k{k_min}_{k_max}"
                reduce_payload = {
                    "run_key": run_key,
                    "dataset": target.get("dataset"),
                    "embedding_engine": target.get("embedding_engine"),
                    "summarizer": target.get("summarizer", "None"),
                    "k_min": int(k_min),
                    "k_max": int(k_max),
                    "tries_per_k": int(tries_per_k),
                }
                reduce_id = self.repository.enqueue_orchestration_job(
                    request_group_id=request_group_id,
                    job_type="reduce_k",
                    job_key=reduce_job_key,
                    base_run_key=run_key,
                    payload_json=reduce_payload,
                    max_attempts=max_attempts,
                    depends_on_job_ids=leaf_job_ids,
                )
                reduce_job_ids.append(int(reduce_id))
                planned += 1

            finalize_job_key = f"{run_key}__finalize_run"
            finalize_payload = {
                "run_key": run_key,
                "dataset": target.get("dataset"),
                "embedding_engine": target.get("embedding_engine"),
                "summarizer": target.get("summarizer", "None"),
                "k_ranges": [[int(lo), int(hi)] for (lo, hi) in k_ranges],
                "tries_per_k": int(tries_per_k),
            }
            self.repository.enqueue_orchestration_job(
                request_group_id=request_group_id,
                job_type="finalize_run",
                job_key=finalize_job_key,
                base_run_key=run_key,
                payload_json=finalize_payload,
                max_attempts=max_attempts,
                depends_on_job_ids=reduce_job_ids,
            )
            planned += 1
        return planned

    def _enqueue_mcq_jobs(
        self,
        *,
        request_group_id: int,
        run_key_to_target: Dict[str, Dict[str, Any]],
        fixed_config: Dict[str, Any],
    ) -> Tuple[int, List[int]]:
        """
        Enqueue one canonical mcq_run job per request run_key.
        """
        max_attempts = max(1, _safe_int(fixed_config.get("max_attempts"), 3))
        planned = 0
        job_ids: List[int] = []
        for run_key, target in run_key_to_target.items():
            payload = dict(target or {})
            payload["run_key"] = run_key
            payload["determinism_class"] = MCQ_DETERMINISM_CLASS
            job_key = f"{run_key}__mcq_run"
            job_id = self.repository.enqueue_orchestration_job(
                request_group_id=request_group_id,
                job_type=JOB_TYPE_MCQ_RUN,
                job_key=job_key,
                base_run_key=run_key,
                payload_json=payload,
                max_attempts=max_attempts,
            )
            job_ids.append(int(job_id))
            planned += 1
        return planned, job_ids

    def _enqueue_mcq_analysis_jobs(
        self,
        *,
        request_group_id: int,
        metadata_json: Dict[str, Any],
        fixed_config: Dict[str, Any],
        depends_on_job_ids: List[int],
    ) -> int:
        """
        Enqueue one analysis_run job per analysis_catalog key for MCQ requests.
        """
        catalog = list(metadata_json.get("analysis_catalog") or [])
        if not catalog:
            return 0
        max_attempts = max(1, _safe_int(fixed_config.get("max_attempts"), 3))
        planned = 0
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            analysis_key = str(entry.get("analysis_key") or "").strip()
            if not analysis_key:
                continue
            payload = {
                "request_id": int(request_group_id),
                "sweep_type": SWEEP_TYPE_MCQ,
                "analysis_key": analysis_key,
                "scope": str(entry.get("scope") or "run"),
                "method_name": str(entry.get("method_name") or analysis_key),
                "method_version": str(entry.get("method_version") or "1.0"),
                "required": bool(entry.get("required")),
                "blocking": bool(entry.get("blocking")),
                "result_keys": list(entry.get("result_keys") or []),
            }
            job_key = f"req{int(request_group_id)}__analysis__{analysis_key}"
            self.repository.enqueue_orchestration_job(
                request_group_id=request_group_id,
                job_type=JOB_TYPE_ANALYSIS_RUN,
                job_key=job_key,
                base_run_key=analysis_key,
                payload_json=payload,
                max_attempts=max_attempts,
                depends_on_job_ids=list(depends_on_job_ids),
            )
            planned += 1
        return planned

    def _plan_orchestration_jobs_for_metadata(
        self,
        *,
        request_group_id: int,
        metadata_json: Dict[str, Any],
        force: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        sweep_type = str(metadata_json.get("sweep_type") or SWEEP_TYPE_CLUSTERING).lower()
        if metadata_json.get("orchestration_planned_at") and not force:
            return 0, metadata_json

        run_key_to_target = dict(metadata_json.get("run_key_to_target") or {})
        if not run_key_to_target:
            return 0, metadata_json

        fixed_config = dict(metadata_json.get("fixed_config") or {})
        if sweep_type == SWEEP_TYPE_MCQ:
            planned_mcq, mcq_job_ids = self._enqueue_mcq_jobs(
                request_group_id=request_group_id,
                run_key_to_target=run_key_to_target,
                fixed_config=fixed_config,
            )
            planned_analysis = 0
            if self.enable_analysis_jobs:
                planned_analysis = self._enqueue_mcq_analysis_jobs(
                    request_group_id=request_group_id,
                    metadata_json=metadata_json,
                    fixed_config=fixed_config,
                    depends_on_job_ids=mcq_job_ids,
                )
            planned_total = int(planned_mcq + planned_analysis)
            metadata_json["orchestration_graph_kind"] = MCQ_ORCHESTRATION_GRAPH_KIND
            metadata_json["historical_backfill_policy"] = MCQ_HISTORICAL_BACKFILL_POLICY
            metadata_json["determinism_class"] = MCQ_DETERMINISM_CLASS
            metadata_json["analysis_execution_mode"] = (
                "orchestration_jobs" if self.enable_analysis_jobs else "compatibility_only"
            )
            metadata_json["analysis_job_count"] = int(planned_analysis)
            metadata_json["orchestration_planned_at"] = _now_iso()
            metadata_json["orchestration_job_count"] = planned_total
            metadata_json["updated_at"] = _now_iso()
            return planned_total, metadata_json

        if sweep_type != SWEEP_TYPE_CLUSTERING:
            return 0, metadata_json

        execution_mode = str(metadata_json.get("execution_mode") or "standalone").lower()
        shard_config = dict(metadata_json.get("shard_config") or {})
        if execution_mode != "sharded":
            k_min = _safe_int(fixed_config.get("k_min"), 2)
            k_max = _safe_int(fixed_config.get("k_max"), max(k_min, 20))
            if k_max < k_min:
                k_min, k_max = k_max, k_min
            k_ranges = [[k, k] for k in range(k_min, k_max + 1)]
            shard_config = {
                "k_ranges": k_ranges,
                "tries_per_k": _safe_int(fixed_config.get("n_restarts"), 50),
                "standalone_special_case": True,
            }

        planned = self._enqueue_sharded_jobs(
            request_group_id=request_group_id,
            run_key_to_target=run_key_to_target,
            fixed_config=fixed_config,
            shard_config=shard_config,
        )
        metadata_json["orchestration_planned_at"] = _now_iso()
        metadata_json["orchestration_job_count"] = int(planned)
        metadata_json["updated_at"] = _now_iso()
        return planned, metadata_json

    def ensure_orchestration_jobs(self, request_id: int, *, force: bool = False) -> int:
        """Idempotently ensure orchestration jobs exist for a request."""
        req_group = self.repository.get_group_by_id(request_id)
        if req_group is None:
            return 0
        if req_group.group_type not in self._request_group_types():
            return 0
        meta = dict(req_group.metadata_json or {})
        planned, meta = self._plan_orchestration_jobs_for_metadata(
            request_group_id=request_id,
            metadata_json=meta,
            force=force,
        )
        if planned > 0:
            req_group.metadata_json = meta
            flag_modified(req_group, "metadata_json")
            self.repository.session.flush()
        return planned

    def create_request(
        self,
        request_name: str,
        algorithm: str,
        fixed_config: Dict[str, Any],
        parameter_axes: Dict[str, Any],
        entry_max: Optional[int],
        n_restarts_suffix: str = "50runs",
        description: Optional[str] = None,
        *,
        sweep_type: str = SWEEP_TYPE_CLUSTERING,
        execution_mode: str = "standalone",
        shard_config: Optional[Dict[str, Any]] = None,
        determinism_class: str = "non_deterministic",
    ) -> int:
        """
        Create a typed sweep request group and optionally pre-plan orchestration jobs.
        """
        adapter = get_sweep_type_adapter(sweep_type)
        target_specs = adapter.build_targets(
            parameter_axes=parameter_axes,
            fixed_config=fixed_config or {},
            entry_max=entry_max,
            n_restarts_suffix=n_restarts_suffix,
        )
        expected_run_keys = [spec.run_key for spec in target_specs]
        run_key_to_target = {spec.run_key: dict(spec.target) for spec in target_specs}
        analysis_defs = adapter.analysis_definitions()
        analysis_catalog = [
            {
                "analysis_key": d.analysis_key,
                "method_name": d.method_name,
                "method_version": d.method_version,
                "scope": d.scope,
                "required": bool(d.required),
                "blocking": bool(d.blocking),
                "result_keys": list(d.result_keys),
            }
            for d in analysis_defs
        ]
        required_analyses = [x["analysis_key"] for x in analysis_catalog if x.get("required")]
        algo = algorithm or adapter.default_algorithm()
        resolved_determinism_class = str(determinism_class or "non_deterministic")
        if adapter.sweep_type == SWEEP_TYPE_MCQ:
            if resolved_determinism_class != MCQ_DETERMINISM_CLASS:
                logger.warning(
                    "MCQ requests are locked to determinism_class=%s (got=%s)",
                    MCQ_DETERMINISM_CLASS,
                    resolved_determinism_class,
                )
            resolved_determinism_class = MCQ_DETERMINISM_CLASS
        metadata_json: Dict[str, Any] = {
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "request_status": REQUEST_STATUS_REQUESTED,
            "sweep_type": adapter.sweep_type,
            "execution_mode": str(execution_mode or "standalone"),
            "shard_config": dict(shard_config or {}),
            "algorithm": algo,
            "determinism_class": resolved_determinism_class,
            "fixed_config": dict(fixed_config or {}),
            "parameter_axes": dict(parameter_axes or {}),
            "entry_max": entry_max,
            "n_restarts_suffix": n_restarts_suffix,
            "expected_run_keys": expected_run_keys,
            "run_key_to_target": run_key_to_target,
            "expected_count": len(expected_run_keys),
            "analysis_catalog": analysis_catalog,
            "required_analyses": required_analyses,
            "completed_analyses": [],
            "failed_analyses": [],
            "analysis_status": (
                ANALYSIS_STATUS_NOT_STARTED
                if required_analyses
                else ANALYSIS_STATUS_NOT_REQUIRED
            ),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        if adapter.sweep_type == SWEEP_TYPE_MCQ:
            metadata_json["orchestration_graph_kind"] = MCQ_ORCHESTRATION_GRAPH_KIND
            metadata_json["historical_backfill_policy"] = MCQ_HISTORICAL_BACKFILL_POLICY

        group_id = self.repository.create_group(
            group_type=adapter.request_group_type,
            name=request_name,
            description=description or f"Sweep request: {request_name} ({algo})",
            metadata_json=metadata_json,
        )

        # Plan orchestration jobs immediately for clustering requests.
        group = self.repository.get_group_by_id(group_id)
        if group is not None:
            planned, meta2 = self._plan_orchestration_jobs_for_metadata(
                request_group_id=group_id,
                metadata_json=dict(group.metadata_json or {}),
            )
            if planned > 0:
                group.metadata_json = meta2
                flag_modified(group, "metadata_json")
                self.repository.session.flush()

        logger.info(
            "Created %s request id=%s name=%s expected_count=%s execution_mode=%s",
            adapter.sweep_type,
            group_id,
            request_name,
            len(expected_run_keys),
            execution_mode,
        )
        return int(group_id)

    def get_request(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Get request group metadata for clustering/mcq request types."""
        group = self.repository.get_group_by_id(request_id)
        if not group or group.group_type not in self._request_group_types():
            return None
        meta = dict(group.metadata_json or {})
        if "sweep_type" not in meta:
            inferred = self._sweep_type_from_group_type(group.group_type)
            if inferred:
                meta["sweep_type"] = inferred
        legacy_meta = dict(meta)
        if self.use_derived_analysis_status:
            derived = self._derive_analysis_state(int(group.id), legacy_meta)
            if self.record_analysis_parity:
                parity = self._attach_analysis_parity(
                    request_id=int(group.id),
                    legacy_meta=legacy_meta,
                    derived_state=derived,
                )
                if parity is not None:
                    meta["analysis_parity"] = parity
            meta["legacy_analysis_status"] = str(
                legacy_meta.get("analysis_status") or ANALYSIS_STATUS_NOT_REQUIRED
            )
            meta["legacy_completed_analyses"] = list(legacy_meta.get("completed_analyses") or [])
            meta["legacy_failed_analyses"] = list(legacy_meta.get("failed_analyses") or [])
            meta["analysis_status"] = str(derived.get("analysis_status") or ANALYSIS_STATUS_NOT_REQUIRED)
            meta["completed_analyses"] = list(derived.get("completed_analyses") or [])
            meta["failed_analyses"] = list(derived.get("failed_analyses") or [])
        return {
            "id": int(group.id),
            "group_type": group.group_type,
            "name": group.name,
            "description": group.description,
            "created_at": group.created_at,
            **meta,
        }

    def list_requests(
        self,
        status: Optional[str] = None,
        include_fulfilled: bool = True,
        sweep_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List sweep requests across registered request group types."""
        from study_query_llm.db.models_v2 import Group

        groups = (
            self.repository.session.query(Group)
            .filter(Group.group_type.in_(list(self._request_group_types())))
            .order_by(Group.created_at.desc())
            .all()
        )
        out: List[Dict[str, Any]] = []
        for g in groups:
            req_full = self.get_request(int(g.id))
            if req_full is None:
                continue
            meta = dict(req_full)
            req_sweep_type = str(
                meta.get("sweep_type")
                or self._sweep_type_from_group_type(g.group_type)
                or ""
            )
            if sweep_type and req_sweep_type != str(sweep_type):
                continue
            req_status = str(meta.get("request_status") or "")
            if status and req_status != status:
                continue
            if not include_fulfilled and req_status == REQUEST_STATUS_FULFILLED:
                continue
            out.append(
                {
                    "id": int(g.id),
                    "name": g.name,
                    "group_type": g.group_type,
                    "sweep_type": req_sweep_type,
                    "request_status": req_status or "?",
                    "expected_count": int(meta.get("expected_count") or 0),
                    "created_at": g.created_at,
                    "analysis_status": str(meta.get("analysis_status") or ""),
                }
            )
        return out

    def compute_progress(self, request_id: int) -> Dict[str, Any]:
        """Compute completed/missing runs for a request from run groups."""
        req = self.get_request(request_id)
        if not req:
            return {
                "expected_count": 0,
                "completed_count": 0,
                "missing_count": 0,
                "completed_run_keys": [],
                "missing_run_keys": [],
                "completed_run_ids": [],
            }

        adapter = get_sweep_type_adapter(req.get("sweep_type"))
        expected_keys = [str(x) for x in (req.get("expected_run_keys") or [])]
        if not expected_keys:
            return {
                "expected_count": 0,
                "completed_count": 0,
                "missing_count": 0,
                "completed_run_keys": [],
                "missing_run_keys": [],
                "completed_run_ids": [],
            }

        from study_query_llm.db.models_v2 import Group

        expected_set = set(expected_keys)
        completed_run_keys: List[str] = []
        completed_run_ids: List[int] = []

        all_runs = (
            self.repository.session.query(Group)
            .filter(Group.group_type == adapter.run_group_type)
            .all()
        )
        for run in all_runs:
            run_key = str((run.metadata_json or {}).get("run_key") or "")
            if run_key and run_key in expected_set:
                completed_run_keys.append(run_key)
                completed_run_ids.append(int(run.id))

        seen_keys = set(completed_run_keys)
        missing_run_keys = [rk for rk in expected_keys if rk not in seen_keys]
        return {
            "expected_count": len(expected_keys),
            "completed_count": len(completed_run_keys),
            "missing_count": len(missing_run_keys),
            "completed_run_keys": completed_run_keys,
            "missing_run_keys": missing_run_keys,
            "completed_run_ids": completed_run_ids,
        }

    def record_delivery(
        self,
        request_id: int,
        run_id: int,
        run_key: str,
    ) -> bool:
        """Idempotently link a delivered run to its request."""
        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type not in self._request_group_types():
            logger.warning("Invalid request_id for record_delivery: %s", request_id)
            return False
        req_meta = dict(req_group.metadata_json or {})
        sweep_type = req_meta.get("sweep_type") or self._sweep_type_from_group_type(req_group.group_type)
        adapter = get_sweep_type_adapter(sweep_type)

        run_group = self.repository.get_group_by_id(run_id)
        if not run_group or run_group.group_type != adapter.run_group_type:
            logger.warning(
                "Invalid run_id for record_delivery: request=%s run=%s expected_group_type=%s",
                request_id,
                run_id,
                adapter.run_group_type,
            )
            return False

        self.repository.create_group_link(
            parent_group_id=request_id,
            child_group_id=run_id,
            link_type="contains",
            metadata_json={"run_key": run_key},
        )

        req_meta["updated_at"] = _now_iso()
        req_meta["request_status"] = REQUEST_STATUS_RUNNING
        req_group.metadata_json = req_meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()
        return True

    def _create_sweep_group(
        self,
        *,
        request_id: int,
        sweep_name: str,
        algorithm: str,
        fixed_config: Dict[str, Any],
        parameter_axes: Dict[str, Any],
        completed_run_ids: List[int],
        sweep_type: str,
        description: str,
    ) -> int:
        if sweep_type == SWEEP_TYPE_CLUSTERING:
            try:
                sweep_id = self.provenance.create_clustering_sweep_group(
                    sweep_name=sweep_name,
                    algorithm=algorithm,
                    fixed_config=fixed_config,
                    parameter_axes=parameter_axes,
                    description=description,
                )
            except IntegrityError:
                from study_query_llm.db.models_v2 import Group

                existing = (
                    self.repository.session.query(Group)
                    .filter(
                        Group.group_type == GROUP_TYPE_CLUSTERING_SWEEP,
                        Group.name == sweep_name,
                    )
                    .first()
                )
                if not existing:
                    raise
                sweep_id = int(existing.id)
            for pos, run_id in enumerate(completed_run_ids):
                self.provenance.link_run_to_clustering_sweep(
                    sweep_id=sweep_id,
                    run_id=run_id,
                    position=pos,
                )
            return int(sweep_id)

        # Generic path for mcq and future sweep types.
        adapter = get_sweep_type_adapter(sweep_type)
        sweep_id = self.repository.create_group(
            group_type=adapter.sweep_group_type,
            name=sweep_name,
            description=description,
            metadata_json={
                "algorithm": algorithm,
                "fixed_config": fixed_config,
                "parameter_axes": parameter_axes,
                "created_from_request_id": request_id,
            },
        )
        for pos, run_id in enumerate(completed_run_ids):
            self.repository.create_group_link(
                parent_group_id=sweep_id,
                child_group_id=run_id,
                link_type="contains",
                position=pos,
            )
        return int(sweep_id)

    def finalize_if_fulfilled(
        self,
        request_id: int,
        sweep_name: Optional[str] = None,
    ) -> Optional[int]:
        """Finalize request into sweep group when all expected runs are present."""
        session = self.repository.session
        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type not in self._request_group_types():
            return None

        if session.get_bind().dialect.name != "sqlite":
            req_group = (
                session.query(type(req_group))
                .filter(type(req_group).id == request_id)
                .with_for_update()
                .first()
            )
            if not req_group:
                return None

        meta = dict(req_group.metadata_json or {})
        if meta.get("request_status") == REQUEST_STATUS_FULFILLED and meta.get("linked_sweep_id"):
            return int(meta["linked_sweep_id"])

        progress = self.compute_progress(request_id)
        if progress["missing_count"] > 0:
            return None

        meta = dict(req_group.metadata_json or {})
        if meta.get("request_status") == REQUEST_STATUS_FULFILLED and meta.get("linked_sweep_id"):
            return int(meta["linked_sweep_id"])

        algorithm = str(meta.get("algorithm") or "unknown_algorithm")
        fixed_config = dict(meta.get("fixed_config") or {})
        parameter_axes = dict(meta.get("parameter_axes") or {})
        completed_run_ids = list(progress.get("completed_run_ids") or [])
        if not completed_run_ids:
            return None

        sweep_type = str(
            meta.get("sweep_type") or self._sweep_type_from_group_type(req_group.group_type) or "clustering"
        )
        default_sweep_name = sweep_name or f"{req_group.name}_sweep"
        description = (
            f"Fulfilled from request {request_id} ({req_group.name}): "
            f"{len(completed_run_ids)} runs"
        )
        sweep_id = self._create_sweep_group(
            request_id=request_id,
            sweep_name=default_sweep_name,
            algorithm=algorithm,
            fixed_config=fixed_config,
            parameter_axes=parameter_axes,
            completed_run_ids=completed_run_ids,
            sweep_type=sweep_type,
            description=description,
        )

        self.repository.create_group_link(
            parent_group_id=request_id,
            child_group_id=sweep_id,
            link_type="generates",
        )
        meta["request_status"] = REQUEST_STATUS_FULFILLED
        meta["fulfilled_at"] = _now_iso()
        meta["linked_sweep_id"] = int(sweep_id)
        meta["updated_at"] = _now_iso()
        req_group.metadata_json = meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()
        return int(sweep_id)

    def mark_analysis_completed(self, request_id: int, analysis_key: str) -> bool:
        """Mark analysis key as completed and recompute request analysis status."""
        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type not in self._request_group_types():
            return False
        meta = dict(req_group.metadata_json or {})
        completed = [str(x) for x in (meta.get("completed_analyses") or [])]
        if analysis_key not in completed:
            completed.append(analysis_key)
        failed_items = [
            x
            for x in (meta.get("failed_analyses") or [])
            if not (isinstance(x, dict) and str(x.get("analysis_key")) == analysis_key)
        ]
        meta["completed_analyses"] = completed
        meta["failed_analyses"] = failed_items
        meta["analysis_status"] = self._recompute_analysis_status(meta)
        meta["updated_at"] = _now_iso()
        req_group.metadata_json = meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()
        return True

    def mark_analysis_failed(self, request_id: int, analysis_key: str, error: str) -> bool:
        """Mark analysis key failed and keep failure details in request metadata."""
        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type not in self._request_group_types():
            return False
        meta = dict(req_group.metadata_json or {})
        failed = list(meta.get("failed_analyses") or [])
        failed = [
            x
            for x in failed
            if not (isinstance(x, dict) and str(x.get("analysis_key")) == analysis_key)
        ]
        failed.append(
            {
                "analysis_key": analysis_key,
                "error": str(error),
                "failed_at": _now_iso(),
            }
        )
        completed = [str(x) for x in (meta.get("completed_analyses") or [])]
        completed = [x for x in completed if x != analysis_key]
        meta["completed_analyses"] = completed
        meta["failed_analyses"] = failed
        meta["analysis_status"] = self._recompute_analysis_status(meta)
        meta["updated_at"] = _now_iso()
        req_group.metadata_json = meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()
        return True

    def record_analysis_result(
        self,
        *,
        request_id: int,
        source_group_id: int,
        analysis_key: str,
        result_key: str,
        result_value: Optional[float] = None,
        result_json: Optional[Dict[str, Any]] = None,
        orchestration_job_id: Optional[int] = None,
        mark_complete: bool = False,
    ) -> int:
        """
        Record analysis result in MethodService and unified provenanced run view.
        """
        req = self.get_request(request_id)
        if not req:
            raise ValueError(f"request_id={request_id} not found")
        analysis_map = self._analysis_catalog_to_dict(list(req.get("analysis_catalog") or []))
        spec = analysis_map.get(analysis_key) or {}
        method_name = str(spec.get("method_name") or analysis_key)
        method_version = str(spec.get("method_version") or "1.0")
        method = self.method_service.get_method(method_name, version=method_version)
        if method is None:
            method_id = self.method_service.register_method(
                name=method_name,
                version=method_version,
                description=f"Auto-registered for analysis_key={analysis_key}",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "analysis_key": {"type": "string"},
                        "request_id": {"type": "integer"},
                    },
                },
            )
        else:
            method_id = int(method.id)

        payload_json = dict(result_json or {})
        payload_json.setdefault(
            "parameters",
            {"analysis_key": analysis_key, "request_id": int(request_id)},
        )
        result_id = self.method_service.record_result(
            method_definition_id=method_id,
            source_group_id=int(source_group_id),
            result_key=result_key,
            result_value=result_value,
            result_json=payload_json,
        )
        if self.unified_execution_writes:
            self.provenanced_runs.record_analysis_execution(
                request_group_id=int(request_id),
                source_group_id=int(source_group_id),
                method_definition_id=method_id,
                analysis_key=analysis_key,
                config_json={"analysis_key": analysis_key, "result_key": result_key},
                metadata_json={
                    "result_id": int(result_id),
                    "analysis_key": str(analysis_key),
                    "result_key": str(result_key),
                    "execution_role": "analysis_execution",
                },
                orchestration_job_id=orchestration_job_id,
                run_status="completed",
            )
        if mark_complete:
            self.mark_analysis_completed(request_id, analysis_key)
        return int(result_id)

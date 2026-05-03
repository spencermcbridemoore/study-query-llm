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
    MCQ_DETERMINISM_CLASS,
    MCQ_HISTORICAL_BACKFILL_POLICY,
    MCQ_ORCHESTRATION_GRAPH_KIND,
    OrchestrationGraphSpec,
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

JOB_TYPE_ANALYSIS_RUN = "analysis_run"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SweepRequestService:
    """Typed sweep request lifecycle manager."""

    def __init__(self, repository: "RawCallRepository"):
        self.repository = repository
        self.provenance = ProvenanceService(repository)
        self.method_service = MethodService(repository)
        self.provenanced_runs = ProvenancedRunService(repository)
        self.use_derived_analysis_status = _env_flag("SQ_DERIVE_ANALYSIS_STATUS_READ", True)
        self.enable_analysis_jobs = _env_flag("SQ_ENABLE_ANALYSIS_JOBS", True)
        self.enable_clustering_analysis_jobs = _env_flag(
            "SQ_ENABLE_CLUSTERING_ANALYSIS_JOBS",
            False,
        )
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

    @staticmethod
    def _normalize_run_key_to_lineage_inputs(
        raw_mapping: Any,
        *,
        allowed_run_keys: Optional[set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if not isinstance(raw_mapping, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for raw_run_key, raw_payload in raw_mapping.items():
            run_key = str(raw_run_key or "").strip()
            if not run_key:
                continue
            if allowed_run_keys is not None and run_key not in allowed_run_keys:
                continue
            if not isinstance(raw_payload, dict):
                continue
            raw_snapshot_ids = raw_payload.get("dataset_snapshot_ids")
            if raw_snapshot_ids is None and raw_payload.get("dataset_snapshot_id") is not None:
                raw_snapshot_ids = [raw_payload.get("dataset_snapshot_id")]
            if raw_snapshot_ids is not None and not isinstance(raw_snapshot_ids, (list, tuple, set)):
                raw_snapshot_ids = [raw_snapshot_ids]
            snapshot_ids: List[int] = []
            for raw_snapshot_id in list(raw_snapshot_ids or []):
                try:
                    snapshot_ids.append(int(raw_snapshot_id))
                except (TypeError, ValueError):
                    continue
            snapshot_ids = sorted(set(snapshot_ids))

            embedding_batch_group_id: Optional[int] = None
            raw_embedding_group_id = raw_payload.get("embedding_batch_group_id")
            if raw_embedding_group_id is not None:
                try:
                    embedding_batch_group_id = int(raw_embedding_group_id)
                except (TypeError, ValueError):
                    embedding_batch_group_id = None

            normalized_payload: Dict[str, Any] = {}
            if snapshot_ids:
                normalized_payload["dataset_snapshot_ids"] = snapshot_ids
            if embedding_batch_group_id is not None:
                normalized_payload["embedding_batch_group_id"] = embedding_batch_group_id
            if normalized_payload:
                out[run_key] = normalized_payload
        return out

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

    def _enqueue_graph_spec(
        self,
        *,
        request_group_id: int,
        graph_spec: OrchestrationGraphSpec,
    ) -> int:
        job_id_by_key: Dict[str, int] = {}
        planned = 0
        for node in graph_spec.jobs:
            depends_on_job_ids: List[int] = []
            for dep_key in node.depends_on_job_keys:
                dep_id = job_id_by_key.get(dep_key)
                if dep_id is None:
                    raise ValueError(
                        f"Invalid orchestration graph: dependency {dep_key!r} "
                        f"must appear before {node.job_key!r}"
                    )
                depends_on_job_ids.append(dep_id)
            job_id = self.repository.enqueue_orchestration_job(
                request_group_id=request_group_id,
                job_type=node.job_type,
                job_key=node.job_key,
                base_run_key=node.base_run_key,
                payload_json=dict(node.payload_json or {}),
                max_attempts=int(node.max_attempts),
                seed_value=node.seed_value,
                depends_on_job_ids=depends_on_job_ids or None,
            )
            job_id_by_key[str(node.job_key)] = int(job_id)
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

        if sweep_type not in {SWEEP_TYPE_CLUSTERING, SWEEP_TYPE_MCQ}:
            return 0, metadata_json

        adapter = get_sweep_type_adapter(sweep_type)
        fixed_config = dict(metadata_json.get("fixed_config") or {})
        execution_mode = str(metadata_json.get("execution_mode") or "standalone").lower()
        shard_config = dict(metadata_json.get("shard_config") or {})
        analysis_catalog = list(metadata_json.get("analysis_catalog") or [])
        lineage_inputs_by_run_key = self._normalize_run_key_to_lineage_inputs(
            metadata_json.get("run_key_to_lineage_inputs"),
            allowed_run_keys=set(run_key_to_target.keys()),
        )
        enable_analysis_jobs = bool(self.enable_analysis_jobs)
        if sweep_type == SWEEP_TYPE_CLUSTERING:
            enable_analysis_jobs = enable_analysis_jobs and bool(
                self.enable_clustering_analysis_jobs
            )
        graph_spec = adapter.build_orchestration_graph(
            request_group_id=request_group_id,
            run_key_to_target=run_key_to_target,
            fixed_config=fixed_config,
            execution_mode=execution_mode,
            shard_config=shard_config,
            analysis_catalog=analysis_catalog,
            enable_analysis_jobs=enable_analysis_jobs,
            lineage_inputs_by_run_key=lineage_inputs_by_run_key,
        )
        planned = self._enqueue_graph_spec(
            request_group_id=request_group_id,
            graph_spec=graph_spec,
        )
        metadata_json.update(dict(graph_spec.metadata_updates or {}))
        metadata_json.setdefault("orchestration_graph_kind", str(graph_spec.graph_kind))
        if sweep_type == SWEEP_TYPE_MCQ:
            metadata_json.setdefault("historical_backfill_policy", MCQ_HISTORICAL_BACKFILL_POLICY)
            metadata_json.setdefault("determinism_class", MCQ_DETERMINISM_CLASS)
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
        run_key_to_lineage_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
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
        normalized_lineage_inputs = self._normalize_run_key_to_lineage_inputs(
            run_key_to_lineage_inputs,
            allowed_run_keys=set(expected_run_keys),
        )
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
        if normalized_lineage_inputs:
            metadata_json["run_key_to_lineage_inputs"] = normalized_lineage_inputs
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

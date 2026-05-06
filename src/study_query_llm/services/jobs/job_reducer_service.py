"""Reducer helpers for sharded orchestration jobs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.exc import DBAPIError, OperationalError

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.ingestion import ingest_result_to_db
from study_query_llm.services.sweep_request_service import SweepRequestService
from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# TODO(request-finalize-concurrency-fix:M2): remove race-compensation retry path
# once request-level single-writer finalization is enabled.
_M1_RACE_COMPENSATION_MAX_ATTEMPTS = 3
_M1_RACE_COMPENSATION_BASE_BACKOFF_SECONDS = 0.25


class _SweepLikeResult:
    """Small adapter to satisfy serialize/ingestion expectations."""

    def __init__(self, payload: Dict[str, Any]):
        self.pca = payload.get("pca", {})
        self.by_k = payload.get("by_k", {})
        self.Z = None
        self.Z_norm = None
        self.dist = None


class JobReducerService:
    """Executes reduce_k and finalize_run jobs."""

    def __init__(self, db: DatabaseConnectionV2, artifacts_dir: Optional[Path] = None):
        self.db = db
        self.artifacts_dir = artifacts_dir or (Path.cwd() / "experimental_results" / "job_shards")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_json(path: Path, payload: Dict[str, Any]) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        return str(path)

    def _get_job(self, repo: RawCallRepository, job_id: int):
        jobs = repo.list_orchestration_jobs()
        for job in jobs:
            if job.id == job_id:
                return job
        return None

    @staticmethod
    def _normalize_snapshot_ids(raw_snapshot_ids: Any) -> List[int]:
        if raw_snapshot_ids is None:
            return []
        if not isinstance(raw_snapshot_ids, (list, tuple, set)):
            raw_snapshot_ids = [raw_snapshot_ids]
        snapshot_ids: List[int] = []
        for raw_snapshot_id in raw_snapshot_ids:
            try:
                snapshot_ids.append(int(raw_snapshot_id))
            except (TypeError, ValueError):
                continue
        return sorted(set(snapshot_ids))

    @classmethod
    def _lineage_inputs_for_run(
        cls,
        *,
        request_metadata: Dict[str, Any],
        run_key: str,
    ) -> Dict[str, Any]:
        mapping = dict(request_metadata.get("run_key_to_lineage_inputs") or {})
        run_entry = mapping.get(str(run_key))
        if not isinstance(run_entry, dict):
            return {}

        snapshot_ids = cls._normalize_snapshot_ids(
            run_entry.get("dataset_snapshot_ids")
            if run_entry.get("dataset_snapshot_ids") is not None
            else run_entry.get("dataset_snapshot_id")
        )
        embedding_batch_group_id: Optional[int] = None
        raw_embedding_batch_group_id = run_entry.get("embedding_batch_group_id")
        if raw_embedding_batch_group_id is not None:
            try:
                embedding_batch_group_id = int(raw_embedding_batch_group_id)
            except (TypeError, ValueError):
                embedding_batch_group_id = None

        out: Dict[str, Any] = {}
        if snapshot_ids:
            out["dataset_snapshot_ids"] = snapshot_ids
        if embedding_batch_group_id is not None:
            out["embedding_batch_group_id"] = embedding_batch_group_id
        return out

    @staticmethod
    def _lookup_existing_run_id_by_run_key(
        repo: RawCallRepository,
        run_key: str,
    ) -> Optional[int]:
        run_key_text = str(run_key or "").strip()
        if not run_key_text:
            return None
        run_rows = (
            repo.session.query(Group)
            .filter(Group.group_type == "clustering_run")
            .all()
        )
        for row in run_rows:
            meta = dict(row.metadata_json or {})
            if str(meta.get("run_key") or "").strip() == run_key_text:
                return int(row.id)
        return None

    @staticmethod
    def _is_retryable_finalize_exception(exc: Exception) -> bool:
        lower_msg = str(exc).lower()
        if "deadlock" in lower_msg or "serialization" in lower_msg:
            return True
        if isinstance(exc, OperationalError):
            return True
        if isinstance(exc, DBAPIError):
            orig = getattr(exc, "orig", None)
            pgcode = str(getattr(orig, "pgcode", "") or "").strip()
            if pgcode in {"40P01", "40001"}:
                return True
        return False

    def reduce_k_job(self, job_id: int) -> str:
        """Combine tries for a K-range into one K aggregate artifact."""
        with self.db.session_scope() as session:
            repo = RawCallRepository(session)
            job = self._get_job(repo, job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            from study_query_llm.db.models_v2 import OrchestrationJobDependency
            dep_rows = session.query(OrchestrationJobDependency).filter_by(job_id=job.id).all()
            dep_ids = [d.depends_on_job_id for d in dep_rows]

            leaf_jobs = [j for j in repo.list_orchestration_jobs() if j.id in dep_ids]
            completed_refs = [j.result_ref for j in leaf_jobs if j.status == "completed" and j.result_ref]
            if not completed_refs:
                raise RuntimeError(f"reduce_k job {job_id} has no completed leaf artifacts")

            best = None
            best_obj = float("inf")
            labels_all: List[List[int]] = []
            objectives: List[float] = []
            tries_payload: List[Dict[str, Any]] = []
            k_key_resolved: Optional[str] = None

            for ref in sorted(completed_refs):
                payload = self._load_json(ref)
                by_k = payload.get("by_k") or {}
                if len(by_k) != 1:
                    raise RuntimeError(
                        f"reduce_k job {job_id}: expected exactly one k bucket per leaf shard "
                        f"(artifact {ref}); got {sorted(by_k.keys())!r}. "
                        "Non-singleton k_range shards violate reducer assumptions."
                    )
                k_key, k_payload = next(iter(by_k.items()))
                if k_key_resolved is None:
                    k_key_resolved = str(k_key)
                elif str(k_key) != str(k_key_resolved):
                    raise RuntimeError(
                        f"reduce_k job {job_id}: mismatched k keys across shards "
                        f"({k_key_resolved!r} vs {k_key!r}) for artifact {ref}"
                    )
                obj = float(k_payload.get("objective", 1e18))
                objectives.append(obj)
                labels = k_payload.get("labels", [])
                if labels:
                    labels_all.append(labels)
                if obj < best_obj:
                    best_obj = obj
                    best = payload
                tries_payload.append(
                    {
                        "try_idx": int(payload.get("try_idx", 0)),
                        "seed_value": payload.get("seed_value"),
                        "objective": k_payload.get("objective"),
                        "labels": k_payload.get("labels"),
                        "profiling": payload.get("profiling"),
                        "k_payload": dict(k_payload),
                    }
                )

            if best is None:
                raise RuntimeError(f"reduce_k job {job_id} failed to select best try")

            k_key = str(k_key_resolved or next(iter(best.get("by_k", {}).keys())))
            tries_payload.sort(key=lambda row: int(row.get("try_idx") or 0))
            out = {
                "pca": best.get("pca", {}),
                "by_k": {
                    k_key: {
                        **best["by_k"][k_key],
                        "objectives": objectives,
                        "labels_all": labels_all if len(labels_all) > 1 else None,
                        "tries": tries_payload,
                    }
                },
                "reduced_at": datetime.now(timezone.utc).isoformat(),
            }
            out_path = self.artifacts_dir / f"reduce_k_job_{job.id}.json"
            result_ref = self._save_json(out_path, out)
            repo.complete_orchestration_job(job.id, result_ref=result_ref)
            return result_ref

    def finalize_run_job(self, job_id: int) -> Optional[int]:
        """Combine reduced K shards into canonical clustering_run and deliver."""
        last_error: Optional[Exception] = None
        for attempt_idx in range(1, _M1_RACE_COMPENSATION_MAX_ATTEMPTS + 1):
            try:
                with self.db.session_scope() as session:
                    repo = RawCallRepository(session)
                    req_svc = SweepRequestService(repo)
                    job = self._get_job(repo, job_id)
                    if not job:
                        raise ValueError(f"Job not found: {job_id}")

                    from study_query_llm.db.models_v2 import OrchestrationJobDependency

                    dep_rows = (
                        session.query(OrchestrationJobDependency).filter_by(job_id=job.id).all()
                    )
                    dep_ids = [d.depends_on_job_id for d in dep_rows]
                    reducer_jobs = [
                        j for j in repo.list_orchestration_jobs() if j.id in dep_ids and j.result_ref
                    ]
                    if not reducer_jobs:
                        raise RuntimeError(f"finalize_run job {job_id} has no reduce_k artifacts")

                    merged_by_k: Dict[str, Any] = {}
                    pca_meta: Dict[str, Any] = {}
                    for rj in reducer_jobs:
                        payload = self._load_json(rj.result_ref)
                        pca_meta = payload.get("pca", pca_meta)
                        for k, kd in (payload.get("by_k") or {}).items():
                            merged_by_k[k] = kd

                    payload = {"pca": pca_meta, "by_k": merged_by_k}
                    out_path = self.artifacts_dir / f"finalize_run_job_{job.id}.json"
                    result_ref = self._save_json(out_path, payload)

                    run_key = str((job.payload_json or {}).get("run_key") or job.base_run_key or "")
                    metadata = {
                        "benchmark_source": (job.payload_json or {}).get("dataset", "unknown"),
                        "embedding_engine": (job.payload_json or {}).get("embedding_engine", "?"),
                        "n_restarts": int((job.payload_json or {}).get("tries_per_k", 1)),
                        "actual_entry_count": int((job.payload_json or {}).get("n_texts", 0)),
                        "request_group_id": int(job.request_group_id),
                        "determinism_class": "pseudo_deterministic",
                        "sweep_config": {
                            "k_min": min(int(k) for k in merged_by_k.keys()),
                            "k_max": max(int(k) for k in merged_by_k.keys()),
                        },
                        "source": "job_table_sharded_finalize",
                        "result_ref": result_ref,
                    }
                    request_group = repo.get_group_by_id(int(job.request_group_id))
                    request_metadata = dict((request_group.metadata_json or {}) if request_group else {})
                    lineage_inputs = self._lineage_inputs_for_run(
                        request_metadata=request_metadata,
                        run_key=run_key,
                    )
                    request_finalizer_enabled = bool(
                        request_metadata.get("request_finalizer_job_enabled")
                    )
                    if lineage_inputs:
                        metadata.update(lineage_inputs)
                    synthetic_result = _SweepLikeResult(payload)
                    ingested_run_id = ingest_result_to_db(
                        synthetic_result,
                        metadata=metadata,
                        ground_truth_labels=np.array([]),
                        db=self.db,
                        run_key=run_key,
                    )
                    resolved_run_id = (
                        int(ingested_run_id)
                        if ingested_run_id is not None
                        else self._lookup_existing_run_id_by_run_key(repo, run_key)
                    )
                    if resolved_run_id is None:
                        raise RuntimeError(
                            f"finalize_run_missing_run_id_for_delivery:{int(job.request_group_id)}:{run_key}"
                        )

                    delivery_ok = req_svc.record_delivery(
                        int(job.request_group_id),
                        int(resolved_run_id),
                        run_key,
                    )
                    if not delivery_ok:
                        raise RuntimeError(
                            f"finalize_run_delivery_link_failed:{int(job.request_group_id)}:{run_key}:{int(resolved_run_id)}"
                        )

                    if not request_finalizer_enabled:
                        req_svc.finalize_if_fulfilled(int(job.request_group_id))
                    # Mark completion only after delivery link + request finalization succeeded.
                    repo.complete_orchestration_job(job.id, result_ref=result_ref)
                    return int(resolved_run_id)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                retryable = self._is_retryable_finalize_exception(exc)
                if retryable and attempt_idx < _M1_RACE_COMPENSATION_MAX_ATTEMPTS:
                    sleep_seconds = _M1_RACE_COMPENSATION_BASE_BACKOFF_SECONDS * attempt_idx
                    logger.warning(
                        "finalize_run_job transient error; retrying (attempt=%s/%s, sleep=%.2fs, job_id=%s): %s",
                        attempt_idx,
                        _M1_RACE_COMPENSATION_MAX_ATTEMPTS,
                        sleep_seconds,
                        job_id,
                        exc,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise
        if last_error is not None:
            raise last_error
        return None

    def finalize_request_job(self, job_id: int) -> Optional[int]:
        """Single-writer request-level finalization job."""
        last_error: Optional[Exception] = None
        for attempt_idx in range(1, _M1_RACE_COMPENSATION_MAX_ATTEMPTS + 1):
            try:
                with self.db.session_scope() as session:
                    repo = RawCallRepository(session)
                    req_svc = SweepRequestService(repo)
                    job = self._get_job(repo, job_id)
                    if not job:
                        raise ValueError(f"Job not found: {job_id}")
                    payload = dict(job.payload_json or {})
                    request_id = int(payload.get("request_id") or job.request_group_id or 0)
                    if request_id <= 0:
                        raise RuntimeError(f"missing_request_id_for_finalize_request:{job_id}")
                    sweep_id = req_svc.finalize_if_fulfilled(request_id)
                    if sweep_id is None:
                        raise RuntimeError(f"request_finalize_not_ready:{request_id}")
                    repo.complete_orchestration_job(
                        int(job.id),
                        result_ref=f"sweep:{int(sweep_id)}",
                    )
                    return int(sweep_id)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                retryable = self._is_retryable_finalize_exception(exc)
                if retryable and attempt_idx < _M1_RACE_COMPENSATION_MAX_ATTEMPTS:
                    sleep_seconds = _M1_RACE_COMPENSATION_BASE_BACKOFF_SECONDS * attempt_idx
                    logger.warning(
                        "finalize_request_job transient error; retrying (attempt=%s/%s, sleep=%.2fs, job_id=%s): %s",
                        attempt_idx,
                        _M1_RACE_COMPENSATION_MAX_ATTEMPTS,
                        sleep_seconds,
                        job_id,
                        exc,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise
        if last_error is not None:
            raise last_error
        return None

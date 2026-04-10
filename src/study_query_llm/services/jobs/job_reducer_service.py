"""Reducer helpers for sharded orchestration jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.ingestion import ingest_result_to_db
from study_query_llm.services.sweep_request_service import SweepRequestService
from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)


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
            for ref in completed_refs:
                payload = self._load_json(ref)
                k_key = next(iter(payload.get("by_k", {}).keys()))
                k_payload = payload["by_k"][k_key]
                obj = float(k_payload.get("objective", 1e18))
                objectives.append(obj)
                labels = k_payload.get("labels", [])
                if labels:
                    labels_all.append(labels)
                if obj < best_obj:
                    best_obj = obj
                    best = payload

            if best is None:
                raise RuntimeError(f"reduce_k job {job_id} failed to select best try")

            k_key = next(iter(best.get("by_k", {}).keys()))
            out = {
                "pca": best.get("pca", {}),
                "by_k": {
                    k_key: {
                        **best["by_k"][k_key],
                        "objectives": objectives,
                        "labels_all": labels_all if len(labels_all) > 1 else None,
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
        with self.db.session_scope() as session:
            repo = RawCallRepository(session)
            req_svc = SweepRequestService(repo)
            job = self._get_job(repo, job_id)
            if not job:
                raise ValueError(f"Job not found: {job_id}")

            from study_query_llm.db.models_v2 import OrchestrationJobDependency
            dep_rows = session.query(OrchestrationJobDependency).filter_by(job_id=job.id).all()
            dep_ids = [d.depends_on_job_id for d in dep_rows]
            reducer_jobs = [j for j in repo.list_orchestration_jobs() if j.id in dep_ids and j.result_ref]
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

            run_key = (job.payload_json or {}).get("run_key") or job.base_run_key
            metadata = {
                "benchmark_source": (job.payload_json or {}).get("dataset", "unknown"),
                "embedding_engine": (job.payload_json or {}).get("embedding_engine", "?"),
                "summarizer": (job.payload_json or {}).get("summarizer", "None"),
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
            synthetic_result = _SweepLikeResult(payload)
            run_id = ingest_result_to_db(
                synthetic_result,
                metadata=metadata,
                ground_truth_labels=np.array([]),
                db=self.db,
                run_key=run_key,
            )
            if run_id:
                req_svc.record_delivery(job.request_group_id, run_id, run_key)
            repo.complete_orchestration_job(job.id, result_ref=result_ref)
            req_svc.finalize_if_fulfilled(job.request_group_id)
            return run_id

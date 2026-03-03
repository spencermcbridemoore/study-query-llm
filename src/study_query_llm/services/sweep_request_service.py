"""
Sweep Request Service — request/delivery lifecycle for clustering_sweep_request.

Manages creation, progress computation, delivery recording, and fulfillment
of sweep requests. Raw clustering_run data remains source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from sqlalchemy import text as sa_text
from sqlalchemy.orm.attributes import flag_modified

from study_query_llm.experiments.sweep_request_types import (
    REQUEST_SCHEMA_VERSION,
    REQUEST_STATUS_FULFILLED,
    REQUEST_STATUS_REQUESTED,
    REQUEST_STATUS_RUNNING,
    expand_parameter_axes,
    targets_to_run_keys,
)
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_CLUSTERING_RUN,
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
    ProvenanceService,
)
from study_query_llm.utils.logging_config import get_logger

if TYPE_CHECKING:
    from study_query_llm.db.raw_call_repository import RawCallRepository

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SweepRequestService:
    """Service for clustering_sweep_request lifecycle: create, deliver, fulfill."""

    def __init__(self, repository: "RawCallRepository"):
        self.repository = repository
        self.provenance = ProvenanceService(repository)

    def create_request(
        self,
        request_name: str,
        algorithm: str,
        fixed_config: Dict[str, Any],
        parameter_axes: Dict[str, Any],
        entry_max: int,
        n_restarts_suffix: str = "50runs",
        description: Optional[str] = None,
    ) -> int:
        """Create a clustering_sweep_request group with expanded run targets.

        Args:
            request_name: Human-readable name for the request.
            algorithm: Algorithm identifier (e.g. "cosine_kllmeans_no_pca").
            fixed_config: Shared config across all runs.
            parameter_axes: Dict with keys datasets, embedding_engines, summarizers.
            entry_max: Sample size (e.g. 300).
            n_restarts_suffix: Suffix for run_key (e.g. "50runs").
            description: Optional description.

        Returns:
            Group ID of the created clustering_sweep_request.
        """
        targets = expand_parameter_axes(
            parameter_axes,
            entry_max=entry_max,
            n_restarts_suffix=n_restarts_suffix,
        )
        expected_run_keys = targets_to_run_keys(targets)
        run_key_to_target = {
            rk: {
                "dataset": t.dataset,
                "embedding_engine": t.embedding_engine,
                "summarizer": t.summarizer,
            }
            for t, rk in zip(targets, expected_run_keys)
        }

        metadata_json = {
            "request_schema_version": REQUEST_SCHEMA_VERSION,
            "request_status": REQUEST_STATUS_REQUESTED,
            "algorithm": algorithm,
            "fixed_config": fixed_config,
            "parameter_axes": parameter_axes,
            "entry_max": entry_max,
            "n_restarts_suffix": n_restarts_suffix,
            "expected_run_keys": expected_run_keys,
            "run_key_to_target": run_key_to_target,
            "expected_count": len(expected_run_keys),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }

        group_id = self.repository.create_group(
            group_type=GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
            name=request_name,
            description=description or f"Sweep request: {request_name} ({algorithm})",
            metadata_json=metadata_json,
        )

        logger.info(
            f"Created sweep request: id={group_id}, name={request_name}, "
            f"expected_count={len(expected_run_keys)}"
        )
        return group_id

    def get_request(self, request_id: int) -> Optional[Dict[str, Any]]:
        """Get request group by ID. Returns None if not found or wrong type."""
        group = self.repository.get_group_by_id(request_id)
        if not group or group.group_type != GROUP_TYPE_CLUSTERING_SWEEP_REQUEST:
            return None
        meta = group.metadata_json or {}
        return {
            "id": group.id,
            "name": group.name,
            "description": group.description,
            "created_at": group.created_at,
            **meta,
        }

    def _is_sqlite(self) -> bool:
        """Return True if the session uses SQLite (for dialect-specific JSON handling)."""
        try:
            return self.repository.session.get_bind().dialect.name == "sqlite"
        except Exception:
            return False

    def list_requests(
        self,
        status: Optional[str] = None,
        include_fulfilled: bool = True,
    ) -> List[Dict[str, Any]]:
        """List clustering_sweep_request groups, optionally filtered by status."""
        session = self.repository.session
        from study_query_llm.db.models_v2 import Group

        query = session.query(Group).filter(
            Group.group_type == GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
        )
        if status or (not include_fulfilled):
            if self._is_sqlite():
                # SQLite: filter in Python (no metadata_json->>'key' support)
                groups = query.order_by(Group.created_at.desc()).all()
                if status:
                    groups = [g for g in groups if (g.metadata_json or {}).get("request_status") == status]
                elif not include_fulfilled:
                    groups = [g for g in groups if (g.metadata_json or {}).get("request_status") != REQUEST_STATUS_FULFILLED]
            else:
                if status:
                    query = query.filter(
                        sa_text("metadata_json->>'request_status' = :st"),
                    ).params(st=status)
                elif not include_fulfilled:
                    query = query.filter(
                        sa_text("metadata_json->>'request_status' != :st"),
                    ).params(st=REQUEST_STATUS_FULFILLED)
                groups = query.order_by(Group.created_at.desc()).all()
        else:
            groups = query.order_by(Group.created_at.desc()).all()
        return [
            {
                "id": g.id,
                "name": g.name,
                "request_status": (g.metadata_json or {}).get("request_status", "?"),
                "expected_count": (g.metadata_json or {}).get("expected_count", 0),
                "created_at": g.created_at,
            }
            for g in groups
        ]

    def compute_progress(self, request_id: int) -> Dict[str, Any]:
        """Compute delivery progress from DB state.

        Returns:
            expected_count, completed_count, missing_count,
            completed_run_keys, missing_run_keys, completed_run_ids
        """
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

        expected_keys = req.get("expected_run_keys") or []
        if not expected_keys:
            return {
                "expected_count": 0,
                "completed_count": 0,
                "missing_count": 0,
                "completed_run_keys": [],
                "missing_run_keys": expected_keys,
                "completed_run_ids": [],
            }

        session = self.repository.session
        from study_query_llm.db.models_v2 import Group

        completed_run_ids: List[int] = []
        completed_run_keys: List[str] = []
        expected_set = set(expected_keys)

        if self._is_sqlite():
            # SQLite: fetch all clustering_run and filter by metadata_json in Python
            all_runs = session.query(Group).filter(
                Group.group_type == GROUP_TYPE_CLUSTERING_RUN,
            ).all()
            for run in all_runs:
                rk = (run.metadata_json or {}).get("run_key")
                if rk and rk in expected_set:
                    completed_run_ids.append(run.id)
                    completed_run_keys.append(rk)
        else:
            # PostgreSQL: use metadata_json->>'run_key' for indexed lookup
            for rk in expected_keys:
                existing = session.query(Group).filter(
                    Group.group_type == GROUP_TYPE_CLUSTERING_RUN,
                    sa_text("metadata_json->>'run_key' = :rk"),
                ).params(rk=rk).first()
                if existing:
                    completed_run_ids.append(existing.id)
                    completed_run_keys.append(rk)

        missing_run_keys = [k for k in expected_keys if k not in set(completed_run_keys)]

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
        """Idempotently link a delivered run to the request.

        Creates GroupLink(request_id -> run_id, link_type="contains").
        Updates metadata counters and updated_at.

        Returns:
            True if link was created or already existed, False if request invalid.
        """
        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type != GROUP_TYPE_CLUSTERING_SWEEP_REQUEST:
            logger.warning(f"Invalid request_id for record_delivery: {request_id}")
            return False

        run_group = self.repository.get_group_by_id(run_id)
        if not run_group or run_group.group_type != GROUP_TYPE_CLUSTERING_RUN:
            logger.warning(f"Invalid run_id for record_delivery: {run_id}")
            return False

        # Idempotent link
        self.repository.create_group_link(
            parent_group_id=request_id,
            child_group_id=run_id,
            link_type="contains",
            metadata_json={"run_key": run_key},
        )

        # Update metadata
        meta = dict(req_group.metadata_json or {})
        meta["updated_at"] = _now_iso()
        meta["request_status"] = REQUEST_STATUS_RUNNING
        req_group.metadata_json = meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()

        logger.debug(f"Recorded delivery: request={request_id} run={run_id} run_key={run_key}")
        return True

    def finalize_if_fulfilled(
        self,
        request_id: int,
        sweep_name: Optional[str] = None,
    ) -> Optional[int]:
        """If all expected runs are delivered, create clustering_sweep and mark fulfilled.

        - Creates clustering_sweep with runs linked via contains
        - Links request -> sweep via generates
        - Sets request_status=fulfilled, fulfilled_at, linked_sweep_id

        Returns:
            Sweep group ID if fulfilled, None if not yet fulfilled or request invalid.
        """
        progress = self.compute_progress(request_id)
        if progress["missing_count"] > 0:
            logger.debug(
                f"Request {request_id} not fulfilled: "
                f"{progress['missing_count']} runs still missing"
            )
            return None

        req_group = self.repository.get_group_by_id(request_id)
        if not req_group or req_group.group_type != GROUP_TYPE_CLUSTERING_SWEEP_REQUEST:
            return None

        meta = req_group.metadata_json or {}
        algorithm = meta.get("algorithm", "cosine_kllmeans_no_pca")
        fixed_config = meta.get("fixed_config", {})
        parameter_axes = meta.get("parameter_axes", {})
        completed_run_ids = progress["completed_run_ids"]

        if not completed_run_ids:
            return None

        default_sweep_name = sweep_name or f"{req_group.name}_sweep"
        description = (
            f"Fulfilled from request {request_id} ({req_group.name}): "
            f"{len(completed_run_ids)} runs"
        )

        sweep_id = self.provenance.create_clustering_sweep_group(
            sweep_name=default_sweep_name,
            algorithm=algorithm,
            fixed_config=fixed_config,
            parameter_axes=parameter_axes,
            description=description,
        )

        for pos, run_id in enumerate(completed_run_ids):
            self.provenance.link_run_to_clustering_sweep(
                sweep_id=sweep_id,
                run_id=run_id,
                position=pos,
            )

        # Link request -> sweep (generates)
        self.repository.create_group_link(
            parent_group_id=request_id,
            child_group_id=sweep_id,
            link_type="generates",
        )

        # Mark request fulfilled
        meta["request_status"] = REQUEST_STATUS_FULFILLED
        meta["fulfilled_at"] = _now_iso()
        meta["linked_sweep_id"] = sweep_id
        meta["updated_at"] = _now_iso()
        req_group.metadata_json = meta
        flag_modified(req_group, "metadata_json")
        self.repository.session.flush()

        logger.info(
            f"Fulfilled request {request_id} -> sweep {sweep_id} "
            f"({len(completed_run_ids)} runs)"
        )
        return sweep_id

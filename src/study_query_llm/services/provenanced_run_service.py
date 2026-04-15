"""
Unified execution-provenance service for method and analysis runs.

`provenanced_runs` is the first-class execution object. Method identity remains
in MethodDefinition and links into each execution row by method_definition_id.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple

from study_query_llm.db.models_v2 import AnalysisResult, Group, GroupLink, MethodDefinition
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

RUN_KIND_EXECUTION = "execution"
EXECUTION_ROLE_METHOD = "method_execution"
EXECUTION_ROLE_ANALYSIS = "analysis_execution"

# Backward-compatible aliases retained for callers/tests.
RUN_KIND_METHOD_EXECUTION = EXECUTION_ROLE_METHOD
RUN_KIND_ANALYSIS_EXECUTION = EXECUTION_ROLE_ANALYSIS

RUN_STATUS_CREATED = "created"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


def canonical_config_hash(config_json: Optional[Dict[str, Any]]) -> str:
    """Return deterministic hash for configuration payloads."""
    payload = config_json or {}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Keys in config_json that describe scheduling/runtime, not algorithmic identity.
_SCHEDULING_ONLY_CONFIG_KEYS = frozenset({
    "max_attempts",
    "job_key",
    "job_id",
    "orchestration_job_id",
    "claimed_by",
    "lease_seconds",
    "lease_expires_at",
    "heartbeat_at",
    "worker_id",
    "worker_slot",
    "claim_lease_seconds",
    "idle_exit_seconds",
    "bundle_size",
    "execution_mode",
    "shard_config",
    "standalone_special_case",
    "concurrency_observed",
})


def _strip_scheduling_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return config dict with scheduling-only keys removed."""
    return {k: v for k, v in config.items() if k not in _SCHEDULING_ONLY_CONFIG_KEYS}


def canonical_run_fingerprint(
    *,
    method_name: Optional[str] = None,
    method_version: Optional[str] = None,
    config_json: Optional[Dict[str, Any]] = None,
    input_snapshot_group_id: Optional[int] = None,
    manifest_hash: Optional[str] = None,
    data_regime: Optional[Dict[str, Any]] = None,
    determinism_class: str = "non_deterministic",
) -> Tuple[Dict[str, Any], str]:
    """Build a canonical run fingerprint for semantic comparability.

    The fingerprint captures *algorithmic identity* (what method, what config,
    what input, what data regime) and deliberately excludes scheduling mechanics
    (job ids, worker ids, claim timestamps, retries, lease info, fan-out shape).

    Returns:
        ``(fingerprint_json, fingerprint_hash)`` where *fingerprint_json* is the
        structured tuple and *fingerprint_hash* is its SHA-256 digest.
    """
    semantic_config = _strip_scheduling_keys(config_json or {})

    fp: Dict[str, Any] = {
        "method_name": method_name or None,
        "method_version": method_version or None,
        "config_hash": canonical_config_hash(semantic_config),
        "input_snapshot_group_id": input_snapshot_group_id,
        "manifest_hash": manifest_hash or None,
        "data_regime": data_regime or None,
        "determinism_class": determinism_class,
    }
    canonical = json.dumps(fp, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fp_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return fp, fp_hash


def fingerprints_match(
    fp_a: Optional[Dict[str, Any]],
    fp_b: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    """Compare two fingerprint dicts and return ``(match, diff)``.

    ``diff`` maps each field name to ``{"a": val_a, "b": val_b}`` for fields
    that differ.  When both fingerprints are ``None``, returns ``(True, {})``.
    """
    if fp_a is None and fp_b is None:
        return True, {}
    if fp_a is None or fp_b is None:
        return False, {"_present": {"a": fp_a is not None, "b": fp_b is not None}}
    all_keys = sorted(set(fp_a) | set(fp_b))
    diff: Dict[str, Any] = {}
    for key in all_keys:
        va = fp_a.get(key)
        vb = fp_b.get(key)
        if va != vb:
            diff[key] = {"a": va, "b": vb}
    return len(diff) == 0, diff


def _primary_snapshot_id_from_run_metadata(meta: Dict[str, Any]) -> Optional[int]:
    """Return deterministic primary snapshot id from legacy run metadata."""
    raw_snapshot_ids = meta.get("dataset_snapshot_ids")
    if raw_snapshot_ids is None and meta.get("dataset_snapshot_id") is not None:
        raw_snapshot_ids = [meta.get("dataset_snapshot_id")]
    if raw_snapshot_ids is None:
        return None
    if not isinstance(raw_snapshot_ids, (list, tuple, set)):
        raw_snapshot_ids = [raw_snapshot_ids]
    snapshot_ids: List[int] = []
    for sid in raw_snapshot_ids:
        try:
            snapshot_ids.append(int(sid))
        except (TypeError, ValueError):
            continue
    if not snapshot_ids:
        return None
    return sorted(set(snapshot_ids))[0]


def _execution_role_from_row(row: Dict[str, Any]) -> str:
    """Resolve execution role from canonical/legacy row shape."""
    meta = dict(row.get("metadata_json") or {})
    role = str(meta.get("execution_role") or "").strip().lower()
    if role in (EXECUTION_ROLE_METHOD, EXECUTION_ROLE_ANALYSIS):
        return role
    raw_kind = str(row.get("run_kind") or "").strip().lower()
    if raw_kind in (EXECUTION_ROLE_METHOD, EXECUTION_ROLE_ANALYSIS):
        return raw_kind
    return EXECUTION_ROLE_METHOD


class ProvenancedRunService:
    """Service wrapper for creating and querying unified execution runs."""

    def __init__(self, repository: RawCallRepository):
        self.repository = repository

    def record_method_execution(
        self,
        *,
        request_group_id: int,
        run_key: str,
        source_group_id: int,
        method_definition_id: Optional[int],
        config_json: Optional[Dict[str, Any]] = None,
        determinism_class: str = "non_deterministic",
        input_snapshot_group_id: Optional[int] = None,
        orchestration_job_id: Optional[int] = None,
        result_group_id: Optional[int] = None,
        result_ref: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        run_status: str = RUN_STATUS_COMPLETED,
    ) -> int:
        """Upsert a method execution run keyed by request+run_key+run_kind."""
        meta = dict(metadata_json or {})
        meta.setdefault("execution_role", EXECUTION_ROLE_METHOD)

        method_name, method_version = self._resolve_method_identity(method_definition_id)
        fp_json, fp_hash = canonical_run_fingerprint(
            method_name=method_name,
            method_version=method_version,
            config_json=config_json,
            input_snapshot_group_id=input_snapshot_group_id,
            manifest_hash=meta.get("manifest_hash"),
            data_regime=meta.get("data_regime"),
            determinism_class=determinism_class,
        )

        return self.repository.create_provenanced_run(
            run_kind=RUN_KIND_EXECUTION,
            run_status=run_status,
            request_group_id=request_group_id,
            source_group_id=source_group_id,
            result_group_id=result_group_id,
            input_snapshot_group_id=input_snapshot_group_id,
            method_definition_id=method_definition_id,
            orchestration_job_id=orchestration_job_id,
            run_key=run_key,
            determinism_class=determinism_class,
            config_hash=canonical_config_hash(config_json),
            config_json=config_json or {},
            result_ref=result_ref,
            metadata_json=meta,
            fingerprint_json=fp_json,
            fingerprint_hash=fp_hash,
        )

    def record_analysis_execution(
        self,
        *,
        request_group_id: int,
        source_group_id: int,
        method_definition_id: Optional[int],
        analysis_key: str,
        analysis_run_key: Optional[str] = None,
        result_ref: Optional[str] = None,
        config_json: Optional[Dict[str, Any]] = None,
        determinism_class: str = "deterministic",
        metadata_json: Optional[Dict[str, Any]] = None,
        run_status: str = RUN_STATUS_COMPLETED,
        orchestration_job_id: Optional[int] = None,
    ) -> int:
        """Create an analysis execution run linked to method definition/spec."""
        synthetic_run_key = analysis_run_key or f"{analysis_key}:{source_group_id}"
        meta = dict(metadata_json or {})
        meta.setdefault("analysis_key", str(analysis_key))
        meta.setdefault("execution_role", EXECUTION_ROLE_ANALYSIS)

        method_name, method_version = self._resolve_method_identity(method_definition_id)
        fp_json, fp_hash = canonical_run_fingerprint(
            method_name=method_name,
            method_version=method_version,
            config_json=config_json,
            manifest_hash=meta.get("manifest_hash"),
            data_regime=meta.get("data_regime"),
            determinism_class=determinism_class,
        )

        return self.repository.create_provenanced_run(
            run_kind=RUN_KIND_EXECUTION,
            run_status=run_status,
            request_group_id=request_group_id,
            source_group_id=source_group_id,
            method_definition_id=method_definition_id,
            orchestration_job_id=orchestration_job_id,
            run_key=synthetic_run_key,
            determinism_class=determinism_class,
            config_hash=canonical_config_hash(config_json),
            config_json=config_json or {},
            result_ref=result_ref,
            metadata_json=meta,
            fingerprint_json=fp_json,
            fingerprint_hash=fp_hash,
        )

    def _resolve_method_identity(
        self, method_definition_id: Optional[int]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Look up method name/version from a definition id."""
        if method_definition_id is None:
            return None, None
        method = (
            self.repository.session.query(MethodDefinition)
            .filter(MethodDefinition.id == method_definition_id)
            .first()
        )
        if method is None:
            return None, None
        return str(method.name), str(method.version) if method.version else None

    def compare_run_fingerprints(
        self, run_a_id: int, run_b_id: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Compare fingerprints of two provenanced_runs by id.

        Returns ``(match, diff)`` — see :func:`fingerprints_match`.
        """
        from study_query_llm.db.models_v2 import ProvenancedRun

        run_a = self.repository.session.query(ProvenancedRun).filter_by(id=run_a_id).first()
        run_b = self.repository.session.query(ProvenancedRun).filter_by(id=run_b_id).first()
        fp_a = run_a.fingerprint_json if run_a else None
        fp_b = run_b.fingerprint_json if run_b else None
        return fingerprints_match(fp_a, fp_b)

    def list_unified_execution_view(self, request_group_id: int) -> List[Dict[str, Any]]:
        """
        Return unified execution rows with compatibility fallback.

        The result includes persisted `provenanced_runs` rows plus synthetic rows
        for legacy clustering/analysis outputs that predate this contract.
        """
        persisted = self.repository.list_provenanced_runs(request_group_id=request_group_id)
        rows: List[Dict[str, Any]] = [run.to_dict() for run in persisted]
        seen: Set[Tuple[str, str, int]] = set()
        for row in rows:
            role = _execution_role_from_row(row)
            row["execution_role"] = role
            if str(row.get("run_kind") or "") != RUN_KIND_EXECUTION:
                row["legacy_run_kind"] = row.get("run_kind")
            row["run_kind"] = RUN_KIND_EXECUTION
            seen.add(
                (
                    role,
                    str(row.get("run_key") or ""),
                    int(row.get("source_group_id") or 0),
                )
            )

        request_group = self.repository.get_group_by_id(request_group_id)
        if request_group is None:
            return rows
        request_meta = dict(request_group.metadata_json or {})
        expected_keys = list(request_meta.get("expected_run_keys") or [])
        expected_set = set(expected_keys)
        sweep_type = str(request_meta.get("sweep_type") or "clustering").lower()
        run_group_type = "mcq_run" if sweep_type == "mcq" else "clustering_run"

        legacy_run_groups = (
            self.repository.session.query(Group).filter(Group.group_type == run_group_type).all()
        )
        linked_primary_snapshot_by_run_id: Dict[int, int] = {}
        if run_group_type == "clustering_run":
            legacy_run_ids = [int(grp.id) for grp in legacy_run_groups]
            if legacy_run_ids:
                # Resolve a deterministic primary snapshot id from explicit run->snapshot links.
                linked_snapshot_rows = (
                    self.repository.session.query(
                        GroupLink.parent_group_id,
                        GroupLink.child_group_id,
                    )
                    .join(Group, Group.id == GroupLink.child_group_id)
                    .filter(
                        GroupLink.parent_group_id.in_(legacy_run_ids),
                        GroupLink.link_type == "depends_on",
                        Group.group_type == "dataset_snapshot",
                    )
                    .all()
                )
                for parent_group_id, child_group_id in linked_snapshot_rows:
                    parent_id = int(parent_group_id)
                    child_id = int(child_group_id)
                    current = linked_primary_snapshot_by_run_id.get(parent_id)
                    if current is None or child_id < current:
                        linked_primary_snapshot_by_run_id[parent_id] = child_id
        source_group_ids: Set[int] = set()
        for grp in legacy_run_groups:
            meta = dict(grp.metadata_json or {})
            run_key = str(meta.get("run_key") or "")
            if not run_key:
                continue
            if expected_set and run_key not in expected_set:
                continue
            primary_snapshot_id = _primary_snapshot_id_from_run_metadata(meta)
            if primary_snapshot_id is None:
                primary_snapshot_id = linked_primary_snapshot_by_run_id.get(int(grp.id))
            signature = (EXECUTION_ROLE_METHOD, run_key, int(grp.id))
            source_group_ids.add(int(grp.id))
            if signature in seen:
                continue
            rows.append(
                {
                    "id": None,
                    "run_kind": RUN_KIND_EXECUTION,
                    "run_status": RUN_STATUS_COMPLETED,
                    "request_group_id": request_group_id,
                    "source_group_id": int(grp.id),
                    "result_group_id": int(grp.id),
                    "input_snapshot_group_id": primary_snapshot_id,
                    "method_definition_id": None,
                    "orchestration_job_id": None,
                    "parent_provenanced_run_id": None,
                    "run_key": run_key,
                    "determinism_class": str(
                        meta.get("determinism_class") or "non_deterministic"
                    ),
                    "config_hash": None,
                    "config_json": {},
                    "result_ref": None,
                    "metadata_json": {
                        "compatibility_mapped": True,
                        "execution_role": EXECUTION_ROLE_METHOD,
                    },
                    "execution_role": EXECUTION_ROLE_METHOD,
                    "created_at": grp.created_at.isoformat() if grp.created_at else None,
                    "updated_at": grp.created_at.isoformat() if grp.created_at else None,
                }
            )
            seen.add(signature)

        if source_group_ids:
            analysis_rows = (
                self.repository.session.query(AnalysisResult)
                .filter(AnalysisResult.source_group_id.in_(source_group_ids))
                .all()
            )
            method_ids = {row.method_definition_id for row in analysis_rows if row.method_definition_id}
            method_rows = (
                self.repository.session.query(MethodDefinition)
                .filter(MethodDefinition.id.in_(method_ids))
                .all()
                if method_ids
                else []
            )
            method_name_by_id = {m.id: m.name for m in method_rows}
            for analysis in analysis_rows:
                method_name = str(method_name_by_id.get(analysis.method_definition_id) or "analysis")
                synthetic_run_key = f"{method_name}:{analysis.result_key}:{analysis.id}"
                signature = (EXECUTION_ROLE_ANALYSIS, synthetic_run_key, int(analysis.source_group_id))
                if signature in seen:
                    continue
                rows.append(
                    {
                        "id": None,
                        "run_kind": RUN_KIND_EXECUTION,
                        "run_status": RUN_STATUS_COMPLETED,
                        "request_group_id": request_group_id,
                        "source_group_id": int(analysis.source_group_id),
                        "result_group_id": analysis.analysis_group_id,
                        "input_snapshot_group_id": None,
                        "method_definition_id": analysis.method_definition_id,
                        "orchestration_job_id": None,
                        "parent_provenanced_run_id": None,
                        "run_key": synthetic_run_key,
                        "determinism_class": "deterministic",
                        "config_hash": None,
                        "config_json": {},
                        "result_ref": None,
                        "metadata_json": {
                            "result_key": analysis.result_key,
                            "compatibility_mapped": True,
                            "execution_role": EXECUTION_ROLE_ANALYSIS,
                        },
                        "execution_role": EXECUTION_ROLE_ANALYSIS,
                        "created_at": analysis.created_at.isoformat()
                        if analysis.created_at
                        else None,
                        "updated_at": analysis.created_at.isoformat()
                        if analysis.created_at
                        else None,
                    }
                )
                seen.add(signature)

        rows.sort(
            key=lambda r: (
                r.get("created_at") or "",
                r.get("execution_role") or "",
                r.get("run_key") or "",
            )
        )
        return rows

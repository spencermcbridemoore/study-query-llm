"""Stage 4: analyze (input group -> analysis_run + provenanced_run + results)."""

from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pyarrow.parquet as pq

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import AnalysisResult, CallArtifact, Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import ProvenanceService

ARTIFACT_TYPE_EMBEDDING_MATRIX = "embedding_matrix"
ARTIFACT_TYPE_DATASET_SNAPSHOT_PARQUET = "dataset_snapshot_parquet"
ARTIFACT_TYPE_ANALYSIS_RESULT_JSON = "analysis_result_json"
ARTIFACT_TYPE_ANALYSIS_RESULT_BLOB = "analysis_result_blob"

_ANALYZE_LOCK_GUARD = threading.Lock()
_ANALYZE_LOCKS: dict[str, threading.Lock] = {}


@dataclass
class AnalysisPayload:
    """Normalized method output payload for analysis stage persistence."""

    scalar_results: dict[str, float]
    structured_results: dict[str, Any]
    artifacts: dict[str, bytes]
    result_ref: str | None = None


AnalysisRunner = Callable[..., AnalysisPayload | Mapping[str, Any]]


def _resolve_db(
    *,
    db: DatabaseConnectionV2 | None,
    database_url: str | None,
) -> tuple[DatabaseConnectionV2, bool]:
    if db is not None:
        return db, False
    resolved = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not resolved:
        raise ValueError("database_url or DATABASE_URL is required when db is not provided")
    created = DatabaseConnectionV2(resolved, enable_pgvector=False)
    created.init_db()
    return created, True


def _analysis_lock(lock_key: str) -> threading.Lock:
    with _ANALYZE_LOCK_GUARD:
        if lock_key not in _ANALYZE_LOCKS:
            _ANALYZE_LOCKS[lock_key] = threading.Lock()
        return _ANALYZE_LOCKS[lock_key]


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _coerce_artifact_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, payload)
        return buf.getvalue()
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _coerce_payload(raw: AnalysisPayload | Mapping[str, Any]) -> AnalysisPayload:
    if isinstance(raw, AnalysisPayload):
        return raw
    scalar = {
        str(k): float(v)
        for k, v in dict(raw.get("scalar_results") or {}).items()
    }
    structured = {str(k): v for k, v in dict(raw.get("structured_results") or {}).items()}
    artifacts = {
        str(k): _coerce_artifact_bytes(v)
        for k, v in dict(raw.get("artifacts") or {}).items()
    }
    result_ref = raw.get("result_ref")
    return AnalysisPayload(
        scalar_results=scalar,
        structured_results=structured,
        artifacts=artifacts,
        result_ref=str(result_ref) if result_ref is not None else None,
    )


def _artifact_type_and_content_type(filename: str) -> tuple[str, str]:
    lowered = filename.lower()
    if lowered.endswith(".json"):
        return ARTIFACT_TYPE_ANALYSIS_RESULT_JSON, "application/json"
    if lowered.endswith(".npy"):
        return ARTIFACT_TYPE_ANALYSIS_RESULT_BLOB, "application/octet-stream"
    if lowered.endswith(".txt"):
        return ARTIFACT_TYPE_ANALYSIS_RESULT_BLOB, "text/plain"
    return ARTIFACT_TYPE_ANALYSIS_RESULT_BLOB, "application/octet-stream"


def _collect_analysis_artifact_uris(session, analysis_group_id: int) -> dict[str, str]:
    artifact_uris: dict[str, str] = {}
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) != int(analysis_group_id):
            continue
        filename = str(metadata.get("logical_filename") or f"artifact_{artifact.id}")
        artifact_uris[filename] = str(artifact.uri)
    return artifact_uris


def _require_group(session, group_id: int) -> Group:
    row = session.query(Group).filter(Group.id == int(group_id)).first()
    if row is None:
        raise ValueError(f"group id={group_id} not found")
    return row


def _latest_artifact_uri_for_group(
    session,
    *,
    group_id: int,
    artifact_type: str,
) -> str | None:
    artifacts = (
        session.query(CallArtifact)
        .filter(CallArtifact.artifact_type == artifact_type)
        .order_by(CallArtifact.id.desc())
        .all()
    )
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) == int(group_id):
            return str(artifact.uri)
    return None


def _load_embedding_matrix(
    session,
    *,
    input_group_id: int,
    artifact_dir: str,
) -> np.ndarray | None:
    uri = _latest_artifact_uri_for_group(
        session,
        group_id=int(input_group_id),
        artifact_type=ARTIFACT_TYPE_EMBEDDING_MATRIX,
    )
    if uri is None:
        return None
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    return np.asarray(artifact_service.load_artifact(uri, ARTIFACT_TYPE_EMBEDDING_MATRIX))


def _infer_snapshot_group_id(session, *, input_group_id: int) -> int | None:
    input_group = _require_group(session, int(input_group_id))
    metadata = dict(input_group.metadata_json or {})
    from_metadata = metadata.get("source_snapshot_group_id")
    if from_metadata is not None:
        return int(from_metadata)

    links = (
        session.query(GroupLink)
        .filter(
            GroupLink.parent_group_id == int(input_group_id),
            GroupLink.link_type == "depends_on",
        )
        .all()
    )
    for link in links:
        child = _require_group(session, int(link.child_group_id))
        if child.group_type == "dataset_snapshot":
            return int(child.id)
    return None


def _load_snapshot_texts(
    session,
    *,
    snapshot_group_id: int,
    artifact_dir: str,
) -> list[str]:
    parquet_uri = _latest_artifact_uri_for_group(
        session,
        group_id=int(snapshot_group_id),
        artifact_type=ARTIFACT_TYPE_DATASET_SNAPSHOT_PARQUET,
    )
    if parquet_uri is None:
        return []
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(parquet_uri)
    table = pq.read_table(io.BytesIO(payload), columns=["text"])
    return [str(value) for value in table.column("text").to_pylist()]


def _resolve_request_group_id(
    repo: RawCallRepository,
    *,
    request_group_id: int | None,
    method_name: str,
    input_group_id: int,
    run_key: str,
) -> int:
    if request_group_id is not None:
        return int(request_group_id)
    provenance = ProvenanceService(repo)
    return provenance.create_analysis_request_group(
        method_name=method_name,
        input_id=int(input_group_id),
        run_key=run_key,
    )


def _resolve_method_definition_id(
    method_service: MethodService,
    *,
    method_name: str,
    method_version: str | None,
) -> int:
    method_row = method_service.get_method(method_name, version=method_version)
    if method_row is None and method_version is None:
        method_row = method_service.get_method(method_name)
    if method_row is None:
        registered_id = method_service.register_method(
            name=method_name,
            version=method_version or "v1",
            code_ref="study_query_llm.pipeline.analyze",
            description=f"Auto-registered by pipeline.analyze for {method_name}",
        )
        return int(registered_id)
    return int(method_row.id)


def _default_method_runner(
    *,
    method_name: str,
    input_group_id: int,
    input_group_type: str,
    input_group_metadata: dict[str, Any],
    embeddings: np.ndarray | None,
    texts: list[str],
    parameters: Mapping[str, Any],
) -> AnalysisPayload:
    if embeddings is None:
        raise ValueError(
            f"analysis method {method_name!r} requires embedding_matrix artifact on input group"
        )
    matrix = np.asarray(embeddings, dtype=np.float64)
    row_count = int(matrix.shape[0])
    dimension = int(matrix.shape[1]) if matrix.ndim == 2 and row_count > 0 else 0
    avg_norm = (
        float(np.linalg.norm(matrix, axis=1).mean())
        if row_count > 0 and dimension > 0
        else 0.0
    )
    scalar_results = {
        "row_count": float(row_count),
        "embedding_dimension": float(dimension),
        "avg_l2_norm": float(avg_norm),
    }
    structured = {
        "method_name": method_name,
        "input_group_id": int(input_group_id),
        "input_group_type": input_group_type,
        "input_group_metadata": input_group_metadata,
        "parameters": dict(parameters or {}),
        "text_count": len(texts),
        "row_count": row_count,
        "embedding_dimension": dimension,
    }
    summary_bytes = json.dumps(
        structured,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return AnalysisPayload(
        scalar_results=scalar_results,
        structured_results={"summary": structured},
        artifacts={"analysis_summary.json": summary_bytes},
        result_ref="analysis_summary.json",
    )


def analyze(
    input_group_id: int,
    *,
    method_name: str,
    run_key: str,
    request_group_id: int | None = None,
    method_version: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    artifact_dir: str = "artifacts",
    method_runner: AnalysisRunner | None = None,
) -> StageResult:
    """
    Execute analysis stage with provenanced_run + analysis_result persistence.
    """
    resolved_params = dict(parameters or {})
    db_conn, _owned_db = _resolve_db(db=db, database_url=database_url)
    with db_conn.session_scope() as session:
        _require_group(session, int(input_group_id))
        repo = RawCallRepository(session)
        resolved_request_group_id = _resolve_request_group_id(
            repo,
            request_group_id=request_group_id,
            method_name=method_name,
            input_group_id=int(input_group_id),
            run_key=run_key,
        )

    lock_key = f"{resolved_request_group_id}:{run_key}:analysis_execution"
    with _analysis_lock(lock_key):
        with db_conn.session_scope() as session:
            repo = RawCallRepository(session)
            existing = repo.get_provenanced_run_by_request_and_key(
                request_group_id=int(resolved_request_group_id),
                run_key=run_key,
                run_kind="analysis_execution",
            )
            if (
                existing is not None
                and existing.run_status == "completed"
                and existing.result_group_id is not None
                and not force
            ):
                artifact_uris = _collect_analysis_artifact_uris(
                    session,
                    int(existing.result_group_id),
                )
                return StageResult(
                    stage_name="analyze",
                    group_id=int(existing.result_group_id),
                    run_id=int(existing.id),
                    artifact_uris=artifact_uris,
                    metadata={
                        "reused": True,
                        "request_group_id": int(resolved_request_group_id),
                        "method_name": method_name,
                    },
                )

            input_group = _require_group(session, int(input_group_id))
            input_group_type = str(input_group.group_type)
            input_group_metadata = dict(input_group.metadata_json or {})
            embeddings = _load_embedding_matrix(
                session,
                input_group_id=int(input_group_id),
                artifact_dir=artifact_dir,
            )
            upstream_snapshot_group_id = _infer_snapshot_group_id(
                session,
                input_group_id=int(input_group_id),
            )
            texts = (
                _load_snapshot_texts(
                    session,
                    snapshot_group_id=int(upstream_snapshot_group_id),
                    artifact_dir=artifact_dir,
                )
                if upstream_snapshot_group_id is not None
                else []
            )

        runner = method_runner or _default_method_runner
        payload_holder: dict[str, AnalysisPayload] = {}

        def _write_analysis_artifacts(
            artifact_service: ArtifactService,
            identity: StageIdentity,
        ) -> dict[str, str]:
            repo = artifact_service.repository
            if repo is None:
                raise RuntimeError("ArtifactService requires repository for analyze stage writes")
            raw_payload = runner(
                method_name=method_name,
                input_group_id=int(input_group_id),
                input_group_type=input_group_type,
                input_group_metadata=input_group_metadata,
                embeddings=embeddings,
                texts=texts,
                parameters=resolved_params,
            )
            payload = _coerce_payload(raw_payload)
            payload_holder["payload"] = payload

            artifact_uris: dict[str, str] = {}
            for logical_name, blob_bytes in payload.artifacts.items():
                artifact_type, content_type = _artifact_type_and_content_type(logical_name)
                artifact_id = artifact_service.store_group_blob_artifact(
                    group_id=identity.group_id,
                    step_name=f"analyze_{method_name}_{run_key}",
                    logical_filename=logical_name,
                    data=blob_bytes,
                    artifact_type=artifact_type,
                    content_type=content_type,
                    metadata={
                        "method_name": method_name,
                        "input_group_id": int(input_group_id),
                        "run_key": run_key,
                    },
                )
                artifact_uris[logical_name] = _call_artifact_uri_by_id(repo, artifact_id)
            return artifact_uris

        def _finalize_analysis(
            repo: RawCallRepository,
            identity: StageIdentity,
            artifact_uris: dict[str, str],
        ) -> dict[str, Any]:
            payload = payload_holder.get("payload")
            if payload is None:
                raise RuntimeError("analysis payload missing; write_artifacts must run first")

            method_service = MethodService(repo)
            method_definition_id = _resolve_method_definition_id(
                method_service,
                method_name=method_name,
                method_version=method_version,
            )

            result_count = 0
            for key, value in payload.scalar_results.items():
                method_service.record_result(
                    method_definition_id=method_definition_id,
                    source_group_id=int(input_group_id),
                    analysis_group_id=identity.group_id,
                    result_key=str(key),
                    result_value=float(value),
                    result_json={"parameters": resolved_params},
                )
                result_count += 1

            for key, value in payload.structured_results.items():
                method_service.record_result(
                    method_definition_id=method_definition_id,
                    source_group_id=int(input_group_id),
                    analysis_group_id=identity.group_id,
                    result_key=str(key),
                    result_json={
                        "parameters": resolved_params,
                        "value": value,
                    },
                )
                result_count += 1

            if artifact_uris:
                method_service.record_result(
                    method_definition_id=method_definition_id,
                    source_group_id=int(input_group_id),
                    analysis_group_id=identity.group_id,
                    result_key="artifacts",
                    result_json={
                        "parameters": resolved_params,
                        "uris": artifact_uris,
                    },
                )
                result_count += 1

            if identity.run_id is not None and payload.result_ref:
                resolved_result_ref = artifact_uris.get(payload.result_ref, payload.result_ref)
                repo.update_provenanced_run(
                    int(identity.run_id),
                    result_ref=str(resolved_result_ref),
                )

            return {
                "method_definition_id": method_definition_id,
                "result_count": result_count,
                "request_group_id": int(resolved_request_group_id),
            }

        result = run_stage(
            db=db_conn,
            stage_name="analyze",
            group_type="analysis_run",
            group_name=f"analyze:{method_name}:{input_group_id}:{run_key}",
            group_description=f"Analysis run for {method_name} on group {input_group_id}",
            group_metadata={
                "method_name": method_name,
                "method_version": method_version,
                "input_group_id": int(input_group_id),
                "run_key": run_key,
                "parameters": resolved_params,
                "request_group_id": int(resolved_request_group_id),
            },
            request_group_id=int(resolved_request_group_id),
            source_group_id=int(input_group_id),
            run_key=run_key,
            run_kind="analysis_execution",
            run_metadata={
                "execution_role": "analysis_execution",
                "method_name": method_name,
            },
            depends_on_group_ids=[int(input_group_id)],
            contains_parent_group_ids=[int(resolved_request_group_id)],
            artifact_dir=artifact_dir,
            write_artifacts=_write_analysis_artifacts,
            finalize_db=_finalize_analysis,
        )
        return StageResult(
            stage_name=result.stage_name,
            group_id=result.group_id,
            run_id=result.run_id,
            artifact_uris=result.artifact_uris,
            metadata={
                **result.metadata,
                "reused": False,
                "request_group_id": int(resolved_request_group_id),
                "method_name": method_name,
            },
        )

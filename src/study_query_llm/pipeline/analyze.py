"""Stage 5: analyze (snapshot + embedding batch -> analysis_run + provenance)."""

from __future__ import annotations

import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pyarrow.parquet as pq

from study_query_llm.algorithms.recipes import (
    COMPOSITE_RECIPES,
    canonical_recipe_hash,
    ensure_composite_recipe,
    register_clustering_components,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri
from study_query_llm.pipeline.clustering import (
    build_effective_recipe_payload,
    build_pipeline_effective_hash,
    get_algorithm_spec,
    is_registry_v1_clustering_method,
    load_rule_set,
    resolve_clustering_resolution,
    resolve_algorithm_runner,
    validate_identity_contract,
    validate_post_selection,
    validate_pre_run,
)
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_service import MethodInputRequirements, MethodService
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.services.provenanced_run_service import ProvenancedRunService

ARTIFACT_TYPE_EMBEDDING_MATRIX = "embedding_matrix"
ARTIFACT_TYPE_SUBQUERY_SPEC = "dataset_subquery_spec"
ARTIFACT_TYPE_ANALYSIS_RESULT_JSON = "analysis_result_json"
ARTIFACT_TYPE_ANALYSIS_RESULT_BLOB = "analysis_result_blob"
REPRESENTATION_FULL = "full"
REPRESENTATION_LABEL_CENTROID = "label_centroid"
REPRESENTATION_LEGACY_INTENT_MEAN = "intent_mean"
REPRESENTATION_SNAPSHOT_ONLY = "snapshot_only"
ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY = "snapshot_only"

_ANALYZE_LOCK_GUARD = threading.Lock()
_ANALYZE_LOCKS: dict[str, threading.Lock] = {}
_CLUSTERING_RULES_PATH = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "rules"
    / "clustering"
    / "rules-v1.0.0.yaml"
)


@dataclass
class AnalysisPayload:
    """Normalized method output payload for analysis stage persistence."""

    scalar_results: dict[str, float]
    structured_results: dict[str, Any]
    artifacts: dict[str, bytes]
    result_ref: str | None = None


@dataclass
class AnalyzeInputBundle:
    """Prepared analyze-stage inputs for either embedding or snapshot mode."""

    analysis_input_group_id: int
    analysis_input_group_type: str
    embedding_batch_group_id: int | None
    embedding_metadata: dict[str, Any]
    dataframe_group_id: int
    resolved_positions: list[int]
    analysis_embeddings: np.ndarray | None
    analysis_texts: list[str]
    representation: str
    representation_meta: dict[str, Any]
    stage_group_name: str
    stage_group_description: str
    stage_depends_on_ids: list[int]


AnalysisRunner = Callable[..., AnalysisPayload | Mapping[str, Any]]


def _resolve_db(
    *,
    db: DatabaseConnectionV2 | None,
    database_url: str | None,
    write_intent: WriteIntent | str | None,
) -> tuple[DatabaseConnectionV2, bool]:
    if db is not None:
        return db, False
    resolved = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not resolved:
        raise ValueError("database_url or DATABASE_URL is required when db is not provided")
    created = DatabaseConnectionV2(
        resolved,
        enable_pgvector=False,
        write_intent=write_intent,
    )
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
    scalar = {str(k): float(v) for k, v in dict(raw.get("scalar_results") or {}).items()}
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


def _require_group(session, group_id: int, *, expected_type: str | None = None) -> Group:
    row = session.query(Group).filter(Group.id == int(group_id)).first()
    if row is None:
        raise ValueError(f"group id={group_id} not found")
    if expected_type is not None and str(row.group_type) != expected_type:
        raise ValueError(
            f"group id={group_id} must be type {expected_type!r}, got {row.group_type!r}"
        )
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
    embedding_batch_group_id: int,
    artifact_dir: str,
) -> np.ndarray:
    uri = _latest_artifact_uri_for_group(
        session,
        group_id=int(embedding_batch_group_id),
        artifact_type=ARTIFACT_TYPE_EMBEDDING_MATRIX,
    )
    if uri is None:
        raise ValueError(
            f"embedding_batch group id={embedding_batch_group_id} has no embedding_matrix artifact"
        )
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    matrix = np.asarray(
        artifact_service.load_artifact(uri, ARTIFACT_TYPE_EMBEDDING_MATRIX),
        dtype=np.float64,
    )
    if matrix.ndim != 2:
        raise ValueError(f"embedding matrix must be 2D, got shape={matrix.shape}")
    return matrix


def _load_snapshot_subquery(
    session,
    *,
    snapshot_group_id: int,
    artifact_dir: str,
) -> dict[str, Any]:
    uri = _latest_artifact_uri_for_group(
        session,
        group_id=int(snapshot_group_id),
        artifact_type=ARTIFACT_TYPE_SUBQUERY_SPEC,
    )
    if uri is None:
        raise ValueError(
            f"dataset_snapshot group id={snapshot_group_id} has no {ARTIFACT_TYPE_SUBQUERY_SPEC} artifact"
        )
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(uri)
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("dataset_subquery_spec payload must be a JSON object")
    return parsed


def _load_dataframe_slice(
    session,
    *,
    dataframe_group_id: int,
    positions: list[int],
    artifact_dir: str,
) -> tuple[list[str], list[int | None], list[str | None]]:
    parquet_uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(parquet_uri)
    table = pq.read_table(
        io.BytesIO(payload),
        columns=["position", "text", "label", "label_name"],
    )
    frame = table.to_pandas().set_index("position", drop=False)
    texts: list[str] = []
    labels: list[int | None] = []
    label_names: list[str | None] = []

    def _as_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            if bool(np.isnan(value)):
                return None
        except Exception:
            pass
        return int(value)

    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        try:
            if bool(np.isnan(value)):
                return None
        except Exception:
            pass
        text = str(value).strip()
        return text or None

    for pos in positions:
        if pos not in frame.index:
            raise ValueError(f"snapshot resolved position {pos} not found in dataframe")
        row = frame.loc[pos]
        texts.append(str(row["text"]))
        labels.append(_as_optional_int(row["label"]))
        label_names.append(_as_optional_str(row["label_name"]))
    return texts, labels, label_names


def _resolve_representation(parameters: Mapping[str, Any]) -> str:
    raw = str(
        parameters.get("representation_type")
        or parameters.get("embedding_representation")
        or REPRESENTATION_FULL
    ).strip().lower()
    if raw == REPRESENTATION_LEGACY_INTENT_MEAN:
        return REPRESENTATION_LABEL_CENTROID
    return raw


def _derive_representation_view(
    *,
    base_embeddings: np.ndarray,
    texts: list[str],
    labels: list[int | None],
    label_names: list[str | None],
    representation: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if representation == REPRESENTATION_FULL:
        return base_embeddings, texts, {}
    if representation != REPRESENTATION_LABEL_CENTROID:
        raise ValueError(f"unsupported analysis representation {representation!r}")

    buckets: dict[int, list[np.ndarray]] = {}
    name_by_label: dict[int, str] = {}
    for idx, label in enumerate(labels):
        if label is None:
            continue
        label_int = int(label)
        buckets.setdefault(label_int, []).append(base_embeddings[idx])
        if label_int not in name_by_label:
            preferred = label_names[idx]
            name_by_label[label_int] = preferred or f"label_{label_int}"
    if not buckets:
        raise ValueError("label_centroid representation requires at least one labeled row")
    ordered_labels = sorted(buckets.keys())
    pooled = np.vstack(
        [
            np.mean(np.asarray(buckets[label], dtype=np.float64), axis=0)
            for label in ordered_labels
        ]
    )
    pooled_texts = [name_by_label[label] for label in ordered_labels]
    meta = {
        "pooled_labels": ordered_labels,
        "pooled_label_count": len(ordered_labels),
    }
    return pooled, pooled_texts, meta


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
    normalized_name = str(method_name).strip().lower()
    if normalized_name in COMPOSITE_RECIPES:
        # Runtime runner dispatch now lives in clustering.registry. Composite
        # recipe metadata remains in algorithms.recipes for MethodDefinition
        # identity and DB registration until both registries are unified.
        register_clustering_components(method_service)
        composite_version = str(method_version or "1.0")
        return int(
            ensure_composite_recipe(
                method_service,
                normalized_name,
                composite_version=composite_version,
                description=(
                    "Composite clustering pipeline "
                    f"({normalized_name}); see method_definitions.recipe_json "
                    "for the stage list."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "k_range": {"type": "array", "items": {"type": "integer"}},
                        "selection_metric": {"type": "string"},
                        "selection_rule": {"type": "string"},
                    },
                },
            )
        )

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


def _resolve_snapshot_dataframe_and_slice(
    session,
    *,
    snapshot_group_id: int,
    artifact_dir: str,
) -> tuple[int, list[int], list[str], list[int | None], list[str | None]]:
    """Resolve snapshot dataframe identity + ordered text slice payload."""
    snapshot_group = _require_group(
        session,
        int(snapshot_group_id),
        expected_type="dataset_snapshot",
    )
    snapshot_metadata = dict(snapshot_group.metadata_json or {})
    snapshot_df_id = int(snapshot_metadata.get("source_dataframe_group_id") or -1)
    if snapshot_df_id <= 0:
        raise ValueError("snapshot metadata must include source_dataframe_group_id")

    snapshot_payload = _load_snapshot_subquery(
        session,
        snapshot_group_id=int(snapshot_group_id),
        artifact_dir=artifact_dir,
    )
    resolved_index_raw = list(snapshot_payload.get("resolved_index") or [])
    resolved_positions: list[int] = []
    for item in resolved_index_raw:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            resolved_positions.append(int(item[0]))
            continue
        raise ValueError("resolved_index entries must be [position, source_id]")
    if not resolved_positions:
        raise ValueError("snapshot resolved_index is empty; nothing to analyze")

    texts, labels, label_names = _load_dataframe_slice(
        session,
        dataframe_group_id=int(snapshot_df_id),
        positions=resolved_positions,
        artifact_dir=artifact_dir,
    )
    return snapshot_df_id, resolved_positions, texts, labels, label_names


def _prepare_analyze_input_bundle(
    session,
    *,
    method_name: str,
    method_input_requirements: MethodInputRequirements,
    snapshot_group_id: int,
    embedding_batch_group_id: int | None,
    artifact_dir: str,
    resolved_params: dict[str, Any],
    run_key: str,
) -> AnalyzeInputBundle:
    """Prepare mode-specific analyze inputs with normalized metadata."""
    (
        snapshot_df_id,
        resolved_positions,
        texts,
        labels,
        label_names,
    ) = _resolve_snapshot_dataframe_and_slice(
        session,
        snapshot_group_id=int(snapshot_group_id),
        artifact_dir=artifact_dir,
    )

    if method_input_requirements.embedding_batch:
        if embedding_batch_group_id is None:
            raise ValueError(
                f"analysis method {method_name!r} requires embedding_batch_group_id"
            )
        embedding_group = _require_group(
            session,
            int(embedding_batch_group_id),
            expected_type="embedding_batch",
        )
        embedding_metadata = dict(embedding_group.metadata_json or {})
        embedding_df_id = int(embedding_metadata.get("source_dataframe_group_id") or -1)
        if embedding_df_id <= 0:
            raise ValueError("embedding metadata must include source_dataframe_group_id")
        if snapshot_df_id != embedding_df_id:
            raise ValueError(
                "snapshot and embedding batch reference different dataframe groups "
                f"(snapshot={snapshot_df_id}, embedding={embedding_df_id})"
            )

        embedding_matrix = _load_embedding_matrix(
            session,
            embedding_batch_group_id=int(embedding_batch_group_id),
            artifact_dir=artifact_dir,
        )
        max_pos = max(resolved_positions)
        if max_pos >= int(embedding_matrix.shape[0]):
            raise ValueError(
                "snapshot resolved_index position exceeds embedding matrix bounds: "
                f"max_position={max_pos} matrix_rows={embedding_matrix.shape[0]}"
            )
        sliced_embeddings = np.asarray(embedding_matrix[resolved_positions], dtype=np.float64)
        if len(texts) != int(sliced_embeddings.shape[0]):
            raise ValueError(
                "text/vector alignment mismatch after resolved-index slicing: "
                f"texts={len(texts)} vectors={sliced_embeddings.shape[0]}"
            )

        representation = _resolve_representation(resolved_params)
        resolved_params.setdefault("representation_type", representation)
        resolved_params.setdefault("embedding_representation", representation)
        analysis_embeddings, analysis_texts, representation_meta = _derive_representation_view(
            base_embeddings=sliced_embeddings,
            texts=texts,
            labels=labels,
            label_names=label_names,
            representation=representation,
        )

        return AnalyzeInputBundle(
            analysis_input_group_id=int(embedding_batch_group_id),
            analysis_input_group_type="embedding_batch",
            embedding_batch_group_id=int(embedding_batch_group_id),
            embedding_metadata=embedding_metadata,
            dataframe_group_id=int(snapshot_df_id),
            resolved_positions=resolved_positions,
            analysis_embeddings=analysis_embeddings,
            analysis_texts=analysis_texts,
            representation=representation,
            representation_meta=representation_meta,
            stage_group_name=(
                f"analyze:{method_name}:{int(embedding_batch_group_id)}:"
                f"{int(snapshot_group_id)}:{run_key}"
            ),
            stage_group_description=(
                f"Analysis run for {method_name} on embedding {int(embedding_batch_group_id)} "
                f"and snapshot {int(snapshot_group_id)}"
            ),
            stage_depends_on_ids=[int(embedding_batch_group_id), int(snapshot_group_id)],
        )

    resolved_params["representation_type"] = REPRESENTATION_SNAPSHOT_ONLY
    resolved_params["embedding_representation"] = REPRESENTATION_SNAPSHOT_ONLY
    return AnalyzeInputBundle(
        analysis_input_group_id=int(snapshot_group_id),
        analysis_input_group_type="dataset_snapshot",
        embedding_batch_group_id=None,
        embedding_metadata={},
        dataframe_group_id=int(snapshot_df_id),
        resolved_positions=resolved_positions,
        analysis_embeddings=None,
        analysis_texts=texts,
        representation=REPRESENTATION_SNAPSHOT_ONLY,
        representation_meta={
            "analysis_input_mode": ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY,
            "embedding_input_used": False,
        },
        stage_group_name=(
            f"analyze:{method_name}:snapshot-only:{int(snapshot_group_id)}:{run_key}"
        ),
        stage_group_description=(
            f"Analysis run for {method_name} on snapshot {int(snapshot_group_id)} (snapshot-only)"
        ),
        stage_depends_on_ids=[int(snapshot_group_id)],
    )


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


def _resolve_builtin_method_runner(method_name: str) -> AnalysisRunner | None:
    return resolve_algorithm_runner(method_name)


def _extract_selection_evidence(payload: AnalysisPayload) -> dict[str, Any] | None:
    candidate = payload.structured_results.get("selection_evidence")
    if isinstance(candidate, Mapping):
        return dict(candidate)
    summary_candidate = payload.structured_results.get("clustering_summary")
    if isinstance(summary_candidate, Mapping):
        maybe = summary_candidate.get("selection_evidence")
        if isinstance(maybe, Mapping):
            return dict(maybe)
    return None


def _extract_cluster_summary(payload: AnalysisPayload) -> dict[str, Any]:
    summary_candidate = payload.structured_results.get("clustering_summary")
    if isinstance(summary_candidate, Mapping):
        return dict(summary_candidate)
    hdbscan_candidate = payload.structured_results.get("hdbscan_summary")
    if isinstance(hdbscan_candidate, Mapping):
        return dict(hdbscan_candidate)
    return {}


def _selected_value_from_evidence(selection_evidence: Mapping[str, Any]) -> tuple[str, int] | None:
    for key in ("chosen_k", "chosen_value", "chosen_min_cluster_size", "chosen_resolution"):
        if key in selection_evidence and selection_evidence[key] is not None:
            return key, int(selection_evidence[key])
    return None


def _patch_terminal_selected_value(
    pipeline_resolved: list[dict[str, Any]],
    *,
    terminal_stage: str,
    selection_evidence: Mapping[str, Any] | None,
) -> None:
    if selection_evidence is None:
        return
    selected = _selected_value_from_evidence(selection_evidence)
    if selected is None:
        return
    selected_key, selected_value = selected
    target_param = "k"
    if terminal_stage == "hdbscan" and selected_key == "chosen_min_cluster_size":
        target_param = "min_cluster_size"
    if terminal_stage not in {"kmeans", "gmm", "hdbscan"}:
        return
    for entry in pipeline_resolved:
        if str(entry.get("stage") or "") != terminal_stage:
            continue
        stage_params = dict(entry.get("params") or {})
        stage_params[target_param] = int(selected_value)
        entry["params"] = stage_params
        break


def _labels_artifact_ref_for_payload(payload: AnalysisPayload) -> str:
    for candidate in (
        "kmeans_labels.json",
        "gmm_labels.json",
        "hdbscan_labels.json",
        "analysis_labels.json",
    ):
        if candidate in payload.artifacts:
            return candidate
    return ""


def _to_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")


def _attach_v1_clustering_provenance(
    *,
    resolution_pre: Any,
    payload: AnalysisPayload,
) -> dict[str, Any]:
    """Patch payload with v1 clustering envelope + identity metadata."""
    resolution = resolution_pre.clone()
    selection_evidence = _extract_selection_evidence(payload)
    _patch_terminal_selected_value(
        resolution.pipeline_resolved,
        terminal_stage=resolution.base_algorithm,
        selection_evidence=selection_evidence,
    )
    validate_post_selection(resolution, selection_evidence=selection_evidence)

    pipeline_effective_hash = build_pipeline_effective_hash(
        resolution.pipeline_resolved,
        resolution.pipeline_effective,
    )
    effective_recipe_payload = build_effective_recipe_payload(
        resolution.pipeline_resolved,
        resolution.pipeline_effective,
    )
    recipe_hash = canonical_recipe_hash(effective_recipe_payload)
    validate_identity_contract(
        pipeline_effective_hash=pipeline_effective_hash,
        recipe_hash=recipe_hash,
    )

    cluster_summary = _extract_cluster_summary(payload)
    labels_artifact_ref = str(
        cluster_summary.get("labels_artifact_ref") or _labels_artifact_ref_for_payload(payload)
    )
    envelope: dict[str, Any] = {
        "operation_type": resolution.operation_type,
        "operation_version": resolution.operation_version,
        "method_name": resolution.method_name,
        "base_algorithm": resolution.base_algorithm,
        "rule_set_version": resolution.rule_set_version,
        "rule_set_hash": resolution.rule_set_hash,
        "rule_inputs": dict(resolution.rule_inputs),
        "input_audit_metadata": dict(resolution.input_audit_metadata),
        "pipeline_declared": list(resolution.pipeline_declared),
        "pipeline_resolved": [dict(entry) for entry in resolution.pipeline_resolved],
        "pipeline_effective": list(resolution.pipeline_effective),
        "pipeline_effective_hash": pipeline_effective_hash,
        "recipe_hash": recipe_hash,
        "skipped_stages": [dict(item) for item in resolution.skipped_stages],
        "aliases": list(resolution.aliases),
        "n_samples": int(cluster_summary.get("n_samples") or 0),
        "n_features": int(cluster_summary.get("n_features") or 0),
        "cluster_count": int(cluster_summary.get("cluster_count") or 0),
        "cluster_ids": list(cluster_summary.get("cluster_ids") or []),
        "cluster_sizes": dict(cluster_summary.get("cluster_sizes") or {}),
        "noise_count": int(cluster_summary.get("noise_count") or 0),
        "noise_fraction": float(cluster_summary.get("noise_fraction") or 0.0),
        "labels_artifact_ref": labels_artifact_ref,
    }
    if selection_evidence is not None:
        envelope["selection_evidence"] = dict(selection_evidence)

    payload.structured_results["clustering_summary"] = envelope
    payload.artifacts.setdefault("clustering_summary.json", _to_json_bytes(envelope))

    hdbscan_summary = payload.structured_results.get("hdbscan_summary")
    if isinstance(hdbscan_summary, Mapping):
        upgraded = dict(hdbscan_summary)
        upgraded.setdefault("operation_type", resolution.operation_type)
        upgraded.setdefault("operation_version", resolution.operation_version)
        upgraded.setdefault("rule_set_version", resolution.rule_set_version)
        upgraded.setdefault("rule_set_hash", resolution.rule_set_hash)
        upgraded.setdefault("rule_inputs", dict(resolution.rule_inputs))
        upgraded.setdefault("pipeline_declared", list(resolution.pipeline_declared))
        upgraded.setdefault("pipeline_resolved", [dict(entry) for entry in resolution.pipeline_resolved])
        upgraded.setdefault("pipeline_effective", list(resolution.pipeline_effective))
        upgraded.setdefault("pipeline_effective_hash", pipeline_effective_hash)
        payload.structured_results["hdbscan_summary"] = upgraded
        payload.artifacts["hdbscan_summary.json"] = _to_json_bytes(upgraded)

    return {
        "resolution": resolution,
        "pipeline_effective_hash": pipeline_effective_hash,
        "recipe_hash": recipe_hash,
        "effective_recipe_payload": effective_recipe_payload,
        "selection_evidence": dict(selection_evidence) if selection_evidence is not None else None,
        "summary": envelope,
    }


def analyze(
    snapshot_group_id: int,
    embedding_batch_group_id: int | None = None,
    *,
    method_name: str,
    run_key: str,
    request_group_id: int | None = None,
    method_version: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    write_intent: WriteIntent | str | None = WriteIntent.CANONICAL,
    artifact_dir: str = "artifacts",
    method_runner: AnalysisRunner | None = None,
) -> StageResult:
    """Execute analysis stage with snapshot input and optional embedding batch."""
    resolved_params = dict(parameters or {})
    clustering_resolution_pre = None
    clustering_rule_set = None
    db_conn, _owned_db = _resolve_db(
        db=db,
        database_url=database_url,
        write_intent=write_intent,
    )
    snapshot_group_id_int = int(snapshot_group_id)
    requested_embedding_batch_group_id: int | None = (
        int(embedding_batch_group_id) if embedding_batch_group_id is not None else None
    )
    with db_conn.session_scope() as session:
        requirement_repo = RawCallRepository(session)
        requirement_method_service = MethodService(requirement_repo)
        method_input_requirements = requirement_method_service.resolve_method_input_requirements(
            name=method_name,
            version=method_version,
        )
        if not method_input_requirements.snapshot:
            raise ValueError(
                f"analysis method {method_name!r} must require snapshot input "
                "(required_inputs.snapshot cannot be false)"
            )
        input_bundle = _prepare_analyze_input_bundle(
            session,
            method_name=method_name,
            method_input_requirements=method_input_requirements,
            snapshot_group_id=snapshot_group_id_int,
            embedding_batch_group_id=requested_embedding_batch_group_id,
            artifact_dir=artifact_dir,
            resolved_params=resolved_params,
            run_key=run_key,
        )
        analysis_input_group_id = int(input_bundle.analysis_input_group_id)
        analysis_input_group_type = str(input_bundle.analysis_input_group_type)
        embedding_batch_input_group_id = input_bundle.embedding_batch_group_id
        embedding_metadata = dict(input_bundle.embedding_metadata)
        dataframe_group_id = int(input_bundle.dataframe_group_id)
        resolved_positions = list(input_bundle.resolved_positions)
        analysis_embeddings = input_bundle.analysis_embeddings
        analysis_texts = list(input_bundle.analysis_texts)
        representation = str(input_bundle.representation)
        representation_meta = dict(input_bundle.representation_meta)
        stage_group_name = str(input_bundle.stage_group_name)
        stage_group_description = str(input_bundle.stage_group_description)
        stage_depends_on_ids = [int(group_id) for group_id in input_bundle.stage_depends_on_ids]
        algorithm_spec = get_algorithm_spec(method_name)
        if is_registry_v1_clustering_method(method_name):
            if analysis_embeddings is None:
                raise ValueError(
                    f"v1 clustering method {method_name!r} requires embedding input"
                )
            clustering_rule_set = load_rule_set(_CLUSTERING_RULES_PATH)
            clustering_resolution_pre = resolve_clustering_resolution(
                method_name=method_name,
                parameters=resolved_params,
                rule_set=clustering_rule_set,
                context={
                    "embedding_dim": int(analysis_embeddings.shape[1]),
                    "n_samples": int(analysis_embeddings.shape[0]),
                },
            )
            validate_pre_run(
                clustering_resolution_pre,
                allowed_context_keys=clustering_rule_set.context_keys,
            )
            if resolved_params.get("determinism_class") is None:
                normalized_method = str(method_name).strip().lower()
                if normalized_method in {"kmeans+silhouette+kneedle", "gmm+bic+argmin"}:
                    resolved_params["determinism_class"] = "pseudo_deterministic"
                else:
                    resolved_params["determinism_class"] = "non_deterministic"
        elif (
            resolved_params.get("determinism_class") is None
            and algorithm_spec is not None
            and str(algorithm_spec.method_name) == "agglomerative+fixed-k"
        ):
            resolved_params["determinism_class"] = str(
                algorithm_spec.default_determinism_class
            )

        repo = RawCallRepository(session)
        resolved_request_group_id = _resolve_request_group_id(
            repo,
            request_group_id=request_group_id,
            method_name=method_name,
            input_group_id=int(analysis_input_group_id),
            run_key=run_key,
        )
        stage_group_metadata = {
            "method_name": method_name,
            "method_version": method_version,
            "snapshot_group_id": int(snapshot_group_id_int),
            "run_key": run_key,
            "representation_type": representation,
            "parameters": resolved_params,
            "request_group_id": int(resolved_request_group_id),
        }
        if embedding_batch_input_group_id is not None:
            stage_group_metadata["embedding_batch_group_id"] = int(embedding_batch_input_group_id)
        else:
            stage_group_metadata["analysis_input_mode"] = ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY

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
                        "representation": representation,
                    },
                )

        runner = method_runner or _resolve_builtin_method_runner(method_name) or _default_method_runner
        payload_holder: dict[str, AnalysisPayload] = {}

        runner_input_metadata = {
            **embedding_metadata,
            "representation": representation,
            "source_snapshot_group_id": int(snapshot_group_id_int),
            "source_dataframe_group_id": int(dataframe_group_id),
            "resolved_index_row_count": len(resolved_positions),
            **representation_meta,
        }

        def _write_analysis_artifacts(
            artifact_service: ArtifactService,
            identity: StageIdentity,
        ) -> dict[str, str]:
            repo = artifact_service.repository
            if repo is None:
                raise RuntimeError("ArtifactService requires repository for analyze stage writes")
            runner_parameters = dict(resolved_params)
            if clustering_resolution_pre is not None:
                # Runner input contract (v1 envelope only): runners may consume
                # these private keys to replay the pre-resolved effective
                # pipeline and rule-input context without recomputing it.
                runner_parameters["_v1_pipeline_resolved"] = [
                    dict(entry) for entry in clustering_resolution_pre.pipeline_resolved
                ]
                runner_parameters["_v1_pipeline_effective"] = list(
                    clustering_resolution_pre.pipeline_effective
                )
                runner_parameters["_v1_rule_inputs"] = dict(clustering_resolution_pre.rule_inputs)
            raw_payload = runner(
                method_name=method_name,
                input_group_id=int(analysis_input_group_id),
                input_group_type=analysis_input_group_type,
                input_group_metadata=runner_input_metadata,
                embeddings=analysis_embeddings,
                texts=analysis_texts,
                parameters=runner_parameters,
            )
            payload = _coerce_payload(raw_payload)
            if clustering_resolution_pre is not None:
                payload_holder["v1_provenance"] = _attach_v1_clustering_provenance(
                    resolution_pre=clustering_resolution_pre,
                    payload=payload,
                )
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
                    metadata=(
                        {
                            "method_name": method_name,
                            "snapshot_group_id": int(snapshot_group_id_int),
                            "run_key": run_key,
                        }
                        | (
                            {
                                "embedding_batch_group_id": int(
                                    embedding_batch_input_group_id
                                ),
                            }
                            if embedding_batch_input_group_id is not None
                            else {
                                "analysis_input_mode": ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY
                            }
                        )
                    ),
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
                    source_group_id=int(analysis_input_group_id),
                    analysis_group_id=identity.group_id,
                    result_key=str(key),
                    result_value=float(value),
                    result_json={"parameters": resolved_params},
                )
                result_count += 1

            for key, value in payload.structured_results.items():
                method_service.record_result(
                    method_definition_id=method_definition_id,
                    source_group_id=int(analysis_input_group_id),
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
                    source_group_id=int(analysis_input_group_id),
                    analysis_group_id=identity.group_id,
                    result_key="artifacts",
                    result_json={
                        "parameters": resolved_params,
                        "uris": artifact_uris,
                    },
                )
                result_count += 1

            resolved_result_ref: str | None = None
            if payload.result_ref:
                resolved_result_ref = str(artifact_uris.get(payload.result_ref, payload.result_ref))

            existing_run_metadata: dict[str, Any] = {}
            if identity.run_id is not None:
                existing_run = repo.get_provenanced_run_by_id(int(identity.run_id))
                if existing_run is not None:
                    existing_run_metadata = dict(existing_run.metadata_json or {})

            execution_metadata: dict[str, Any] = dict(existing_run_metadata)
            execution_metadata["stage_name"] = "analyze"
            execution_metadata["analysis_key"] = str(method_name)
            execution_metadata["analysis_group_id"] = int(identity.group_id)
            execution_metadata["input_snapshot_group_id"] = int(snapshot_group_id_int)
            execution_metadata["request_group_id"] = int(resolved_request_group_id)
            execution_metadata["representation_type"] = representation
            if embedding_batch_input_group_id is not None:
                execution_metadata["embedding_batch_group_id"] = int(
                    embedding_batch_input_group_id
                )
            else:
                execution_metadata["analysis_input_mode"] = ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY

            manifest_hash = embedding_metadata.get("manifest_hash")
            if manifest_hash is not None:
                execution_metadata.setdefault("manifest_hash", str(manifest_hash))

            data_regime = {
                key: resolved_params[key]
                for key in (
                    "dataset_slug",
                    "representation_type",
                    "embedding_provider",
                    "embedding_deployment",
                )
                if key in resolved_params and resolved_params[key] is not None
            }
            if data_regime:
                execution_metadata.setdefault("data_regime", data_regime)

            canonical_config: dict[str, Any] = {
                "parameters": dict(resolved_params),
                "snapshot_group_id": int(snapshot_group_id_int),
                "representation_type": representation,
            }
            composite_recipe = COMPOSITE_RECIPES.get(str(method_name).strip().lower())
            if composite_recipe is not None:
                # TODO(clustering-registry-followup): Resolve runtime mismatch for
                # cosine_kllmeans_no_pca by wiring a dedicated runner or formally
                # deprecating/removing it from analyze runtime paths.
                canonical_config.setdefault(
                    "recipe_hash",
                    canonical_recipe_hash(dict(composite_recipe)),
                )
            if embedding_batch_input_group_id is not None:
                canonical_config["embedding_batch_group_id"] = int(
                    embedding_batch_input_group_id
                )
            else:
                canonical_config["analysis_input_mode"] = ANALYSIS_INPUT_MODE_SNAPSHOT_ONLY
            v1_provenance = payload_holder.get("v1_provenance")
            if isinstance(v1_provenance, Mapping):
                recipe_hash = v1_provenance.get("recipe_hash")
                if recipe_hash is not None:
                    canonical_config["recipe_hash"] = str(recipe_hash)
                summary_payload = v1_provenance.get("summary")
                if isinstance(summary_payload, Mapping):
                    canonical_config["operation_type"] = str(
                        summary_payload.get("operation_type") or ""
                    )
                    canonical_config["operation_version"] = str(
                        summary_payload.get("operation_version") or ""
                    )
                    canonical_config["rule_set_version"] = str(
                        summary_payload.get("rule_set_version") or ""
                    )
                    canonical_config["rule_set_hash"] = str(
                        summary_payload.get("rule_set_hash") or ""
                    )
                    canonical_config["rule_inputs"] = dict(
                        summary_payload.get("rule_inputs") or {}
                    )
                    canonical_config["pipeline_declared"] = list(
                        summary_payload.get("pipeline_declared") or []
                    )
                    canonical_config["pipeline_resolved"] = list(
                        summary_payload.get("pipeline_resolved") or []
                    )
                    canonical_config["pipeline_effective"] = list(
                        summary_payload.get("pipeline_effective") or []
                    )
                    canonical_config["pipeline_effective_hash"] = str(
                        summary_payload.get("pipeline_effective_hash") or ""
                    )
                    execution_metadata["operation_type"] = str(
                        summary_payload.get("operation_type") or ""
                    )
                    execution_metadata["operation_version"] = str(
                        summary_payload.get("operation_version") or ""
                    )
            if method_version is not None:
                canonical_config["method_version"] = str(method_version)

            determinism_class = str(
                resolved_params.get("determinism_class") or "non_deterministic"
            )
            provenanced_run_service = ProvenancedRunService(repo)
            enriched_run_id = int(
                provenanced_run_service.record_analysis_execution(
                    request_group_id=int(resolved_request_group_id),
                    source_group_id=int(analysis_input_group_id),
                    method_definition_id=method_definition_id,
                    analysis_key=method_name,
                    analysis_run_key=run_key,
                    result_ref=resolved_result_ref,
                    config_json=canonical_config,
                    determinism_class=determinism_class,
                    metadata_json=execution_metadata,
                    input_snapshot_group_id=int(snapshot_group_id_int),
                )
            )

            if identity.run_id is not None and int(identity.run_id) != enriched_run_id:
                raise RuntimeError(
                    "analysis execution provenance upsert targeted a different run row "
                    f"(stage_run_id={identity.run_id}, enriched_run_id={enriched_run_id})"
                )

            return {
                "method_definition_id": method_definition_id,
                "result_count": result_count,
                "request_group_id": int(resolved_request_group_id),
                "analysis_execution_run_id": enriched_run_id,
                "representation": representation,
            }

        result = run_stage(
            db=db_conn,
            stage_name="analyze",
            group_type="analysis_run",
            group_name=stage_group_name,
            group_description=stage_group_description,
            group_metadata=stage_group_metadata,
            request_group_id=int(resolved_request_group_id),
            source_group_id=int(analysis_input_group_id),
            run_key=run_key,
            run_kind="analysis_execution",
            run_metadata={
                "execution_role": "analysis_execution",
                "method_name": method_name,
            },
            depends_on_group_ids=[int(group_id) for group_id in stage_depends_on_ids],
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
                "representation": representation,
            },
        )

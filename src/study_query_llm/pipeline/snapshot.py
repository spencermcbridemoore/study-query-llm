"""Stage 3: snapshot (dataframe + subquery spec -> resolved row index artifact)."""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult, SubquerySpec
from study_query_llm.services.artifact_service import ArtifactService

ARTIFACT_TYPE_SUBQUERY_SPEC = "dataset_subquery_spec"


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


def _collect_snapshot_artifact_uris(session, snapshot_group_id: int) -> dict[str, str]:
    repo = RawCallRepository(session)
    artifacts = repo.list_group_artifacts(
        group_id=int(snapshot_group_id),
        artifact_types=[ARTIFACT_TYPE_SUBQUERY_SPEC],
    )
    artifact_uris: dict[str, str] = {}
    for artifact in artifacts:
        if artifact.artifact_type == ARTIFACT_TYPE_SUBQUERY_SPEC:
            artifact_uris["subquery_spec.json"] = str(artifact.uri)
    return artifact_uris


def _normalize_spec(spec: SubquerySpec | None) -> SubquerySpec:
    return spec or SubquerySpec()


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decode_extra(payload: Any, *, position: int, source_id: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, str) or payload == "":
        raise ValueError(
            f"snapshot row position={position} source_id={source_id!r} "
            "has invalid extra_json payload"
        )
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"snapshot row position={position} source_id={source_id!r} "
            f"has malformed extra_json: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise ValueError(
            f"snapshot row position={position} source_id={source_id!r} "
            "has non-object extra_json payload"
        )
    return decoded


def _typed_equals(actual: Any, candidate: Any) -> bool:
    return type(actual) is type(candidate) and actual == candidate


def _apply_category_filter(
    filtered: pd.DataFrame,
    *,
    category_filter: Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    if filtered.empty:
        return filtered

    decoded = filtered.apply(
        lambda row: _decode_extra(
            row["extra_json"],
            position=int(row["position"]),
            source_id=str(row["source_id"]),
        ),
        axis=1,
    )
    mask = pd.Series(True, index=filtered.index)
    for key, allowed in category_filter.items():
        candidates = tuple(allowed)
        mask &= decoded.map(
            lambda parsed, filter_key=key, filter_candidates=candidates: any(
                _typed_equals(parsed.get(filter_key), candidate)
                for candidate in filter_candidates
            )
        )
    return filtered[mask]


def _load_dataframe_frame(
    *,
    dataframe_parquet_uri: str,
    artifact_dir: str,
) -> pd.DataFrame:
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(dataframe_parquet_uri)
    table = pq.read_table(
        source=io.BytesIO(payload),
        columns=["position", "source_id", "text", "label", "label_name", "extra_json"],
    )
    frame = table.to_pandas()
    frame["position"] = frame["position"].astype(int)
    frame["source_id"] = frame["source_id"].astype(str)
    return frame


def _find_existing_snapshot_group(
    session,
    *,
    dataframe_group_id: int,
    spec_hash: str,
    resolved_index_hash: str,
) -> int | None:
    repo = RawCallRepository(session)
    return repo.find_group_id_by_metadata(
        group_type="dataset_snapshot",
        metadata_eq={
            "source_dataframe_group_id": int(dataframe_group_id),
            "spec_hash": str(spec_hash),
            "resolved_index_hash": str(resolved_index_hash),
        },
    )


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _apply_subquery(
    frame: pd.DataFrame,
    *,
    spec: SubquerySpec,
) -> pd.DataFrame:
    normalized = spec.to_canonical_dict()
    label_mode = str(normalized.get("label_mode") or "all")
    filtered = frame

    if label_mode == "labeled":
        filtered = filtered[filtered["label"].notna()]
    elif label_mode == "unlabeled":
        filtered = filtered[filtered["label"].isna()]
    elif label_mode == "all":
        filtered = filtered
    else:
        raise ValueError("label_mode must be one of {'all','labeled','unlabeled'}")

    filter_expr = normalized.get("filter_expr")
    if filter_expr:
        try:
            filtered = filtered.query(str(filter_expr), engine="python")
        except Exception as exc:
            raise ValueError(f"invalid filter_expr {filter_expr!r}: {exc}") from exc

    category_filter = normalized.get("category_filter")
    if category_filter:
        filtered = _apply_category_filter(filtered, category_filter=category_filter)

    sample_n = normalized.get("sample_n")
    sample_fraction = normalized.get("sample_fraction")
    sampling_seed = normalized.get("sampling_seed")
    if sample_n is not None and sample_fraction is not None:
        raise ValueError("SubquerySpec cannot set both sample_n and sample_fraction")
    if (sample_n is not None or sample_fraction is not None) and sampling_seed is None:
        raise ValueError(
            "Subquery sampling requires sampling_seed for deterministic snapshot ids"
        )

    if sample_n is not None:
        sample_n_int = int(sample_n)
        if sample_n_int <= 0:
            raise ValueError("sample_n must be > 0 when provided")
        sample_n_int = min(sample_n_int, len(filtered))
        filtered = filtered.sample(n=sample_n_int, random_state=int(sampling_seed))
    elif sample_fraction is not None:
        sample_fraction_float = float(sample_fraction)
        if sample_fraction_float <= 0.0 or sample_fraction_float > 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        filtered = filtered.sample(frac=sample_fraction_float, random_state=int(sampling_seed))

    return filtered.sort_values(by="position", kind="mergesort").reset_index(drop=True)


def snapshot(
    dataframe_group_id: int,
    *,
    subquery_spec: SubquerySpec | None = None,
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    write_intent: WriteIntent | str | None = WriteIntent.CANONICAL,
    artifact_dir: str = "artifacts",
) -> StageResult:
    """
    Resolve a deterministic snapshot row index from canonical dataframe and subquery spec.
    """
    db_conn, _owned_db = _resolve_db(
        db=db,
        database_url=database_url,
        write_intent=write_intent,
    )
    with db_conn.session_scope() as session:
        dataframe_group = (
            session.query(Group)
            .filter(
                Group.id == int(dataframe_group_id),
                Group.group_type == "dataset_dataframe",
            )
            .first()
        )
        if dataframe_group is None:
            raise ValueError(f"dataset_dataframe group id={dataframe_group_id} not found")
        group_metadata = dict(dataframe_group.metadata_json or {})
        dataset_slug = str(group_metadata.get("dataset_slug") or "")
        if not dataset_slug:
            raise ValueError(
                "dataset_dataframe group is missing metadata_json['dataset_slug']"
            )
        dataframe_parquet_uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))

    spec_obj = _normalize_spec(subquery_spec)
    normalized_spec = spec_obj.to_canonical_dict()
    spec_hash = _hash_payload(normalized_spec)
    filtered = _apply_subquery(
        _load_dataframe_frame(
            dataframe_parquet_uri=dataframe_parquet_uri,
            artifact_dir=artifact_dir,
        ),
        spec=spec_obj,
    )
    resolved_index = [
        [int(row.position), str(row.source_id)]
        for row in filtered.itertuples(index=False)
    ]
    resolved_index_hash = _hash_payload(resolved_index)
    payload = {
        "subquery_spec": normalized_spec,
        "spec_hash": spec_hash,
        "resolved_index": resolved_index,
        "resolved_index_hash": resolved_index_hash,
        "row_count": len(resolved_index),
        "source_dataframe_group_id": int(dataframe_group_id),
    }
    payload_bytes = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    with db_conn.session_scope() as session:
        existing_group_id = _find_existing_snapshot_group(
            session,
            dataframe_group_id=int(dataframe_group_id),
            spec_hash=spec_hash,
            resolved_index_hash=resolved_index_hash,
        )
        if existing_group_id is not None and not force:
            artifact_uris = _collect_snapshot_artifact_uris(session, existing_group_id)
            return StageResult(
                stage_name="snapshot",
                group_id=existing_group_id,
                run_id=None,
                artifact_uris=artifact_uris,
                metadata={
                    "reused": True,
                    "spec_hash": spec_hash,
                    "resolved_index_hash": resolved_index_hash,
                    "row_count": len(resolved_index),
                },
            )

    def _write_snapshot_artifacts(artifact_service, identity: StageIdentity) -> dict[str, str]:
        repo = artifact_service.repository
        if repo is None:
            raise RuntimeError("ArtifactService requires repository for snapshot stage writes")
        payload_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="snapshot",
            logical_filename="subquery_spec.json",
            data=payload_bytes,
            artifact_type=ARTIFACT_TYPE_SUBQUERY_SPEC,
            content_type="application/json",
            metadata={
                "spec_hash": spec_hash,
                "resolved_index_hash": resolved_index_hash,
                "row_count": len(resolved_index),
            },
        )
        return {
            "subquery_spec.json": _call_artifact_uri_by_id(repo, payload_artifact_id),
        }

    result = run_stage(
        db=db_conn,
        stage_name="snapshot",
        group_type="dataset_snapshot",
        group_name=f"snap:{dataset_slug}:{dataframe_group_id}:{spec_hash[:8]}",
        group_description=f"Dataset snapshot view over dataframe {dataframe_group_id}",
        group_metadata={
            "dataset_slug": dataset_slug,
            "source_dataframe_group_id": int(dataframe_group_id),
            "spec_hash": spec_hash,
            "resolved_index_hash": resolved_index_hash,
            "row_count": len(resolved_index),
            "subquery_spec": normalized_spec,
        },
        depends_on_group_ids=[int(dataframe_group_id)],
        artifact_dir=artifact_dir,
        write_artifacts=_write_snapshot_artifacts,
    )
    return StageResult(
        stage_name=result.stage_name,
        group_id=result.group_id,
        run_id=result.run_id,
        artifact_uris=result.artifact_uris,
        metadata={
            **result.metadata,
            "reused": False,
            "spec_hash": spec_hash,
            "resolved_index_hash": resolved_index_hash,
            "row_count": len(resolved_index),
        },
    )

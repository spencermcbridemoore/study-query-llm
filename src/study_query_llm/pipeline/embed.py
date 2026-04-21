"""Stage 3: embed (snapshot parquet -> embedding matrix artifact)."""

from __future__ import annotations

import asyncio
import io
import json
import os
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.embeddings.constants import CACHE_KEY_VERSION
from study_query_llm.services.embeddings.helpers import fetch_embeddings_async

ARTIFACT_TYPE_SNAPSHOT_PARQUET = "dataset_snapshot_parquet"
ARTIFACT_TYPE_EMBEDDING_MATRIX = "embedding_matrix"
ARTIFACT_TYPE_SPARSE_SIDECAR = "embedding_sparse_sidecar"

REPRESENTATION_FULL = "full"
REPRESENTATION_LABEL_CENTROID = "label_centroid"
REPRESENTATION_LABEL_CENTROID_LEGACY = "intent_mean"
REPRESENTATION_SPARSE = "sparse"

LEGACY_REPRESENTATION_ALIASES: dict[str, str] = {
    REPRESENTATION_LABEL_CENTROID_LEGACY: REPRESENTATION_LABEL_CENTROID,
}
ALLOWED_REPRESENTATIONS = {
    REPRESENTATION_FULL,
    REPRESENTATION_LABEL_CENTROID,
    REPRESENTATION_SPARSE,
}

EmbeddingFetcher = Callable[..., np.ndarray]


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


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _collect_embedding_artifact_uris(session, embedding_group_id: int) -> dict[str, str]:
    artifact_uris: dict[str, str] = {}
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) != int(embedding_group_id):
            continue
        if artifact.artifact_type == ARTIFACT_TYPE_EMBEDDING_MATRIX:
            artifact_uris["embedding_matrix.npy"] = str(artifact.uri)
        elif artifact.artifact_type == ARTIFACT_TYPE_SPARSE_SIDECAR:
            artifact_uris["sparse_sidecar.json"] = str(artifact.uri)
    return artifact_uris


def _find_snapshot_parquet_uri(session, snapshot_group_id: int) -> str:
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.desc()).all()
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) != int(snapshot_group_id):
            continue
        if artifact.artifact_type == ARTIFACT_TYPE_SNAPSHOT_PARQUET:
            return str(artifact.uri)
    raise ValueError(
        f"snapshot group id={snapshot_group_id} has no {ARTIFACT_TYPE_SNAPSHOT_PARQUET} artifact"
    )


def _load_snapshot_texts_and_labels(
    *,
    snapshot_parquet_uri: str,
    artifact_dir: str,
) -> tuple[list[str], list[int | None]]:
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(snapshot_parquet_uri)
    table = pq.read_table(io.BytesIO(payload), columns=["text", "label"])
    texts = [str(value) for value in table.column("text").to_pylist()]
    labels = [
        int(value) if value is not None else None
        for value in table.column("label").to_pylist()
    ]
    return texts, labels


def _default_embedding_fetcher(
    *,
    texts: list[str],
    deployment: str,
    provider: str,
    db: DatabaseConnectionV2,
    dataset_key: str,
    entry_max: int,
    snapshot_group_id: int,
    chunk_size: int | None,
    timeout: float,
) -> np.ndarray:
    matrix = asyncio.run(
        fetch_embeddings_async(
            texts_list=texts,
            deployment=deployment,
            db=db,
            timeout=timeout,
            chunk_size=chunk_size,
            provider_name=provider,
            l3_cache_key=dataset_key,
            l3_entry_max=entry_max,
            l3_snapshot_group_id=snapshot_group_id,
        )
    )
    return np.asarray(matrix, dtype=np.float64)


def _pool_label_centroid(
    *,
    matrix: np.ndarray,
    labels: list[int | None],
) -> tuple[np.ndarray, dict[str, Any]]:
    buckets: dict[int, list[np.ndarray]] = {}
    for idx, label in enumerate(labels):
        if label is None:
            continue
        buckets.setdefault(int(label), []).append(matrix[idx])
    if not buckets:
        raise ValueError("label_centroid representation requires at least one labeled row")
    sorted_labels = sorted(buckets.keys())
    pooled = np.vstack(
        [
            np.mean(np.asarray(buckets[label], dtype=np.float64), axis=0)
            for label in sorted_labels
        ]
    )
    metadata = {
        "pooled_label_count": len(sorted_labels),
        "pooled_labels": sorted_labels,
    }
    return pooled, metadata


def _normalize_representation(representation: str) -> tuple[str, list[str]]:
    input_repr = str(representation).strip().lower()
    canonical_repr = LEGACY_REPRESENTATION_ALIASES.get(input_repr, input_repr)
    if canonical_repr not in ALLOWED_REPRESENTATIONS:
        allowed_tokens = sorted(
            ALLOWED_REPRESENTATIONS.union(LEGACY_REPRESENTATION_ALIASES.keys())
        )
        raise ValueError(
            f"representation must be one of {allowed_tokens}, got {representation!r}"
        )
    lookup_reprs = [canonical_repr]
    if input_repr != canonical_repr:
        lookup_reprs.append(input_repr)
    return canonical_repr, lookup_reprs


def embed(
    snapshot_group_id: int,
    *,
    deployment: str,
    provider: str = "azure",
    representation: str = "full",
    force: bool = False,
    entry_max: int | None = None,
    key_version: str = CACHE_KEY_VERSION,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    artifact_dir: str = "artifacts",
    embedding_fetcher: EmbeddingFetcher | None = None,
    chunk_size: int | None = None,
    timeout: float = 600.0,
) -> StageResult:
    """
    Build/reuse an embedding matrix artifact for a snapshot group.
    """
    canonical_repr, lookup_reprs = _normalize_representation(representation)

    db_conn, _owned_db = _resolve_db(db=db, database_url=database_url)
    with db_conn.session_scope() as session:
        snapshot_group = (
            session.query(Group)
            .filter(
                Group.id == int(snapshot_group_id),
                Group.group_type == "dataset_snapshot",
            )
            .first()
        )
        if snapshot_group is None:
            raise ValueError(f"dataset_snapshot group id={snapshot_group_id} not found")
        snapshot_parquet_uri = _find_snapshot_parquet_uri(session, int(snapshot_group_id))
        snapshot_metadata = dict(snapshot_group.metadata_json or {})
        dataset_slug = str(snapshot_metadata.get("dataset_slug") or "unknown_dataset")

    texts, labels = _load_snapshot_texts_and_labels(
        snapshot_parquet_uri=snapshot_parquet_uri,
        artifact_dir=artifact_dir,
    )
    if not texts:
        raise ValueError("snapshot parquet has no rows to embed")

    dataset_key = f"snap:{int(snapshot_group_id)}:{canonical_repr}"
    lookup_dataset_keys = [
        f"snap:{int(snapshot_group_id)}:{lookup_repr}" for lookup_repr in lookup_reprs
    ]
    initial_entry_max = int(entry_max if entry_max is not None else len(texts))

    with db_conn.session_scope() as session:
        repo = RawCallRepository(session)
        artifacts = ArtifactService(repository=repo, artifact_dir=artifact_dir)
        hit = None
        hit_dataset_key = dataset_key
        for lookup_dataset_key in lookup_dataset_keys:
            hit = artifacts.find_embedding_matrix_artifact(
                dataset_key=lookup_dataset_key,
                embedding_engine=deployment,
                provider=provider,
                entry_max=initial_entry_max,
                key_version=key_version,
            )
            if hit is not None:
                hit_dataset_key = lookup_dataset_key
                break
        if hit is not None and not force:
            group_id = int(hit.get("group_id") or 0)
            artifact_uris = {"embedding_matrix.npy": str(hit["uri"])}
            if group_id > 0:
                artifact_uris = _collect_embedding_artifact_uris(session, group_id)
                artifact_uris.setdefault("embedding_matrix.npy", str(hit["uri"]))
            return StageResult(
                stage_name="embed",
                group_id=group_id,
                run_id=None,
                artifact_uris=artifact_uris,
                metadata={
                    "reused": True,
                    "dataset_key": dataset_key,
                    "representation": canonical_repr,
                    "matched_dataset_key": hit_dataset_key,
                },
            )

    fetcher = embedding_fetcher or _default_embedding_fetcher
    base_matrix = np.asarray(
        fetcher(
            texts=texts,
            deployment=deployment,
            provider=provider,
            db=db_conn,
            dataset_key=dataset_key,
            entry_max=initial_entry_max,
            snapshot_group_id=int(snapshot_group_id),
            chunk_size=chunk_size,
            timeout=timeout,
        ),
        dtype=np.float64,
    )
    if base_matrix.ndim != 2:
        raise ValueError(f"embedding_fetcher returned shape {base_matrix.shape}, expected 2D")
    if int(base_matrix.shape[0]) != len(texts):
        raise ValueError(
            "embedding_fetcher row count mismatch: "
            f"{base_matrix.shape[0]} vectors for {len(texts)} snapshot rows"
        )

    matrix = base_matrix
    matrix_metadata: dict[str, Any] = {
        "representation": canonical_repr,
        "source_snapshot_group_id": int(snapshot_group_id),
    }
    if canonical_repr == REPRESENTATION_LABEL_CENTROID:
        matrix, pooled_meta = _pool_label_centroid(matrix=base_matrix, labels=labels)
        matrix_metadata.update(pooled_meta)

    effective_entry_max = int(matrix.shape[0])
    sparse_sidecar: dict[str, Any] | None = None
    if canonical_repr == REPRESENTATION_SPARSE:
        nnz = int(np.count_nonzero(matrix))
        total = int(matrix.size)
        sparse_sidecar = {
            "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "nnz": nnz,
            "density": float(nnz / total) if total > 0 else 0.0,
        }
        matrix_metadata.update(sparse_sidecar)

    def _write_embedding_artifacts(
        artifact_service: ArtifactService,
        identity: StageIdentity,
    ) -> dict[str, str]:
        repo = artifact_service.repository
        if repo is None:
            raise RuntimeError("ArtifactService requires repository for embed stage writes")
        matrix_artifact_id = artifact_service.store_embedding_matrix(
            identity.group_id,
            matrix,
            dataset_key=dataset_key,
            embedding_engine=deployment,
            provider=provider,
            entry_max=effective_entry_max,
            key_version=key_version,
            metadata=matrix_metadata,
        )
        artifact_uris = {
            "embedding_matrix.npy": _call_artifact_uri_by_id(repo, matrix_artifact_id),
        }
        if sparse_sidecar is not None:
            sidecar_id = artifact_service.store_group_blob_artifact(
                group_id=identity.group_id,
                step_name="embedding_matrix",
                logical_filename="sparse_sidecar.json",
                data=json.dumps(
                    sparse_sidecar,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8"),
                artifact_type=ARTIFACT_TYPE_SPARSE_SIDECAR,
                content_type="application/json",
                metadata={
                    "representation": canonical_repr,
                    "dataset_key": dataset_key,
                },
            )
            artifact_uris["sparse_sidecar.json"] = _call_artifact_uri_by_id(repo, sidecar_id)
        return artifact_uris

    result = run_stage(
        db=db_conn,
        stage_name="embed",
        group_type="embedding_batch",
        group_name=f"embed:{dataset_slug}:{snapshot_group_id}:{deployment}:{canonical_repr}",
        group_description=f"Embedding batch for snapshot {snapshot_group_id}",
        group_metadata={
            "dataset_slug": dataset_slug,
            "dataset_key": dataset_key,
            "representation": canonical_repr,
            "provider": provider,
            "embedding_engine": deployment,
            "entry_max": effective_entry_max,
            "key_version": key_version,
            "source_snapshot_group_id": int(snapshot_group_id),
        },
        depends_on_group_ids=[int(snapshot_group_id)],
        artifact_dir=artifact_dir,
        write_artifacts=_write_embedding_artifacts,
    )
    return StageResult(
        stage_name=result.stage_name,
        group_id=result.group_id,
        run_id=result.run_id,
        artifact_uris=result.artifact_uris,
        metadata={
            **result.metadata,
            "reused": False,
            "dataset_key": dataset_key,
            "representation": canonical_repr,
            "row_count": int(matrix.shape[0]),
            "dimension": int(matrix.shape[1]) if matrix.size else 0,
        },
    )

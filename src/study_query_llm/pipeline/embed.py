"""Stage 4: embed (canonical dataframe -> shared full embedding matrix artifact)."""

from __future__ import annotations

import asyncio
import io
import os
from typing import Callable

import numpy as np
import pyarrow.parquet as pq

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.embeddings.constants import CACHE_KEY_VERSION
from study_query_llm.services.embeddings.helpers import fetch_embeddings_async

ARTIFACT_TYPE_EMBEDDING_MATRIX = "embedding_matrix"
REPRESENTATION_FULL = "full"
LEGACY_NON_FULL_ALIASES = {"intent_mean": "label_centroid"}

EmbeddingFetcher = Callable[..., np.ndarray]


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
    return artifact_uris


def _load_dataframe_texts(
    *,
    dataframe_parquet_uri: str,
    artifact_dir: str,
) -> list[str]:
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(dataframe_parquet_uri)
    table = pq.read_table(io.BytesIO(payload), columns=["text"])
    return [str(value) for value in table.column("text").to_pylist()]


def _default_embedding_fetcher(
    *,
    texts: list[str],
    deployment: str,
    provider: str,
    db: DatabaseConnectionV2,
    dataset_key: str,
    entry_max: int,
    chunk_size: int | None,
    chunk_worker_concurrency: int,
    chunk_circuit_breaker_enabled: bool,
    chunk_failure_fallback_threshold: int,
    max_retries: int,
    initial_wait: float,
    max_wait: float,
    singleflight_lease_seconds: int,
    singleflight_wait_timeout_seconds: float,
    singleflight_poll_seconds: float,
    timeout: float,
) -> np.ndarray:
    matrix = asyncio.run(
        fetch_embeddings_async(
            texts_list=texts,
            deployment=deployment,
            db=db,
            timeout=timeout,
            chunk_size=chunk_size,
            chunk_worker_concurrency=chunk_worker_concurrency,
            chunk_circuit_breaker_enabled=chunk_circuit_breaker_enabled,
            chunk_failure_fallback_threshold=chunk_failure_fallback_threshold,
            provider_name=provider,
            max_retries=max_retries,
            initial_wait=initial_wait,
            max_wait=max_wait,
            singleflight_lease_seconds=singleflight_lease_seconds,
            singleflight_wait_timeout_seconds=singleflight_wait_timeout_seconds,
            singleflight_poll_seconds=singleflight_poll_seconds,
            l3_cache_key=dataset_key,
            l3_entry_max=entry_max,
            l3_snapshot_group_id=None,
        )
    )
    return np.asarray(matrix, dtype=np.float64)


def _normalize_representation(representation: str) -> str:
    canonical = LEGACY_NON_FULL_ALIASES.get(str(representation).strip().lower(), str(representation).strip().lower())
    if canonical != REPRESENTATION_FULL:
        raise ValueError(
            "embed only supports representation='full'. Non-full representations "
            "must be derived in analyze from snapshot-sliced full vectors."
        )
    return canonical


def embed(
    dataframe_group_id: int,
    *,
    deployment: str,
    provider: str = "azure",
    representation: str = "full",
    force: bool = False,
    entry_max: int | None = None,
    key_version: str = CACHE_KEY_VERSION,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    write_intent: WriteIntent | str | None = WriteIntent.CANONICAL,
    artifact_dir: str = "artifacts",
    embedding_fetcher: EmbeddingFetcher | None = None,
    chunk_size: int | None = None,
    chunk_worker_concurrency: int = 1,
    chunk_circuit_breaker_enabled: bool = False,
    chunk_failure_fallback_threshold: int = 2,
    max_retries: int = 6,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    singleflight_lease_seconds: int = 45,
    singleflight_wait_timeout_seconds: float = 90.0,
    singleflight_poll_seconds: float = 0.1,
    timeout: float = 600.0,
) -> StageResult:
    """Build/reuse a dataframe-scoped full embedding matrix artifact."""
    canonical_repr = _normalize_representation(representation)
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
        dataframe_metadata = dict(dataframe_group.metadata_json or {})
        dataset_slug = str(dataframe_metadata.get("dataset_slug") or "unknown_dataset")
        dataframe_parquet_uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))

    texts = _load_dataframe_texts(
        dataframe_parquet_uri=dataframe_parquet_uri,
        artifact_dir=artifact_dir,
    )
    if not texts:
        raise ValueError("canonical dataframe has no text rows to embed")
    if entry_max is not None:
        bounded_entry_max = int(entry_max)
        if bounded_entry_max <= 0:
            raise ValueError("entry_max must be a positive integer when provided")
        # entry_max is intended to limit how many dataframe rows are embedded.
        # Keep behavior explicit here so cache keys and matrix shape align.
        texts = texts[:bounded_entry_max]

    dataset_key = f"dataframe:{int(dataframe_group_id)}:{canonical_repr}"
    initial_entry_max = int(len(texts))

    with db_conn.session_scope() as session:
        repo = RawCallRepository(session)
        artifacts = ArtifactService(repository=repo, artifact_dir=artifact_dir)
        hit = artifacts.find_embedding_matrix_artifact(
            dataset_key=dataset_key,
            embedding_engine=deployment,
            provider=provider,
            entry_max=initial_entry_max,
            key_version=key_version,
        )
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
                },
            )

    fetcher = embedding_fetcher or _default_embedding_fetcher
    matrix = np.asarray(
        fetcher(
            texts=texts,
            deployment=deployment,
            provider=provider,
            db=db_conn,
            dataset_key=dataset_key,
            entry_max=initial_entry_max,
            chunk_size=chunk_size,
            chunk_worker_concurrency=int(chunk_worker_concurrency),
            chunk_circuit_breaker_enabled=bool(chunk_circuit_breaker_enabled),
            chunk_failure_fallback_threshold=int(chunk_failure_fallback_threshold),
            max_retries=int(max_retries),
            initial_wait=float(initial_wait),
            max_wait=float(max_wait),
            singleflight_lease_seconds=int(singleflight_lease_seconds),
            singleflight_wait_timeout_seconds=float(singleflight_wait_timeout_seconds),
            singleflight_poll_seconds=float(singleflight_poll_seconds),
            timeout=timeout,
        ),
        dtype=np.float64,
    )
    if matrix.ndim != 2:
        raise ValueError(f"embedding_fetcher returned shape {matrix.shape}, expected 2D")
    if int(matrix.shape[0]) != len(texts):
        raise ValueError(
            "embedding_fetcher row count mismatch: "
            f"{matrix.shape[0]} vectors for {len(texts)} dataframe rows"
        )

    effective_entry_max = int(matrix.shape[0])

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
            metadata={
                "representation": canonical_repr,
                "source_dataframe_group_id": int(dataframe_group_id),
            },
        )
        return {
            "embedding_matrix.npy": _call_artifact_uri_by_id(repo, matrix_artifact_id),
        }

    result = run_stage(
        db=db_conn,
        stage_name="embed",
        group_type="embedding_batch",
        group_name=f"embed:{dataset_slug}:{dataframe_group_id}:{deployment}:{canonical_repr}",
        group_description=f"Embedding batch for dataframe {dataframe_group_id}",
        group_metadata={
            "dataset_slug": dataset_slug,
            "dataset_key": dataset_key,
            "representation": canonical_repr,
            "provider": provider,
            "embedding_engine": deployment,
            "entry_max": effective_entry_max,
            "chunk_worker_concurrency": int(chunk_worker_concurrency),
            "chunk_circuit_breaker_enabled": bool(chunk_circuit_breaker_enabled),
            "chunk_failure_fallback_threshold": int(chunk_failure_fallback_threshold),
            "key_version": key_version,
            "source_dataframe_group_id": int(dataframe_group_id),
        },
        depends_on_group_ids=[int(dataframe_group_id)],
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

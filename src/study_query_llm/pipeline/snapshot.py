"""Stage 2: snapshot (dataset acquisition artifacts -> snapshot parquet)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.parser_protocol import ParserCallable, ParserContext
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import SnapshotRow, StageResult
from study_query_llm.services.artifact_service import ArtifactService

ARTIFACT_TYPE_ACQ_FILE = "dataset_acquisition_file"
ARTIFACT_TYPE_ACQ_MANIFEST = "dataset_acquisition_manifest"
ARTIFACT_TYPE_SNAPSHOT_PARQUET = "dataset_snapshot_parquet"
ARTIFACT_TYPE_SNAPSHOT_MANIFEST = "dataset_snapshot_manifest"


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


def _collect_acquisition_artifact_uris(session, dataset_group_id: int) -> dict[str, str]:
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
    artifact_uris: dict[str, str] = {}
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) != int(dataset_group_id):
            continue
        if artifact.artifact_type == ARTIFACT_TYPE_ACQ_FILE:
            rel_path = str(metadata.get("relative_path") or "").strip()
            if rel_path:
                artifact_uris[rel_path] = str(artifact.uri)
        elif artifact.artifact_type == ARTIFACT_TYPE_ACQ_MANIFEST:
            artifact_uris["acquisition.json"] = str(artifact.uri)
    return artifact_uris


def _collect_snapshot_artifact_uris(session, snapshot_group_id: int) -> dict[str, str]:
    artifacts = session.query(CallArtifact).order_by(CallArtifact.id.asc()).all()
    artifact_uris: dict[str, str] = {}
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if int(metadata.get("group_id") or -1) != int(snapshot_group_id):
            continue
        if artifact.artifact_type == ARTIFACT_TYPE_SNAPSHOT_PARQUET:
            artifact_uris["snapshot.parquet"] = str(artifact.uri)
        elif artifact.artifact_type == ARTIFACT_TYPE_SNAPSHOT_MANIFEST:
            artifact_uris["snapshot_index.json"] = str(artifact.uri)
    return artifact_uris


def _resolve_parser(
    *,
    dataset_slug: str,
    parser: ParserCallable | None,
) -> ParserCallable:
    if parser is not None:
        return parser
    cfg = ACQUIRE_REGISTRY.get(dataset_slug)
    if cfg is None or cfg.default_parser is None:
        raise ValueError(
            f"No parser provided and no default parser registered for dataset slug {dataset_slug!r}"
        )
    return cfg.default_parser


def _parser_identity(parser: ParserCallable) -> str:
    return f"{getattr(parser, '__module__', 'unknown')}.{getattr(parser, '__qualname__', repr(parser))}"


def _materialize_acquisition_files(
    *,
    artifact_uris: dict[str, str],
    target_dir: Path,
    artifact_dir: str,
) -> None:
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    for relative_path, uri in artifact_uris.items():
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact_service.storage.read_from_uri(uri)
        destination.write_bytes(payload)


def _build_snapshot_payload(rows: list[SnapshotRow]) -> tuple[bytes, bytes, dict[str, int], str]:
    rows_sorted = sorted(rows, key=lambda row: int(row.position))
    extra_json = [
        json.dumps(dict(row.extra or {}), sort_keys=True, ensure_ascii=False)
        for row in rows_sorted
    ]
    table = pa.table(
        {
            "position": pa.array([int(row.position) for row in rows_sorted], type=pa.int64()),
            "source_id": pa.array([str(row.source_id) for row in rows_sorted], type=pa.string()),
            "text": pa.array([str(row.text) for row in rows_sorted], type=pa.string()),
            "label": pa.array([row.label for row in rows_sorted], type=pa.int64()),
            "label_name": pa.array([row.label_name for row in rows_sorted], type=pa.string()),
            "extra_json": pa.array(extra_json, type=pa.string()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(table, buf)
    parquet_bytes = buf.getvalue()
    snapshot_hash = hashlib.sha256(parquet_bytes).hexdigest()
    label_count = len({int(row.label) for row in rows_sorted if row.label is not None})
    index_payload = {
        "row_count": len(rows_sorted),
        "label_count": label_count,
    }
    index_bytes = json.dumps(index_payload, indent=2, ensure_ascii=False).encode("utf-8")
    return parquet_bytes, index_bytes, index_payload, snapshot_hash


def _find_existing_snapshot_group(
    session,
    *,
    dataset_group_id: int,
    parser_identity: str,
    representation: str,
    snapshot_hash: str,
) -> int | None:
    snapshot_groups = (
        session.query(Group)
        .filter(Group.group_type == "dataset_snapshot")
        .order_by(Group.id.desc())
        .all()
    )
    for group in snapshot_groups:
        metadata = dict(group.metadata_json or {})
        if int(metadata.get("source_dataset_group_id") or -1) != int(dataset_group_id):
            continue
        if str(metadata.get("parser_identity") or "") != parser_identity:
            continue
        if str(metadata.get("representation") or "") != representation:
            continue
        if str(metadata.get("snapshot_manifest_hash") or "") != snapshot_hash:
            continue
        return int(group.id)
    return None


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def snapshot(
    dataset_group_id: int,
    *,
    parser: ParserCallable | None = None,
    representation: str = "raw",
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    artifact_dir: str = "artifacts",
) -> StageResult:
    """
    Build deterministic snapshot parquet from an acquired dataset group.
    """
    db_conn, _owned_db = _resolve_db(db=db, database_url=database_url)
    with db_conn.session_scope() as session:
        dataset_group = (
            session.query(Group)
            .filter(Group.id == int(dataset_group_id), Group.group_type == "dataset")
            .first()
        )
        if dataset_group is None:
            raise ValueError(f"dataset group id={dataset_group_id} not found")
        group_metadata = dict(dataset_group.metadata_json or {})
        dataset_slug = str(group_metadata.get("dataset_slug") or "")
        if not dataset_slug:
            raise ValueError(
                f"dataset group id={dataset_group_id} is missing metadata_json['dataset_slug']"
            )
        acquisition_artifact_uris = _collect_acquisition_artifact_uris(session, dataset_group_id)
        if not acquisition_artifact_uris:
            raise ValueError(
                f"dataset group id={dataset_group_id} has no acquisition artifacts to snapshot"
            )

    resolved_parser = _resolve_parser(dataset_slug=dataset_slug, parser=parser)
    parser_id = _parser_identity(resolved_parser)

    with tempfile.TemporaryDirectory(prefix=f"snapshot_{dataset_group_id}_") as temp_dir:
        local_dir = Path(temp_dir)
        _materialize_acquisition_files(
            artifact_uris=acquisition_artifact_uris,
            target_dir=local_dir,
            artifact_dir=artifact_dir,
        )
        parser_ctx = ParserContext(
            dataset_group_id=int(dataset_group_id),
            artifact_uris=acquisition_artifact_uris,
            artifact_dir_local=local_dir,
            source_metadata=group_metadata.get("source") or {},
        )
        rows = list(resolved_parser(parser_ctx))

    parquet_bytes, index_bytes, index_payload, snapshot_hash = _build_snapshot_payload(rows)
    with db_conn.session_scope() as session:
        existing_group_id = _find_existing_snapshot_group(
            session,
            dataset_group_id=int(dataset_group_id),
            parser_identity=parser_id,
            representation=str(representation),
            snapshot_hash=snapshot_hash,
        )
        if existing_group_id is not None and not force:
            artifact_uris = _collect_snapshot_artifact_uris(session, existing_group_id)
            return StageResult(
                stage_name="snapshot",
                group_id=existing_group_id,
                run_id=None,
                artifact_uris=artifact_uris,
                metadata={"reused": True, "snapshot_manifest_hash": snapshot_hash},
            )

    def _write_snapshot_artifacts(artifact_service, identity: StageIdentity) -> dict[str, str]:
        repo = artifact_service.repository
        if repo is None:
            raise RuntimeError("ArtifactService requires repository for snapshot stage writes")
        parquet_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="snapshot",
            logical_filename="snapshot.parquet",
            data=parquet_bytes,
            artifact_type=ARTIFACT_TYPE_SNAPSHOT_PARQUET,
            content_type="application/octet-stream",
            metadata={
                "snapshot_manifest_hash": snapshot_hash,
                "row_count": index_payload["row_count"],
                "label_count": index_payload["label_count"],
                "representation": representation,
            },
        )
        index_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="snapshot",
            logical_filename="snapshot_index.json",
            data=index_bytes,
            artifact_type=ARTIFACT_TYPE_SNAPSHOT_MANIFEST,
            content_type="application/json",
            metadata={
                "snapshot_manifest_hash": snapshot_hash,
                "representation": representation,
            },
        )
        return {
            "snapshot.parquet": _call_artifact_uri_by_id(repo, parquet_artifact_id),
            "snapshot_index.json": _call_artifact_uri_by_id(repo, index_artifact_id),
        }

    result = run_stage(
        db=db_conn,
        stage_name="snapshot",
        group_type="dataset_snapshot",
        group_name=f"snap:{dataset_slug}:{representation}:{dataset_group_id}",
        group_description=f"Dataset snapshot: {dataset_slug} ({representation})",
        group_metadata={
            "dataset_slug": dataset_slug,
            "source_dataset_group_id": int(dataset_group_id),
            "representation": str(representation),
            "parser_identity": parser_id,
            "snapshot_manifest_hash": snapshot_hash,
            "row_count": index_payload["row_count"],
            "label_count": index_payload["label_count"],
        },
        depends_on_group_ids=[int(dataset_group_id)],
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
            "snapshot_manifest_hash": snapshot_hash,
            "row_count": index_payload["row_count"],
            "label_count": index_payload["label_count"],
        },
    )

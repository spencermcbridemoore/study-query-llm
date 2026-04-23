"""Stage 2: parse (acquired artifacts -> canonical dataframe parquet)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.parser_protocol import ParserCallable, ParserContext
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import StageIdentity, allow_no_run_stage, run_stage
from study_query_llm.pipeline.types import SnapshotRow, StageResult
from study_query_llm.services.artifact_service import ArtifactService

ARTIFACT_TYPE_ACQ_FILE = "dataset_acquisition_file"
ARTIFACT_TYPE_ACQ_MANIFEST = "dataset_acquisition_manifest"
ARTIFACT_TYPE_CANONICAL_PARQUET = "dataset_canonical_parquet"
ARTIFACT_TYPE_DATAFRAME_MANIFEST = "dataset_dataframe_manifest"


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
    repo = RawCallRepository(session)
    artifacts = repo.list_group_artifacts(
        group_id=int(dataset_group_id),
        artifact_types=[ARTIFACT_TYPE_ACQ_FILE, ARTIFACT_TYPE_ACQ_MANIFEST],
    )
    artifact_uris: dict[str, str] = {}
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if artifact.artifact_type == ARTIFACT_TYPE_ACQ_FILE:
            rel_path = str(metadata.get("relative_path") or "").strip()
            if rel_path:
                artifact_uris[rel_path] = str(artifact.uri)
        elif artifact.artifact_type == ARTIFACT_TYPE_ACQ_MANIFEST:
            artifact_uris["acquisition.json"] = str(artifact.uri)
    return artifact_uris


def _collect_dataframe_artifact_uris(session, dataframe_group_id: int) -> dict[str, str]:
    repo = RawCallRepository(session)
    artifacts = repo.list_group_artifacts(
        group_id=int(dataframe_group_id),
        artifact_types=[ARTIFACT_TYPE_CANONICAL_PARQUET, ARTIFACT_TYPE_DATAFRAME_MANIFEST],
    )
    artifact_uris: dict[str, str] = {}
    for artifact in artifacts:
        metadata = dict(artifact.metadata_json or {})
        if artifact.artifact_type == ARTIFACT_TYPE_CANONICAL_PARQUET:
            artifact_uris["dataframe.parquet"] = str(artifact.uri)
        elif artifact.artifact_type == ARTIFACT_TYPE_DATAFRAME_MANIFEST:
            artifact_uris["dataframe_manifest.json"] = str(artifact.uri)
    return artifact_uris


def _materialize_acquisition_files(
    *,
    artifact_uris: dict[str, str],
    target_dir: Path,
    artifact_dir: str,
) -> None:
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    for relative_path, uri in artifact_uris.items():
        if relative_path == "acquisition.json":
            continue
        destination = target_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = artifact_service.storage.read_from_uri(uri)
        destination.write_bytes(payload)


def _resolve_parser_and_identity(
    *,
    dataset_slug: str,
    parser: ParserCallable | None,
    parser_id: str | None,
    parser_version: str | None,
) -> tuple[ParserCallable, str, str]:
    cfg = ACQUIRE_REGISTRY.get(dataset_slug)
    resolved_parser = parser
    resolved_parser_id = (parser_id or "").strip()
    resolved_parser_version = (parser_version or "").strip()
    if resolved_parser is None:
        if cfg is None or cfg.default_parser is None:
            raise ValueError(
                f"No parser provided and no default parser registered for dataset slug {dataset_slug!r}"
            )
        resolved_parser = cfg.default_parser
        resolved_parser_id = str(cfg.default_parser_id or "").strip()
        resolved_parser_version = str(cfg.default_parser_version or "").strip()
    elif not resolved_parser_id or not resolved_parser_version:
        if (
            cfg is not None
            and cfg.default_parser is resolved_parser
            and cfg.default_parser_id
            and cfg.default_parser_version
        ):
            resolved_parser_id = str(cfg.default_parser_id)
            resolved_parser_version = str(cfg.default_parser_version)
        else:
            raise ValueError(
                "parser overrides must provide parser_id and parser_version "
                "for stable parse idempotency."
            )
    if not resolved_parser_id or not resolved_parser_version:
        raise ValueError("resolved parser identity is incomplete")
    return resolved_parser, resolved_parser_id, resolved_parser_version


def _build_dataframe_payload(
    rows: list[SnapshotRow],
) -> tuple[bytes, bytes, dict[str, Any], str]:
    if not rows:
        raise ValueError("parser returned no rows; cannot build canonical dataframe")
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
    parquet_buf = io.BytesIO()
    pq.write_table(table, parquet_buf)
    parquet_bytes = parquet_buf.getvalue()
    dataframe_hash = hashlib.sha256(parquet_bytes).hexdigest()
    label_count = len({int(row.label) for row in rows_sorted if row.label is not None})
    manifest_payload = {
        "row_count": len(rows_sorted),
        "label_count": label_count,
        "dataframe_hash": dataframe_hash,
    }
    manifest_bytes = json.dumps(
        manifest_payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return parquet_bytes, manifest_bytes, manifest_payload, dataframe_hash


def _find_existing_dataframe_group(
    session,
    *,
    dataset_group_id: int,
    parser_id: str,
    parser_version: str,
    dataframe_hash: str,
) -> int | None:
    repo = RawCallRepository(session)
    return repo.find_group_id_by_metadata(
        group_type="dataset_dataframe",
        metadata_eq={
            "source_dataset_group_id": int(dataset_group_id),
            "parser_id": str(parser_id),
            "parser_version": str(parser_version),
            "dataframe_hash": str(dataframe_hash),
        },
    )


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


@allow_no_run_stage
def find_dataframe_parquet_uri(session, dataframe_group_id: int) -> str:
    """Return canonical dataframe parquet URI for a dataframe group."""
    repo = RawCallRepository(session)
    artifacts = repo.list_group_artifacts(
        group_id=int(dataframe_group_id),
        artifact_types=[ARTIFACT_TYPE_CANONICAL_PARQUET],
        newest_first=True,
    )
    for artifact in artifacts:
        if artifact.artifact_type == ARTIFACT_TYPE_CANONICAL_PARQUET:
            return str(artifact.uri)
    raise ValueError(
        f"dataframe group id={dataframe_group_id} has no {ARTIFACT_TYPE_CANONICAL_PARQUET} artifact"
    )


def parse(
    dataset_group_id: int,
    *,
    parser: ParserCallable | None = None,
    parser_id: str | None = None,
    parser_version: str | None = None,
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    artifact_dir: str = "artifacts",
) -> StageResult:
    """Build deterministic canonical dataframe parquet from an acquired dataset group."""
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
                f"dataset group id={dataset_group_id} missing metadata_json['dataset_slug']"
            )
        content_fingerprint = str(group_metadata.get("content_fingerprint") or "")
        acquisition_artifact_uris = _collect_acquisition_artifact_uris(session, int(dataset_group_id))
        if not acquisition_artifact_uris:
            raise ValueError(
                f"dataset group id={dataset_group_id} has no acquisition artifacts to parse"
            )

    resolved_parser, resolved_parser_id, resolved_parser_version = _resolve_parser_and_identity(
        dataset_slug=dataset_slug,
        parser=parser,
        parser_id=parser_id,
        parser_version=parser_version,
    )

    with tempfile.TemporaryDirectory(prefix=f"parse_{dataset_group_id}_") as temp_dir:
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
            parser_id=resolved_parser_id,
            parser_version=resolved_parser_version,
        )
        rows = list(resolved_parser(parser_ctx))

    parquet_bytes, manifest_bytes, manifest_payload, dataframe_hash = _build_dataframe_payload(rows)
    with db_conn.session_scope() as session:
        existing_group_id = _find_existing_dataframe_group(
            session,
            dataset_group_id=int(dataset_group_id),
            parser_id=resolved_parser_id,
            parser_version=resolved_parser_version,
            dataframe_hash=dataframe_hash,
        )
        if existing_group_id is not None and not force:
            artifact_uris = _collect_dataframe_artifact_uris(session, existing_group_id)
            return StageResult(
                stage_name="parse",
                group_id=existing_group_id,
                run_id=None,
                artifact_uris=artifact_uris,
                metadata={
                    "reused": True,
                    "parser_id": resolved_parser_id,
                    "parser_version": resolved_parser_version,
                    "dataframe_hash": dataframe_hash,
                },
            )

    def _write_dataframe_artifacts(
        artifact_service: ArtifactService,
        identity: StageIdentity,
    ) -> dict[str, str]:
        repo = artifact_service.repository
        if repo is None:
            raise RuntimeError("ArtifactService requires repository for parse stage writes")
        parquet_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="parse",
            logical_filename="dataframe.parquet",
            data=parquet_bytes,
            artifact_type=ARTIFACT_TYPE_CANONICAL_PARQUET,
            content_type="application/octet-stream",
            metadata={
                "dataframe_hash": dataframe_hash,
                "row_count": manifest_payload["row_count"],
                "label_count": manifest_payload["label_count"],
                "parser_id": resolved_parser_id,
                "parser_version": resolved_parser_version,
            },
        )
        manifest_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="parse",
            logical_filename="dataframe_manifest.json",
            data=manifest_bytes,
            artifact_type=ARTIFACT_TYPE_DATAFRAME_MANIFEST,
            content_type="application/json",
            metadata={
                "dataframe_hash": dataframe_hash,
                "parser_id": resolved_parser_id,
                "parser_version": resolved_parser_version,
            },
        )
        return {
            "dataframe.parquet": _call_artifact_uri_by_id(repo, parquet_artifact_id),
            "dataframe_manifest.json": _call_artifact_uri_by_id(repo, manifest_artifact_id),
        }

    result = run_stage(
        db=db_conn,
        stage_name="parse",
        group_type="dataset_dataframe",
        group_name=f"parse:{dataset_slug}:{resolved_parser_id}:{dataset_group_id}",
        group_description=f"Canonical dataframe for dataset {dataset_group_id}",
        group_metadata={
            "dataset_slug": dataset_slug,
            "source_dataset_group_id": int(dataset_group_id),
            "parser_id": resolved_parser_id,
            "parser_version": resolved_parser_version,
            "dataframe_hash": dataframe_hash,
            "row_count": manifest_payload["row_count"],
            "label_count": manifest_payload["label_count"],
            "content_fingerprint": content_fingerprint,
        },
        depends_on_group_ids=[int(dataset_group_id)],
        artifact_dir=artifact_dir,
        write_artifacts=_write_dataframe_artifacts,
    )
    return StageResult(
        stage_name=result.stage_name,
        group_id=result.group_id,
        run_id=result.run_id,
        artifact_uris=result.artifact_uris,
        metadata={
            **result.metadata,
            "reused": False,
            "parser_id": resolved_parser_id,
            "parser_version": resolved_parser_version,
            "dataframe_hash": dataframe_hash,
            "row_count": manifest_payload["row_count"],
            "label_count": manifest_payload["label_count"],
        },
    )

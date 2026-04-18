"""Stage 1: acquisition (source bytes -> dataset group + artifacts)."""

from __future__ import annotations

import json
import os
from typing import Callable

from study_query_llm.datasets.acquisition import (
    FetchedFile,
    acquisition_manifest_sha256,
    build_acquisition_manifest,
    content_fingerprint,
    download_acquisition_files,
    fetch_url,
)
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import StageIdentity, run_stage
from study_query_llm.pipeline.types import StageResult

ARTIFACT_TYPE_ACQ_FILE = "dataset_acquisition_file"
ARTIFACT_TYPE_ACQ_MANIFEST = "dataset_acquisition_manifest"


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
    artifacts = (
        session.query(CallArtifact)
        .order_by(CallArtifact.id.asc())
        .all()
    )
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


def _find_dataset_group_by_fingerprint(
    session,
    *,
    fingerprint: str,
) -> int | None:
    dataset_groups = (
        session.query(Group)
        .filter(Group.group_type == "dataset")
        .order_by(Group.id.desc())
        .all()
    )
    for group in dataset_groups:
        metadata = dict(group.metadata_json or {})
        if str(metadata.get("content_fingerprint") or "") == fingerprint:
            return int(group.id)
    return None


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def acquire(
    spec: DatasetAcquireConfig,
    *,
    force: bool = False,
    db: DatabaseConnectionV2 | None = None,
    database_url: str | None = None,
    artifact_dir: str = "artifacts",
    fetch: Callable[[str], bytes] = fetch_url,
) -> StageResult:
    """
    Acquire source bytes and persist a dataset group with acquisition artifacts.
    """
    db_conn, _owned_db = _resolve_db(db=db, database_url=database_url)
    fetched_files: list[FetchedFile] = download_acquisition_files(
        list(spec.file_specs()),
        fetch=fetch,
    )
    source_metadata = dict(spec.source_metadata() or {})
    manifest = build_acquisition_manifest(
        dataset_slug=spec.slug,
        source=source_metadata,
        files=fetched_files,
        runner_script="study_query_llm.pipeline.acquire",
    )
    manifest_hash = acquisition_manifest_sha256(manifest)
    fingerprint = content_fingerprint(dataset_slug=spec.slug, manifest=manifest)

    with db_conn.session_scope() as session:
        existing_group_id = _find_dataset_group_by_fingerprint(
            session,
            fingerprint=fingerprint,
        )
        if existing_group_id is not None and not force:
            artifact_uris = _collect_acquisition_artifact_uris(session, existing_group_id)
            return StageResult(
                stage_name="acquire",
                group_id=existing_group_id,
                run_id=None,
                artifact_uris=artifact_uris,
                metadata={"reused": True, "content_fingerprint": fingerprint},
            )

    manifest_bytes = json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    def _write_acquisition_artifacts(artifact_service, identity: StageIdentity) -> dict[str, str]:
        repo = artifact_service.repository
        if repo is None:
            raise RuntimeError("ArtifactService requires repository for stage acquire writes")
        artifact_uris: dict[str, str] = {}
        for fetched in fetched_files:
            logical_filename = fetched.relative_path.replace("/", "_").replace("\\", "_")
            artifact_id = artifact_service.store_group_blob_artifact(
                group_id=identity.group_id,
                step_name="acquisition",
                logical_filename=logical_filename,
                data=fetched.data,
                artifact_type=ARTIFACT_TYPE_ACQ_FILE,
                content_type="application/octet-stream",
                metadata={
                    "source_url": fetched.url,
                    "sha256": fetched.sha256,
                    "relative_path": fetched.relative_path,
                    "byte_size": fetched.byte_size,
                },
            )
            artifact_uris[fetched.relative_path] = _call_artifact_uri_by_id(repo, artifact_id)

        manifest_artifact_id = artifact_service.store_group_blob_artifact(
            group_id=identity.group_id,
            step_name="acquisition",
            logical_filename="acquisition.json",
            data=manifest_bytes,
            artifact_type=ARTIFACT_TYPE_ACQ_MANIFEST,
            content_type="application/json",
            metadata={
                "manifest_hash": manifest_hash,
                "dataset_slug": spec.slug,
            },
        )
        artifact_uris["acquisition.json"] = _call_artifact_uri_by_id(repo, manifest_artifact_id)
        return artifact_uris

    result = run_stage(
        db=db_conn,
        stage_name="acquire",
        group_type="dataset",
        group_name=f"acq:{spec.slug}:{fingerprint[:8]}",
        group_description=f"Dataset acquisition: {spec.slug}",
        group_metadata={
            "dataset_slug": spec.slug,
            "source": source_metadata,
            "content_fingerprint": fingerprint,
            "manifest_hash": manifest_hash,
            "file_count": len(fetched_files),
        },
        artifact_dir=artifact_dir,
        write_artifacts=_write_acquisition_artifacts,
    )
    return StageResult(
        stage_name=result.stage_name,
        group_id=result.group_id,
        run_id=result.run_id,
        artifact_uris=result.artifact_uris,
        metadata={
            **result.metadata,
            "reused": False,
            "content_fingerprint": fingerprint,
            "manifest_hash": manifest_hash,
        },
    )

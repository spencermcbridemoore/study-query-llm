"""Shared stage runner enforcing persistence ordering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.types import StageResult
from study_query_llm.services.artifact_service import ArtifactService

RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_RUNNING = "running"


@dataclass(frozen=True)
class StageIdentity:
    """DB identifiers claimed for a stage execution."""

    stage_name: str
    group_id: int
    run_id: int | None
    group_name: str
    group_type: str
    request_group_id: int | None


StageArtifactWriter = Callable[[ArtifactService, StageIdentity], dict[str, str] | None]
StageFinalizeHook = Callable[
    [RawCallRepository, StageIdentity, dict[str, str]],
    dict[str, Any] | None,
]


def cap_group_name(name: str, max_len: int = 180) -> tuple[str, str | None]:
    """
    Cap group name length while preserving a deterministic suffix hash.

    Returns `(capped_name, full_name_or_none)`.
    """
    if max_len < 16:
        raise ValueError("max_len must be >= 16")
    if len(name) <= max_len:
        return name, None
    suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    prefix_len = max_len - 9  # ':' + 8 hash chars
    capped = f"{name[:prefix_len]}:{suffix}"
    return capped, name


def allow_no_run_stage(func: Callable[..., Any]) -> Callable[..., Any]:
    """Escape hatch decorator for non-stage helper functions."""
    return func


def _normalize_artifact_uris(raw: dict[str, str] | None) -> dict[str, str]:
    if not raw:
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def run_stage(
    *,
    db: DatabaseConnectionV2,
    stage_name: str,
    group_type: str,
    group_name: str,
    group_description: str | None = None,
    group_metadata: dict[str, Any] | None = None,
    request_group_id: int | None = None,
    source_group_id: int | None = None,
    run_key: str | None = None,
    run_kind: str = "execution",
    run_metadata: dict[str, Any] | None = None,
    depends_on_group_ids: Sequence[int] | None = None,
    artifact_dir: str = "artifacts",
    write_artifacts: StageArtifactWriter | None = None,
    finalize_db: StageFinalizeHook | None = None,
) -> StageResult:
    """
    Execute a stage with strict ordering:
    1) Claim DB identity
    2) Write artifacts
    3) Mark run completed (or failed on error)
    """
    depends_on = [int(group_id) for group_id in (depends_on_group_ids or ())]
    identity: StageIdentity | None = None
    artifact_uris: dict[str, str] = {}
    result_metadata: dict[str, Any] = {}

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        capped_name, full_name = cap_group_name(group_name)
        metadata_json = dict(group_metadata or {})
        if full_name is not None:
            metadata_json.setdefault("full_name", full_name)
        group_id = repo.create_group(
            group_type=group_type,
            name=capped_name,
            description=group_description,
            metadata_json=metadata_json,
        )
        for dependency_group_id in depends_on:
            repo.create_group_link(
                parent_group_id=group_id,
                child_group_id=dependency_group_id,
                link_type="depends_on",
            )

        run_id: int | None = None
        if request_group_id is not None and run_key:
            run_meta = {"stage_name": stage_name}
            if run_metadata:
                run_meta.update(dict(run_metadata))
            run_id = repo.create_provenanced_run(
                run_kind=run_kind,
                run_status=RUN_STATUS_RUNNING,
                request_group_id=int(request_group_id),
                source_group_id=int(source_group_id) if source_group_id is not None else None,
                result_group_id=int(group_id),
                run_key=str(run_key),
                metadata_json=run_meta,
            )
        identity = StageIdentity(
            stage_name=stage_name,
            group_id=int(group_id),
            run_id=int(run_id) if run_id is not None else None,
            group_name=capped_name,
            group_type=group_type,
            request_group_id=int(request_group_id) if request_group_id is not None else None,
        )

    try:
        if write_artifacts is not None:
            with db.session_scope() as session:
                artifact_repo = RawCallRepository(session)
                artifact_service = ArtifactService(
                    repository=artifact_repo,
                    artifact_dir=artifact_dir,
                )
                artifact_uris = _normalize_artifact_uris(
                    write_artifacts(artifact_service, identity)
                )

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            if finalize_db is not None:
                result_metadata = dict(finalize_db(repo, identity, artifact_uris) or {})
            if identity.run_id is not None:
                repo.update_provenanced_run(
                    identity.run_id,
                    run_status=RUN_STATUS_COMPLETED,
                )
    except Exception as exc:
        if identity is not None and identity.run_id is not None:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                row = repo.get_provenanced_run_by_id(identity.run_id)
                metadata = dict((row.metadata_json or {}) if row else {})
                metadata["stage_failure"] = str(exc)
                repo.update_provenanced_run(
                    identity.run_id,
                    run_status=RUN_STATUS_FAILED,
                    metadata_json=metadata,
                )
        raise

    return StageResult(
        stage_name=stage_name,
        group_id=identity.group_id,
        run_id=identity.run_id,
        artifact_uris=artifact_uris,
        metadata=result_metadata,
    )

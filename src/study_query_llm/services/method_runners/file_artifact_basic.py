"""File-artifact source runner for polymorphic method execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
)


class FileArtifactBasicParams(BaseModel):
    """Boundary-validated parameter payload for source file artifact storage."""

    file_path: str = Field(min_length=1)
    registered_name: str = Field(min_length=1)
    registered_version: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    expected_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("file_path", "registered_name", "registered_version", "content_type")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


async def run_file_artifact_basic(
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Persist caller-provided file bytes as a source-stage artifact."""
    try:
        parsed = FileArtifactBasicParams.model_validate(parameters or {})
    except ValidationError as exc:
        raise ValueError(f"invalid_file_artifact_basic_parameters: {exc}") from exc

    file_path = Path(parsed.file_path).expanduser()
    if not file_path.exists():
        raise ValueError(f"file_not_found:{file_path}")
    if not file_path.is_file():
        raise ValueError(f"file_path_not_file:{file_path}")

    file_bytes = file_path.read_bytes()
    actual_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if actual_sha256 != parsed.expected_sha256:
        raise ValueError(
            "file_sha256_mismatch:"
            f"expected={parsed.expected_sha256},actual={actual_sha256}"
        )

    logical_filename = file_path.name or f"{parsed.registered_name}.bin"
    artifact_service = ArtifactService(repository=context.repository)
    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=int(context.source_group_id),
        step_name="file_artifact",
        logical_filename=logical_filename,
        data=file_bytes,
        artifact_type="dataset_source_file",
        content_type=parsed.content_type,
        metadata={
            "registered_name": parsed.registered_name,
            "registered_version": parsed.registered_version,
            "sha256": actual_sha256,
            "byte_size": len(file_bytes),
            "source_file_path": str(file_path),
        },
    )
    artifact_uri = _call_artifact_uri_by_id(context.repository, int(artifact_id))

    return MethodRunnerResult(
        output_json={
            "dataset_name": parsed.registered_name,
            "dataset_version": parsed.registered_version,
            "content_type": parsed.content_type,
            "sha256": actual_sha256,
            "byte_size": len(file_bytes),
            "artifact_id": int(artifact_id),
            "artifact_uri": artifact_uri,
        },
        pipeline_stage_role="source",
        result_ref=artifact_uri,
        pipeline_stage_context={
            "dataset_name": parsed.registered_name,
            "dataset_version": parsed.registered_version,
            "content_type": parsed.content_type,
            "sha256": actual_sha256,
            "byte_size": len(file_bytes),
            "logical_filename": logical_filename,
        },
        metadata_json={
            "dataset_name": parsed.registered_name,
            "dataset_version": parsed.registered_version,
            "content_type": parsed.content_type,
            "sha256": actual_sha256,
            "byte_size": len(file_bytes),
            "artifact_id": int(artifact_id),
        },
    )


"""Execution service for polymorphic runtime-invokable methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    ensure_default_method_runtimes_registered,
    get_method_runtime,
    run_method_runtime,
)
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenanced_run_service import ProvenancedRunService

_PIPELINE_STAGE_ROLES = frozenset({"source", "imported", "subset", "export"})


class MethodExecutionRequest(BaseModel):
    """Boundary-validated request payload for method execution."""

    request_group_id: int = Field(gt=0)
    base_run_key: str = Field(min_length=1)
    method_name: str = Field(min_length=1)
    method_version: str = Field(min_length=1)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    source_group_id: Optional[int] = Field(default=None, gt=0)
    imported_run_id: Optional[int] = Field(default=None, gt=0)
    node_id: Optional[str] = None
    invocation_id: Optional[str] = None
    force: bool = False

    model_config = ConfigDict(extra="forbid")

    @field_validator("base_run_key", "method_name", "method_version")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text

    @field_validator("node_id")
    @classmethod
    def _normalize_optional_node_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("invocation_id")
    @classmethod
    def _validate_invocation_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        # Boundary validation: invocation ids are UUID strings.
        UUID(text)
        return text


@dataclass(frozen=True)
class MethodExecutionOutcome:
    """Structured response returned by MethodExecutionService.execute()."""

    run_id: int
    run_key: str
    reused: bool
    result_ref: Optional[str]
    output_json: Dict[str, Any]
    pipeline_stage_role: str
    pipeline_stage_context: Dict[str, Any]
    metadata_json: Dict[str, Any]


def compose_method_run_key(
    *,
    base_run_key: str,
    method_name: str,
    method_version: str,
    node_id: Optional[str] = None,
    invocation_id: Optional[str] = None,
) -> str:
    """Build deterministic method execution run key with fixed suffix ordering."""
    key = (
        f"{base_run_key}__method__{method_name}@{method_version}"
        # Ordering is locked by plan contract.
        f"{f'__node__{node_id}' if node_id else ''}"
        f"{f'__inv__{invocation_id}' if invocation_id else ''}"
    )
    return key


class MethodExecutionService:
    """Execute one registered method runner and persist canonical provenance."""

    def __init__(self, repository: RawCallRepository):
        self.repository = repository
        self.method_service = MethodService(repository)
        self.provenanced_run_service = ProvenancedRunService(repository)
        ensure_default_method_runtimes_registered()

    async def execute(
        self,
        *,
        request_group_id: int,
        base_run_key: str,
        method_name: str,
        method_version: str,
        parameters: Optional[Dict[str, Any]] = None,
        source_group_id: Optional[int] = None,
        imported_run_id: Optional[int] = None,
        node_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        force: bool = False,
    ) -> MethodExecutionOutcome:
        """Execute one method runtime and write `provenanced_runs`."""
        try:
            req = MethodExecutionRequest(
                request_group_id=request_group_id,
                base_run_key=base_run_key,
                method_name=method_name,
                method_version=method_version,
                parameters=dict(parameters or {}),
                source_group_id=source_group_id,
                imported_run_id=imported_run_id,
                node_id=node_id,
                invocation_id=invocation_id,
                force=force,
            )
        except ValidationError as exc:
            raise ValueError(f"invalid_method_execution_request: {exc}") from exc

        method_row = self.method_service.get_method(
            req.method_name,
            version=req.method_version,
        )
        if method_row is None:
            raise ValueError(
                f"method_not_registered:{req.method_name}@{req.method_version}; "
                "register methods via scripts/register_inference_methods.py"
            )

        runtime_spec = get_method_runtime(req.method_name, req.method_version)
        if runtime_spec is None:
            raise ValueError(
                f"runtime_not_registered:{req.method_name}@{req.method_version}"
            )

        run_key = compose_method_run_key(
            base_run_key=req.base_run_key,
            method_name=req.method_name,
            method_version=req.method_version,
            node_id=req.node_id,
            invocation_id=req.invocation_id,
        )
        existing = self.repository.get_provenanced_run_by_request_and_key(
            request_group_id=int(req.request_group_id),
            run_key=run_key,
            run_kind="execution",
        )
        if existing is not None and str(existing.run_status or "") == "completed" and not req.force:
            existing_meta = dict(existing.metadata_json or {})
            stage_context = dict(existing_meta.get("pipeline_stage_context") or {})
            return MethodExecutionOutcome(
                run_id=int(existing.id),
                run_key=run_key,
                reused=True,
                result_ref=existing.result_ref,
                output_json=dict(existing_meta.get("result_payload") or {}),
                pipeline_stage_role=str(
                    existing_meta.get("pipeline_stage_role") or "export"
                ),
                pipeline_stage_context=stage_context,
                metadata_json=existing_meta,
            )

        resolved_source_group_id = int(req.source_group_id or req.request_group_id)
        imported_metadata: Dict[str, Any] = {}
        if req.imported_run_id is not None:
            imported_row = self.repository.get_provenanced_run_by_id(int(req.imported_run_id))
            if imported_row is None:
                raise ValueError(f"imported_run_not_found:{req.imported_run_id}")
            imported_metadata = dict(imported_row.metadata_json or {})
            for required_key in ("execution_role", "pipeline_stage_role"):
                value = imported_metadata.get(required_key)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "malformed_imported_run_metadata:"
                        f"{required_key} missing or invalid on run_id={req.imported_run_id}"
                    )
        runner_context = MethodRunnerContext(
            repository=self.repository,
            request_group_id=int(req.request_group_id),
            source_group_id=resolved_source_group_id,
            method_name=req.method_name,
            method_version=req.method_version,
            run_key=run_key,
            imported_run_id=(
                int(req.imported_run_id) if req.imported_run_id is not None else None
            ),
            imported_run_metadata=imported_metadata,
        )
        runner_result = await run_method_runtime(
            spec=runtime_spec,
            parameters=req.parameters,
            context=runner_context,
        )

        stage_role = str(runner_result.pipeline_stage_role or "").strip().lower()
        if stage_role not in _PIPELINE_STAGE_ROLES:
            raise ValueError(
                "invalid_pipeline_stage_role:"
                f"{runner_result.pipeline_stage_role!r}; expected one of "
                f"{sorted(_PIPELINE_STAGE_ROLES)}"
            )

        execution_metadata: Dict[str, Any] = dict(runner_result.metadata_json or {})
        execution_metadata["execution_role"] = "method_execution"
        execution_metadata["pipeline_stage_role"] = stage_role
        execution_metadata["pipeline_stage_context"] = dict(
            runner_result.pipeline_stage_context or {}
        )
        execution_metadata["result_payload"] = dict(runner_result.output_json or {})
        if req.imported_run_id is not None:
            execution_metadata["imported_run_ref"] = {
                "run_id": int(req.imported_run_id),
                "execution_role": str(imported_metadata.get("execution_role")),
                "pipeline_stage_role": str(imported_metadata.get("pipeline_stage_role")),
            }

        canonical_config = {
            "parameters": dict(req.parameters),
            "method_name": req.method_name,
            "method_version": req.method_version,
        }
        if req.node_id:
            canonical_config["node_id"] = req.node_id
        if req.invocation_id:
            canonical_config["invocation_id"] = req.invocation_id

        run_id = int(
            self.provenanced_run_service.record_method_execution(
                request_group_id=int(req.request_group_id),
                run_key=run_key,
                source_group_id=resolved_source_group_id,
                method_definition_id=int(method_row.id),
                config_json=canonical_config,
                determinism_class=str(
                    req.parameters.get("determinism_class") or "non_deterministic"
                ),
                result_ref=runner_result.result_ref,
                metadata_json=execution_metadata,
                run_status="completed",
            )
        )
        return MethodExecutionOutcome(
            run_id=run_id,
            run_key=run_key,
            reused=False,
            result_ref=runner_result.result_ref,
            output_json=dict(runner_result.output_json or {}),
            pipeline_stage_role=stage_role,
            pipeline_stage_context=dict(runner_result.pipeline_stage_context or {}),
            metadata_json=execution_metadata,
        )


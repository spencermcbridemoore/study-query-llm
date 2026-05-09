"""CSV parse transform runner for polymorphic method execution."""

from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
)


class CsvParseBasicParams(BaseModel):
    """Boundary-validated parameter payload for CSV parse transform."""

    dataset_name: str = Field(min_length=1)
    dataset_version: Optional[str] = None
    encoding: str = "utf-8"
    delimiter: str = ","
    expected_columns: Optional[List[str]] = None
    expected_dtypes: Optional[Dict[str, str]] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("dataset_name", "encoding", "delimiter")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text

    @field_validator("dataset_version")
    @classmethod
    def _strip_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("expected_columns")
    @classmethod
    def _normalize_expected_columns(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned: List[str] = []
        for column in value:
            token = str(column or "").strip()
            if token:
                cleaned.append(token)
        if not cleaned:
            return None
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("expected_columns must not contain duplicates")
        return cleaned

    @field_validator("expected_dtypes")
    @classmethod
    def _normalize_expected_dtypes(
        cls,
        value: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        if value is None:
            return None
        normalized: Dict[str, str] = {}
        for raw_name, raw_dtype in dict(value).items():
            col_name = str(raw_name or "").strip()
            dtype = str(raw_dtype or "").strip()
            if not col_name or not dtype:
                raise ValueError("expected_dtypes must have non-empty keys and values")
            normalized[col_name] = dtype
        return normalized

    @model_validator(mode="after")
    def _validate_expected_contracts(self) -> "CsvParseBasicParams":
        if self.expected_dtypes is not None and self.expected_columns is None:
            raise ValueError("expected_dtypes requires expected_columns")
        if self.expected_dtypes is not None and self.expected_columns is not None:
            unknown_keys = sorted(set(self.expected_dtypes.keys()) - set(self.expected_columns))
            if unknown_keys:
                raise ValueError(
                    "expected_dtypes contains columns not present in expected_columns: "
                    f"{unknown_keys}"
                )
        return self


def _call_artifact_uri_by_id(repo: RawCallRepository, artifact_id: int) -> str:
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return sanitized or "dataset"


def _build_schema_payload(frame: pd.DataFrame) -> Dict[str, Any]:
    columns: List[Dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        columns.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "nullable": bool(null_count > 0),
                "null_count": null_count,
            }
        )
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": columns,
    }


def _validate_expected_columns(
    *,
    actual_columns: List[str],
    expected_columns: List[str],
) -> None:
    expected_set = set(expected_columns)
    actual_set = set(actual_columns)
    if actual_set == expected_set:
        return
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    raise ValueError(
        "csv_columns_mismatch:"
        f"missing={missing},unexpected={unexpected}"
    )


def _validate_expected_dtypes(
    *,
    actual_dtypes: Dict[str, str],
    expected_dtypes: Dict[str, str],
) -> None:
    mismatches: Dict[str, Dict[str, str]] = {}
    for column, expected_dtype in expected_dtypes.items():
        actual_dtype = actual_dtypes.get(column)
        if actual_dtype != expected_dtype:
            mismatches[column] = {
                "expected": str(expected_dtype),
                "actual": str(actual_dtype),
            }
    if mismatches:
        raise ValueError(f"csv_dtypes_mismatch:{mismatches}")


async def run_csv_parse_basic(
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Parse upstream CSV artifact into a parquet imported artifact."""
    try:
        parsed = CsvParseBasicParams.model_validate(parameters or {})
    except ValidationError as exc:
        raise ValueError(f"invalid_csv_parse_basic_parameters: {exc}") from exc

    if context.imported_run_id is None:
        raise ValueError("csv_parse_requires_imported_run_id")
    imported_run = context.repository.get_provenanced_run_by_id(int(context.imported_run_id))
    if imported_run is None:
        raise ValueError(f"imported_run_not_found:{context.imported_run_id}")
    source_result_ref = str(imported_run.result_ref or "").strip()
    if not source_result_ref:
        raise ValueError(
            f"imported_run_missing_result_ref:{context.imported_run_id}"
        )

    artifact_service = ArtifactService(repository=context.repository)
    csv_bytes = artifact_service.storage.read_from_uri(source_result_ref)

    try:
        frame = pd.read_csv(
            io.BytesIO(csv_bytes),
            sep=parsed.delimiter,
            encoding=parsed.encoding,
            dtype_backend="numpy_nullable",
        )
    except Exception as exc:
        raise ValueError(f"csv_parse_failed:{exc}") from exc

    actual_columns = [str(column) for column in frame.columns.tolist()]
    schema_payload = _build_schema_payload(frame)
    actual_dtypes = {
        str(col["name"]): str(col["dtype"])
        for col in schema_payload["columns"]
    }

    if parsed.expected_columns is not None:
        _validate_expected_columns(
            actual_columns=actual_columns,
            expected_columns=parsed.expected_columns,
        )
    if parsed.expected_dtypes is not None:
        _validate_expected_dtypes(
            actual_dtypes=actual_dtypes,
            expected_dtypes=parsed.expected_dtypes,
        )

    parquet_buf = io.BytesIO()
    frame.to_parquet(parquet_buf, engine="pyarrow", index=False)
    parquet_bytes = parquet_buf.getvalue()

    base_name = _sanitize_filename(parsed.dataset_name)
    if parsed.dataset_version:
        version = _sanitize_filename(parsed.dataset_version)
        logical_filename = f"{base_name}_{version}.parquet"
    else:
        logical_filename = f"{base_name}.parquet"

    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=int(context.source_group_id),
        step_name="csv_parse",
        logical_filename=logical_filename,
        data=parquet_bytes,
        artifact_type="dataset_canonical_parquet",
        content_type="application/octet-stream",
        metadata={
            "dataset_name": parsed.dataset_name,
            "dataset_version": parsed.dataset_version,
            "row_count": schema_payload["row_count"],
            "column_count": schema_payload["column_count"],
            "source_imported_run_id": int(context.imported_run_id),
        },
    )
    artifact_uri = _call_artifact_uri_by_id(context.repository, int(artifact_id))

    pipeline_stage_context: Dict[str, Any] = {
        "dataset_name": parsed.dataset_name,
        "row_count": schema_payload["row_count"],
        "column_count": schema_payload["column_count"],
        "columns": schema_payload["columns"],
        "serialization": {"format": "parquet", "engine": "pyarrow"},
    }
    if parsed.dataset_version:
        pipeline_stage_context["dataset_version"] = parsed.dataset_version

    return MethodRunnerResult(
        output_json={
            "dataset_name": parsed.dataset_name,
            "dataset_version": parsed.dataset_version,
            "row_count": schema_payload["row_count"],
            "column_count": schema_payload["column_count"],
            "artifact_id": int(artifact_id),
            "artifact_uri": artifact_uri,
        },
        pipeline_stage_role="imported",
        result_ref=artifact_uri,
        pipeline_stage_context=pipeline_stage_context,
        metadata_json={
            "dataset_name": parsed.dataset_name,
            "dataset_version": parsed.dataset_version,
            "source_imported_run_id": int(context.imported_run_id),
            "expected_columns_enforced": parsed.expected_columns is not None,
            "expected_dtypes_enforced": parsed.expected_dtypes is not None,
            "artifact_id": int(artifact_id),
        },
    )


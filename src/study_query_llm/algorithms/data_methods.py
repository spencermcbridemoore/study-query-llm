"""
Register-first data method definitions for file artifact and CSV parse lanes.

Definitions are registered first so runtime dispatch and provenance writers
share canonical ``(name, version)`` identities before execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from study_query_llm.utils.logging_config import get_logger

if TYPE_CHECKING:
    from study_query_llm.services.method_service import MethodService

logger = get_logger(__name__)


MATURITY_RUNNER_WIRED = "runner_wired"


DATA_METHODS: List[Dict[str, Any]] = [
    {
        "name": "file_artifact.basic",
        "version": "0.1",
        "role": "method_execution",
        "code_ref": "src/study_query_llm/services/method_runners/file_artifact_basic.py",
        "description": (
            "Persist caller-provided file bytes as a source-stage artifact after "
            "boundary sha256 verification."
        ),
        "parameters_schema": {
            "type": "object",
            "required": [
                "file_path",
                "registered_name",
                "registered_version",
                "content_type",
                "expected_sha256",
            ],
            "properties": {
                "file_path": {"type": "string"},
                "registered_name": {"type": "string"},
                "registered_version": {"type": "string"},
                "content_type": {"type": "string"},
                "expected_sha256": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
            },
            "additionalProperties": False,
        },
        "maturity": MATURITY_RUNNER_WIRED,
    },
    {
        "name": "csv_parse.basic",
        "version": "0.1",
        "role": "method_execution",
        "code_ref": "src/study_query_llm/services/method_runners/csv_parse_basic.py",
        "description": (
            "Parse an upstream CSV artifact into canonical parquet and capture the "
            "resulting schema in pipeline_stage_context."
        ),
        "parameters_schema": {
            "type": "object",
            "required": ["dataset_name"],
            "properties": {
                "dataset_name": {"type": "string"},
                "encoding": {"type": "string"},
                "delimiter": {"type": "string"},
                "expected_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "expected_dtypes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
        "maturity": MATURITY_RUNNER_WIRED,
    },
]


def register_data_methods(method_service: "MethodService") -> Dict[str, int]:
    """Register all data methods idempotently."""
    registered: Dict[str, int] = {}
    for spec in DATA_METHODS:
        name = str(spec["name"])
        version = str(spec["version"])
        key = f"{name}@{version}"
        existing = method_service.get_method(name, version=version)
        if existing is not None:
            registered[key] = int(existing.id)
            logger.debug("Data method already registered: %s (id=%s)", key, existing.id)
            continue
        method_id = method_service.register_method(
            name=name,
            version=version,
            code_ref=spec.get("code_ref"),
            description=spec.get("description"),
            parameters_schema=spec.get("parameters_schema"),
        )
        registered[key] = int(method_id)
        logger.info("Registered data method: %s (id=%s)", key, method_id)
    return registered


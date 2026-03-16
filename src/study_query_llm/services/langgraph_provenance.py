"""LangGraph method provenance helper.

Centralizes method resolution, parameter redaction, and result envelope assembly
for langgraph_run job provenance recording. Best-effort; failures log warnings.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Keys to redact from parameters before persistence
_SENSITIVE_KEYS = frozenset(
    {"api_key", "token", "secret", "password", "authorization", "bearer"}
)

# Default method name/version when not specified in payload
_DEFAULT_METHOD_NAME = "langgraph_run.default"
_DEFAULT_METHOD_VERSION = "1"

# Standard result_key for langgraph job outcomes
RESULT_KEY_JOB_OUTCOME = "job_outcome"


def _redact_sensitive(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of d with sensitive keys masked. Case-insensitive key match."""
    out = copy.deepcopy(d)
    for k in list(out.keys()):
        if k.lower() in _SENSITIVE_KEYS:
            out[k] = "***REDACTED***"
    return out


def _resolve_method_identity(
    payload_json: Dict[str, Any],
    job_key: str,
) -> tuple[str, str]:
    """Resolve method name and version from payload/config or defaults."""
    raw_config = payload_json.get("config")
    config = raw_config if isinstance(raw_config, dict) else {}
    name = config.get("method_name") or payload_json.get("method_name")
    version = config.get("method_version") or payload_json.get("method_version")

    if name is None:
        # Derive from job_key prefix if meaningful (e.g., "lg_task1_1" -> "langgraph_run.task1")
        prefix = job_key.split("_")[0] if job_key else ""
        if prefix and prefix != "lg":
            name = f"langgraph_run.{prefix}"
        else:
            name = _DEFAULT_METHOD_NAME
    if version is None:
        version = _DEFAULT_METHOD_VERSION

    return str(name), str(version)


def _ensure_method_registered(
    method_svc: Any,
    name: str,
    version: str,
) -> Optional[int]:
    """Get or register method definition. Returns method_id or None on failure."""
    method = method_svc.get_method(name=name, version=version)
    if method is not None:
        return int(method.id)
    try:
        return method_svc.register_method(
            name=name,
            version=version,
            code_ref="src/study_query_llm/services/langgraph_job_runner.py",
            description="LangGraph run (minimal echo or custom graph)",
            parameters_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "config": {"type": "object"},
                },
            },
        )
    except Exception as e:
        logger.warning("Failed to register method %s@%s: %s", name, version, e)
        return None


def build_result_envelope(
    *,
    job_id: int,
    job_key: str,
    payload_json: Dict[str, Any],
    status: str,
    result_ref: Optional[str] = None,
    error: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    checkpoint_refs: Optional[Dict[str, Any]] = None,
    method_name: Optional[str] = None,
    method_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build standard result_json envelope for langgraph job outcomes."""
    parameters = _redact_sensitive(dict(payload_json))

    envelope: Dict[str, Any] = {
        "status": status,
        "job_id": job_id,
        "job_key": job_key,
        "result_ref": result_ref,
        "error": error,
        "parameters": parameters,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if method_name is not None:
        envelope["method"] = {
            "name": method_name,
            "version": method_version or _DEFAULT_METHOD_VERSION,
        }
    if state is not None:
        envelope["state"] = state
    if checkpoint_refs:
        envelope["checkpoint_refs"] = checkpoint_refs

    return envelope


def record_langgraph_job_outcome(
    *,
    method_svc: Any,
    request_group_id: int,
    job_id: int,
    job_key: str,
    payload_json: Dict[str, Any],
    status: str,
    result_ref: Optional[str] = None,
    error: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    checkpoint_refs: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Record langgraph job outcome to analysis_results. Best-effort; returns result_id or None."""
    name, version = _resolve_method_identity(payload_json, job_key)
    method_id = _ensure_method_registered(method_svc, name, version)
    if method_id is None:
        logger.warning("Skipping provenance record: could not resolve method for job %s", job_id)
        return None

    envelope = build_result_envelope(
        job_id=job_id,
        job_key=job_key,
        payload_json=payload_json,
        status=status,
        result_ref=result_ref,
        error=error,
        state=state,
        checkpoint_refs=checkpoint_refs,
        method_name=name,
        method_version=version,
    )

    try:
        result_id = method_svc.record_result(
            method_definition_id=method_id,
            source_group_id=request_group_id,
            result_key=RESULT_KEY_JOB_OUTCOME,
            result_json=envelope,
        )
        logger.debug("Recorded langgraph provenance: result_id=%s, job_id=%s", result_id, job_id)
        return result_id
    except Exception as e:
        logger.warning("Failed to record langgraph provenance for job %s: %s", job_id, e)
        return None

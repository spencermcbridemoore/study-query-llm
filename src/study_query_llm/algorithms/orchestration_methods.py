"""Register-first method catalog for orchestration and MCQ analysis lanes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from study_query_llm.utils.logging_config import get_logger

if TYPE_CHECKING:
    from study_query_llm.services.method_service import MethodService

logger = get_logger(__name__)


# Legacy orchestration/MCQ methods still execute through analysis/experiment flows
# rather than dedicated method-runner modules. Keep code_ref anchored to the
# closest implementation module (not call-site consumer services).
ORCHESTRATION_METHODS: List[Dict[str, Any]] = [
    {
        "name": "langgraph_run.default",
        "version": "1",
        "code_ref": "src/study_query_llm/services/jobs/langgraph_job_runner.py",
        "description": "Default LangGraph job outcome method identity.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "config": {"type": "object"},
            },
        },
    },
    {
        "name": "mcq_answer_position_probe",
        "version": "1.0",
        "code_ref": "src/study_query_llm/experiments/mcq_answer_position_probe.py",
        "description": "MCQ answer-position probe execution method.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "deployment": {"type": "string"},
                "level": {"type": "string"},
                "subject": {"type": "string"},
                "options_per_question": {"type": "integer"},
                "questions_per_test": {"type": "integer"},
                "label_style": {"type": "string"},
                "spread_correct_answer_uniformly": {"type": "boolean"},
                "samples_per_combo": {"type": "integer"},
                "template_version": {"type": "string"},
            },
        },
    },
    {
        "name": "mcq_compliance_metrics",
        "version": "1.0",
        "code_ref": "src/study_query_llm/analysis/mcq_from_run.py",
        "description": "MCQ compliance analysis result method identity.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "analysis_key": {"type": "string"},
                "request_id": {"type": "integer"},
            },
        },
    },
    {
        "name": "mcq_answer_position_distribution",
        "version": "1.0",
        "code_ref": "src/study_query_llm/analysis/mcq_from_run.py",
        "description": "MCQ answer-position distribution analysis method identity.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "analysis_key": {"type": "string"},
                "request_id": {"type": "integer"},
            },
        },
    },
    {
        "name": "mcq_answer_position_chi_square",
        "version": "1.0",
        "code_ref": "src/study_query_llm/analysis/mcq_from_run.py",
        "description": "MCQ chi-square answer-position analysis method identity.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "analysis_key": {"type": "string"},
                "request_id": {"type": "integer"},
            },
        },
    },
]


def register_orchestration_methods(method_service: "MethodService") -> Dict[str, int]:
    """Idempotently register orchestration/MCQ method identities."""
    registered: Dict[str, int] = {}
    for spec in ORCHESTRATION_METHODS:
        name = str(spec["name"])
        version = str(spec["version"])
        key = f"{name}@{version}"
        existing = method_service.get_method(name, version=version)
        if existing is not None:
            registered[key] = int(existing.id)
            logger.debug(
                "Orchestration method already registered: %s (id=%s)",
                key,
                existing.id,
            )
            continue
        method_id = method_service.register_method(
            name=name,
            version=version,
            code_ref=spec.get("code_ref"),
            description=spec.get("description"),
            parameters_schema=spec.get("parameters_schema"),
        )
        registered[key] = int(method_id)
        logger.info("Registered orchestration method: %s (id=%s)", key, method_id)
    return registered


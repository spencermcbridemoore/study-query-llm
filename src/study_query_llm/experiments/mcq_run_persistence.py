"""Persist MCQ probe results as mcq_run groups and record sweep request delivery."""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import text as sa_text
from sqlalchemy.orm.attributes import flag_modified

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN
from study_query_llm.services.provenanced_run_service import ProvenancedRunService
from study_query_llm.services.sweep_request_service import SweepRequestService

MCQ_RUN_METADATA_VERSION = 1
MCQ_METHOD_DETERMINISM_CLASS = "non_deterministic"
MCQ_METHOD_VERSION = "1.0"


def mcq_run_key_exists_in_db(db: DatabaseConnectionV2, run_key: str) -> bool:
    """Compatibility helper for historical scripts.

    Active request-driven workers should use claim-or-skip via `SweepRunClaim`
    ownership and treat this pre-check as legacy compatibility only.
    """
    with db.session_scope() as session:
        existing = (
            session.query(Group)
            .filter(
                Group.group_type == GROUP_TYPE_MCQ_RUN,
                sa_text("metadata_json->>'run_key' = :rk"),
            )
            .params(rk=run_key)
            .first()
        )
        return existing is not None


def persist_mcq_probe_result(
    db: DatabaseConnectionV2,
    request_id: int,
    run_key: str,
    target: Dict[str, Any],
    probe_details: Dict[str, Any],
) -> int:
    """Create mcq_run group with full probe payload, record_delivery. Returns run group id."""
    summary = probe_details.get("summary") or {}
    metadata_json: Dict[str, Any] = {
        "run_key": run_key,
        "sweep_type": "mcq",
        "deployment": target.get("deployment"),
        "level": target.get("level"),
        "subject": target.get("subject"),
        "options_per_question": target.get("options_per_question"),
        "questions_per_test": target.get("questions_per_test"),
        "label_style": target.get("label_style"),
        "spread_correct_answer_uniformly": target.get("spread_correct_answer_uniformly"),
        "samples_per_combo": target.get("samples_per_combo"),
        "template_version": target.get("template_version"),
        "result_summary": summary,
        "probe_details": probe_details,
        "mcq_metadata_version": MCQ_RUN_METADATA_VERSION,
    }
    safe_name = run_key.replace("/", "_")[:120]
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name=f"mcq_run_{safe_name}",
            metadata_json=metadata_json,
        )
        grp = repo.get_group_by_id(run_id)
        if grp is not None:
            meta = dict(grp.metadata_json or {})
            meta["_group_id"] = int(run_id)
            grp.metadata_json = meta
            flag_modified(grp, "metadata_json")
            session.flush()
        svc = SweepRequestService(repo)
        request = svc.get_request(request_id) or {}
        method_name = str(request.get("algorithm") or "mcq_answer_position_probe")
        method_service = MethodService(repo)
        method = method_service.get_method(method_name, version=MCQ_METHOD_VERSION)
        if method is None:
            method_id = method_service.register_method(
                name=method_name,
                version=MCQ_METHOD_VERSION,
                description="MCQ answer-position probe execution",
                parameters_schema={
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
            )
        else:
            method_id = int(method.id)
        ProvenancedRunService(repo).record_method_execution(
            request_group_id=int(request_id),
            run_key=str(run_key),
            source_group_id=int(run_id),
            result_group_id=int(run_id),
            method_definition_id=int(method_id),
            determinism_class=MCQ_METHOD_DETERMINISM_CLASS,
            config_json=dict(target or {}),
            metadata_json={
                "mcq_metadata_version": MCQ_RUN_METADATA_VERSION,
                "probe_result_summary": summary,
            },
            run_status="completed",
        )
        svc.record_delivery(request_id, run_id, run_key)
    return run_id

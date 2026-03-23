"""MCQ-focused tests for typed SweepRequestService behavior."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_MCQ_RUN,
    GROUP_TYPE_MCQ_SWEEP,
)
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _mcq_axes() -> dict:
    return {
        "levels": ["high school"],
        "subjects": ["physics"],
        "deployments": ["gpt-4o-mini"],
        "options_per_question": [4, 5],
        "questions_per_test": [20],
        "label_styles": ["upper"],
        "spread_correct_answer_uniformly": [False],
    }


def test_create_mcq_request_expands_expected_keys_and_analysis_contract():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_req",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50, "template_version": "v1"},
            parameter_axes=_mcq_axes(),
            entry_max=None,
            sweep_type="mcq",
        )
        req = svc.get_request(req_id)
        assert req is not None
        assert req["sweep_type"] == "mcq"
        assert req["expected_count"] == 2
        assert "mcq_compliance" in (req.get("required_analyses") or [])
        assert req.get("analysis_status") == "not_started"

        keys = req.get("expected_run_keys") or []
        assert any("4opt_20q_upper_no_spread_50samples_v1" in k for k in keys)
        assert any("5opt_20q_upper_no_spread_50samples_v1" in k for k in keys)


def test_mcq_progress_and_finalize_creates_mcq_sweep():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_finalize",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50},
            parameter_axes={
                **_mcq_axes(),
                "options_per_question": [4],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        req = svc.get_request(req_id)
        run_key = req["expected_run_keys"][0]
        run_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="mcq_run_1",
            metadata_json={"run_key": run_key},
        )

        progress = svc.compute_progress(req_id)
        assert progress["completed_count"] == 1
        assert progress["missing_count"] == 0

        sweep_id = svc.finalize_if_fulfilled(req_id)
        assert sweep_id is not None

        sweep_group = repo.get_group_by_id(sweep_id)
        assert sweep_group is not None
        assert sweep_group.group_type == GROUP_TYPE_MCQ_SWEEP

        contains = session.query(GroupLink).filter_by(
            parent_group_id=sweep_id,
            child_group_id=run_id,
            link_type="contains",
        ).first()
        assert contains is not None

        generates = session.query(GroupLink).filter_by(
            parent_group_id=req_id,
            child_group_id=sweep_id,
            link_type="generates",
        ).first()
        assert generates is not None


def test_record_delivery_enforces_mcq_run_group_type():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_delivery",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50},
            parameter_axes={
                **_mcq_axes(),
                "options_per_question": [4],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        req = svc.get_request(req_id)
        run_key = req["expected_run_keys"][0]

        wrong_run_id = repo.create_group(
            group_type="clustering_run",
            name="wrong_type_run",
            metadata_json={"run_key": run_key},
        )
        assert svc.record_delivery(req_id, wrong_run_id, run_key) is False

        right_run_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="right_type_run",
            metadata_json={"run_key": run_key},
        )
        assert svc.record_delivery(req_id, right_run_id, run_key) is True


def test_mcq_analysis_status_and_result_recording():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_analysis",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50},
            parameter_axes={
                **_mcq_axes(),
                "options_per_question": [4],
            },
            entry_max=None,
            sweep_type="mcq",
        )

        source_group_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="analysis_source_run",
            metadata_json={"run_key": "rk"},
        )

        # Fail one required analysis, then complete both required analyses.
        assert svc.mark_analysis_failed(req_id, "mcq_compliance", "parse error")
        req = svc.get_request(req_id)
        assert req["analysis_status"] == "failed"

        result_id = svc.record_analysis_result(
            request_id=req_id,
            source_group_id=source_group_id,
            analysis_key="mcq_compliance",
            result_key="format_compliance_rate",
            result_value=0.92,
            mark_complete=True,
        )
        assert result_id > 0
        assert svc.mark_analysis_completed(req_id, "mcq_answer_position_distribution")

        req = svc.get_request(req_id)
        assert req["analysis_status"] == "complete"
        assert "mcq_compliance" in (req.get("completed_analyses") or [])
        assert "mcq_answer_position_distribution" in (req.get("completed_analyses") or [])

        method_svc = MethodService(repo)
        method = method_svc.get_method("mcq_compliance_metrics", version="1.0")
        assert method is not None

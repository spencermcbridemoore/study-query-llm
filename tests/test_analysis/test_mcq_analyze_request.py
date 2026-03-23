"""Integration tests for MCQ analysis driver."""

from __future__ import annotations

from sqlalchemy.orm.attributes import flag_modified

from study_query_llm.analysis.mcq_analyze_request import run_mcq_analyses_for_request
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN
from study_query_llm.services.sweep_request_service import SweepRequestService


def _mcq_axes():
    return {
        "levels": ["high school"],
        "subjects": ["physics"],
        "deployments": ["gpt-4o-mini"],
        "options_per_question": [4],
        "questions_per_test": [5],
        "label_styles": ["upper"],
        "spread_correct_answer_uniformly": [False],
    }


def test_run_mcq_analyses_dry_run():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_analyze_dry",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 2, "concurrency": 1},
            parameter_axes=_mcq_axes(),
            entry_max=None,
            sweep_type="mcq",
        )
        req = svc.get_request(req_id)
        run_key = req["expected_run_keys"][0]
        run_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="t",
            metadata_json={
                "run_key": run_key,
                "probe_details": {
                    "summary": {
                        "samples_attempted": 4,
                        "samples_with_successful_call": 4,
                        "samples_with_valid_answer_key": 3,
                        "heading_present_count": 2,
                        "pooled_distribution": {
                            "A": {"count": 5, "pct": 0.25},
                            "B": {"count": 5, "pct": 0.25},
                            "C": {"count": 5, "pct": 0.25},
                            "D": {"count": 5, "pct": 0.25},
                        },
                    },
                    "call_errors": [],
                    "parse_failures": [],
                },
            },
        )
        g = session.query(Group).filter(Group.id == run_id).first()
        meta = dict(g.metadata_json or {})
        meta["_group_id"] = int(run_id)
        g.metadata_json = meta
        flag_modified(g, "metadata_json")
        session.flush()
        svc.record_delivery(req_id, run_id, run_key)

    report = run_mcq_analyses_for_request(db, req_id, dry_run=True)
    assert report["dry_run"] is True
    assert any("mcq_compliance" in str(x) for x in report["recorded"])

    report2 = run_mcq_analyses_for_request(db, req_id, dry_run=False)
    assert report2.get("dry_run") is False
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req2 = svc.get_request(req_id)
    assert "mcq_compliance" in (req2.get("completed_analyses") or [])

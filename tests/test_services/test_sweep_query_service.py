"""Tests for SweepQueryService DB sweep reconstruction."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN
from study_query_llm.services.sweep_query_service import SweepQueryService
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _mcq_axes():
    return {
        "levels": ["high school"],
        "subjects": ["physics"],
        "deployments": ["gpt-4o-mini"],
        "options_per_question": [4],
        "questions_per_test": [20],
        "label_styles": ["upper"],
        "spread_correct_answer_uniformly": [False],
    }


def _summary_with_pooled():
    return {
        "samples_attempted": 10,
        "samples_with_successful_call": 10,
        "samples_with_valid_answer_key": 8,
        "heading_present_count": 9,
        "answer_count_total": 40,
        "chi_square_vs_uniform": 2.5,
        "deployment": "gpt-4o-mini",
        "subject": "physics",
        "level": "high school",
        "labels": ["A", "B", "C", "D"],
        "pooled_distribution": {
            "A": {"count": 10, "pct": 0.25},
            "B": {"count": 10, "pct": 0.25},
            "C": {"count": 10, "pct": 0.25},
            "D": {"count": 10, "pct": 0.25},
        },
    }


def test_get_mcq_metrics_df_empty():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepQueryService(repo)
        df = svc.get_mcq_metrics_df()
    assert df.empty
    assert "pct_A" in df.columns
    assert "format_compliance_rate" in df.columns


def test_get_mcq_metrics_df_all_runs_and_request_filter():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        rsvc = SweepRequestService(repo)
        req_id = rsvc.create_request(
            request_name="mcq_qs",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50, "template_version": "v1"},
            parameter_axes=_mcq_axes(),
            entry_max=None,
            sweep_type="mcq",
        )
        req = rsvc.get_request(req_id)
        run_key = req["expected_run_keys"][0]

        meta = {
            "run_key": run_key,
            "deployment": "gpt-4o-mini",
            "subject": "physics",
            "level": "high school",
            "options_per_question": 4,
            "questions_per_test": 20,
            "label_style": "upper",
            "spread_correct_answer_uniformly": False,
            "samples_per_combo": 50,
            "template_version": "v1",
            "result_summary": _summary_with_pooled(),
        }
        run_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="mcq_run_test",
            metadata_json=meta,
        )
        repo.create_group_link(
            parent_group_id=req_id,
            child_group_id=run_id,
            link_type="contains",
            metadata_json={"run_key": run_key},
        )

        qsvc = SweepQueryService(repo)
        df_all = qsvc.get_mcq_metrics_df()
        assert len(df_all) == 1
        row = df_all.iloc[0]
        assert row["run_key"] == run_key
        assert row["k"] == 1
        assert abs(float(row["pct_A"]) - 0.25) < 1e-6
        assert abs(float(row["format_compliance_rate"]) - 0.9) < 1e-6
        assert abs(float(row["answer_key_parse_rate"]) - 0.8) < 1e-6

        df_req = qsvc.get_mcq_metrics_df(mcq_request_id=req_id)
        assert len(df_req) == 1

        orphan_meta = {**meta, "run_key": "orphan_only"}
        repo.create_group(
            group_type=GROUP_TYPE_MCQ_RUN,
            name="mcq_orphan",
            metadata_json=orphan_meta,
        )
        session.flush()

        df_req2 = qsvc.get_mcq_metrics_df(mcq_request_id=req_id)
        assert len(df_req2) == 1

        df_all2 = qsvc.get_mcq_metrics_df()
        assert len(df_all2) == 2

        missing = qsvc.get_mcq_metrics_df(mcq_request_id=999999)
        assert missing.empty


def test_get_mcq_metrics_df_request_with_no_links_returns_empty():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        rsvc = SweepRequestService(repo)
        req_id = rsvc.create_request(
            request_name="empty_req",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 50},
            parameter_axes=_mcq_axes(),
            entry_max=None,
            sweep_type="mcq",
        )
        qsvc = SweepQueryService(repo)
        df = qsvc.get_mcq_metrics_df(mcq_request_id=req_id)
        assert df.empty

"""Tests for new analysis_* group helpers on ProvenanceService."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_ANALYSIS_REQUEST,
    GROUP_TYPE_ANALYSIS_RUN,
    ProvenanceService,
)


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_analysis_run_group_caps_name_and_preserves_full_name() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        long_run_key = "rk_" + ("x" * 240)
        group_id = provenance.create_analysis_run_group(
            method_name="stability_report",
            input_id=123,
            run_key=long_run_key,
            request_group_id=45,
        )
        group = repo.get_group_by_id(group_id)
        assert group is not None
        assert group.group_type == GROUP_TYPE_ANALYSIS_RUN
        assert len(group.name) <= 180
        assert group.metadata_json is not None
        assert group.metadata_json["method_name"] == "stability_report"
        assert group.metadata_json["input_id"] == 123
        assert group.metadata_json["run_key"] == long_run_key
        assert "full_name" in group.metadata_json
        assert str(group.metadata_json["full_name"]).startswith("analyze:stability_report:123:")


def test_create_analysis_request_group_is_idempotent_by_method_input_run_key() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        first_id = provenance.create_analysis_request_group(
            method_name="cluster_count",
            input_id=55,
            run_key="rk_shared",
        )
        second_id = provenance.create_analysis_request_group(
            method_name="cluster_count",
            input_id=55,
            run_key="rk_shared",
        )
        assert first_id == second_id

        group = repo.get_group_by_id(first_id)
        assert group is not None
        assert group.group_type == GROUP_TYPE_ANALYSIS_REQUEST
        assert group.metadata_json is not None
        assert group.metadata_json["method_name"] == "cluster_count"
        assert group.metadata_json["input_id"] == 55
        assert group.metadata_json["run_key"] == "rk_shared"

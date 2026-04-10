"""Tests for langgraph_provenance helper."""

from __future__ import annotations

import pytest

from study_query_llm.services.langgraph_provenance import (
    RESULT_KEY_JOB_OUTCOME,
    build_result_envelope,
    record_langgraph_job_outcome,
)


def test_build_result_envelope_basic():
    """Envelope has required fields and redacts sensitive keys."""
    envelope = build_result_envelope(
        job_id=1,
        job_key="lg_1",
        payload_json={"prompt": "hi", "api_key": "secret"},
        status="completed",
        result_ref="/path/to/out.json",
        method_name="langgraph_run.default",
        method_version="1",
    )
    assert envelope["status"] == "completed"
    assert envelope["job_id"] == 1
    assert envelope["job_key"] == "lg_1"
    assert envelope["result_ref"] == "/path/to/out.json"
    assert envelope["parameters"]["prompt"] == "hi"
    assert envelope["parameters"]["api_key"] == "***REDACTED***"
    assert envelope["method"] == {"name": "langgraph_run.default", "version": "1"}
    assert "recorded_at" in envelope


def test_build_result_envelope_failure():
    """Failure envelope includes error."""
    envelope = build_result_envelope(
        job_id=2,
        job_key="lg_2",
        payload_json={"prompt": "x"},
        status="failed",
        error="payload_validation_error",
        method_name="langgraph_run.default",
        method_version="1",
    )
    assert envelope["status"] == "failed"
    assert envelope["error"] == "payload_validation_error"
    assert envelope["result_ref"] is None


def test_build_result_envelope_with_checkpoint_refs():
    """Checkpoint refs included when provided."""
    envelope = build_result_envelope(
        job_id=3,
        job_key="lg_3",
        payload_json={"prompt": "p"},
        status="completed",
        checkpoint_refs={"thread_id": "job_3", "checkpoint_id": "abc-123"},
        method_name="langgraph_run.default",
        method_version="1",
    )
    assert envelope["checkpoint_refs"] == {"thread_id": "job_3", "checkpoint_id": "abc-123"}


def test_record_langgraph_job_outcome_success():
    """record_langgraph_job_outcome creates analysis_results row."""
    from study_query_llm.db.connection_v2 import DatabaseConnectionV2
    from study_query_llm.db.raw_call_repository import RawCallRepository
    from study_query_llm.services.method_service import MethodService
    from study_query_llm.services.provenanced_run_service import ProvenancedRunService

    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        run_svc = ProvenancedRunService(repo)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="provenance_test",
            metadata_json={},
        )

        result_id = record_langgraph_job_outcome(
            method_svc=method_svc,
            provenanced_run_svc=run_svc,
            request_group_id=req_id,
            job_id=99,
            job_key="lg_test",
            payload_json={"prompt": "test"},
            status="completed",
            result_ref="/tmp/out.json",
        )
        assert result_id is not None

        results = method_svc.query_results(source_group_id=req_id, result_key=RESULT_KEY_JOB_OUTCOME)
        assert len(results) == 1
        assert results[0].result_json["parameters"]["prompt"] == "test"
        assert results[0].result_json["status"] == "completed"
        execution_rows = repo.list_provenanced_runs(
            request_group_id=req_id,
            run_kind="analysis_execution",
        )
        assert len(execution_rows) == 1
        assert execution_rows[0].run_kind == "execution"

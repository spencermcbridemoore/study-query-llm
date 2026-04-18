"""Unit tests for scripts/backup_mcq_db_to_json.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "backup_mcq_db_to_json.py"


@pytest.fixture(scope="module")
def backup_mod():
    spec = importlib.util.spec_from_file_location("backup_mcq_db_to_json", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_mcq_fixture(db: DatabaseConnectionV2, artifact_uri: str) -> None:
    from study_query_llm.db.models_v2 import (
        AnalysisResult,
        CallArtifact,
        Group,
        GroupMember,
        MethodDefinition,
        ProvenancedRun,
        RawCall,
    )

    with db.session_scope() as session:
        request_group = Group(
            group_type="mcq_sweep_request",
            name="mcq_req_1",
            metadata_json={},
        )
        run_group = Group(
            group_type="mcq_run",
            name="mcq_run_1",
            metadata_json={},
        )
        unrelated_group = Group(
            group_type="custom",
            name="unrelated_group",
            metadata_json={},
        )
        session.add_all([request_group, run_group, unrelated_group])
        session.flush()

        raw_call = RawCall(
            provider="mock",
            model="mock-model",
            modality="text",
            status="success",
            request_json={"prompt": "q"},
            response_json={"text": "a"},
            metadata_json={},
        )
        session.add(raw_call)
        session.flush()

        session.add(GroupMember(group_id=run_group.id, call_id=raw_call.id))
        session.add(
            CallArtifact(
                call_id=raw_call.id,
                artifact_type="mcq_payload",
                uri=artifact_uri,
                content_type="application/json",
                byte_size=12,
                metadata_json={"kind": "mcq"},
            )
        )
        # Should not be exported: call artifact from non-MCQ-linked call.
        unrelated_call = RawCall(
            provider="mock",
            model="mock-model",
            modality="text",
            status="success",
            request_json={"prompt": "x"},
            response_json={"text": "y"},
            metadata_json={},
        )
        session.add(unrelated_call)
        session.flush()
        session.add(
            CallArtifact(
                call_id=unrelated_call.id,
                artifact_type="other",
                uri=f"{artifact_uri}.other",
                metadata_json={},
            )
        )

        method = MethodDefinition(
            name="mcq_metric",
            version="1.0.0",
            is_active=True,
        )
        session.add(method)
        session.flush()

        session.add(
            ProvenancedRun(
                run_kind="execution",
                run_status="completed",
                request_group_id=request_group.id,
                source_group_id=run_group.id,
                result_group_id=run_group.id,
                run_key="rk1",
                determinism_class="deterministic",
            )
        )
        session.add(
            AnalysisResult(
                method_definition_id=method.id,
                source_group_id=run_group.id,
                analysis_group_id=run_group.id,
                result_key="accuracy",
                result_value=0.75,
            )
        )
        session.flush()


def test_collect_exports_expected_mcq_surface(backup_mod, tmp_path: Path) -> None:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    artifact_uri = str((tmp_path / "artifact_a.json").resolve())
    _seed_mcq_fixture(db, artifact_uri)

    with db.session_scope() as session:
        payload = backup_mod._collect(session)

    assert payload["counts"] == {
        "groups": 2,
        "provenanced_runs": 1,
        "analysis_results": 1,
        "call_artifacts": 1,
    }
    assert len(payload["groups"]) == 2
    assert {row["group_type"] for row in payload["groups"]} == {
        "mcq_run",
        "mcq_sweep_request",
    }
    assert len(payload["provenanced_runs"]) == 1
    assert payload["provenanced_runs"][0]["run_key"] == "rk1"
    assert len(payload["analysis_results"]) == 1
    assert payload["analysis_results"][0]["result_key"] == "accuracy"
    assert len(payload["call_artifacts"]) == 1
    assert payload["call_artifacts"][0]["uri"] == artifact_uri


def test_build_documents_includes_counts_and_manifest(backup_mod) -> None:
    payload = {
        "groups": [{"id": 1}],
        "provenanced_runs": [{"id": 2}],
        "analysis_results": [{"id": 3}],
        "call_artifacts": [{"id": 4}],
        "counts": {
            "groups": 1,
            "provenanced_runs": 1,
            "analysis_results": 1,
            "call_artifacts": 1,
        },
        "mcq_group_index": [{"id": 1, "group_type": "mcq_run", "name": "r"}],
    }
    export_doc, manifest_doc = backup_mod._build_documents(
        payload=payload,
        source_url_redacted="postgresql://study:***@localhost:5432/db",
    )

    assert export_doc["backup_metadata"]["row_counts"]["groups"] == 1
    assert "groups" in export_doc and len(export_doc["groups"]) == 1
    assert manifest_doc["row_counts"]["analysis_results"] == 1
    assert manifest_doc["mcq_group_index"][0]["group_type"] == "mcq_run"

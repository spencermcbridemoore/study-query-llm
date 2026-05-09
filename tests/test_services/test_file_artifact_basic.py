"""Tests for file_artifact.basic method runner."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_runners.file_artifact_basic import (
    run_file_artifact_basic,
)
from study_query_llm.services.method_runtime_registry import MethodRunnerContext


@pytest.fixture
def db_connection():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.mark.asyncio
async def test_file_artifact_basic_persists_source_artifact(
    db_connection,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "source.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")
    expected_sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="file-artifact-runner-test",
                metadata_json={},
            )
        )
        context = MethodRunnerContext(
            repository=repo,
            request_group_id=group_id,
            source_group_id=group_id,
            method_name="file_artifact.basic",
            method_version="0.1",
            run_key="rk",
        )

        out = await run_file_artifact_basic(
            {
                "file_path": str(file_path),
                "registered_name": "midterm2_questions",
                "registered_version": "v1",
                "content_type": "text/csv",
                "expected_sha256": expected_sha256,
            },
            context,
        )
        assert out.pipeline_stage_role == "source"
        assert out.output_json["sha256"] == expected_sha256
        assert out.result_ref is not None
        assert Path(str(out.result_ref)).exists()

        artifacts = session.query(CallArtifact).all()
        assert len(artifacts) == 1
        assert artifacts[0].artifact_type == "dataset_source_file"
        assert (artifacts[0].metadata_json or {}).get("sha256") == expected_sha256


@pytest.mark.asyncio
async def test_file_artifact_basic_hash_mismatch_fails_before_write(
    db_connection,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "source.csv"
    file_path.write_text("a,b\n1,2\n", encoding="utf-8")

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="file-artifact-runner-test",
                metadata_json={},
            )
        )
        context = MethodRunnerContext(
            repository=repo,
            request_group_id=group_id,
            source_group_id=group_id,
            method_name="file_artifact.basic",
            method_version="0.1",
            run_key="rk",
        )
        before_count = int(session.query(CallArtifact).count())
        with pytest.raises(ValueError, match="file_sha256_mismatch"):
            await run_file_artifact_basic(
                {
                    "file_path": str(file_path),
                    "registered_name": "midterm2_questions",
                    "registered_version": "v1",
                    "content_type": "text/csv",
                    "expected_sha256": "0" * 64,
                },
                context,
            )
        after_count = int(session.query(CallArtifact).count())
        assert before_count == after_count


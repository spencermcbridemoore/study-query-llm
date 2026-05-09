"""Tests for csv_parse.basic method runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_runners.csv_parse_basic import run_csv_parse_basic
from study_query_llm.services.method_runtime_registry import MethodRunnerContext


@pytest.fixture
def db_connection():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _seed_source_run(
    *,
    repo: RawCallRepository,
    group_id: int,
    artifact_dir: str,
    csv_payload: bytes,
) -> int:
    artifact_service = ArtifactService(repository=repo, artifact_dir=artifact_dir)
    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=group_id,
        step_name="file_artifact",
        logical_filename="source.csv",
        data=csv_payload,
        artifact_type="dataset_source_file",
        content_type="text/csv",
        metadata={},
    )
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    assert artifact is not None
    return int(
        repo.create_provenanced_run(
            run_kind="execution",
            run_status="completed",
            request_group_id=group_id,
            source_group_id=group_id,
            run_key="source-file-run",
            result_ref=str(artifact.uri),
            metadata_json={
                "execution_role": "method_execution",
                "pipeline_stage_role": "source",
            },
        )
    )


def _build_context(
    *,
    repo: RawCallRepository,
    group_id: int,
    imported_run_id: int,
) -> MethodRunnerContext:
    return MethodRunnerContext(
        repository=repo,
        request_group_id=group_id,
        source_group_id=group_id,
        method_name="csv_parse.basic",
        method_version="0.1",
        run_key="csv-parse-run",
        imported_run_id=imported_run_id,
        imported_run_metadata={
            "execution_role": "method_execution",
            "pipeline_stage_role": "source",
        },
    )


@pytest.mark.asyncio
async def test_csv_parse_basic_omitted_strict_fields_happy_path(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    artifact_dir = str((tmp_path / "artifacts").resolve())
    csv_payload = (
        "ItemID,Flag,Score,Note\n"
        "1,true,5,\n"
        "2,false,7,hello\n"
    ).encode("utf-8")

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        group_id = int(
            repo.create_group(group_type="analysis_request", name="csv-parse-test")
        )
        source_run_id = _seed_source_run(
            repo=repo,
            group_id=group_id,
            artifact_dir=artifact_dir,
            csv_payload=csv_payload,
        )
        context = _build_context(
            repo=repo,
            group_id=group_id,
            imported_run_id=source_run_id,
        )

        out = await run_csv_parse_basic({"dataset_name": "midterm2_questions"}, context)
        assert out.pipeline_stage_role == "imported"
        assert out.result_ref is not None
        assert Path(str(out.result_ref)).exists()
        stage_context = dict(out.pipeline_stage_context or {})
        assert stage_context["dataset_name"] == "midterm2_questions"
        assert stage_context["row_count"] == 2
        assert stage_context["column_count"] == 4
        assert [entry["name"] for entry in stage_context["columns"]] == [
            "ItemID",
            "Flag",
            "Score",
            "Note",
        ]


@pytest.mark.asyncio
async def test_csv_parse_basic_expected_columns_mismatch_fails_before_write(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    artifact_dir = str((tmp_path / "artifacts").resolve())
    csv_payload = (
        "ItemID,Flag,Score,Note\n"
        "1,true,5,\n"
        "2,false,7,hello\n"
    ).encode("utf-8")

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        group_id = int(
            repo.create_group(group_type="analysis_request", name="csv-parse-test")
        )
        source_run_id = _seed_source_run(
            repo=repo,
            group_id=group_id,
            artifact_dir=artifact_dir,
            csv_payload=csv_payload,
        )
        context = _build_context(
            repo=repo,
            group_id=group_id,
            imported_run_id=source_run_id,
        )
        before_count = int(session.query(CallArtifact).count())
        with pytest.raises(ValueError, match="csv_columns_mismatch"):
            await run_csv_parse_basic(
                {
                    "dataset_name": "midterm2_questions",
                    "expected_columns": ["ItemID", "Score", "MissingColumn"],
                },
                context,
            )
        after_count = int(session.query(CallArtifact).count())
        assert before_count == after_count


@pytest.mark.asyncio
async def test_csv_parse_basic_expected_dtypes_mismatch_fails_before_write(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    artifact_dir = str((tmp_path / "artifacts").resolve())
    csv_payload = (
        "ItemID,Flag,Score,Note\n"
        "1,true,5,\n"
        "2,false,7,hello\n"
    ).encode("utf-8")

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        group_id = int(
            repo.create_group(group_type="analysis_request", name="csv-parse-test")
        )
        source_run_id = _seed_source_run(
            repo=repo,
            group_id=group_id,
            artifact_dir=artifact_dir,
            csv_payload=csv_payload,
        )
        context = _build_context(
            repo=repo,
            group_id=group_id,
            imported_run_id=source_run_id,
        )
        before_count = int(session.query(CallArtifact).count())
        with pytest.raises(ValueError, match="csv_dtypes_mismatch"):
            await run_csv_parse_basic(
                {
                    "dataset_name": "midterm2_questions",
                    "expected_columns": ["ItemID", "Flag", "Score", "Note"],
                    "expected_dtypes": {"ItemID": "string"},
                },
                context,
            )
        after_count = int(session.query(CallArtifact).count())
        assert before_count == after_count


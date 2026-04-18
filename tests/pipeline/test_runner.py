"""Tests for pipeline runner ordering and status transitions."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.runner import run_stage


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "stage_runner.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _seed_groups(db: DatabaseConnectionV2) -> tuple[int, int]:
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = repo.create_group(
            group_type="analysis_request",
            name="request_1",
            metadata_json={},
        )
        source_group_id = repo.create_group(
            group_type="dataset_snapshot",
            name="source_snapshot_1",
            metadata_json={},
        )
        return request_group_id, source_group_id


def test_run_stage_claims_db_identity_before_artifact_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    request_group_id, source_group_id = _seed_groups(db)
    artifact_dir = (tmp_path / "artifacts").resolve()
    events: list[str] = []

    def writer(artifact_service, identity):
        run_row = artifact_service.repository.get_provenanced_run_by_id(identity.run_id)
        assert run_row is not None
        assert run_row.run_status == "running"
        events.append("db_identity_claimed")
        logical_path = f"{identity.group_id}/stage/test_payload.txt"
        uri = artifact_service.storage.write(
            logical_path=logical_path,
            data=b"payload",
            content_type="text/plain",
        )
        events.append("artifact_written")
        return {"payload": uri}

    def finalize(repo, identity, artifact_uris):
        events.append("finalized")
        return {"artifact_count": len(artifact_uris)}

    result = run_stage(
        db=db,
        stage_name="test_stage",
        group_type="analysis_run",
        group_name="analyze:test_stage:1:rk1",
        request_group_id=request_group_id,
        source_group_id=source_group_id,
        run_key="rk1",
        depends_on_group_ids=[source_group_id],
        artifact_dir=str(artifact_dir),
        write_artifacts=writer,
        finalize_db=finalize,
    )

    assert events == ["db_identity_claimed", "artifact_written", "finalized"]
    payload_uri = result.artifact_uris["payload"]
    assert Path(payload_uri).is_file()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        assert run_row.run_status == "completed"
        depends_on_link = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == source_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert depends_on_link is not None


def test_run_stage_marks_failed_when_artifact_writer_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    request_group_id, source_group_id = _seed_groups(db)
    artifact_dir = (tmp_path / "artifacts").resolve()

    def failing_writer(_artifact_service, _identity):
        raise RuntimeError("artifact write failure")

    with pytest.raises(RuntimeError, match="artifact write failure"):
        run_stage(
            db=db,
            stage_name="test_failure",
            group_type="analysis_run",
            group_name="analyze:test_failure:1:rk_fail",
            request_group_id=request_group_id,
            source_group_id=source_group_id,
            run_key="rk_fail",
            artifact_dir=str(artifact_dir),
            write_artifacts=failing_writer,
        )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_request_and_key(
            request_group_id=request_group_id,
            run_key="rk_fail",
            run_kind="execution",
        )
        assert run_row is not None
        assert run_row.run_status == "failed"
    assert not artifact_dir.exists() or not any(artifact_dir.rglob("*"))

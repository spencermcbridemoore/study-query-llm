"""Tests for pipeline.acquire stage."""

from __future__ import annotations

from pathlib import Path

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "acquire.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [
            FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv"),
            FileFetchSpec(relative_path="data/test.csv", url="https://example.test/test.csv"),
        ]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {
                "dataset": "fixture_dataset",
                "revision": "abc123",
            },
        }

    return DatasetAcquireConfig(
        slug="fixture_dataset",
        file_specs=file_specs,
        source_metadata=source_metadata,
    )


def test_acquire_is_idempotent_without_force(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    spec = _fixture_spec()
    payload_by_url = {
        "https://example.test/train.csv": b"id,text\n1,hello\n2,world\n",
        "https://example.test/test.csv": b"id,text\n3,test\n",
    }

    def fake_fetch(url: str) -> bytes:
        return payload_by_url[url]

    artifact_dir = str((tmp_path / "artifacts").resolve())
    first = acquire(spec, db=db, artifact_dir=artifact_dir, fetch=fake_fetch)
    second = acquire(spec, db=db, artifact_dir=artifact_dir, fetch=fake_fetch)
    forced = acquire(spec, db=db, artifact_dir=artifact_dir, fetch=fake_fetch, force=True)

    assert first.group_id == second.group_id
    assert first.group_id != forced.group_id
    assert "acquisition.json" in first.artifact_uris
    assert "data/train.csv" in first.artifact_uris
    assert "data/test.csv" in first.artifact_uris

    with db.session_scope() as session:
        dataset_groups = (
            session.query(Group)
            .filter(Group.group_type == "dataset")
            .order_by(Group.id.asc())
            .all()
        )
        assert len(dataset_groups) == 2
        first_meta = dict(dataset_groups[0].metadata_json or {})
        assert first_meta["dataset_slug"] == "fixture_dataset"
        assert "content_fingerprint" in first_meta
        assert "manifest_hash" in first_meta

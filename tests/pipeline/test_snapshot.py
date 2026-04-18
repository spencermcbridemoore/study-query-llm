"""Tests for pipeline.snapshot stage."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "snapshot.sqlite3").resolve()
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
            "pinning_identity": {"dataset": "fixture_dataset", "revision": "abc123"},
        }

    return DatasetAcquireConfig(
        slug="fixture_dataset",
        file_specs=file_specs,
        source_metadata=source_metadata,
    )


def _fixture_parser(ctx) -> list[SnapshotRow]:
    assert (ctx.artifact_dir_local / "data" / "train.csv").is_file()
    assert (ctx.artifact_dir_local / "data" / "test.csv").is_file()
    return [
        SnapshotRow(position=0, source_id="row-0", text="hello", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="row-1", text="world", label=1, label_name="b"),
        SnapshotRow(position=2, source_id="row-2", text="again", label=1, label_name="b"),
    ]


def test_snapshot_writes_parquet_manifest_and_depends_on_link(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    spec = _fixture_spec()
    payload_by_url = {
        "https://example.test/train.csv": b"id,text\n1,hello\n2,world\n",
        "https://example.test/test.csv": b"id,text\n3,test\n",
    }
    artifact_dir = str((tmp_path / "artifacts").resolve())

    dataset_result = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda url: payload_by_url[url],
    )
    first = snapshot(
        dataset_result.group_id,
        parser=_fixture_parser,
        db=db,
        artifact_dir=artifact_dir,
    )
    second = snapshot(
        dataset_result.group_id,
        parser=_fixture_parser,
        db=db,
        artifact_dir=artifact_dir,
    )

    assert first.group_id == second.group_id
    parquet_uri = first.artifact_uris["snapshot.parquet"]
    index_uri = first.artifact_uris["snapshot_index.json"]
    assert Path(parquet_uri).is_file()
    assert Path(index_uri).is_file()

    table = pq.read_table(parquet_uri)
    assert table.column("position").to_pylist() == [0, 1, 2]
    assert table.column("source_id").to_pylist() == ["row-0", "row-1", "row-2"]
    assert table.column("label").to_pylist() == [0, 1, 1]

    with db.session_scope() as session:
        link = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == first.group_id,
                GroupLink.child_group_id == dataset_result.group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert link is not None

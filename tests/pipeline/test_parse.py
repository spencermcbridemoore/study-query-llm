"""Tests for pipeline.parse stage."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.types import SnapshotRow


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "parse.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _fixture_parser(_ctx) -> list[SnapshotRow]:
    return [
        SnapshotRow(position=0, source_id="id-0", text="alpha", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="id-1", text="beta", label=1, label_name="b"),
    ]


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv")]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "parse_fixture", "revision": "r1"},
        }

    return DatasetAcquireConfig(
        slug="parse_fixture",
        file_specs=file_specs,
        source_metadata=source_metadata,
        default_parser=_fixture_parser,
        default_parser_id="parse_fixture.default",
        default_parser_version="v1",
    )


def test_parse_idempotent_and_writes_canonical_parquet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    acquired = acquire(
        _fixture_spec(),
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: b"id,text\n1,hello\n",
    )

    first = parse(
        acquired.group_id,
        parser=_fixture_parser,
        parser_id="parse_fixture.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    second = parse(
        acquired.group_id,
        parser=_fixture_parser,
        parser_id="parse_fixture.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )

    assert first.group_id == second.group_id
    assert second.metadata["reused"] is True
    parquet_uri = first.artifact_uris["dataframe.parquet"]
    table = pq.read_table(parquet_uri)
    assert table.column("position").to_pylist() == [0, 1]
    assert table.column("text").to_pylist() == ["alpha", "beta"]
    assert first.metadata["parser_id"] == "parse_fixture.default"
    assert first.metadata["parser_version"] == "v1"


def test_parse_version_change_invalidates_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    acquired = acquire(
        _fixture_spec(),
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: b"id,text\n1,hello\n",
    )

    v1 = parse(
        acquired.group_id,
        parser=_fixture_parser,
        parser_id="parse_fixture.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    v2 = parse(
        acquired.group_id,
        parser=_fixture_parser,
        parser_id="parse_fixture.default",
        parser_version="v2",
        db=db,
        artifact_dir=artifact_dir,
    )
    assert v1.group_id != v2.group_id

"""Tests for pipeline.embed stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "embed.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [
            FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv"),
        ]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "embed_fixture", "revision": "r1"},
        }

    return DatasetAcquireConfig(
        slug="embed_fixture",
        file_specs=file_specs,
        source_metadata=source_metadata,
        default_parser=_fixture_parser,
        default_parser_id="embed_fixture.default",
        default_parser_version="v1",
    )


def _fixture_parser(_ctx) -> list[SnapshotRow]:
    return [
        SnapshotRow(position=0, source_id="id-0", text="alpha", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="id-1", text="beta", label=1, label_name="b"),
        SnapshotRow(position=2, source_id="id-2", text="gamma", label=1, label_name="b"),
    ]


def _prepare_snapshot(
    *,
    db: DatabaseConnectionV2,
    artifact_dir: str,
) -> tuple[int, int, int]:
    acquired = acquire(
        _fixture_spec(),
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: b"id,text\n1,hello\n2,world\n",
    )
    parsed = parse(
        acquired.group_id,
        parser=_fixture_parser,
        parser_id="embed_fixture.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    snapped = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    return int(acquired.group_id), int(parsed.group_id), int(snapped.group_id)


def test_embed_reuses_existing_matrix_across_dataframe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )

    calls = {"count": 0}

    def fake_fetcher(**kwargs):
        calls["count"] += 1
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    first = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    second = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )

    assert calls["count"] == 1
    assert first.group_id == second.group_id
    assert first.metadata["reused"] is False
    assert second.metadata["reused"] is True
    assert Path(first.artifact_uris["embedding_matrix.npy"]).is_file()

    # A distinct snapshot over the same dataframe should still reuse the same full matrix.
    snapshot(
        dataframe_group_id,
        subquery_spec=SubquerySpec(sample_n=2, sampling_seed=9, label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    third = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    assert third.group_id == first.group_id


def test_embed_rejects_non_full_representations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )

    try:
        embed(
            dataframe_group_id,
            deployment="test-embedding-model",
            provider="test-provider",
            representation="label_centroid",
            db=db,
            artifact_dir=artifact_dir,
        )
    except ValueError as exc:
        assert "only supports representation='full'" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-full representation rejection")

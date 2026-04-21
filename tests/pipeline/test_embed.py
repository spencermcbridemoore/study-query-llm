"""Tests for pipeline.embed stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow
from study_query_llm.services.artifact_service import ArtifactService


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
) -> int:
    acquired = acquire(
        _fixture_spec(),
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: b"id,text\n1,hello\n2,world\n",
    )
    snapped = snapshot(
        acquired.group_id,
        parser=_fixture_parser,
        db=db,
        artifact_dir=artifact_dir,
    )
    return int(snapped.group_id)


def test_embed_reuses_existing_matrix_and_depends_on_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    snapshot_group_id = _prepare_snapshot(db=db, artifact_dir=artifact_dir)

    calls = {"count": 0}

    def fake_fetcher(**kwargs):
        calls["count"] += 1
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    first = embed(
        snapshot_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    second = embed(
        snapshot_group_id,
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

    with db.session_scope() as session:
        link = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == first.group_id,
                GroupLink.child_group_id == snapshot_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert link is not None


def test_embed_label_centroid_representation_pools_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    snapshot_group_id = _prepare_snapshot(db=db, artifact_dir=artifact_dir)

    def fake_fetcher(**_kwargs):
        return np.asarray(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 5.0],
            ],
            dtype=np.float64,
        )

    result = embed(
        snapshot_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        representation="label_centroid",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    pooled = artifact_service.load_artifact(
        result.artifact_uris["embedding_matrix.npy"],
        "embedding_matrix",
    )
    assert pooled.shape == (2, 2)
    np.testing.assert_allclose(pooled[0], np.asarray([1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(pooled[1], np.asarray([4.0, 4.0], dtype=np.float64))
    assert result.metadata["row_count"] == 2


def test_embed_intent_mean_alias_maps_to_label_centroid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    snapshot_group_id = _prepare_snapshot(db=db, artifact_dir=artifact_dir)

    def fake_fetcher(**_kwargs):
        return np.asarray(
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [5.0, 5.0],
            ],
            dtype=np.float64,
        )

    result = embed(
        snapshot_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        representation="intent_mean",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    assert result.metadata["representation"] == "label_centroid"
    assert str(result.metadata["dataset_key"]).endswith(":label_centroid")


def test_embed_sparse_representation_writes_sidecar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    snapshot_group_id = _prepare_snapshot(db=db, artifact_dir=artifact_dir)

    def fake_fetcher(**_kwargs):
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float64,
        )

    result = embed(
        snapshot_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        representation="sparse",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    assert "sparse_sidecar.json" in result.artifact_uris
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(result.artifact_uris["sparse_sidecar.json"])
    sidecar = json.loads(payload.decode("utf-8"))
    assert sidecar["shape"] == [3, 3]
    assert sidecar["nnz"] == 3
    assert sidecar["density"] == 3 / 9

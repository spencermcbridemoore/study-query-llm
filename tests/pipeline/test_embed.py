"""Tests for pipeline.embed stage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time

import numpy as np

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import EmbeddingCacheLease
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.embed import _matrix_lease_key, embed
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec
from study_query_llm.services.embeddings.constants import CACHE_KEY_VERSION


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


def test_embed_entry_max_bounds_rows_and_reuse(tmp_path: Path, monkeypatch) -> None:
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
        return np.asarray([[float(i)] for i in range(len(texts))], dtype=np.float64)

    bounded_first = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        entry_max=2,
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    bounded_second = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        entry_max=2,
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )

    assert calls["count"] == 1
    assert bounded_first.metadata["row_count"] == 2
    assert bounded_first.metadata["reused"] is False
    assert bounded_second.metadata["reused"] is True
    assert bounded_first.group_id == bounded_second.group_id

    wider = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        entry_max=3,
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    assert calls["count"] == 2
    assert wider.metadata["row_count"] == 3
    assert wider.metadata["reused"] is False
    assert wider.group_id != bounded_first.group_id


def test_embed_threads_runtime_knobs_to_fetcher(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )

    captured: dict[str, object] = {}

    def fake_fetcher(**kwargs):
        captured.update(kwargs)
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    _ = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
        chunk_size=2,
        chunk_worker_concurrency=3,
        chunk_circuit_breaker_enabled=True,
        chunk_failure_fallback_threshold=4,
        max_retries=9,
        initial_wait=0.25,
        max_wait=4.0,
        singleflight_lease_seconds=11,
        singleflight_wait_timeout_seconds=22.0,
        singleflight_poll_seconds=0.2,
        timeout=123.0,
    )

    assert captured["chunk_size"] == 2
    assert captured["chunk_worker_concurrency"] == 3
    assert captured["chunk_circuit_breaker_enabled"] is True
    assert captured["chunk_failure_fallback_threshold"] == 4
    assert captured["max_retries"] == 9
    assert captured["initial_wait"] == 0.25
    assert captured["max_wait"] == 4.0
    assert captured["singleflight_lease_seconds"] == 11
    assert captured["singleflight_wait_timeout_seconds"] == 22.0
    assert captured["singleflight_poll_seconds"] == 0.2
    assert captured["timeout"] == 123.0


def test_embed_matrix_dedupe_sequential_idempotency(tmp_path: Path, monkeypatch) -> None:
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
    assert second.metadata["reused"] is True
    assert second.metadata["matrix_dedupe_reused"] is True


def test_embed_matrix_dedupe_chunked_mode_reuses_single_batch(tmp_path: Path, monkeypatch) -> None:
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
        assert kwargs["chunk_size"] == 2
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    first = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
        chunk_size=2,
    )
    second = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
        chunk_size=2,
    )

    assert calls["count"] == 1
    assert first.group_id == second.group_id
    assert second.metadata["reused"] is True


def test_embed_matrix_dedupe_distinct_keys_produce_distinct_batches(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )

    def fake_fetcher(**kwargs):
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    base = embed(
        dataframe_group_id,
        deployment="engine-a",
        provider="provider-a",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    changed_engine = embed(
        dataframe_group_id,
        deployment="engine-b",
        provider="provider-a",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    changed_provider = embed(
        dataframe_group_id,
        deployment="engine-a",
        provider="provider-b",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    changed_entry_max = embed(
        dataframe_group_id,
        deployment="engine-a",
        provider="provider-a",
        entry_max=2,
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    changed_key_version = embed(
        dataframe_group_id,
        deployment="engine-a",
        provider="provider-a",
        key_version="raw_v2",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )

    ids = {
        int(base.group_id),
        int(changed_engine.group_id),
        int(changed_provider.group_id),
        int(changed_entry_max.group_id),
        int(changed_key_version.group_id),
    }
    assert len(ids) == 5


def test_embed_matrix_dedupe_concurrent_calls_share_single_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )

    calls = {"count": 0}
    lock = threading.Lock()

    def fake_fetcher(**kwargs):
        with lock:
            calls["count"] += 1
        time.sleep(0.2)
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    def run_once():
        return embed(
            dataframe_group_id,
            deployment="test-embedding-model",
            provider="test-provider",
            db=db,
            artifact_dir=artifact_dir,
            embedding_fetcher=fake_fetcher,
            singleflight_wait_timeout_seconds=5.0,
            singleflight_poll_seconds=0.05,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(run_once)
        fut_b = pool.submit(run_once)
        result_a = fut_a.result()
        result_b = fut_b.result()

    assert calls["count"] == 1
    assert result_a.group_id == result_b.group_id
    assert bool(result_a.metadata["reused"]) != bool(result_b.metadata["reused"])


def test_embed_matrix_dedupe_takes_over_expired_lease_without_integrityerror(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _dataset_group_id, dataframe_group_id, _snapshot_group_id = _prepare_snapshot(
        db=db,
        artifact_dir=artifact_dir,
    )
    dataset_key = f"dataframe:{int(dataframe_group_id)}:full"
    _logical_key, stale_storage_key = _matrix_lease_key(
        dataset_key=dataset_key,
        embedding_engine="test-embedding-model",
        provider="test-provider",
        entry_max=3,
        key_version=CACHE_KEY_VERSION,
    )
    now = datetime.now(timezone.utc)
    with db.session_scope() as session:
        session.add(
            EmbeddingCacheLease(
                cache_key=stale_storage_key,
                lease_owner="stale-owner",
                lease_expires_at=now - timedelta(minutes=5),
                created_at=now - timedelta(minutes=10),
                updated_at=now - timedelta(minutes=10),
            )
        )
        session.flush()

    calls = {"count": 0}

    def fake_fetcher(**kwargs):
        calls["count"] += 1
        texts = kwargs["texts"]
        return np.asarray([[float(i), float(i + 1)] for i in range(len(texts))], dtype=np.float64)

    result = embed(
        dataframe_group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    assert result.metadata["reused"] is False
    assert calls["count"] == 1
    with db.session_scope() as session:
        lease = (
            session.query(EmbeddingCacheLease)
            .filter(EmbeddingCacheLease.cache_key == stale_storage_key)
            .first()
        )
        assert lease is None

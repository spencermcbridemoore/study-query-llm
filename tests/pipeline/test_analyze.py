"""Tests for pipeline.analyze stage."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import AnalysisResult, Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.analyze import analyze
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.hdbscan_runner import run_hdbscan_analysis
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow


def _db(tmp_path: Path) -> tuple[DatabaseConnectionV2, str]:
    db_path = (tmp_path / "analyze.sqlite3").resolve()
    database_url = f"sqlite:///{db_path.as_posix()}"
    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()
    return db, database_url


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [
            FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv"),
        ]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "analyze_fixture", "revision": "r1"},
        }

    return DatasetAcquireConfig(
        slug="analyze_fixture",
        file_specs=file_specs,
        source_metadata=source_metadata,
    )


def _fixture_parser(_ctx) -> list[SnapshotRow]:
    return [
        SnapshotRow(position=0, source_id="id-0", text="alpha", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="id-1", text="beta", label=1, label_name="b"),
        SnapshotRow(position=2, source_id="id-2", text="gamma", label=1, label_name="b"),
    ]


def _prepare_embedding_input(
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

    def fake_fetcher(**kwargs):
        texts = kwargs["texts"]
        return np.asarray(
            [[float(i), float(i + 1), float(i + 2)] for i in range(len(texts))],
            dtype=np.float64,
        )

    embedded = embed(
        snapped.group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    return int(embedded.group_id)


def _create_request_group(db: DatabaseConnectionV2, name: str) -> int:
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        return int(
            repo.create_group(
                group_type="analysis_request",
                name=name,
                metadata_json={},
            )
        )


def test_analyze_claims_running_before_runner_and_completes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    request_group_id = _create_request_group(db, "request_status")
    events: list[str] = []

    def runner(**kwargs):
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            run_row = repo.get_provenanced_run_by_request_and_key(
                request_group_id=request_group_id,
                run_key="rk_status",
                run_kind="analysis_execution",
            )
            assert run_row is not None
            assert run_row.run_status == "running"
        events.append("runner")
        return {
            "scalar_results": {"row_count": float(kwargs["embeddings"].shape[0])},
            "structured_results": {"summary": {"ok": True}},
            "artifacts": {"analysis_summary.json": b'{"ok": true}'},
            "result_ref": "analysis_summary.json",
        }

    result = analyze(
        input_group_id,
        method_name="status_method",
        run_key="rk_status",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )

    assert events == ["runner"]
    assert "analysis_summary.json" in result.artifact_uris

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        assert run_row.run_status == "completed"

        depends_on = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == input_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert depends_on is not None

        contains = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == request_group_id,
                GroupLink.child_group_id == result.group_id,
                GroupLink.link_type == "contains",
            )
            .first()
        )
        assert contains is not None


def test_analyze_marks_failed_when_runner_errors_without_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    request_group_id = _create_request_group(db, "request_failure")

    def failing_runner(**_kwargs):
        raise RuntimeError("runner failure")

    with pytest.raises(RuntimeError, match="runner failure"):
        analyze(
            input_group_id,
            method_name="failure_method",
            run_key="rk_failure",
            request_group_id=request_group_id,
            db=db,
            artifact_dir=artifact_dir,
            method_runner=failing_runner,
        )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_request_and_key(
            request_group_id=request_group_id,
            run_key="rk_failure",
            run_kind="analysis_execution",
        )
        assert run_row is not None
        assert run_row.run_status == "failed"
        count = (
            session.query(AnalysisResult)
            .filter(AnalysisResult.analysis_group_id == run_row.result_group_id)
            .count()
        )
        assert count == 0


def test_analyze_auto_request_group_and_idempotent_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    calls = {"count": 0}

    def runner(**kwargs):
        calls["count"] += 1
        return {
            "scalar_results": {"row_count": float(kwargs["embeddings"].shape[0])},
            "structured_results": {},
            "artifacts": {"summary.json": b"{}"},
            "result_ref": "summary.json",
        }

    first = analyze(
        input_group_id,
        method_name="auto_request_method",
        run_key="rk_auto",
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )
    second = analyze(
        input_group_id,
        method_name="auto_request_method",
        run_key="rk_auto",
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )

    assert calls["count"] == 1
    assert first.group_id == second.group_id
    assert first.run_id == second.run_id
    assert first.metadata["reused"] is False
    assert second.metadata["reused"] is True

    with db.session_scope() as session:
        request_groups = (
            session.query(Group)
            .filter(Group.group_type == "analysis_request")
            .all()
        )
        matches = []
        for group in request_groups:
            metadata = dict(group.metadata_json or {})
            if (
                metadata.get("method_name") == "auto_request_method"
                and int(metadata.get("input_id") or -1) == input_group_id
                and metadata.get("run_key") == "rk_auto"
            ):
                matches.append(group.id)
        assert len(matches) == 1


def test_analyze_enriches_execution_row_with_canonical_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    request_group_id = _create_request_group(db, "request_canonical")

    def runner(**kwargs):
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        return {
            "scalar_results": {"row_count": float(embeddings.shape[0])},
            "structured_results": {"canonical": {"ok": True}},
            "artifacts": {"canonical.json": b'{"canonical":true}'},
            "result_ref": "canonical.json",
        }

    result = analyze(
        input_group_id,
        method_name="canonical_identity_method",
        run_key="rk_canonical",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters={
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
            "determinism_class": "non_deterministic",
        },
        method_runner=runner,
    )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_request_and_key(
            request_group_id=request_group_id,
            run_key="rk_canonical",
            run_kind="analysis_execution",
        )
        assert run_row is not None
        assert int(run_row.id) == int(result.run_id or -1)
        assert run_row.method_definition_id is not None
        assert run_row.config_json is not None
        assert run_row.config_json["parameters"]["dataset_slug"] == "analyze_fixture"
        assert run_row.config_hash is not None
        assert len(str(run_row.config_hash)) == 64
        assert run_row.fingerprint_json is not None
        assert run_row.fingerprint_json["method_name"] == "canonical_identity_method"
        assert run_row.fingerprint_json["determinism_class"] == "non_deterministic"
        assert run_row.fingerprint_hash is not None
        assert len(str(run_row.fingerprint_hash)) == 64


def test_analyze_hdbscan_runner_persists_expected_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("hdbscan")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    request_group_id = _create_request_group(db, "request_hdbscan")

    result = analyze(
        input_group_id,
        method_name="phase1_hdbscan_fixture",
        run_key="rk_hdbscan",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        method_runner=run_hdbscan_analysis,
        parameters={
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
            "hdbscan_min_cluster_size": 2,
            "hdbscan_min_samples": 1,
            "hdbscan_metric": "euclidean",
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_normalize_embeddings": True,
        },
    )

    assert "hdbscan_summary.json" in result.artifact_uris
    assert "hdbscan_labels.json" in result.artifact_uris

    with db.session_scope() as session:
        result_rows = (
            session.query(AnalysisResult)
            .filter(AnalysisResult.analysis_group_id == result.group_id)
            .all()
        )
        result_keys = {row.result_key for row in result_rows}
        assert "hdbscan_summary" in result_keys
        assert "hdbscan_cluster_labels" in result_keys
        assert "cluster_count" in result_keys
        assert "noise_count" in result_keys

        labels_row = next(
            row for row in result_rows if row.result_key == "hdbscan_cluster_labels"
        )
        labels_payload = labels_row.result_json["value"]
        assert len(labels_payload["cluster_labels"]) == 3


def test_analyze_race_same_key_runs_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    input_group_id = _prepare_embedding_input(db=db, artifact_dir=artifact_dir)
    request_group_id = _create_request_group(db, "request_race")
    calls = {"count": 0}
    count_lock = threading.Lock()

    def runner(**kwargs):
        with count_lock:
            calls["count"] += 1
        time.sleep(0.1)
        return {
            "scalar_results": {"row_count": float(kwargs["embeddings"].shape[0])},
            "structured_results": {"race": True},
            "artifacts": {"race.json": b'{"race": true}'},
            "result_ref": "race.json",
        }

    def invoke():
        return analyze(
            input_group_id,
            method_name="race_method",
            run_key="rk_race",
            request_group_id=request_group_id,
            database_url=database_url,
            artifact_dir=artifact_dir,
            method_runner=runner,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(invoke)
        future_b = pool.submit(invoke)
        result_a = future_a.result()
        result_b = future_b.result()

    assert calls["count"] == 1
    assert result_a.group_id == result_b.group_id
    assert result_a.run_id == result_b.run_id

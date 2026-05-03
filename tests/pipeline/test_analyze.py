"""Tests for pipeline.analyze stage."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from study_query_llm.algorithms.recipes import COMPOSITE_RECIPES, canonical_recipe_hash
from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import AnalysisResult, Group, GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.analyze import analyze
from study_query_llm.pipeline.clustering import (
    run_gmm_bic_argmin_analysis,
    run_kmeans_silhouette_kneedle_analysis,
)
from study_query_llm.pipeline.clustering.agglomerative_preproc_runner import (
    run_agglomerative_preproc_fixed_k_analysis,
)
from study_query_llm.pipeline.clustering.dbscan_fixed_eps_runner import (
    run_dbscan_fixed_eps_analysis,
)
from study_query_llm.pipeline.clustering.gmm_fixed_k_runner import run_gmm_fixed_k_analysis
from study_query_llm.pipeline.clustering.hdbscan_preproc_fixed_runner import (
    run_hdbscan_preproc_fixed_analysis,
)
from study_query_llm.pipeline.clustering.kmeans_fixed_k_runner import (
    run_kmeans_fixed_k_analysis,
)
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.hdbscan_runner import run_hdbscan_analysis
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec
from study_query_llm.services.method_service import MethodService


_V1_ENVELOPE_FIELDS: tuple[str, ...] = (
    "rule_set_hash",
    "rule_set_version",
    "rule_inputs",
    "pipeline_declared",
    "pipeline_resolved",
    "pipeline_effective",
    "pipeline_effective_hash",
    "operation_type",
    "operation_version",
)


def _assert_no_v1_envelope_fields(config_json: Mapping[str, Any]) -> None:
    """Slice 1.5 invariant: bundled-grammar methods emit zero v1-envelope fields.

    The deprecation+removal pass dropped the ``cluster_pipeline`` envelope; new
    write paths must not leak the legacy provenance keys into ``config_json``
    on ``provenanced_runs``. Used by the rewritten v1-envelope integration
    tests to pin the simplified single-envelope model.
    """
    cfg = dict(config_json or {})
    leaked = [field for field in _V1_ENVELOPE_FIELDS if field in cfg]
    assert not leaked, f"v1 envelope fields leaked into config_json: {leaked}"


def _db(tmp_path: Path) -> tuple[DatabaseConnectionV2, str]:
    db_path = (tmp_path / "analyze.sqlite3").resolve()
    database_url = f"sqlite:///{db_path.as_posix()}"
    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()
    return db, database_url


def _fixture_spec() -> DatasetAcquireConfig:
    def file_specs():
        return [FileFetchSpec(relative_path="data/train.csv", url="https://example.test/train.csv")]

    def source_metadata():
        return {
            "kind": "fixture",
            "pinning_identity": {"dataset": "analyze_fixture", "revision": "r1"},
        }

    return DatasetAcquireConfig(
        slug="analyze_fixture",
        file_specs=file_specs,
        source_metadata=source_metadata,
        default_parser=_fixture_parser,
        default_parser_id="analyze_fixture.default",
        default_parser_version="v1",
    )


def _fixture_parser(_ctx) -> list[SnapshotRow]:
    return [
        SnapshotRow(position=0, source_id="id-0", text="alpha", label=0, label_name="a"),
        SnapshotRow(position=1, source_id="id-1", text="beta", label=1, label_name="b"),
        SnapshotRow(position=2, source_id="id-2", text="gamma", label=1, label_name="b"),
        SnapshotRow(position=3, source_id="id-3", text="delta", label=0, label_name="a"),
    ]


def _prepare_inputs(
    *,
    db: DatabaseConnectionV2,
    artifact_dir: str,
    subquery_spec: SubquerySpec | None = None,
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
        parser_id="analyze_fixture.default",
        parser_version="v1",
        db=db,
        artifact_dir=artifact_dir,
    )
    snapped = snapshot(
        parsed.group_id,
        subquery_spec=subquery_spec or SubquerySpec(label_mode="all"),
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
        parsed.group_id,
        deployment="test-embedding-model",
        provider="test-provider",
        db=db,
        artifact_dir=artifact_dir,
        embedding_fetcher=fake_fetcher,
    )
    return int(parsed.group_id), int(snapped.group_id), int(embedded.group_id)


def _create_request_group(db: DatabaseConnectionV2, name: str) -> int:
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        return int(repo.create_group(group_type="analysis_request", name=name, metadata_json={}))


def test_analyze_claims_running_and_preserves_text_vector_alignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
        subquery_spec=SubquerySpec(sample_n=2, sampling_seed=11, label_mode="all"),
    )
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
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        texts = list(kwargs["texts"])
        assert embeddings.shape[0] == len(texts)
        events.append("runner")
        return {
            "scalar_results": {"row_count": float(embeddings.shape[0])},
            "structured_results": {"texts": texts},
            "artifacts": {"analysis_summary.json": b'{"ok": true}'},
            "result_ref": "analysis_summary.json",
        }

    result = analyze(
        snapshot_group_id,
        embedding_group_id,
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
        assert int(run_row.source_group_id or -1) == int(embedding_group_id)
        cfg = dict(run_row.config_json or {})
        assert int(cfg.get("embedding_batch_group_id") or -1) == int(embedding_group_id)
        assert "analysis_input_mode" not in cfg

        dep_embedding = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == embedding_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        dep_snapshot = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == snapshot_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert dep_embedding is not None
        assert dep_snapshot is not None


def test_analyze_auto_request_group_and_idempotent_reuse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    calls = {"count": 0}

    def runner(**kwargs):
        calls["count"] += 1
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        return {
            "scalar_results": {"row_count": float(embeddings.shape[0])},
            "structured_results": {},
            "artifacts": {"summary.json": b"{}"},
            "result_ref": "summary.json",
        }

    first = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="auto_request_method",
        run_key="rk_auto",
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )
    second = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="auto_request_method",
        run_key="rk_auto",
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )

    assert calls["count"] == 1
    assert first.group_id == second.group_id
    assert first.run_id == second.run_id
    assert second.metadata["reused"] is True

    with db.session_scope() as session:
        request_groups = session.query(Group).filter(Group.group_type == "analysis_request").all()
        assert len(request_groups) == 1


def test_analyze_requires_embedding_by_default_when_contract_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, _embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="requires embedding_batch_group_id"):
        analyze(
            snapshot_group_id,
            None,
            method_name="default_requires_embedding",
            run_key="rk_missing_embedding",
            db=db,
            artifact_dir=artifact_dir,
        )


def test_analyze_malformed_required_inputs_defaults_to_embedding_required(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, _embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    with db.session_scope() as session:
        method_service = MethodService(RawCallRepository(session))
        method_service.register_method(
            name="legacy_malformed_contract",
            version="v1",
            code_ref="tests.pipeline.test_analyze",
            input_schema={
                "required_inputs": {
                    "snapshot": True,
                    "embedding_batch": True,
                }
            },
        )
        method_row = method_service.get_method("legacy_malformed_contract", version="v1")
        assert method_row is not None
        method_row.input_schema = {
            "required_inputs": {
                "snapshot": "true",
                "embedding_batch": "false",
            }
        }
        session.flush()

    with pytest.raises(ValueError, match="requires embedding_batch_group_id"):
        analyze(
            snapshot_group_id,
            None,
            method_name="legacy_malformed_contract",
            method_version="v1",
            run_key="rk_legacy_malformed_contract",
            db=db,
            artifact_dir=artifact_dir,
        )


def test_analyze_snapshot_only_method_uses_snapshot_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_snapshot_only")
    with db.session_scope() as session:
        method_service = MethodService(RawCallRepository(session))
        method_service.register_method(
            name="snapshot_only_fixture",
            version="v1",
            code_ref="tests.pipeline.test_analyze",
            input_schema={
                "required_inputs": {
                    "snapshot": True,
                    "embedding_batch": False,
                }
            },
        )

    def runner(**kwargs):
        assert kwargs["input_group_id"] == snapshot_group_id
        assert kwargs["input_group_type"] == "dataset_snapshot"
        assert kwargs["embeddings"] is None
        texts = list(kwargs["texts"])
        return {
            "scalar_results": {"row_count": float(len(texts))},
            "structured_results": {"mode": "snapshot_only"},
            "artifacts": {"snapshot_only.json": b'{"mode":"snapshot_only"}'},
            "result_ref": "snapshot_only.json",
        }

    result = analyze(
        snapshot_group_id,
        None,
        method_name="snapshot_only_fixture",
        method_version="v1",
        run_key="rk_snapshot_only",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
    )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        assert int(run_row.source_group_id or -1) == int(snapshot_group_id)
        cfg = dict(run_row.config_json or {})
        assert cfg["analysis_input_mode"] == "snapshot_only"
        assert "embedding_batch_group_id" not in cfg

        dep_snapshot = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == snapshot_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        dep_embedding = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == result.group_id,
                GroupLink.child_group_id == embedding_group_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert dep_snapshot is not None
        assert dep_embedding is None


def test_analyze_fingerprint_includes_snapshot_and_representation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_fingerprint")

    def runner(**kwargs):
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        return {
            "scalar_results": {"row_count": float(embeddings.shape[0])},
            "structured_results": {"ok": True},
            "artifacts": {"fingerprint.json": b"{}"},
            "result_ref": "fingerprint.json",
        }

    full = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="fp_method",
        run_key="rk_fp_full",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        method_runner=runner,
        parameters={"representation_type": "full"},
    )
    with pytest.raises(ValueError, match=r"retired in Slice 1\.6"):
        analyze(
            snapshot_group_id,
            embedding_group_id,
            method_name="fp_method",
            run_key="rk_fp_centroid",
            request_group_id=request_group_id,
            db=db,
            artifact_dir=artifact_dir,
            method_runner=runner,
            parameters={"representation_type": "label_centroid"},
        )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_full = repo.get_provenanced_run_by_id(int(full.run_id or 0))
        assert run_full is not None
        assert run_full.input_snapshot_group_id == snapshot_group_id


def test_analyze_hdbscan_migration_error_fires_before_runner_non_full_check(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Slice 1.6: label_centroid raises in analyze before HDBSCAN runner sees embeddings."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_hdbscan_non_full")

    with pytest.raises(ValueError, match=r"retired in Slice 1\.6"):
        analyze(
            snapshot_group_id,
            embedding_group_id,
            method_name="phase1_hdbscan_fixture",
            run_key="rk_hdbscan_non_full",
            request_group_id=request_group_id,
            db=db,
            artifact_dir=artifact_dir,
            method_runner=run_hdbscan_analysis,
            parameters={
                "dataset_slug": "analyze_fixture",
                "representation_type": "label_centroid",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "hdbscan_min_cluster_size": 2,
                "hdbscan_min_samples": 1,
            },
        )


def test_analyze_label_centroid_loud_rejects_with_migration_message(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_slice16_migration")

    def runner(**kwargs):
        embeddings = np.asarray(kwargs["embeddings"], dtype=np.float64)
        return {
            "scalar_results": {"row_count": float(embeddings.shape[0])},
            "structured_results": {"ok": True},
            "artifacts": {"out.json": b"{}"},
            "result_ref": "out.json",
        }

    for rep in ("label_centroid", "intent_mean"):
        with pytest.raises(ValueError) as excinfo:
            analyze(
                snapshot_group_id,
                embedding_group_id,
                method_name="slice16_migration",
                run_key=f"rk_{rep}",
                request_group_id=request_group_id,
                db=db,
                artifact_dir=artifact_dir,
                method_runner=runner,
                parameters={"representation_type": rep},
            )
        msg = str(excinfo.value)
        assert "label_centroid" in msg
        assert "retired in Slice 1.6" in msg


def test_hdbscan_runner_uses_deterministic_defaults_and_echoes_parameters(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def fit_predict(self, matrix):
            rows = int(np.asarray(matrix).shape[0])
            self.probabilities_ = np.ones(rows, dtype=np.float64)
            self.outlier_scores_ = np.zeros(rows, dtype=np.float64)
            return np.asarray([0, 0, -1], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN))

    result = run_hdbscan_analysis(
        method_name="phase1_hdbscan_fixture",
        input_group_id=123,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64),
        texts=["a", "b", "c"],
        parameters={"hdbscan_min_cluster_size": 2},
    )

    kwargs = dict(captured.get("kwargs") or {})
    assert kwargs["metric"] == "cosine"
    assert kwargs["random_state"] == 0
    assert kwargs["core_dist_n_jobs"] == 1
    assert kwargs["approx_min_span_tree"] is False

    used = (
        result["structured_results"]["hdbscan_summary"]["parameters"]  # type: ignore[index]
    )
    assert used["hdbscan_metric"] == "cosine"
    assert used["hdbscan_random_state"] == 0
    assert used["hdbscan_core_dist_n_jobs"] == 1
    assert used["hdbscan_approx_min_span_tree"] is False


def test_hdbscan_runner_allows_explicit_policy_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeHDBSCAN:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def fit_predict(self, matrix):
            rows = int(np.asarray(matrix).shape[0])
            self.probabilities_ = np.ones(rows, dtype=np.float64)
            self.outlier_scores_ = np.zeros(rows, dtype=np.float64)
            return np.asarray([0, 1, -1], dtype=np.int64)

    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN))

    result = run_hdbscan_analysis(
        method_name="phase1_hdbscan_fixture",
        input_group_id=124,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64),
        texts=["a", "b", "c"],
        parameters={
            "hdbscan_min_cluster_size": 2,
            "hdbscan_metric": "euclidean",
            "hdbscan_random_state": 7,
            "hdbscan_core_dist_n_jobs": 2,
            "hdbscan_approx_min_span_tree": True,
        },
    )

    kwargs = dict(captured.get("kwargs") or {})
    assert kwargs["metric"] == "euclidean"
    assert kwargs["random_state"] == 7
    assert kwargs["core_dist_n_jobs"] == 2
    assert kwargs["approx_min_span_tree"] is True

    used = (
        result["structured_results"]["hdbscan_summary"]["parameters"]  # type: ignore[index]
    )
    assert used["hdbscan_metric"] == "euclidean"
    assert used["hdbscan_random_state"] == 7
    assert used["hdbscan_core_dist_n_jobs"] == 2
    assert used["hdbscan_approx_min_span_tree"] is True


def test_hdbscan_runner_rejects_zero_core_dist_jobs(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "hdbscan",
        types.SimpleNamespace(HDBSCAN=object),
    )

    with pytest.raises(ValueError, match="core_dist_n_jobs"):
        run_hdbscan_analysis(
            method_name="phase1_hdbscan_fixture",
            input_group_id=125,
            input_group_type="embedding_batch",
            input_group_metadata={"representation": "full"},
            embeddings=np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float64),
            texts=["a", "b", "c"],
            parameters={
                "hdbscan_min_cluster_size": 2,
                "hdbscan_core_dist_n_jobs": 0,
            },
        )


def test_analyze_kmeans_normalize_pca_sweep_writes_clustering_summary_no_v1_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Slice 1.5: the bundled-grammar kmeans method emits the runner's own
    summary/labels/selection-evidence rows; no v1-envelope provenance fields
    leak into config_json or the AnalysisResult value."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_kmeans_normalize_pca_sweep")

    result = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="kmeans+normalize+pca+sweep",
        run_key="rk_kmeans_normalize_pca_sweep",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters={
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
        },
    )
    assert "kmeans_summary.json" in result.artifact_uris
    assert "kmeans_labels.json" in result.artifact_uris

    expected_recipe_hash = canonical_recipe_hash(
        dict(COMPOSITE_RECIPES["kmeans+normalize+pca+sweep"])
    )
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        cfg = dict(run_row.config_json or {})
        _assert_no_v1_envelope_fields(cfg)
        assert cfg.get("recipe_hash") == expected_recipe_hash

        summary_result = (
            session.query(AnalysisResult)
            .filter(
                AnalysisResult.analysis_group_id == int(result.group_id),
                AnalysisResult.result_key == "clustering_summary",
            )
            .first()
        )
        assert summary_result is not None
        value = dict(summary_result.result_json or {}).get("value") or {}
        assert int(value["n_samples"]) == 4
        assert int(value["cluster_count"]) >= 1
        assert int(value["parameters"]["chosen_k"]) in value["parameters"]["k_range"]
        # No v1 envelope leakage in the runner's summary either.
        for legacy_field in (
            "operation_type",
            "operation_version",
            "rule_set_hash",
            "pipeline_resolved",
            "pipeline_effective_hash",
        ):
            assert legacy_field not in value


def test_analyze_gmm_normalize_pca_sweep_writes_clustering_summary_no_v1_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Slice 1.5: the bundled-grammar gmm method emits the runner's own
    summary/labels/selection-evidence rows; no v1-envelope provenance fields
    leak into config_json or the AnalysisResult value."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_gmm_normalize_pca_sweep")

    result = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="gmm+normalize+pca+sweep",
        run_key="rk_gmm_normalize_pca_sweep",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters={
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
        },
    )
    assert "gmm_summary.json" in result.artifact_uris
    assert "gmm_labels.json" in result.artifact_uris

    expected_recipe_hash = canonical_recipe_hash(
        dict(COMPOSITE_RECIPES["gmm+normalize+pca+sweep"])
    )
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        cfg = dict(run_row.config_json or {})
        _assert_no_v1_envelope_fields(cfg)
        assert cfg.get("recipe_hash") == expected_recipe_hash


@pytest.mark.parametrize(
    ("method_name", "explicit_runner", "parameters"),
    [
        (
            "hdbscan+fixed",
            run_hdbscan_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "hdbscan_min_cluster_size": 2,
                "hdbscan_min_samples": 1,
            },
        ),
        (
            "kmeans+normalize+pca+sweep",
            run_kmeans_silhouette_kneedle_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
            },
        ),
        (
            "gmm+normalize+pca+sweep",
            run_gmm_bic_argmin_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
            },
        ),
        (
            "kmeans+normalize+fixed-k",
            run_kmeans_fixed_k_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "k": 2,
            },
        ),
        (
            "gmm+pca+fixed-k",
            run_gmm_fixed_k_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "k": 2,
                "pca_n_components": 3,
            },
        ),
        (
            "dbscan+normalize+pca+fixed-eps",
            run_dbscan_fixed_eps_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "eps": 10.0,
                "min_samples": 2,
                "pca_n_components": 3,
            },
        ),
        (
            "spherical-kmeans+approx+fixed-k",
            run_kmeans_fixed_k_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "k": 2,
            },
        ),
        (
            "agglomerative+normalize+fixed-k",
            run_agglomerative_preproc_fixed_k_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "k": 2,
            },
        ),
        (
            "hdbscan+pca+fixed",
            run_hdbscan_preproc_fixed_analysis,
            {
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
                "min_cluster_size": 2,
                "pca_n_components": 3,
            },
        ),
    ],
)
def test_analyze_registry_dispatch_matches_explicit_runner_fingerprint(
    method_name: str,
    explicit_runner,
    parameters: dict[str, object],
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )

    if method_name.startswith("hdbscan"):
        class _FakeHDBSCAN:
            def __init__(self, **kwargs):
                self._kwargs = dict(kwargs)

            def fit_predict(self, matrix):
                rows = int(np.asarray(matrix).shape[0])
                self.probabilities_ = np.ones(rows, dtype=np.float64)
                self.outlier_scores_ = np.zeros(rows, dtype=np.float64)
                return np.asarray([0, 0, 1, 1], dtype=np.int64)

        monkeypatch.setitem(
            sys.modules,
            "hdbscan",
            types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN),
        )

    request_group_id = _create_request_group(db, f"request_registry_parity_{method_name}")
    via_registry = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name=method_name,
        run_key=f"rk_registry_{method_name}",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters=parameters,
        method_runner=None,
    )
    via_explicit = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name=method_name,
        run_key=f"rk_explicit_{method_name}",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters=parameters,
        method_runner=explicit_runner,
    )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_registry = repo.get_provenanced_run_by_id(int(via_registry.run_id or 0))
        run_explicit = repo.get_provenanced_run_by_id(int(via_explicit.run_id or 0))
        assert run_registry is not None
        assert run_explicit is not None
        assert run_registry.fingerprint_hash == run_explicit.fingerprint_hash


def test_analyze_cosine_kllmeans_falls_back_to_default_runner_and_records_recipe_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_cosine_kllmeans_default")
    result = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="cosine_kllmeans_no_pca",
        run_key="rk_cosine_kllmeans_default",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters={
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
        },
    )

    assert "analysis_summary.json" in result.artifact_uris
    assert "clustering_summary.json" not in result.artifact_uris

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        cfg = dict(run_row.config_json or {})
        expected_recipe_hash = canonical_recipe_hash(
            dict(COMPOSITE_RECIPES["cosine_kllmeans_no_pca"])
        )
        assert cfg.get("recipe_hash") == expected_recipe_hash
        assert "operation_type" not in cfg
        summary_result = (
            session.query(AnalysisResult)
            .filter(
                AnalysisResult.analysis_group_id == int(result.group_id),
                AnalysisResult.result_key == "summary",
            )
            .first()
        )
        assert summary_result is not None


def test_analyze_agglomerative_fixed_k_runs_with_envelope_none(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Slice 1.5: ``agglomerative+fixed-k`` is one of four bundled-grammar
    methods that all share ``provenance_envelope=none`` after the v1 envelope
    retirement; the prior 'outside v1' framing no longer applies."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(db, "request_agglomerative_fixed_k")
    result = analyze(
        snapshot_group_id,
        embedding_group_id,
        method_name="agglomerative+fixed-k",
        run_key="rk_agglomerative_fixed_k",
        request_group_id=request_group_id,
        db=db,
        artifact_dir=artifact_dir,
        parameters={
            "k": 2,
            "dataset_slug": "analyze_fixture",
            "representation_type": "full",
            "embedding_provider": "test-provider",
            "embedding_deployment": "test-embedding-model",
        },
        method_runner=None,
    )

    assert "agglomerative_summary.json" in result.artifact_uris
    assert "agglomerative_labels.json" in result.artifact_uris
    assert "clustering_summary.json" not in result.artifact_uris

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        run_row = repo.get_provenanced_run_by_id(int(result.run_id or 0))
        assert run_row is not None
        assert run_row.determinism_class == "deterministic"
        cfg = dict(run_row.config_json or {})
        assert "operation_type" not in cfg
        assert "rule_set_hash" not in cfg
        assert cfg.get("recipe_hash") is None

        summary_result = (
            session.query(AnalysisResult)
            .filter(
                AnalysisResult.analysis_group_id == int(result.group_id),
                AnalysisResult.result_key == "clustering_summary",
            )
            .first()
        )
        assert summary_result is not None
        summary_value = dict(summary_result.result_json or {}).get("value") or {}
        assert summary_value.get("base_algorithm") == "agglomerative"


@pytest.mark.parametrize(
    "deprecated_name",
    ["hdbscan", "kmeans+silhouette+kneedle", "gmm+bic+argmin"],
)
def test_deprecated_method_name_rejected_even_with_explicit_method_runner(
    deprecated_name: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Slice 1.5: the deprecation guard fires before runner resolution so an
    explicit ``method_runner`` injection cannot bypass it. The stub runner
    here would raise ``AssertionError`` if the guard let it through; the
    test passes only if the guard's ``ValueError`` is raised first.
    """
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db, _database_url = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    _df_group_id, snapshot_group_id, embedding_group_id = _prepare_inputs(
        db=db,
        artifact_dir=artifact_dir,
    )
    request_group_id = _create_request_group(
        db, f"request_deprecated_guard_{deprecated_name}"
    )

    def _stub_runner(**_kwargs):
        raise AssertionError(
            "runner must not be called when method_name is deprecated"
        )

    with pytest.raises(ValueError, match="deprecated in Slice 1.5"):
        analyze(
            snapshot_group_id,
            embedding_group_id,
            method_name=deprecated_name,
            run_key=f"rk_deprecated_{deprecated_name}",
            request_group_id=request_group_id,
            db=db,
            artifact_dir=artifact_dir,
            parameters={
                "dataset_slug": "analyze_fixture",
                "representation_type": "full",
                "embedding_provider": "test-provider",
                "embedding_deployment": "test-embedding-model",
            },
            method_runner=_stub_runner,
        )

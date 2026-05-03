"""Tests for analysis_run branching in sweep worker runtime."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments import sweep_worker_main as worker_main
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_run_one_analysis_run_job_clustering_calls_analyze(monkeypatch) -> None:
    db = _db()
    run_key = "dbpedia_engine_a_None_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="cluster_analysis_success",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
        )
        run_id = repo.create_group(
            group_type="clustering_run",
            name="run_with_lineage",
            metadata_json={
                "run_key": run_key,
                "dataset_snapshot_ids": [101],
                "embedding_batch_group_id": 202,
            },
        )
        assert svc.record_delivery(request_id, int(run_id), run_key) is True

    captured: dict = {}

    def _fake_analyze(**kwargs):  # noqa: ANN003
        captured.update(dict(kwargs))
        return None

    monkeypatch.setattr(worker_main, "run_pipeline_analyze", _fake_analyze)

    job_snapshot = {
        "id": 41,
        "request_group_id": int(request_id),
        "job_type": "analysis_run",
        "job_key": f"req{int(request_id)}__{run_key}__analysis__bundle_eval",
        "base_run_key": run_key,
        "payload_json": {
            "request_id": int(request_id),
            "sweep_type": "clustering",
            "analysis_key": "bundle_eval",
            "run_key": run_key,
            "method_name": "kmeans+normalize+pca+sweep",
            "method_version": "1.0",
            "parameters": {"top_n": 5},
            "force": False,
        },
    }

    job_id, result_ref, error = worker_main.run_one_analysis_run_job(
        job_snapshot=job_snapshot,
        db=db,
        worker_label="test-worker",
    )
    assert job_id == 41
    assert error is None
    assert result_ref == f"analysis:bundle_eval:{run_key}"
    assert int(captured.get("snapshot_group_id") or -1) == 101
    assert int(captured.get("embedding_batch_group_id") or -1) == 202
    assert int(captured.get("request_group_id") or -1) == int(request_id)
    assert str(captured.get("run_key") or "") == run_key
    assert dict(captured.get("parameters") or {}) == {"top_n": 5}
    assert bool(captured.get("force")) is False


def test_run_one_analysis_run_job_clustering_missing_embedding_fails() -> None:
    db = _db()
    run_key = "dbpedia_engine_a_None_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="cluster_analysis_missing_embedding",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
        )
        run_id = repo.create_group(
            group_type="clustering_run",
            name="run_missing_embedding",
            metadata_json={
                "run_key": run_key,
                "dataset_snapshot_ids": [101],
            },
        )
        assert svc.record_delivery(request_id, int(run_id), run_key) is True

    job_snapshot = {
        "id": 42,
        "request_group_id": int(request_id),
        "job_type": "analysis_run",
        "job_key": f"req{int(request_id)}__{run_key}__analysis__bundle_eval",
        "base_run_key": run_key,
        "payload_json": {
            "request_id": int(request_id),
            "sweep_type": "clustering",
            "analysis_key": "bundle_eval",
            "run_key": run_key,
            "method_name": "kmeans+normalize+pca+sweep",
            "method_version": "1.0",
        },
    }
    job_id, result_ref, error = worker_main.run_one_analysis_run_job(
        job_snapshot=job_snapshot,
        db=db,
        worker_label="test-worker",
    )
    assert job_id == 42
    assert result_ref is None
    assert str(error or "").startswith("missing_embedding_batch_group_id:")

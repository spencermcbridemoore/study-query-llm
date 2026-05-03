"""Tests for reducer service lineage propagation behavior."""

from __future__ import annotations

import json
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.jobs.job_reducer_service import JobReducerService
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_finalize_run_job_propagates_lineage_inputs_from_request_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = _db()
    run_key = "dbpedia_engine_a_None_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="lineage_propagation_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 2]], "tries_per_k": 1},
            run_key_to_lineage_inputs={
                run_key: {
                    "dataset_snapshot_ids": [9, 7, 9],
                    "embedding_batch_group_id": 18,
                }
            },
        )
        reduce_payload = {
            "pca": {},
            "by_k": {
                "2": {
                    "labels": [0, 1],
                    "labels_all": [[0, 1]],
                    "objectives": [0.5],
                    "objective": 0.5,
                    "representatives": ["a", "b"],
                }
            },
        }
        reduce_ref = tmp_path / "reduce_payload.json"
        reduce_ref.write_text(json.dumps(reduce_payload), encoding="utf-8")
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="reduce_k",
            job_key=f"{run_key}__reduce_k2_2",
            payload_json={"run_key": run_key},
        )
        repo.complete_orchestration_job(int(reduce_id), result_ref=str(reduce_ref))
        finalize_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="finalize_run",
            job_key=f"{run_key}__finalize_run",
            payload_json={
                "run_key": run_key,
                "dataset": "dbpedia",
                "embedding_engine": "engine/a",
                "summarizer": "None",
                "tries_per_k": 1,
            },
            depends_on_job_ids=[int(reduce_id)],
        )

    captured: dict = {}

    def _fake_ingest(result, metadata, ground_truth_labels, db, run_key):  # noqa: ANN001
        captured["metadata"] = dict(metadata or {})
        return 999

    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.ingest_result_to_db",
        _fake_ingest,
    )

    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    run_id = reducer.finalize_run_job(int(finalize_id))
    assert run_id == 999
    metadata = dict(captured.get("metadata") or {})
    assert metadata.get("dataset_snapshot_ids") == [7, 9]
    assert int(metadata.get("embedding_batch_group_id") or -1) == 18
    assert int(metadata.get("request_group_id") or -1) > 0

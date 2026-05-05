"""Deterministic stress checks for request-level finalization under worker fanout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments import sweep_worker_main as worker_main
from study_query_llm.experiments.sweep_request_types import build_run_key
from study_query_llm.services.jobs.job_reducer_service import JobReducerService
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _write_leaf_result(path: Path, k: int) -> str:
    payload = {
        "pca": {},
        "by_k": {
            str(int(k)): {
                "labels": [0, 1, 0, 1],
                "objective": 0.5,
                "representatives": ["a", "b"],
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _drain_request_jobs(
    *,
    db: DatabaseConnectionV2,
    request_id: int,
    worker_count: int,
    tmp_path: Path,
) -> None:
    reducer = JobReducerService(db, artifacts_dir=tmp_path / "artifacts")
    job_types = ["run_k_try", "reduce_k", "finalize_run", "finalize_request", "analysis_run"]

    for _ in range(500):
        claimed_any = False
        for worker_idx in range(int(worker_count)):
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                job = repo.claim_next_orchestration_job(
                    worker_id=f"stress-worker-{worker_idx}",
                    lease_seconds=60,
                    request_group_id=int(request_id),
                    job_types=job_types,
                )
                if job is None:
                    continue
                claimed_any = True
                snapshot = {
                    "id": int(job.id),
                    "request_group_id": int(job.request_group_id),
                    "job_type": str(job.job_type),
                    "job_key": str(job.job_key),
                    "base_run_key": str(job.base_run_key or ""),
                    "payload_json": dict(job.payload_json or {}),
                }

            job_type = str(snapshot["job_type"])
            if job_type == "run_k_try":
                payload = dict(snapshot["payload_json"] or {})
                k_min = int(payload.get("k_min") or 2)
                ref = _write_leaf_result(
                    tmp_path / f"leaf_job_{int(snapshot['id'])}.json",
                    k=k_min,
                )
                with db.session_scope() as session:
                    RawCallRepository(session).complete_orchestration_job(
                        int(snapshot["id"]),
                        result_ref=ref,
                    )
            elif job_type == "reduce_k":
                reducer.reduce_k_job(int(snapshot["id"]))
            elif job_type == "finalize_run":
                run_id = reducer.finalize_run_job(int(snapshot["id"]))
                assert run_id is not None
            elif job_type == "finalize_request":
                sweep_id = reducer.finalize_request_job(int(snapshot["id"]))
                assert sweep_id is not None
            elif job_type == "analysis_run":
                job_id, result_ref, error = worker_main.run_one_analysis_run_job(
                    job_snapshot=snapshot,
                    db=db,
                    worker_label=f"stress-worker-{worker_idx}",
                )
                assert int(job_id) == int(snapshot["id"])
                assert error is None
                with db.session_scope() as session:
                    RawCallRepository(session).complete_orchestration_job(
                        int(snapshot["id"]),
                        result_ref=str(result_ref or ""),
                    )
            else:
                raise AssertionError(f"unexpected_job_type:{job_type}")

        with db.session_scope() as session:
            repo = RawCallRepository(session)
            jobs = repo.list_orchestration_jobs(request_group_id=int(request_id))
            active = [j for j in jobs if str(j.status) in {"pending", "ready", "claimed"}]
        if not active:
            return
        if not claimed_any:
            raise AssertionError("orchestration_drain_stalled_with_active_jobs")

    raise AssertionError("orchestration_drain_iteration_limit_exceeded")


@pytest.mark.parametrize("worker_count", [1, 4, 16])
def test_request_finalization_stress_deterministic_lane(
    monkeypatch,
    tmp_path: Path,
    worker_count: int,
) -> None:
    db = _db()
    monkeypatch.setattr(worker_main, "run_pipeline_analyze", lambda **kwargs: None)

    datasets = [f"dataset_{idx}" for idx in range(1, 7)]
    embedding_engine = "engine/a"
    entry_max = 50
    run_key_to_lineage_inputs = {
        build_run_key(dataset=name, embedding_engine=embedding_engine, entry_max=entry_max): {
            "dataset_snapshot_ids": [1000 + idx],
            "embedding_batch_group_id": 2000 + idx,
        }
        for idx, name in enumerate(datasets, start=1)
    }

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        svc.enable_analysis_jobs = True
        svc.enable_clustering_analysis_jobs = True
        svc.use_request_finalizer_job = True
        request_id = svc.create_request(
            request_name=f"stress_request_workers_{worker_count}",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": list(datasets),
                "embedding_engines": [embedding_engine],
            },
            entry_max=entry_max,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 2]], "tries_per_k": 1},
            clustering_analysis_selection=[
                {
                    "method_name": "kmeans+fixed-k",
                    "parameters": {"k": 2},
                }
            ],
            run_key_to_lineage_inputs=run_key_to_lineage_inputs,
        )

    _drain_request_jobs(
        db=db,
        request_id=int(request_id),
        worker_count=int(worker_count),
        tmp_path=tmp_path / f"stress_w{worker_count}",
    )

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request = svc.get_request(int(request_id))
        assert request is not None
        assert str(request.get("request_status") or "") == "fulfilled"
        assert int(request.get("linked_sweep_id") or 0) > 0

        progress = svc.compute_progress(int(request_id))
        assert int(progress["expected_count"]) == len(datasets)
        assert int(progress["completed_count"]) == len(datasets)
        assert int(progress["missing_count"]) == 0

        jobs = repo.list_orchestration_jobs(request_group_id=int(request_id))
        failed = [j for j in jobs if str(j.status) == "failed"]
        assert failed == []
        assert all(str(j.status) == "completed" for j in jobs)

        analysis_failures = [
            dict(j.error_json or {})
            for j in jobs
            if j.job_type == "analysis_run" and dict(j.error_json or {})
        ]
        assert analysis_failures == []

        contains_links = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == int(request_id),
                GroupLink.link_type == "contains",
            )
            .all()
        )
        assert len(contains_links) == len(datasets)

"""Tests for reducer service lineage propagation behavior."""

from __future__ import annotations

import json
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
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
    run_key = "dbpedia_engine_a_50_50runs"
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
    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.SweepRequestService.record_delivery",
        lambda *args, **kwargs: True,
    )

    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    run_id = reducer.finalize_run_job(int(finalize_id))
    assert run_id == 999
    metadata = dict(captured.get("metadata") or {})
    assert metadata.get("dataset_snapshot_ids") == [7, 9]
    assert int(metadata.get("embedding_batch_group_id") or -1) == 18
    assert int(metadata.get("request_group_id") or -1) > 0


def test_finalize_run_job_repairs_delivery_link_when_ingest_returns_existing_run_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = _db()
    run_key = "dbpedia_engine_a_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="lineage_repair_existing_run_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
            },
            entry_max=50,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 2]], "tries_per_k": 1},
        )
        existing_run_id = repo.create_group(
            group_type="clustering_run",
            name="existing_run",
            metadata_json={
                "run_key": run_key,
                "dataset_snapshot_ids": [7, 9],
                "embedding_batch_group_id": 18,
            },
        )
        reduce_ref = tmp_path / "reduce_payload_existing_run.json"
        reduce_ref.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="reduce_k",
            job_key=f"{run_key}__reduce_k2_2__existing",
            payload_json={"run_key": run_key},
        )
        repo.complete_orchestration_job(int(reduce_id), result_ref=str(reduce_ref))
        finalize_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="finalize_run",
            job_key=f"{run_key}__finalize_run__existing",
            payload_json={
                "run_key": run_key,
                "dataset": "dbpedia",
                "embedding_engine": "engine/a",
                "tries_per_k": 1,
            },
            depends_on_job_ids=[int(reduce_id)],
        )

    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.ingest_result_to_db",
        lambda *args, **kwargs: None,
    )
    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    repaired_run_id = reducer.finalize_run_job(int(finalize_id))

    assert repaired_run_id == int(existing_run_id)
    with db.session_scope() as session:
        links = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == int(request_id),
                GroupLink.child_group_id == int(existing_run_id),
                GroupLink.link_type == "contains",
            )
            .all()
        )
        assert len(links) == 1


def test_finalize_run_job_retry_repairs_link_after_partial_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = _db()
    run_key = "dbpedia_engine_a_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="lineage_repair_retry_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
            },
            entry_max=50,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 2]], "tries_per_k": 1},
        )
        reduce_ref = tmp_path / "reduce_payload_retry.json"
        reduce_ref.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="reduce_k",
            job_key=f"{run_key}__reduce_k2_2__retry",
            payload_json={"run_key": run_key},
        )
        repo.complete_orchestration_job(int(reduce_id), result_ref=str(reduce_ref))
        finalize_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="finalize_run",
            job_key=f"{run_key}__finalize_run__retry",
            payload_json={
                "run_key": run_key,
                "dataset": "dbpedia",
                "embedding_engine": "engine/a",
                "tries_per_k": 1,
            },
            depends_on_job_ids=[int(reduce_id)],
        )

    state: dict[str, int | None] = {"ingest_calls": 0, "run_id": None, "delivery_calls": 0}
    real_record_delivery = SweepRequestService.record_delivery

    def _fake_ingest(result, metadata, ground_truth_labels, db, run_key):  # noqa: ANN001
        state["ingest_calls"] = int(state["ingest_calls"] or 0) + 1
        if int(state["ingest_calls"] or 0) == 1:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                run_id = repo.create_group(
                    group_type="clustering_run",
                    name="retry_recovery_run",
                    metadata_json={
                        "run_key": run_key,
                        "dataset_snapshot_ids": [7, 9],
                        "embedding_batch_group_id": 18,
                    },
                )
            state["run_id"] = int(run_id)
            return int(run_id)
        return None

    def _flaky_record_delivery(self, request_id, run_id, run_key):  # noqa: ANN001
        state["delivery_calls"] = int(state["delivery_calls"] or 0) + 1
        if int(state["delivery_calls"] or 0) == 1:
            raise RuntimeError("simulated_delivery_failure")
        return real_record_delivery(self, request_id, run_id, run_key)

    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.ingest_result_to_db",
        _fake_ingest,
    )
    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.SweepRequestService.record_delivery",
        _flaky_record_delivery,
    )

    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    try:
        reducer.finalize_run_job(int(finalize_id))
    except RuntimeError as exc:
        assert "simulated_delivery_failure" in str(exc)

    repaired_run_id = reducer.finalize_run_job(int(finalize_id))
    assert repaired_run_id == int(state["run_id"] or -1)

    with db.session_scope() as session:
        links = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == int(request_id),
                GroupLink.child_group_id == int(state["run_id"] or -1),
                GroupLink.link_type == "contains",
            )
            .all()
        )
        assert len(links) == 1


def test_finalize_run_job_duplicate_finalize_paths_keep_single_delivery_link(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = _db()
    run_key = "dbpedia_engine_a_50_50runs"
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="lineage_repair_duplicate_finalize_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
            },
            entry_max=50,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 2]], "tries_per_k": 1},
        )
        existing_run_id = repo.create_group(
            group_type="clustering_run",
            name="existing_run_duplicate_finalize",
            metadata_json={
                "run_key": run_key,
                "dataset_snapshot_ids": [7, 9],
                "embedding_batch_group_id": 18,
            },
        )
        reduce_ref = tmp_path / "reduce_payload_duplicate_finalize.json"
        reduce_ref.write_text(
            json.dumps(
                {
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
            ),
            encoding="utf-8",
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="reduce_k",
            job_key=f"{run_key}__reduce_k2_2__dup_finalize",
            payload_json={"run_key": run_key},
        )
        repo.complete_orchestration_job(int(reduce_id), result_ref=str(reduce_ref))
        finalize_a = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="finalize_run",
            job_key=f"{run_key}__finalize_run__dup_a",
            payload_json={
                "run_key": run_key,
                "dataset": "dbpedia",
                "embedding_engine": "engine/a",
                "tries_per_k": 1,
            },
            depends_on_job_ids=[int(reduce_id)],
        )
        finalize_b = repo.enqueue_orchestration_job(
            request_group_id=request_id,
            job_type="finalize_run",
            job_key=f"{run_key}__finalize_run__dup_b",
            payload_json={
                "run_key": run_key,
                "dataset": "dbpedia",
                "embedding_engine": "engine/a",
                "tries_per_k": 1,
            },
            depends_on_job_ids=[int(reduce_id)],
        )

    monkeypatch.setattr(
        "study_query_llm.services.jobs.job_reducer_service.ingest_result_to_db",
        lambda *args, **kwargs: None,
    )
    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    run_id_a = reducer.finalize_run_job(int(finalize_a))
    run_id_b = reducer.finalize_run_job(int(finalize_b))

    assert run_id_a == int(existing_run_id)
    assert run_id_b == int(existing_run_id)
    with db.session_scope() as session:
        links = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == int(request_id),
                GroupLink.child_group_id == int(existing_run_id),
                GroupLink.link_type == "contains",
            )
            .all()
        )
        assert len(links) == 1


def test_reduce_k_job_collects_sorted_tries_from_leaf_shards(tmp_path: Path) -> None:
    db = _db()
    objectives = [0.8, 0.3, 0.5]
    try_indices = [2, 0, 1]
    shard_refs: list[str] = []
    for i, (obj, tidx) in enumerate(zip(objectives, try_indices)):
        path = tmp_path / f"leaf_{i}.json"
        path.write_text(
            json.dumps(
                {
                    "pca": {},
                    "by_k": {
                        "2": {
                            "labels": [0, 1],
                            "objective": obj,
                            "representatives": ["a", "b"],
                        }
                    },
                    "try_idx": tidx,
                    "seed_value": 100 + tidx,
                    "profiling": {"run_sweep_seconds": float(tidx)},
                }
            ),
            encoding="utf-8",
        )
        shard_refs.append(str(path))

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="reduce_tries_req",
            metadata_json={},
        )
        leaf_ids: list[int] = []
        for ref in shard_refs:
            lid = int(
                repo.enqueue_orchestration_job(
                    request_group_id=int(req_id),
                    job_type="run_k_try",
                    job_key=f"rk_try_{Path(ref).stem}",
                    payload_json={},
                )
            )
            repo.complete_orchestration_job(lid, result_ref=ref)
            leaf_ids.append(lid)
        rid = int(
            repo.enqueue_orchestration_job(
                request_group_id=int(req_id),
                job_type="reduce_k",
                job_key="rk_reduce_tries",
                depends_on_job_ids=leaf_ids,
            )
        )

    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    out_ref = reducer.reduce_k_job(rid)
    payload = json.loads(Path(out_ref).read_text(encoding="utf-8"))
    tries = payload["by_k"]["2"]["tries"]
    assert len(tries) == 3
    assert [t["try_idx"] for t in tries] == [0, 1, 2]
    assert payload["by_k"]["2"]["objective"] == 0.3
    assert len(payload["by_k"]["2"]["objectives"]) == 3


def test_reduce_k_job_rejects_multi_k_leaf_shard(tmp_path: Path) -> None:
    db = _db()
    path = tmp_path / "leaf_multi.json"
    path.write_text(
        json.dumps(
            {
                "by_k": {
                    "2": {"labels": [0, 1], "objective": 0.5},
                    "3": {"labels": [0, 1, 0], "objective": 0.4},
                },
                "try_idx": 0,
            }
        ),
        encoding="utf-8",
    )
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="bad_shard_req",
            metadata_json={},
        )
        lid = int(
            repo.enqueue_orchestration_job(
                request_group_id=int(req_id),
                job_type="run_k_try",
                job_key="rk_bad",
                payload_json={},
            )
        )
        repo.complete_orchestration_job(lid, result_ref=str(path))
        rid = int(
            repo.enqueue_orchestration_job(
                request_group_id=int(req_id),
                job_type="reduce_k",
                job_key="rk_reduce_bad",
                depends_on_job_ids=[lid],
            )
        )

    reducer = JobReducerService(db, artifacts_dir=tmp_path)
    try:
        reducer.reduce_k_job(rid)
    except RuntimeError as exc:
        assert "exactly one k bucket" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

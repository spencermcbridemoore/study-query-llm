"""Tests for generic orchestration job table and sharded request planning."""

from __future__ import annotations

import json
from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.jobs import build_p0_baseline_snapshot
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _baseline_fixture_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "p0_baseline"
        / "baseline_snapshot.json"
    )


def test_p0_baseline_fixture_matches_generated_snapshot():
    fixture_path = _baseline_fixture_path()
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))
    actual = build_p0_baseline_snapshot()
    assert actual == expected


def test_enqueue_claim_dependency_and_promote():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="req",
            metadata_json={},
        )

        leaf_id = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__k2_3__try0",
            payload_json={"embedding_engine": "engine/a"},
            seed_value=123,
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="reduce_k",
            job_key="rk__reduce_k2_3",
            depends_on_job_ids=[leaf_id],
        )
        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        status_by_id = {j.id: j.status for j in jobs}
        assert status_by_id[leaf_id] == "ready"
        assert status_by_id[reduce_id] == "pending"

        claimed = repo.claim_next_orchestration_job(
            worker_id="w1",
            lease_seconds=60,
            request_group_id=req_id,
            job_types=["run_k_try"],
        )
        assert claimed is not None
        assert claimed.id == leaf_id
        repo.complete_orchestration_job(leaf_id, result_ref="leaf.json")
        repo.promote_ready_orchestration_jobs(request_group_id=req_id)

        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        status_by_id = {j.id: j.status for j in jobs}
        assert status_by_id[reduce_id] == "ready"


def test_sharded_request_planner_creates_leaf_reducer_finalize_jobs():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        req_id = svc.create_request(
            request_name="sharded_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=300,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 3], [4, 5]], "tries_per_k": 3},
        )

        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        types = [j.job_type for j in jobs]
        # leaf: 2 k_ranges * 3 tries = 6
        assert types.count("run_k_try") == 6
        # reducers: 2 reduce_k + 1 finalize
        assert types.count("reduce_k") == 2
        assert types.count("finalize_run") == 1

        # Seed values should be persisted for leaf jobs.
        leaf_jobs = [j for j in jobs if j.job_type == "run_k_try"]
        assert all(j.seed_value is not None for j in leaf_jobs)


def test_claim_orchestration_job_batch_respects_limit():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="req",
            metadata_json={},
        )
        for i in range(5):
            repo.enqueue_orchestration_job(
                request_group_id=req_id,
                job_type="run_k_try",
                job_key=f"rk__k2_3__try{i}",
                payload_json={"embedding_engine": "e1"},
                seed_value=100 + i,
            )
        snapshots = repo.claim_orchestration_job_batch(
            request_group_id=req_id,
            job_types=["run_k_try"],
            claim_owner="batch-claim",
            lease_seconds=60,
            limit=2,
        )
        assert len(snapshots) == 2
        claimed_ids = {s["id"] for s in snapshots}
        assert len(claimed_ids) == 2
        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        status_by_id = {j.id: j.status for j in jobs}
        claimed_count = sum(1 for s in status_by_id.values() if s == "claimed")
        assert claimed_count == 2
        ready_count = sum(1 for s in status_by_id.values() if s == "ready")
        assert ready_count == 3


def test_claim_orchestration_job_batch_respects_filter_payload():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="req",
            metadata_json={},
        )
        repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__e1__try0",
            payload_json={"embedding_engine": "e1"},
            seed_value=1,
        )
        repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__e2__try0",
            payload_json={"embedding_engine": "e2"},
            seed_value=2,
        )
        snapshots = repo.claim_orchestration_job_batch(
            request_group_id=req_id,
            job_types=["run_k_try"],
            claim_owner="batch-claim",
            lease_seconds=60,
            limit=10,
            filter_payload={"embedding_engine": "e1"},
        )
        assert len(snapshots) == 1
        assert snapshots[0]["payload_json"]["embedding_engine"] == "e1"
        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        status_by_payload = {
            (j.payload_json or {}).get("embedding_engine"): j.status for j in jobs
        }
        assert status_by_payload.get("e1") == "claimed"
        assert status_by_payload.get("e2") == "ready"


def test_claim_orchestration_job_batch_respects_dependency():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="req",
            metadata_json={},
        )
        leaf_id = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__k2_3__try0",
            payload_json={"embedding_engine": "e1"},
            seed_value=123,
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="reduce_k",
            job_key="rk__reduce_k2_3",
            depends_on_job_ids=[leaf_id],
        )
        snapshots = repo.claim_orchestration_job_batch(
            request_group_id=req_id,
            job_types=["run_k_try", "reduce_k"],
            claim_owner="batch-claim",
            lease_seconds=60,
            limit=10,
        )
        types_claimed = [s["job_type"] for s in snapshots]
        assert "reduce_k" not in types_claimed
        assert "run_k_try" in types_claimed
        assert len(snapshots) == 1


def test_complete_orchestration_jobs_batch_promotes_once():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req_id = repo.create_group(
            group_type="clustering_sweep_request",
            name="req",
            metadata_json={},
        )
        leaf1 = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__k2_3__try0",
            payload_json={"embedding_engine": "e1"},
            seed_value=1,
        )
        leaf2 = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="run_k_try",
            job_key="rk__k2_3__try1",
            payload_json={"embedding_engine": "e1"},
            seed_value=2,
        )
        reduce_id = repo.enqueue_orchestration_job(
            request_group_id=req_id,
            job_type="reduce_k",
            job_key="rk__reduce_k2_3",
            depends_on_job_ids=[leaf1, leaf2],
        )
        snapshots = repo.claim_orchestration_job_batch(
            request_group_id=req_id,
            job_types=["run_k_try"],
            claim_owner="batch-claim",
            lease_seconds=60,
            limit=10,
        )
        assert len(snapshots) == 2
        batch = [(s["id"], "ref_" + str(s["id"])) for s in snapshots]
        repo.complete_orchestration_jobs_batch(batch)
        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        status_by_id = {j.id: j.status for j in jobs}
        assert status_by_id[reduce_id] == "ready"


def test_complete_orchestration_jobs_batch_multiple_request_groups():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        req1 = repo.create_group(
            group_type="clustering_sweep_request",
            name="req1",
            metadata_json={},
        )
        req2 = repo.create_group(
            group_type="clustering_sweep_request",
            name="req2",
            metadata_json={},
        )
        leaf1 = repo.enqueue_orchestration_job(
            request_group_id=req1,
            job_type="run_k_try",
            job_key="rk__req1__try0",
            payload_json={},
            seed_value=1,
        )
        leaf2 = repo.enqueue_orchestration_job(
            request_group_id=req2,
            job_type="run_k_try",
            job_key="rk__req2__try0",
            payload_json={},
            seed_value=2,
        )
        snapshots1 = repo.claim_orchestration_job_batch(
            request_group_id=req1,
            job_types=["run_k_try"],
            claim_owner="batch",
            lease_seconds=60,
            limit=10,
        )
        snapshots2 = repo.claim_orchestration_job_batch(
            request_group_id=req2,
            job_types=["run_k_try"],
            claim_owner="batch",
            lease_seconds=60,
            limit=10,
        )
        batch = [
            (snapshots1[0]["id"], "r1"),
            (snapshots2[0]["id"], "r2"),
        ]
        repo.complete_orchestration_jobs_batch(batch)
        jobs1 = repo.list_orchestration_jobs(request_group_id=req1)
        jobs2 = repo.list_orchestration_jobs(request_group_id=req2)
        assert any(j.status == "completed" for j in jobs1)
        assert any(j.status == "completed" for j in jobs2)


def test_mcq_request_planner_creates_single_mcq_job_per_run_key():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_jobs",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 12, "template_version": "v1"},
            parameter_axes={
                "levels": ["high school"],
                "subjects": ["physics"],
                "deployments": ["gpt-4o-mini"],
                "options_per_question": [4, 5],
                "questions_per_test": [20],
                "label_styles": ["upper"],
                "spread_correct_answer_uniformly": [False],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        req = svc.get_request(req_id)
        assert req is not None
        expected = int(req["expected_count"])
        jobs = repo.list_orchestration_jobs(request_group_id=req_id)
        mcq_jobs = [j for j in jobs if j.job_type == "mcq_run"]
        analysis_jobs = [j for j in jobs if j.job_type == "analysis_run"]
        assert len(mcq_jobs) == expected
        assert all(j.status == "ready" for j in mcq_jobs)
        assert len(analysis_jobs) >= 1
        assert all(j.status == "pending" for j in analysis_jobs)


def test_mcq_orchestration_jobs_can_be_claimed_and_completed():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req_id = svc.create_request(
            request_name="mcq_claim_complete",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 3, "template_version": "v1"},
            parameter_axes={
                "levels": ["high school"],
                "subjects": ["physics"],
                "deployments": ["gpt-4o-mini"],
                "options_per_question": [4],
                "questions_per_test": [10],
                "label_styles": ["upper"],
                "spread_correct_answer_uniformly": [False],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        job = repo.claim_next_orchestration_job(
            worker_id="mcq-worker-1",
            lease_seconds=60,
            request_group_id=req_id,
            job_types=["mcq_run"],
        )
        assert job is not None
        assert job.job_type == "mcq_run"
        repo.complete_orchestration_job(int(job.id), result_ref="run_id:1")
        refreshed = {
            int(j.id): j for j in repo.list_orchestration_jobs(request_group_id=req_id)
        }
        assert int(job.id) in refreshed
        assert refreshed[int(job.id)].status == "completed"

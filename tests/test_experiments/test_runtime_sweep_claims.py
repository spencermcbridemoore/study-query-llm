"""Claim-or-skip behavior tests for request-driven runtime sweeps."""

from __future__ import annotations

from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import SweepRunClaim
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments import sweep_worker_main as worker_main
from study_query_llm.experiments.runtime_sweeps import _claim_run_target, _complete_run_claim


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "runtime_claims.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _request_group_id(db: DatabaseConnectionV2) -> int:
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        return int(
            repo.create_group(
                group_type="clustering_sweep_request",
                name="claim_test_request",
                metadata_json={},
            )
        )


def test_claim_run_target_single_active_owner(tmp_path: Path) -> None:
    db = _db(tmp_path)
    request_id = _request_group_id(db)

    assert _claim_run_target(
        db=db,
        request_id=request_id,
        run_key="run_key_1",
        worker_id="worker_a",
        lease_seconds=60,
    )
    assert not _claim_run_target(
        db=db,
        request_id=request_id,
        run_key="run_key_1",
        worker_id="worker_b",
        lease_seconds=60,
    )
    # Same owner can refresh lease.
    assert _claim_run_target(
        db=db,
        request_id=request_id,
        run_key="run_key_1",
        worker_id="worker_a",
        lease_seconds=60,
    )

    with db.session_scope() as session:
        claims = (
            session.query(SweepRunClaim)
            .filter(
                SweepRunClaim.request_group_id == request_id,
                SweepRunClaim.run_key == "run_key_1",
            )
            .all()
        )
        assert len(claims) == 1
        assert str(claims[0].claim_status) == "claimed"
        assert str(claims[0].claimed_by) == "worker_a"


def test_claim_run_target_skips_completed_runs(tmp_path: Path) -> None:
    db = _db(tmp_path)
    request_id = _request_group_id(db)

    assert _claim_run_target(
        db=db,
        request_id=request_id,
        run_key="run_key_done",
        worker_id="worker_a",
        lease_seconds=60,
    )
    _complete_run_claim(
        db=db,
        request_id=request_id,
        run_key="run_key_done",
        run_id=101,
        worker_id="worker_a",
    )

    assert not _claim_run_target(
        db=db,
        request_id=request_id,
        run_key="run_key_done",
        worker_id="worker_b",
        lease_seconds=60,
    )


def test_standalone_worker_preserves_mcq_fallback_path(tmp_path: Path, monkeypatch) -> None:
    db = _db(tmp_path)
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        request_id = int(
            repo.create_group(
                group_type="mcq_sweep_request",
                name="mcq_fallback_request",
                metadata_json={
                    "request_status": "requested",
                    "sweep_type": "mcq",
                    "run_key_to_target": {},
                    "expected_run_keys": [],
                    "analysis_catalog": [],
                    "required_analyses": [],
                },
            )
        )

    worker_main.db = db
    called = {"mcq": False, "cluster": False, "sharded": False}

    def _fake_mcq(**kwargs):
        called["mcq"] = True
        return 9

    def _fake_cluster(**kwargs):
        called["cluster"] = True
        return 0

    def _fake_sharded(**kwargs):
        called["sharded"] = True
        return 0

    monkeypatch.setattr(worker_main, "_run_mcq_standalone_worker_loop", _fake_mcq)
    monkeypatch.setattr(worker_main, "_run_clustering_standalone_worker_loop", _fake_cluster)
    monkeypatch.setattr(worker_main, "_run_sharded_worker_loop", _fake_sharded)

    result = worker_main._run_standalone_worker_loop(
        request_id=request_id,
        worker_id="w-test",
        worker_slot=0,
        embedding_engine=None,
        tei_endpoint=None,
        provider_label="local",
        embedding_provider_name=None,
        claim_lease_seconds=60,
        max_runs=1,
        idle_exit_seconds=1,
        force=False,
        repo_root=tmp_path,
    )
    assert result == 9
    assert called["mcq"] is True
    assert called["cluster"] is False
    assert called["sharded"] is False

"""Claim-or-skip behavior tests for request-driven runtime sweeps."""

from __future__ import annotations

from pathlib import Path

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import SweepRunClaim
from study_query_llm.db.raw_call_repository import RawCallRepository
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

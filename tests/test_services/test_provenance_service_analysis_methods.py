"""Tests for new analysis_* group helpers on ProvenanceService."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_ANALYSIS_REQUEST,
    GROUP_TYPE_ANALYSIS_RUN,
    ProvenanceService,
)


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_analysis_run_group_caps_name_and_preserves_full_name() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        long_run_key = "rk_" + ("x" * 240)
        group_id = provenance.create_analysis_run_group(
            method_name="stability_report",
            input_id=123,
            run_key=long_run_key,
            request_group_id=45,
        )
        group = repo.get_group_by_id(group_id)
        assert group is not None
        assert group.group_type == GROUP_TYPE_ANALYSIS_RUN
        assert len(group.name) <= 180
        assert group.metadata_json is not None
        assert group.metadata_json["method_name"] == "stability_report"
        assert group.metadata_json["input_id"] == 123
        assert group.metadata_json["run_key"] == long_run_key
        assert "full_name" in group.metadata_json
        assert str(group.metadata_json["full_name"]).startswith("analyze:stability_report:123:")


def test_create_analysis_request_group_is_idempotent_by_method_input_run_key() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        first_id = provenance.create_analysis_request_group(
            method_name="cluster_count",
            input_id=55,
            run_key="rk_shared",
        )
        second_id = provenance.create_analysis_request_group(
            method_name="cluster_count",
            input_id=55,
            run_key="rk_shared",
        )
        assert first_id == second_id

        group = repo.get_group_by_id(first_id)
        assert group is not None
        assert group.group_type == GROUP_TYPE_ANALYSIS_REQUEST
        assert group.metadata_json is not None
        assert group.metadata_json["method_name"] == "cluster_count"
        assert group.metadata_json["input_id"] == 55
        assert group.metadata_json["run_key"] == "rk_shared"


def _file_db(tmp_path: Path) -> DatabaseConnectionV2:
    """File-backed SQLite shared across threads (in-memory SQLite is per-connection)."""
    db_path = (tmp_path / "analysis_request_race.sqlite3").resolve()
    db = DatabaseConnectionV2(
        f"sqlite:///{db_path.as_posix()}", enable_pgvector=False, quiet=True
    )
    db.init_db()
    return db


def test_create_analysis_request_group_concurrent_same_identity_creates_one(
    tmp_path: Path,
) -> None:
    """Parallel callers sharing one identity must converge on a single group.

    This is a real-thread race, not a sequential idempotency check: every worker
    resolves the same (method_name, input_id, run_key) at once against a shared
    file-backed SQLite DB. Exactly one ``analysis_request`` row must be created
    and every caller must receive that same group id. Without the DB-enforced
    unique index + conflict-safe insert, the scan-then-insert get-or-create would
    create duplicate rows here.
    """
    db = _file_db(tmp_path)
    workers = 6
    rounds = 6
    failures: list[str] = []

    for round_idx in range(rounds):
        identity = {
            "method_name": "cluster_count",
            "input_id": 100 + round_idx,
            "run_key": f"rk_round_{round_idx}",
        }
        barrier = threading.Barrier(workers)
        lock = threading.Lock()
        ids: list[int] = []

        def _resolve(_identity=identity, _barrier=barrier) -> None:
            try:
                _barrier.wait(timeout=10.0)
                with db.session_scope() as session:
                    prov = ProvenanceService(RawCallRepository(session))
                    gid = prov.create_analysis_request_group(**_identity)
                with lock:
                    ids.append(int(gid))
            except Exception as exc:  # surfaced via assertion below
                with lock:
                    failures.append(f"{type(exc).__name__}: {exc}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_resolve) for _ in range(workers)]
            for fut in futures:
                fut.result(timeout=20.0)

        assert not failures, f"round {round_idx} raised: {failures}"
        assert len(ids) == workers
        assert len(set(ids)) == 1, (
            f"round {round_idx} created multiple groups: {sorted(set(ids))}"
        )

    with db.session_scope() as session:
        groups = (
            session.query(Group)
            .filter(Group.group_type == GROUP_TYPE_ANALYSIS_REQUEST)
            .all()
        )
        assert len(groups) == rounds, sorted(g.id for g in groups)


def test_analysis_request_identity_unique_index_blocks_direct_duplicate(
    tmp_path: Path,
) -> None:
    """The DB-level partial unique index rejects a second identical identity.

    Guards against a regression that silently drops the constraint and lets the
    original duplicate-creation race reappear.
    """
    db = _file_db(tmp_path)
    identity_meta = {"method_name": "m", "input_id": 7, "run_key": "rk"}
    with db.session_scope() as session:
        RawCallRepository(session).create_group(
            group_type=GROUP_TYPE_ANALYSIS_REQUEST,
            name="first",
            metadata_json=dict(identity_meta),
        )
    with pytest.raises(IntegrityError):
        with db.session_scope() as session:
            RawCallRepository(session).create_group(
                group_type=GROUP_TYPE_ANALYSIS_REQUEST,
                name="second",
                metadata_json=dict(identity_meta),
            )

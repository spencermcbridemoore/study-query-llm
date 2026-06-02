"""Unit tests for scripts/remediate_analysis_request_duplicates.py core logic.

These exercise the dialect-agnostic ``remediate_duplicates`` core on SQLite by
dropping the unique index first to recreate the pre-fix state in which duplicate
``analysis_request`` rows could exist. This validates the repoint/delete logic
the canonical-database run will perform.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy import text

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import (
    ANALYSIS_REQUEST_UNIQUE_INDEX_NAME,
    Group,
    GroupLink,
    GroupMember,
    ProvenancedRun,
    RawCall,
)
from study_query_llm.db.raw_call_repository import RawCallRepository

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "remediate_analysis_request_duplicates.py"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "remediate_analysis_request_duplicates", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db_without_unique_index() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False, quiet=True)
    db.init_db()
    # Recreate the pre-fix world: drop the guard so duplicates can be inserted.
    with db.engine.begin() as conn:
        conn.execute(text(f"DROP INDEX IF EXISTS {ANALYSIS_REQUEST_UNIQUE_INDEX_NAME}"))
    return db


def _make_request(repo: RawCallRepository, identity: dict, name: str) -> int:
    return int(
        repo.create_group(
            group_type="analysis_request", name=name, metadata_json=dict(identity)
        )
    )


def test_remediate_collapses_duplicates_and_repoints_all_references() -> None:
    mod = _mod()
    db = _db_without_unique_index()
    id_x = {"method_name": "m", "input_id": 1, "run_key": "rkX"}
    id_y = {"method_name": "m", "input_id": 2, "run_key": "rkY"}

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        keeper_x = _make_request(repo, id_x, "keeperX")
        loser_x = _make_request(repo, id_x, "loserX")
        keeper_y = _make_request(repo, id_y, "keeperY")
        loser_y = _make_request(repo, id_y, "loserY")

        raw = RawCall(provider="test", modality="text", status="success", request_json={})
        session.add(raw)
        session.flush()
        session.add(GroupMember(group_id=loser_x, call_id=raw.id))

        # contains edge keeperX -> loserX becomes a self-link after repoint.
        session.add(
            GroupLink(parent_group_id=keeper_x, child_group_id=loser_x, link_type="contains")
        )

        # provenanced_run under loserX with a distinct run_key -> clean repoint.
        run_x = repo.create_provenanced_run(
            run_kind="execution", request_group_id=loser_x, run_key="loserX_run"
        )

        # analysis_run group carrying a JSON request_group_id pointer to loserX.
        analysis_run_group = repo.create_group(
            group_type="analysis_run",
            name="arX",
            metadata_json={"request_group_id": loser_x, "method_name": "m"},
        )

        # collision: keeperY and loserY both own a run keyed 'dup' (execution).
        repo.create_provenanced_run(run_kind="execution", request_group_id=keeper_y, run_key="dup")
        repo.create_provenanced_run(run_kind="execution", request_group_id=loser_y, run_key="dup")

    # Dry run must report duplicates but change nothing.
    with db.session_scope() as session:
        dry = mod.remediate_duplicates(session, apply=False)
    assert dry["duplicate_identities"] == 2
    assert dry["remaining_duplicates"] == 2
    with db.session_scope() as session:
        assert (
            session.query(Group).filter(Group.group_type == "analysis_request").count() == 4
        )

    # Apply.
    with db.session_scope() as session:
        report = mod.remediate_duplicates(session, apply=True)
    assert report["remaining_duplicates"] == 0

    with db.session_scope() as session:
        remaining_ids = {
            g.id
            for g in session.query(Group)
            .filter(Group.group_type == "analysis_request")
            .all()
        }
        assert remaining_ids == {keeper_x, keeper_y}

        run = session.get(ProvenancedRun, run_x)
        assert run is not None and run.request_group_id == keeper_x

        members = session.query(GroupMember).all()
        assert len(members) == 1 and members[0].group_id == keeper_x

        assert session.query(GroupLink).all() == []  # self-link cleaned up

        arg = session.get(Group, analysis_run_group)
        assert arg.metadata_json["request_group_id"] == keeper_x

        dup_runs = (
            session.query(ProvenancedRun)
            .filter(
                ProvenancedRun.request_group_id == keeper_y,
                ProvenancedRun.run_key == "dup",
            )
            .all()
        )
        assert len(dup_runs) == 1  # loserY's colliding run was dropped


def test_find_duplicate_identities_ignores_incomplete_identities() -> None:
    mod = _mod()
    db = _db_without_unique_index()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        # Empty-metadata container rows must never be treated as duplicates.
        repo.create_group(group_type="analysis_request", name="empty1", metadata_json={})
        repo.create_group(group_type="analysis_request", name="empty2", metadata_json={})
        dups = mod.find_duplicate_identities(session)
    assert dups == {}

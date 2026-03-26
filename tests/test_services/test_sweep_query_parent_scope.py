"""Tests for parent scope id resolution (group_links contains → parent group)."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import (
    GROUP_TYPE_CLUSTERING_RUN,
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
)
from study_query_llm.services.sweep_query_service import _parent_scope_id_by_child_run


def _db():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_parent_scope_prefers_clustering_sweep_over_request():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        sweep_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_SWEEP,
            name="sw",
            metadata_json={"algorithm": "cosine_kllmeans_x"},
        )
        req_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
            name="req",
            metadata_json={"request_status": "fulfilled"},
        )
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run",
            metadata_json={
                "algorithm": "cosine_kllmeans_no_pca",
                "dataset": "d",
            },
        )
        repo.create_group_link(sweep_id, run_id, "contains")
        repo.create_group_link(req_id, run_id, "contains")
        m = _parent_scope_id_by_child_run(
            session,
            [run_id],
            allowed_parent_types=(
                GROUP_TYPE_CLUSTERING_SWEEP,
                GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
            ),
            preference_order=(
                GROUP_TYPE_CLUSTERING_SWEEP,
                GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
            ),
        )
        assert m[run_id] == sweep_id


def test_parent_scope_empty_run_ids():
    db = _db()
    with db.session_scope() as session:
        m = _parent_scope_id_by_child_run(
            session,
            [],
            allowed_parent_types=(GROUP_TYPE_CLUSTERING_SWEEP,),
            preference_order=(GROUP_TYPE_CLUSTERING_SWEEP,),
        )
    assert m == {}

"""Tests for dataset snapshot delta lineage links."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import GroupLink
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import ProvenanceService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_dataset_snapshot_with_parent_delta_lineage() -> None:
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)
        parent_id = provenance.create_dataset_snapshot_group(
            snapshot_name="base_snapshot",
            source_dataset="dbpedia",
            sample_size=100,
            label_mode="labeled",
            sampling_method="seeded",
            sampling_seed=42,
        )
        child_id = provenance.create_dataset_snapshot_group(
            snapshot_name="delta_snapshot",
            source_dataset="dbpedia",
            sample_size=120,
            label_mode="labeled",
            sampling_method="delta_append",
            sampling_seed=43,
            parent_snapshot_group_id=parent_id,
            delta_metadata={"added_rows": 20},
        )
        child = repo.get_group_by_id(child_id)
        assert child is not None
        meta = dict(child.metadata_json or {})
        assert int(meta["parent_snapshot_group_id"]) == int(parent_id)
        assert meta["snapshot_lineage_mode"] == "delta"
        assert dict(meta["delta_metadata"] or {}).get("added_rows") == 20

        link = (
            session.query(GroupLink)
            .filter(
                GroupLink.parent_group_id == child_id,
                GroupLink.child_group_id == parent_id,
                GroupLink.link_type == "depends_on",
            )
            .first()
        )
        assert link is not None
        assert dict(link.metadata_json or {}).get("relation") == "delta_parent_snapshot"

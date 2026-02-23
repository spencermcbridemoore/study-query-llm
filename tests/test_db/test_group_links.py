"""
Tests for GroupLink functionality.

Tests group relationships, step sequences, and dependency traversal.
"""

import pytest
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_group_link(db_connection):
    """Test creating a link between groups."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Create parent and child groups
        parent_id = repo.create_group(
            group_type="clustering_run",
            name="test_run",
            description="Test run",
        )

        child_id = repo.create_group(
            group_type="clustering_step",
            name="step_1",
            description="First step",
        )

        # Create link
        link_id = repo.create_group_link(
            parent_group_id=parent_id,
            child_group_id=child_id,
            link_type="clustering_step",
            position=1,
        )

        assert link_id > 0

        # Verify link was created
        from study_query_llm.db.models_v2 import GroupLink

        link = session.query(GroupLink).filter_by(id=link_id).first()
        assert link is not None
        assert link.parent_group_id == parent_id
        assert link.child_group_id == child_id
        assert link.link_type == "clustering_step"
        assert link.position == 1


def test_get_group_children(db_connection):
    """Test getting child groups for a parent."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Create run with multiple steps
        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        step1_id = repo.create_group(group_type="clustering_step", name="step_1")
        step2_id = repo.create_group(group_type="clustering_step", name="step_2")
        step3_id = repo.create_group(group_type="clustering_step", name="step_3")

        # Create links
        repo.create_group_link(run_id, step1_id, "clustering_step", position=1)
        repo.create_group_link(run_id, step2_id, "clustering_step", position=2)
        repo.create_group_link(run_id, step3_id, "clustering_step", position=3)

        # Get children
        children = repo.get_group_children(run_id, link_type="clustering_step")

        assert len(children) == 3
        assert {c.id for c in children} == {step1_id, step2_id, step3_id}


def test_get_group_parents(db_connection):
    """Test getting parent groups for a child."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Create parent and child
        parent_id = repo.create_group(group_type="clustering_run", name="test_run")
        child_id = repo.create_group(group_type="clustering_step", name="step_1")

        # Create link
        repo.create_group_link(parent_id, child_id, "clustering_step", position=1)

        # Get parents
        parents = repo.get_group_parents(child_id)

        assert len(parents) == 1
        assert parents[0].id == parent_id


def test_get_run_step_sequence(db_connection):
    """Test getting ordered step sequence for a run."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Create run with steps
        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        step1_id = repo.create_group(group_type="clustering_step", name="step_1")
        step2_id = repo.create_group(group_type="clustering_step", name="step_2")
        step3_id = repo.create_group(group_type="clustering_step", name="step_3")

        # Create links in specific order
        repo.create_group_link(run_id, step2_id, "clustering_step", position=2)
        repo.create_group_link(run_id, step1_id, "clustering_step", position=1)
        repo.create_group_link(run_id, step3_id, "clustering_step", position=3)

        # Get step sequence
        steps = repo.get_run_step_sequence(run_id)

        assert len(steps) == 3
        assert steps[0].id == step1_id  # Position 1
        assert steps[1].id == step2_id  # Position 2
        assert steps[2].id == step3_id  # Position 3


def test_group_link_different_types(db_connection):
    """Test different link types (step, depends_on, generates, contains)."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        # Create groups
        embedding_batch_id = repo.create_group(
            group_type="embedding_batch", name="emb_batch"
        )
        run_id = repo.create_group(group_type="clustering_run", name="test_run")
        step_id = repo.create_group(group_type="clustering_step", name="step_1")

        # Create different link types
        repo.create_group_link(embedding_batch_id, run_id, "generates")
        repo.create_group_link(run_id, step_id, "clustering_step", position=1)

        # Verify links
        generated = repo.get_group_children(embedding_batch_id, link_type="generates")
        assert len(generated) == 1
        assert generated[0].id == run_id

        steps = repo.get_group_children(run_id, link_type="clustering_step")
        assert len(steps) == 1
        assert steps[0].id == step_id


def test_create_group_link_duplicate(db_connection):
    """Test that creating duplicate link returns existing link ID."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)

        parent_id = repo.create_group(group_type="clustering_run", name="test_run")
        child_id = repo.create_group(group_type="clustering_step", name="step_1")

        # Create link
        link_id1 = repo.create_group_link(
            parent_id, child_id, "clustering_step", position=1
        )

        # Try to create duplicate
        link_id2 = repo.create_group_link(
            parent_id, child_id, "clustering_step", position=1
        )

        # Should return same ID
        assert link_id1 == link_id2

        # Verify only one link exists
        from study_query_llm.db.models_v2 import GroupLink

        links = session.query(GroupLink).filter_by(
            parent_group_id=parent_id,
            child_group_id=child_id,
            link_type="clustering_step",
        ).all()

        assert len(links) == 1

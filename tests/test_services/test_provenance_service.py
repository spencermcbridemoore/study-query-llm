"""
Tests for ProvenanceService.

Tests run/experiment provenance tracking via Groups and GroupMember.
"""

import pytest
from study_query_llm.services.provenance_service import (
    ProvenanceService,
    GROUP_TYPE_RUN,
    GROUP_TYPE_STEP,
    GROUP_TYPE_DATASET,
    GROUP_TYPE_EMBEDDING_BATCH,
    GROUP_TYPE_SUMMARIZATION_BATCH,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_create_run_group(db_connection):
    """Test creating a run group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        run_id = provenance.create_run_group(
            algorithm="pca_kllmeans_sweep",
            config={"k_min": 2, "k_max": 10, "pca_dim": 64},
        )

        # Verify group was created
        group = repo.get_group_by_id(run_id)
        assert group is not None
        assert group.group_type == GROUP_TYPE_RUN
        assert "pca_kllmeans_sweep" in group.name
        assert group.metadata_json is not None
        assert group.metadata_json["algorithm"] == "pca_kllmeans_sweep"
        assert group.metadata_json["config"]["k_min"] == 2
        assert group.metadata_json["config"]["k_max"] == 10


def test_create_step_group(db_connection):
    """Test creating a step group within a run."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # Create parent run
        run_id = provenance.create_run_group(
            algorithm="pca_kllmeans_sweep",
            config={"k_min": 2, "k_max": 10},
        )

        # Create step
        step_id = provenance.create_step_group(
            parent_run_id=run_id,
            step_name="pca_projection",
            step_type="pca",
        )

        # Verify step was created
        step_group = repo.get_group_by_id(step_id)
        assert step_group is not None
        assert step_group.group_type == GROUP_TYPE_STEP
        assert step_group.metadata_json["parent_run_id"] == run_id
        assert step_group.metadata_json["step_name"] == "pca_projection"
        assert step_group.metadata_json["step_type"] == "pca"


def test_link_raw_calls_to_group(db_connection):
    """Test linking RawCalls to a group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # Create a run group
        run_id = provenance.create_run_group(algorithm="test_algorithm")

        # Create some test RawCalls
        call_id1 = repo.insert_raw_call(
            provider="test",
            request_json={"input": "test1"},
            modality="embedding",
            status="success",
        )
        call_id2 = repo.insert_raw_call(
            provider="test",
            request_json={"input": "test2"},
            modality="embedding",
            status="success",
        )

        # Link calls to group
        member_ids = provenance.link_raw_calls_to_group(
            run_id, [call_id1, call_id2], role="input"
        )

        assert len(member_ids) == 2

        # Verify calls are in the group
        calls = repo.get_calls_in_group(run_id)
        assert len(calls) == 2
        assert {c.id for c in calls} == {call_id1, call_id2}


def test_link_artifacts_to_group(db_connection):
    """Test creating and linking artifacts to a group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # Create a run group
        run_id = provenance.create_run_group(algorithm="test_algorithm")

        # Create a dummy RawCall to link artifacts to (CallArtifact requires call_id)
        call_id = repo.insert_raw_call(
            provider="test",
            request_json={"type": "artifact_placeholder"},
            modality="text",
            status="success",
        )

        # Create artifacts
        artifacts = [
            {
                "call_id": call_id,
                "artifact_type": "json",
                "uri": "file:///path/to/results.json",
                "content_type": "application/json",
                "byte_size": 1024,
                "metadata_json": {"key": "value"},
            },
            {
                "call_id": call_id,
                "artifact_type": "npy",
                "uri": "file:///path/to/vectors.npy",
                "content_type": "application/octet-stream",
                "byte_size": 2048,
            },
        ]

        artifact_ids = provenance.link_artifacts_to_group(run_id, artifacts)

        assert len(artifact_ids) == 2

        # Verify artifacts were created
        from study_query_llm.db.models_v2 import CallArtifact

        for artifact_id in artifact_ids:
            artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
            assert artifact is not None
            assert artifact.metadata_json["group_id"] == run_id


def test_get_run_provenance(db_connection):
    """Test querying run provenance."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        # Create a run
        run_id = provenance.create_run_group(
            algorithm="test_algorithm", config={"param": "value"}
        )

        # Create RawCalls and link them
        call_id1 = repo.insert_raw_call(
            provider="test",
            request_json={"input": "test1"},
            modality="embedding",
            status="success",
        )
        provenance.link_raw_calls_to_group(run_id, [call_id1], role="input")

        # Create a step
        step_id = provenance.create_step_group(
            parent_run_id=run_id, step_name="test_step"
        )

        # Create an artifact (need a call_id)
        artifact_call_id = repo.insert_raw_call(
            provider="test",
            request_json={"type": "artifact_placeholder"},
            modality="text",
            status="success",
        )
        
        artifact_id = provenance.link_artifacts_to_group(
            run_id,
            [
                {
                    "call_id": artifact_call_id,
                    "artifact_type": "json",
                    "uri": "file:///test.json",
                }
            ],
        )[0]

        # Get provenance
        prov = provenance.get_run_provenance(run_id)

        assert prov["run_group"].id == run_id
        assert len(prov["raw_calls"]) == 1
        assert prov["raw_calls"][0].id == call_id1
        assert len(prov["step_groups"]) == 1
        assert prov["step_groups"][0].id == step_id
        assert len(prov["artifacts"]) == 1
        assert prov["artifacts"][0].id == artifact_id
        assert prov["metadata"]["algorithm"] == "test_algorithm"
        assert prov["metadata"]["config"]["param"] == "value"


def test_create_dataset_group(db_connection):
    """Test creating a dataset group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        dataset_id = provenance.create_dataset_group(
            name="test_dataset",
            description="Test dataset",
            metadata={"source": "file.csv", "size": 1000},
        )

        group = repo.get_group_by_id(dataset_id)
        assert group.group_type == GROUP_TYPE_DATASET
        assert group.name == "test_dataset"
        assert group.metadata_json["source"] == "file.csv"


def test_create_embedding_batch_group(db_connection):
    """Test creating an embedding batch group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        batch_id = provenance.create_embedding_batch_group(
            deployment="text-embedding-ada-002",
            metadata={"count": 100},
        )

        group = repo.get_group_by_id(batch_id)
        assert group.group_type == GROUP_TYPE_EMBEDDING_BATCH
        assert group.metadata_json["deployment"] == "text-embedding-ada-002"
        assert group.metadata_json["count"] == 100


def test_create_summarization_batch_group(db_connection):
    """Test creating a summarization batch group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        batch_id = provenance.create_summarization_batch_group(
            llm_deployment="gpt-4",
            metadata={"count": 50},
        )

        group = repo.get_group_by_id(batch_id)
        assert group.group_type == GROUP_TYPE_SUMMARIZATION_BATCH
        assert group.metadata_json["llm_deployment"] == "gpt-4"
        assert group.metadata_json["count"] == 50


def test_get_run_provenance_nonexistent(db_connection):
    """Test that get_run_provenance raises error for nonexistent run."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        provenance = ProvenanceService(repo)

        with pytest.raises(ValueError, match="not found"):
            provenance.get_run_provenance(99999)

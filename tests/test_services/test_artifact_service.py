"""
Tests for ArtifactService.

Tests artifact storage, loading, and group linkage.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.fixture
def temp_artifact_dir():
    """Fixture for temporary artifact directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_store_sweep_results(db_connection, temp_artifact_dir):
    """Test storing sweep results as JSON artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        # Create a run group
        run_id = repo.create_group(
            group_type="clustering_run",
            name="test_run",
            description="Test run",
        )

        # Create a placeholder RawCall for the artifact (CallArtifact requires call_id)
        call_id = repo.insert_raw_call(
            provider="test",
            request_json={"type": "artifact_placeholder"},
            modality="text",
            status="success",
        )

        sweep_results = {
            "by_k": {
                5: {"labels": [0, 1, 0, 1, 0], "objective": 0.5},
                10: {"labels": [0, 1, 2, 1, 0], "objective": 0.3},
            },
            "pca_meta": {"dim": 64, "variance_explained": 0.95},
        }

        artifact_id = service.store_sweep_results(
            run_id=run_id,
            sweep_results=sweep_results,
            step_name="sweep_complete",
        )

        assert artifact_id > 0

        # Verify artifact was created
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact is not None
        assert artifact.artifact_type == "sweep_results"
        assert artifact.metadata_json["group_id"] == run_id
        assert artifact.content_type == "application/json"

        # Verify file exists and can be loaded
        assert Path(artifact.uri).exists()
        loaded = service.load_artifact(artifact.uri, "sweep_results")
        # JSON converts integer keys to strings
        assert loaded["by_k"]["5"]["objective"] == 0.5
        assert loaded["by_k"]["10"]["objective"] == 0.3


def test_store_cluster_labels(db_connection, temp_artifact_dir):
    """Test storing cluster labels as NPY artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        # Service will create placeholder RawCall automatically
        labels = np.array([0, 1, 0, 1, 2, 0, 1])

        artifact_id = service.store_cluster_labels(
            run_id=run_id,
            labels=labels,
            step_name="clustering_k=3",
            k=3,
        )

        assert artifact_id > 0

        # Verify artifact
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact.artifact_type == "cluster_labels"
        assert artifact.metadata_json["k"] == 3
        assert artifact.metadata_json["shape"] == [7]

        # Verify file can be loaded
        loaded = service.load_artifact(artifact.uri, "cluster_labels")
        np.testing.assert_array_equal(loaded, labels)


def test_store_pca_components(db_connection, temp_artifact_dir):
    """Test storing PCA components as NPY artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        # Service will create placeholder RawCall automatically
        components = np.random.rand(100, 64)  # 100 points, 64 dims

        artifact_id = service.store_pca_components(
            run_id=run_id,
            components=components,
            step_name="pca_projection",
        )

        assert artifact_id > 0

        # Verify file can be loaded
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        loaded = service.load_artifact(artifact.uri, "pca_components")
        np.testing.assert_array_equal(loaded, components)


def test_store_metrics(db_connection, temp_artifact_dir):
    """Test storing metrics as JSON artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        # Service will create placeholder RawCall automatically
        metrics = {
            "silhouette": 0.5,
            "ari": 0.8,
            "coverage": 0.9,
        }

        artifact_id = service.store_metrics(
            run_id=run_id,
            metrics=metrics,
            step_name="metrics_k=5",
        )

        assert artifact_id > 0

        # Verify file can be loaded
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        loaded = service.load_artifact(artifact.uri, "metrics")
        assert loaded["silhouette"] == 0.5
        assert loaded["ari"] == 0.8


def test_store_representatives(db_connection, temp_artifact_dir):
    """Test storing representatives as CSV artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        run_id = repo.create_group(group_type="clustering_run", name="test_run")

        # Service will create placeholder RawCall automatically
        representatives = ["Question 1", "Question 2", "Question 3"]

        artifact_id = service.store_representatives(
            run_id=run_id,
            representatives=representatives,
            step_name="representatives_k=5",
            k=5,
        )

        assert artifact_id > 0

        # Verify file can be loaded
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        loaded = service.load_artifact(artifact.uri, "representatives")
        assert loaded == representatives
        assert artifact.metadata_json["k"] == 5


def test_artifact_service_without_repository(temp_artifact_dir):
    """Test that service works without repository (no DB persistence)."""
    service = ArtifactService(repository=None, artifact_dir=temp_artifact_dir)

    sweep_results = {"by_k": {5: {"labels": [0, 1, 0]}}}

    # Should still save file, but return 0 for artifact_id
    artifact_id = service.store_sweep_results(
        run_id=1,
        sweep_results=sweep_results,
        step_name="test",
    )

    assert artifact_id == 0  # No persistence

    # But file should still exist
    uri = service._generate_uri(1, "test", "sweep_results", "json")
    assert Path(uri).exists()


def test_load_artifact_nonexistent():
    """Test that loading nonexistent artifact raises error."""
    service = ArtifactService(repository=None, artifact_dir="nonexistent")

    with pytest.raises(FileNotFoundError):
        service.load_artifact("nonexistent/path.json", "sweep_results")


def test_load_artifact_unknown_type(temp_artifact_dir):
    """Test that loading unknown artifact type raises error."""
    service = ArtifactService(repository=None, artifact_dir=temp_artifact_dir)

    # Create a dummy file
    dummy_file = Path(temp_artifact_dir) / "dummy.txt"
    dummy_file.write_text("test")

    with pytest.raises(ValueError, match="Unknown artifact type"):
        service.load_artifact(str(dummy_file), "unknown_type")

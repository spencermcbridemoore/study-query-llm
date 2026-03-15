"""
Tests for ArtifactService.

Tests artifact storage, loading, and group linkage.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from unittest.mock import MagicMock, patch
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


def test_store_dataset_snapshot_manifest(db_connection, temp_artifact_dir):
    """Test storing dataset snapshot manifest as JSON artifact."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)

        snapshot_group_id = repo.create_group(
            group_type="dataset_snapshot",
            name="dbpedia_286_seed42_labeled",
            description="Test snapshot",
        )

        entries = [
            {"position": 0, "source_id": 123, "text": "Alpha", "label": 1},
            {"position": 1, "source_id": 456, "text": "Beta", "label": 0},
        ]
        artifact_id = service.store_dataset_snapshot_manifest(
            snapshot_group_id=snapshot_group_id,
            snapshot_name="dbpedia_286_seed42_labeled",
            entries=entries,
            metadata={"label_mode": "labeled"},
        )

        assert artifact_id > 0

        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact is not None
        assert artifact.artifact_type == "dataset_snapshot_manifest"
        assert artifact.metadata_json["group_id"] == snapshot_group_id
        assert artifact.metadata_json["entry_count"] == 2
        assert artifact.metadata_json["label_mode"] == "labeled"
        assert "manifest_hash" in artifact.metadata_json

        loaded = service.load_artifact(artifact.uri, "dataset_snapshot_manifest")
        assert loaded["snapshot_name"] == "dbpedia_286_seed42_labeled"
        assert len(loaded["entries"]) == 2


def test_store_and_find_embedding_matrix(db_connection, temp_artifact_dir):
    """Test storing/reusing embedding_matrix artifact metadata."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)
        embedding_batch_group_id = repo.create_group(
            group_type="embedding_batch",
            name="batch_test",
        )
        matrix = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        artifact_id = service.store_embedding_matrix(
            embedding_batch_group_id,
            matrix,
            dataset_key="dbpedia:entry_max=2",
            embedding_engine="text-embedding-3-small",
            provider="azure",
            entry_max=2,
            key_version="raw_v1",
        )
        assert artifact_id > 0
        hit = service.find_embedding_matrix_artifact(
            dataset_key="dbpedia:entry_max=2",
            embedding_engine="text-embedding-3-small",
            provider="azure",
            entry_max=2,
            key_version="raw_v1",
        )
        assert hit is not None
        loaded = service.load_artifact(hit["uri"], "embedding_matrix")
        np.testing.assert_array_equal(loaded, matrix)


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
    logical_path = service._generate_logical_path(1, "test", "sweep_results", "json")
    uri = service.storage.get_uri(logical_path)
    assert service.storage.exists(logical_path)


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


def test_artifact_service_default_backend_uses_local_when_env_unset(
    db_connection, temp_artifact_dir
):
    """Default backend selection uses local when ARTIFACT_STORAGE_BACKEND is unset."""
    import os

    orig = os.environ.pop("ARTIFACT_STORAGE_BACKEND", None)
    try:
        with db_connection.session_scope() as session:
            repo = RawCallRepository(session)
            service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)
            run_id = repo.create_group(group_type="clustering_run", name="test_run")
            sweep_results = {"by_k": {5: {"labels": [0, 1, 0]}}}
            artifact_id = service.store_sweep_results(
                run_id=run_id, sweep_results=sweep_results, step_name="sweep_complete"
            )
            assert artifact_id > 0
            from study_query_llm.db.models_v2 import CallArtifact

            artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
            assert artifact is not None
            # Local backend produces file path, not https URL
            assert not str(artifact.uri).startswith("https://")
            assert Path(artifact.uri).exists()
    finally:
        if orig is not None:
            os.environ["ARTIFACT_STORAGE_BACKEND"] = orig


def test_artifact_service_explicit_storage_backend_override(temp_artifact_dir):
    """Explicit storage_backend parameter overrides env-driven default."""
    from study_query_llm.storage.local import LocalStorageBackend

    backend = LocalStorageBackend(base_dir=temp_artifact_dir)
    service = ArtifactService(
        repository=None, artifact_dir=temp_artifact_dir, storage_backend=backend
    )
    assert service.storage is backend
    sweep_results = {"by_k": {5: {"labels": [0, 1, 0]}}}
    artifact_id = service.store_sweep_results(
        run_id=1, sweep_results=sweep_results, step_name="test"
    )
    assert artifact_id == 0
    assert service.storage.exists("1/test/sweep_results.json")


def test_store_sweep_results_adds_integrity_and_governance_metadata(
    db_connection, temp_artifact_dir
):
    """Stored artifacts include integrity + governance metadata fields."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)
        run_id = repo.create_group(group_type="clustering_run", name="test_run")
        artifact_id = service.store_sweep_results(
            run_id=run_id,
            sweep_results={"by_k": {2: {"labels": [0, 1]}}},
            step_name="sweep_complete",
        )
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact is not None
        md = artifact.metadata_json or {}
        assert md.get("schema_version") == "artifact.v1"
        assert md.get("governance_version") == "blob_ops_phase2"
        assert md.get("storage_backend") == "local"
        assert isinstance(md.get("created_at"), str)
        assert isinstance(md.get("sha256"), str) and len(md["sha256"]) == 64
        assert int(md.get("byte_size", -1)) == int(artifact.byte_size)


def test_load_artifact_integrity_mismatch_raises(db_connection, temp_artifact_dir):
    """load_artifact raises when checksum/size expectations mismatch."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)
        run_id = repo.create_group(group_type="clustering_run", name="test_run")
        artifact_id = service.store_sweep_results(
            run_id=run_id,
            sweep_results={"by_k": {2: {"labels": [0, 1]}}},
            step_name="sweep_complete",
        )
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact is not None

        with pytest.raises(ValueError, match="checksum mismatch"):
            service.load_artifact(
                artifact.uri,
                "sweep_results",
                expected_sha256="0" * 64,
                expected_byte_size=artifact.byte_size,
            )

        with pytest.raises(ValueError, match="byte size mismatch"):
            service.load_artifact(
                artifact.uri,
                "sweep_results",
                expected_byte_size=(artifact.byte_size or 0) + 1,
            )


def test_artifact_service_strict_mode_disallows_local(monkeypatch, temp_artifact_dir):
    """Strict mode should fail fast when local backend is selected."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.setenv("ARTIFACT_RUNTIME_ENV", "stage")
    monkeypatch.setenv("ARTIFACT_STORAGE_STRICT_MODE", "true")

    with pytest.raises(ValueError, match="Local artifact backend is disallowed"):
        ArtifactService(repository=None, artifact_dir=temp_artifact_dir)


def test_artifact_service_uses_env_scoped_container(monkeypatch, temp_artifact_dir):
    """Container selection uses explicit AZURE_STORAGE_CONTAINER_* when set."""
    fake_backend = MagicMock()
    fake_backend.backend_type = "azure_blob"
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setenv("ARTIFACT_RUNTIME_ENV", "stage")
    monkeypatch.setenv("ARTIFACT_AUTH_MODE", "managed_identity")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_STAGE", "artifacts-stage")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")
    monkeypatch.setenv("AZURE_STORAGE_PREFIX", "study-query")

    with patch(
        "study_query_llm.storage.factory.StorageBackendFactory.create",
        return_value=fake_backend,
    ) as mock_create:
        service = ArtifactService(repository=None, artifact_dir=temp_artifact_dir)
        assert service.storage is fake_backend
        kwargs = mock_create.call_args.kwargs
        assert kwargs["container_name"] == "artifacts-stage"
        assert kwargs["auth_mode"] == "managed_identity"
        assert kwargs["runtime_env"] == "stage"
        assert kwargs["blob_prefix"] == "study-query"


def test_artifact_service_derives_container_from_base(monkeypatch, temp_artifact_dir):
    """Container is derived as {AZURE_STORAGE_CONTAINER}-{lane} when lane vars are unset."""
    fake_backend = MagicMock()
    fake_backend.backend_type = "azure_blob"
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setenv("ARTIFACT_RUNTIME_ENV", "stage")
    monkeypatch.setenv("ARTIFACT_AUTH_MODE", "managed_identity")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER", "artifacts")
    monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://example.blob.core.windows.net")
    monkeypatch.delenv("AZURE_STORAGE_CONTAINER_STAGE", raising=False)

    with patch(
        "study_query_llm.storage.factory.StorageBackendFactory.create",
        return_value=fake_backend,
    ) as mock_create:
        service = ArtifactService(repository=None, artifact_dir=temp_artifact_dir)
        assert service.storage is fake_backend
        kwargs = mock_create.call_args.kwargs
        assert kwargs["container_name"] == "artifacts-stage"
        assert kwargs["runtime_env"] == "stage"


def test_artifact_service_strict_mode_raises_when_azure_unavailable(
    monkeypatch, temp_artifact_dir
):
    """Strict mode should raise instead of silently falling back."""
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "azure_blob")
    monkeypatch.setenv("ARTIFACT_RUNTIME_ENV", "prod")
    monkeypatch.setenv("ARTIFACT_AUTH_MODE", "connection_string")
    monkeypatch.setenv("AZURE_STORAGE_CONTAINER_PROD", "artifacts-prod")
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "")

    with patch(
        "study_query_llm.storage.factory.StorageBackendFactory.create",
        side_effect=ValueError("missing credentials"),
    ):
        with pytest.raises(RuntimeError, match="strict mode"):
            ArtifactService(repository=None, artifact_dir=temp_artifact_dir)


def test_artifact_service_operation_counters(db_connection, temp_artifact_dir):
    """Write/read operations should be reflected in local counters."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        service = ArtifactService(repository=repo, artifact_dir=temp_artifact_dir)
        run_id = repo.create_group(group_type="clustering_run", name="test_run")
        artifact_id = service.store_sweep_results(
            run_id=run_id,
            sweep_results={"by_k": {2: {"labels": [0, 1]}}},
            step_name="sweep_complete",
        )
        from study_query_llm.db.models_v2 import CallArtifact

        artifact = session.query(CallArtifact).filter_by(id=artifact_id).first()
        assert artifact is not None
        service.load_artifact(
            artifact.uri,
            "sweep_results",
            expected_sha256=(artifact.metadata_json or {}).get("sha256"),
            expected_byte_size=artifact.byte_size,
        )

        counts = service.get_operation_counts()
        assert counts.get("write.success", 0) >= 1
        assert counts.get("read.success", 0) >= 1

"""
Tests for ingest_sweep_to_db script.

Covers call_artifacts mode, idempotency, URI/backend guard, and session safety.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_ingest_module():
    """Load ingest_sweep_to_db as a module for unit testing."""
    script_path = PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"
    spec = importlib.util.spec_from_file_location("ingest_sweep", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ingest_sweep"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_script_has_source_mode_flags():
    """Script defines --source-mode and --dry-run."""
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"),
            "--help",
        ],
        env={**os.environ, "DATABASE_URL": "sqlite:///:memory:"},
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "--source-mode" in out
    assert "call_artifacts" in out
    assert "local_pkl" in out
    assert "--dry-run" in out


def test_call_artifacts_dry_run_no_artifacts_exits():
    """call_artifacts mode with no artifacts exits 1."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"),
                "--source-mode",
                "call_artifacts",
                "--dry-run",
            ],
            env=env,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 1
        assert "No sweep_results CallArtifact" in (result.stdout + result.stderr)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_call_artifacts_dry_run_with_seeded_artifact():
    """call_artifacts --dry-run with seeded CallArtifact runs and prints."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        # Seed artifact and run via DB
        from study_query_llm.db.connection_v2 import DatabaseConnectionV2
        from study_query_llm.db.models_v2 import CallArtifact
        from study_query_llm.db.raw_call_repository import RawCallRepository
        from study_query_llm.services.artifact_service import ArtifactService
        from study_query_llm.services.provenance_service import ProvenanceService

        db = DatabaseConnectionV2(f"sqlite:///{db_path}", enable_pgvector=False)
        db.init_db()

        artifact_dir = tempfile.mkdtemp()
        try:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                prov = ProvenanceService(repo)
                run_id = prov.create_run_group(
                    algorithm="cosine_kllmeans_no_pca",
                    config={"test": True},
                    name="seed_run",
                )
                svc = ArtifactService(repository=repo, artifact_dir=artifact_dir)
                sweep_data = {
                    "dist": [[0.0, 0.5], [0.5, 0.0]],
                    "by_k": {
                        "2": {
                            "labels": [0, 1],
                            "labels_all": [[0, 1]],
                            "objectives": [0.5],
                            "representatives": ["a", "b"],
                        },
                    },
                }
                aid = svc.store_sweep_results(
                    run_id=run_id,
                    sweep_results=sweep_data,
                    step_name="sweep_complete",
                    metadata={
                        "run_key": "test_dry_run_key",
                        "dataset": "test",
                        "embedding_engine": "test_eng",
                        "summarizer": "None",
                        "n_restarts": 1,
                        "n_samples": 2,
                        "data_type": "sweep",
                    },
                )
                assert aid > 0

            env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"),
                    "--source-mode",
                    "call_artifacts",
                    "--dry-run",
                ],
                env=env,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result.returncode == 0
            out = result.stdout + result.stderr
            assert "DRY RUN" in out or "Would ingest" in out
            assert "test_dry_run_key" in out
        finally:
            import shutil
            shutil.rmtree(artifact_dir, ignore_errors=True)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_call_artifacts_idempotent():
    """call_artifacts mode does not create duplicate runs for same run_key."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        from study_query_llm.db.connection_v2 import DatabaseConnectionV2
        from study_query_llm.db.models_v2 import CallArtifact, Group
        from study_query_llm.db.raw_call_repository import RawCallRepository
        from study_query_llm.services.artifact_service import ArtifactService
        from study_query_llm.services.provenance_service import ProvenanceService
        from sqlalchemy import text as sa_text

        db = DatabaseConnectionV2(f"sqlite:///{db_path}", enable_pgvector=False)
        db.init_db()

        artifact_dir = tempfile.mkdtemp()
        try:
            with db.session_scope() as session:
                repo = RawCallRepository(session)
                prov = ProvenanceService(repo)
                run_id = prov.create_run_group(
                    algorithm="cosine_kllmeans_no_pca",
                    config={"test": True},
                    name="seed_run",
                )
                svc = ArtifactService(repository=repo, artifact_dir=artifact_dir)
                sweep_data = {
                    "dist": [[0.0, 0.5], [0.5, 0.0]],
                    "by_k": {
                        "2": {
                            "labels": [0, 1],
                            "labels_all": [[0, 1]],
                            "objectives": [0.5],
                            "representatives": ["a", "b"],
                        },
                    },
                }
                svc.store_sweep_results(
                    run_id=run_id,
                    sweep_results=sweep_data,
                    step_name="sweep_complete",
                    metadata={
                        "run_key": "test_idempotent_key",
                        "dataset": "test",
                        "embedding_engine": "test_eng",
                        "summarizer": "None",
                        "n_restarts": 1,
                        "n_samples": 2,
                        "data_type": "sweep",
                    },
                )

            env = {**os.environ, "DATABASE_URL": f"sqlite:///{db_path}"}
            result1 = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"),
                    "--source-mode",
                    "call_artifacts",
                ],
                env=env,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result1.returncode == 0

            result2 = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "ingest_sweep_to_db.py"),
                    "--source-mode",
                    "call_artifacts",
                ],
                env=env,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
            )
            assert result2.returncode == 0

            with db.session_scope() as session:
                runs = (
                    session.query(Group)
                    .filter(
                        Group.group_type == "clustering_run",
                        sa_text("metadata_json->>'run_key' = :rk"),
                    )
                    .params(rk="test_idempotent_key")
                    .all()
                )
                assert len(runs) == 1
        finally:
            import shutil
            shutil.rmtree(artifact_dir, ignore_errors=True)
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass


def test_uri_backend_mismatch_guard_blob_with_local():
    """_assert_uri_backend_compatible raises when blob URI with local backend."""
    mod = _load_ingest_module()
    from study_query_llm.storage.local import LocalStorageBackend

    class MockArtifactService:
        storage = LocalStorageBackend(base_dir=tempfile.mkdtemp())

    uri = "https://storage.blob.core.windows.net/artifacts/1/sweep.json"
    with pytest.raises(ValueError, match="Azure Blob.*backend is local"):
        mod._assert_uri_backend_compatible(uri, MockArtifactService())


def test_uri_backend_compatible_local():
    """_assert_uri_backend_compatible passes for local URI with local backend."""
    mod = _load_ingest_module()
    from study_query_llm.storage.local import LocalStorageBackend

    class MockArtifactService:
        storage = LocalStorageBackend(base_dir=tempfile.mkdtemp())

    uri = "/artifacts/1/sweep_complete/sweep_results.json"
    mod._assert_uri_backend_compatible(uri, MockArtifactService())

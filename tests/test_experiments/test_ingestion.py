"""
Tests for ingestion module (ingest_result_to_db, run_key_exists_in_db).
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.experiments.ingestion import ingest_result_to_db, run_key_exists_in_db
from study_query_llm.algorithms.sweep import SweepResult, SweepConfig, run_sweep


@pytest.fixture
def db_connection():
    """In-memory SQLite for tests."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


@pytest.fixture
def temp_artifact_dir():
    """Temporary artifact directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_run_key_exists_in_db(db_connection):
    """run_key_exists_in_db returns False when key absent, True when present."""
    assert run_key_exists_in_db(db_connection, "nonexistent_key") is False

    # Create a run with run_key via ingest_result_to_db
    from study_query_llm.algorithms.sweep import SweepResult

    result = SweepResult(
        pca={},
        by_k={
            2: {
                "labels": [0, 1],
                "labels_all": [[0, 1]],
                "objectives": [0.5],
                "representatives": ["a", "b"],
            },
            3: {
                "labels": [0, 1, 0],
                "labels_all": [[0, 1, 0]],
                "objectives": [0.4],
                "representatives": ["a", "b", "c"],
            },
        },
    )
    metadata = {
        "benchmark_source": "test_dataset",
        "embedding_engine": "test_engine",
        "summarizer": "None",
        "n_restarts": 1,
        "actual_entry_count": 3,
    }
    run_id = ingest_result_to_db(
        result, metadata, np.array([0, 1, 0]), db_connection, "test_run_key_123"
    )
    assert run_id is not None
    assert run_key_exists_in_db(db_connection, "test_run_key_123") is True


def test_ingest_result_to_db_persists_artifact(db_connection, temp_artifact_dir):
    """ingest_result_to_db creates run and CallArtifact linked to run."""
    result = SweepResult(
        pca={},
        by_k={
            2: {
                "labels": [0, 1],
                "labels_all": [[0, 1]],
                "objectives": [0.5],
                "representatives": ["a", "b"],
            },
        },
    )
    metadata = {
        "benchmark_source": "test_ds",
        "embedding_engine": "test_eng",
        "summarizer": "None",
        "n_restarts": 1,
        "actual_entry_count": 2,
    }
    run_id = ingest_result_to_db(
        result, metadata, np.array([0, 1]), db_connection, "test_artifact_run_key"
    )
    assert run_id is not None

    from study_query_llm.db.models_v2 import Group, CallArtifact

    with db_connection.session_scope() as session:
        run = session.query(Group).filter_by(id=run_id).first()
        assert run is not None
        meta = run.metadata_json or {}
        assert "run_key" in meta
        assert meta["run_key"] == "test_artifact_run_key"

        artifacts = (
            session.query(CallArtifact)
            .filter(CallArtifact.artifact_type == "sweep_results")
            .all()
        )
        linked = [a for a in artifacts if (a.metadata_json or {}).get("group_id") == run_id]
        assert len(linked) >= 1
        assert linked[0].uri


def test_ingest_result_to_db_idempotent(db_connection):
    """Re-ingesting same run_key returns None (skipped)."""
    result = SweepResult(
        pca={},
        by_k={
            2: {
                "labels": [0, 1],
                "labels_all": [[0, 1]],
                "objectives": [0.5],
                "representatives": ["a", "b"],
            },
        },
    )
    metadata = {
        "benchmark_source": "idem_ds",
        "embedding_engine": "idem_eng",
        "summarizer": "None",
        "n_restarts": 1,
        "actual_entry_count": 2,
    }
    run_key = "idempotent_test_key"
    run_id1 = ingest_result_to_db(result, metadata, np.array([0, 1]), db_connection, run_key)
    assert run_id1 is not None

    run_id2 = ingest_result_to_db(result, metadata, np.array([0, 1]), db_connection, run_key)
    assert run_id2 is None

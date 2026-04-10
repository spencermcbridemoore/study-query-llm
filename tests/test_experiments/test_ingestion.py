"""
Tests for ingestion module (ingest_result_to_db, run_key_exists_in_db).
"""

import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group, ProvenancedRun
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.ingestion import ingest_result_to_db, run_key_exists_in_db
from study_query_llm.algorithms.sweep import SweepResult, SweepConfig, run_sweep
from study_query_llm.services.sweep_request_service import SweepRequestService


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


def test_ingest_result_to_db_sets_primary_snapshot_on_provenanced_run(db_connection):
    """Primary snapshot id is persisted into provenanced_runs for request-linked writes."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        req_service = SweepRequestService(repo)
        request_id = req_service.create_request(
            request_name="snapshot_primary_req",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 2, "n_restarts": 1},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=2,
            sweep_type="clustering",
            execution_mode="standalone",
        )
        primary_snapshot_id = repo.create_group(
            group_type="dataset_snapshot",
            name="dbpedia_2_seed42_labeled",
            metadata_json={
                "snapshot_name": "dbpedia_2_seed42_labeled",
                "source_dataset": "dbpedia",
                "sample_size": 2,
                "label_mode": "labeled",
                "sampling_method": "seeded",
            },
        )
        secondary_snapshot_id = repo.create_group(
            group_type="dataset_snapshot",
            name="dbpedia_2_seed99_labeled",
            metadata_json={
                "snapshot_name": "dbpedia_2_seed99_labeled",
                "source_dataset": "dbpedia",
                "sample_size": 2,
                "label_mode": "labeled",
                "sampling_method": "seeded",
            },
        )

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
        "benchmark_source": "dbpedia",
        "embedding_engine": "engine/a",
        "summarizer": "None",
        "n_restarts": 1,
        "actual_entry_count": 2,
        "request_group_id": int(request_id),
        # Deliberately unsorted with duplicates to verify deterministic normalization.
        "dataset_snapshot_ids": [
            int(secondary_snapshot_id),
            int(primary_snapshot_id),
            int(secondary_snapshot_id),
        ],
    }
    run_key = "snapshot_primary_test_key"
    run_id = ingest_result_to_db(result, metadata, np.array([0, 1]), db_connection, run_key)
    assert run_id is not None

    with db_connection.session_scope() as session:
        run_group = session.query(Group).filter(Group.id == int(run_id)).first()
        assert run_group is not None
        run_meta = dict(run_group.metadata_json or {})
        assert run_meta.get("dataset_snapshot_ids") == [
            int(primary_snapshot_id),
            int(secondary_snapshot_id),
        ]

        row = (
            session.query(ProvenancedRun)
            .filter(
                ProvenancedRun.request_group_id == int(request_id),
                ProvenancedRun.run_kind == "execution",
                ProvenancedRun.run_key == run_key,
            )
            .first()
        )
        assert row is not None
        assert str((row.metadata_json or {}).get("execution_role") or "") == "method_execution"
        assert int(row.input_snapshot_group_id or 0) == int(primary_snapshot_id)

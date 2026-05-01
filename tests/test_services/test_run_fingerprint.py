"""Tests for canonical run fingerprint generation and comparison."""

from __future__ import annotations

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.jobs import build_p0_baseline_snapshot
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import GROUP_TYPE_CLUSTERING_RUN
from study_query_llm.services.provenanced_run_service import (
    ProvenancedRunService,
    canonical_run_fingerprint,
    fingerprints_match,
)
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def test_canonical_run_fingerprint_deterministic():
    fp1, h1 = canonical_run_fingerprint(
        method_name="linear_probe",
        method_version="1.0",
        config_json={"C": 1.0, "penalty": "l2"},
        input_snapshot_group_id=42,
        manifest_hash="abc123",
        determinism_class="deterministic",
    )
    fp2, h2 = canonical_run_fingerprint(
        method_name="linear_probe",
        method_version="1.0",
        config_json={"penalty": "l2", "C": 1.0},
        input_snapshot_group_id=42,
        manifest_hash="abc123",
        determinism_class="deterministic",
    )
    assert h1 == h2
    assert fp1 == fp2


def test_canonical_run_fingerprint_strips_scheduling_keys():
    fp_with, h_with = canonical_run_fingerprint(
        method_name="knn",
        method_version="1.0",
        config_json={
            "k": 5,
            "metric": "cosine",
            "max_attempts": 3,
            "worker_id": "w-1",
            "lease_seconds": 600,
            "bundle_size": 10,
        },
        determinism_class="deterministic",
    )
    fp_without, h_without = canonical_run_fingerprint(
        method_name="knn",
        method_version="1.0",
        config_json={"k": 5, "metric": "cosine"},
        determinism_class="deterministic",
    )
    assert h_with == h_without
    assert fp_with == fp_without


def test_canonical_run_fingerprint_analysis_input_mode_affects_hash():
    base_config = {
        "snapshot_group_id": 42,
        "representation_type": "snapshot_only",
        "parameters": {"alpha": 1.0},
    }
    _, hash_without_mode = canonical_run_fingerprint(
        method_name="snapshot_method",
        method_version="1.0",
        config_json=base_config,
        input_snapshot_group_id=42,
        determinism_class="deterministic",
    )
    _, hash_with_mode = canonical_run_fingerprint(
        method_name="snapshot_method",
        method_version="1.0",
        config_json={
            **base_config,
            "analysis_input_mode": "snapshot_only",
        },
        input_snapshot_group_id=42,
        determinism_class="deterministic",
    )
    assert hash_without_mode != hash_with_mode


def test_p0_baseline_fingerprint_canary_matches():
    snapshot = build_p0_baseline_snapshot()
    canary = dict(snapshot.get("fingerprint_canary") or {})
    assert canary.get("matches") is True
    assert canary.get("scheduling_hash") == canary.get("canonical_hash")


def test_canonical_run_fingerprint_different_method_different_hash():
    _, h1 = canonical_run_fingerprint(method_name="knn", method_version="1.0")
    _, h2 = canonical_run_fingerprint(method_name="linear_probe", method_version="1.0")
    assert h1 != h2


def test_canonical_run_fingerprint_different_snapshot_different_hash():
    _, h1 = canonical_run_fingerprint(
        method_name="knn", input_snapshot_group_id=1
    )
    _, h2 = canonical_run_fingerprint(
        method_name="knn", input_snapshot_group_id=2
    )
    assert h1 != h2


def test_canonical_run_fingerprint_data_regime_matters():
    _, h1 = canonical_run_fingerprint(
        method_name="knn",
        data_regime={"label_mode": "labeled", "split_mode": "train_test"},
    )
    _, h2 = canonical_run_fingerprint(
        method_name="knn",
        data_regime={"label_mode": "unlabeled"},
    )
    assert h1 != h2


def test_fingerprints_match_identical():
    fp, _ = canonical_run_fingerprint(method_name="x", method_version="1")
    match, diff = fingerprints_match(fp, fp)
    assert match is True
    assert diff == {}


def test_fingerprints_match_different():
    fp_a, _ = canonical_run_fingerprint(method_name="x", method_version="1")
    fp_b, _ = canonical_run_fingerprint(method_name="y", method_version="1")
    match, diff = fingerprints_match(fp_a, fp_b)
    assert match is False
    assert "method_name" in diff


def test_fingerprints_match_none_handling():
    fp, _ = canonical_run_fingerprint(method_name="x")
    match_nn, diff_nn = fingerprints_match(None, None)
    assert match_nn is True
    match_na, diff_na = fingerprints_match(fp, None)
    assert match_na is False
    assert "_present" in diff_na


def test_record_method_execution_writes_fingerprint():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="fp_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3, "n_restarts": 2},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
            sweep_type="clustering",
        )
        req = svc.get_request(request_id)
        run_key = str(req["expected_run_keys"][0])
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r_fp",
            metadata_json={"run_key": run_key},
        )
        method_svc = MethodService(repo)
        method_id = method_svc.register_method(
            name="cosine_kllmeans_no_pca", version="1.0", description="t"
        )
        pr_svc = ProvenancedRunService(repo)
        pr_id = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=run_key,
            source_group_id=run_id,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3},
            determinism_class="pseudo_deterministic",
        )

        from study_query_llm.db.models_v2 import ProvenancedRun

        row = session.query(ProvenancedRun).filter_by(id=pr_id).first()
        assert row is not None
        assert row.fingerprint_hash is not None
        assert isinstance(row.fingerprint_json, dict)
        assert row.fingerprint_json["method_name"] == "cosine_kllmeans_no_pca"
        assert row.fingerprint_json["method_version"] == "1.0"
        assert row.fingerprint_json["determinism_class"] == "pseudo_deterministic"


def test_compare_run_fingerprints_same_method():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="fp_cmp",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a", "engine/b"],
                "summarizers": ["None"],
            },
            entry_max=50,
            sweep_type="clustering",
        )
        req = svc.get_request(request_id)
        keys = req["expected_run_keys"]
        run_a = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r_a",
            metadata_json={"run_key": keys[0]},
        )
        run_b = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r_b",
            metadata_json={"run_key": keys[1]},
        )
        method_svc = MethodService(repo)
        method_id = method_svc.register_method(
            name="cosine_kllmeans_no_pca", version="1.0", description="t"
        )
        pr_svc = ProvenancedRunService(repo)
        pr_a = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=keys[0],
            source_group_id=run_a,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3},
        )
        pr_b = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=keys[1],
            source_group_id=run_b,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3},
        )
        match, diff = pr_svc.compare_run_fingerprints(pr_a, pr_b)
        assert match is True
        assert diff == {}


def test_compare_run_fingerprints_different_method():
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="fp_cmp2",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a", "engine/b"],
                "summarizers": ["None"],
            },
            entry_max=50,
            sweep_type="clustering",
        )
        req = svc.get_request(request_id)
        keys = req["expected_run_keys"]
        run_a = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r_x",
            metadata_json={"run_key": keys[0]},
        )
        run_b = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="r_y",
            metadata_json={"run_key": keys[1]},
        )
        method_svc = MethodService(repo)
        mid_a = method_svc.register_method(name="method_a", version="1.0", description="a")
        mid_b = method_svc.register_method(name="method_b", version="1.0", description="b")
        pr_svc = ProvenancedRunService(repo)
        pr_a = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=keys[0],
            source_group_id=run_a,
            method_definition_id=mid_a,
            config_json={"k_min": 2},
        )
        pr_b = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=keys[1],
            source_group_id=run_b,
            method_definition_id=mid_b,
            config_json={"k_min": 2},
        )
        match, diff = pr_svc.compare_run_fingerprints(pr_a, pr_b)
        assert match is False
        assert "method_name" in diff

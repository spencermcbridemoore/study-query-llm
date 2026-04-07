"""
Tests for SweepRequestService — clustering_sweep_request lifecycle.

Covers: create_request, compute_progress, idempotent record_delivery,
fulfillment into clustering_sweep, and backward compatibility.
"""

import pytest
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.provenance_service import (
    ProvenanceService,
    GROUP_TYPE_CLUSTERING_RUN,
    GROUP_TYPE_CLUSTERING_SWEEP,
    GROUP_TYPE_CLUSTERING_SWEEP_REQUEST,
)
from study_query_llm.services.sweep_request_service import SweepRequestService
from study_query_llm.experiments.sweep_request_types import (
    SWEEP_TYPE_CLUSTERING,
    SWEEP_TYPE_MCQ,
    build_run_key,
    build_mcq_run_key,
    expand_parameter_axes,
    get_sweep_type_adapter,
    list_registered_sweep_types,
    normalize_summarizer,
    targets_to_run_keys,
    RunTarget,
    REQUEST_STATUS_REQUESTED,
    REQUEST_STATUS_RUNNING,
    REQUEST_STATUS_FULFILLED,
)


@pytest.fixture
def db_connection():
    """Fixture for in-memory SQLite database (v2 schema)."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# sweep_request_types (no DB)
# ---------------------------------------------------------------------------


def test_normalize_summarizer():
    """normalize_summarizer: None -> 'None', str unchanged."""
    assert normalize_summarizer(None) == "None"
    assert normalize_summarizer("gpt-4o") == "gpt-4o"


def test_build_run_key():
    """build_run_key produces deterministic format matching ingestion convention."""
    rk = build_run_key(
        dataset="dbpedia",
        embedding_engine="embed-v-4-0",
        summarizer="gpt-4o-mini",
        entry_max=300,
        n_restarts_suffix="50runs",
    )
    assert rk == "dbpedia_embed_v_4_0_gpt_4o_mini_300_50runs"


def test_build_run_key_summarizer_none():
    """build_run_key with summarizer None uses 'None' in key."""
    rk = build_run_key(
        dataset="yahoo",
        embedding_engine="text-embedding-3-small",
        summarizer=None,
        entry_max=300,
    )
    assert "None" in rk
    assert rk == "yahoo_text_embedding_3_small_None_300_50runs"


def test_expand_parameter_axes():
    """expand_parameter_axes produces deterministic RunTarget list."""
    axes = {
        "datasets": ["dbpedia", "yahoo"],
        "embedding_engines": ["embed-v-4-0"],
        "summarizers": [None, "gpt-4o"],
    }
    targets = expand_parameter_axes(axes, entry_max=300)
    assert len(targets) == 4  # 2 * 1 * 2
    assert targets[0].dataset == "dbpedia"
    assert targets[0].embedding_engine == "embed-v-4-0"
    assert targets[0].summarizer == "None"
    assert targets[1].summarizer == "gpt-4o"
    assert targets[2].dataset == "yahoo"


def test_targets_to_run_keys():
    """targets_to_run_keys converts RunTargets to run_key strings."""
    targets = [
        RunTarget("dbpedia", "e1", "s1", 300, "50runs"),
        RunTarget("yahoo", "e2", "s2", 300, "50runs"),
    ]
    keys = targets_to_run_keys(targets)
    assert len(keys) == 2
    assert keys[0] == "dbpedia_e1_s1_300_50runs"
    assert keys[1] == "yahoo_e2_s2_300_50runs"


def test_list_registered_sweep_types_contains_expected():
    """Registry includes clustering and mcq sweep types."""
    types = set(list_registered_sweep_types())
    assert SWEEP_TYPE_CLUSTERING in types
    assert SWEEP_TYPE_MCQ in types


def test_get_sweep_type_adapter_shapes():
    """Adapters expose request/run/sweep group type contracts."""
    clustering = get_sweep_type_adapter(SWEEP_TYPE_CLUSTERING)
    assert clustering.request_group_type == "clustering_sweep_request"
    assert clustering.run_group_type == "clustering_run"
    assert clustering.sweep_group_type == "clustering_sweep"

    mcq = get_sweep_type_adapter(SWEEP_TYPE_MCQ)
    assert mcq.request_group_type == "mcq_sweep_request"
    assert mcq.run_group_type == "mcq_run"
    assert mcq.sweep_group_type == "mcq_sweep"


def test_build_mcq_run_key_deterministic():
    """MCQ run key includes tuple dimensions and is deterministic."""
    key_1 = build_mcq_run_key(
        deployment="gpt-4o-mini",
        level="high school",
        subject="physics",
        options_per_question=5,
        questions_per_test=20,
        label_style="upper",
        spread_correct_answer_uniformly=False,
        samples_per_combo=50,
        template_version="v1",
    )
    key_2 = build_mcq_run_key(
        deployment="gpt-4o-mini",
        level="high school",
        subject="physics",
        options_per_question=5,
        questions_per_test=20,
        label_style="upper",
        spread_correct_answer_uniformly=False,
        samples_per_combo=50,
        template_version="v1",
    )
    assert key_1 == key_2
    assert key_1.startswith("mcq_")
    assert key_1.endswith("_v1")


# ---------------------------------------------------------------------------
# SweepRequestService
# ---------------------------------------------------------------------------


def test_create_request_computes_expected_keys(db_connection):
    """create_request stores correct expected_run_keys from parameter axes."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        axes = {
            "datasets": ["dbpedia"],
            "embedding_engines": ["embed-v-4-0"],
            "summarizers": [None, "gpt-4o"],
        }
        req_id = svc.create_request(
            request_name="test_sweep",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 10},
            parameter_axes=axes,
            entry_max=300,
        )

        req = svc.get_request(req_id)
        assert req is not None
        assert req["request_status"] == REQUEST_STATUS_REQUESTED
        expected = req["expected_run_keys"]
        assert len(expected) == 2
        assert "dbpedia_embed_v_4_0_None_300_50runs" in expected
        assert "dbpedia_embed_v_4_0_gpt_4o_300_50runs" in expected


def test_compute_progress_partial_deliveries(db_connection):
    """compute_progress reports completed vs missing from DB state."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        axes = {
            "datasets": ["dbpedia"],
            "embedding_engines": ["embed-v-4-0"],
            "summarizers": ["gpt-4o"],
        }
        req_id = svc.create_request(
            request_name="partial_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes=axes,
            entry_max=300,
        )

        expected_key = "dbpedia_embed_v_4_0_gpt_4o_300_50runs"
        progress = svc.compute_progress(req_id)
        assert progress["expected_count"] == 1
        assert progress["completed_count"] == 0
        assert progress["missing_count"] == 1
        assert progress["missing_run_keys"] == [expected_key]

        # Create a clustering_run with run_key
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="test_run",
            description="Test",
            metadata_json={"run_key": expected_key},
        )

        progress = svc.compute_progress(req_id)
        assert progress["completed_count"] == 1
        assert progress["missing_count"] == 0
        assert progress["completed_run_keys"] == [expected_key]
        assert progress["completed_run_ids"] == [run_id]


def test_record_delivery_idempotent(db_connection):
    """record_delivery creates link once; duplicate calls do not create duplicate links."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        from study_query_llm.db.models_v2 import GroupLink

        axes = {"datasets": ["dbpedia"], "embedding_engines": ["e1"], "summarizers": ["s1"]}
        req_id = svc.create_request(
            request_name="idempotent_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes=axes,
            entry_max=300,
        )

        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run1",
            metadata_json={"run_key": "dbpedia_e1_s1_300_50runs"},
        )

        ok1 = svc.record_delivery(req_id, run_id, "dbpedia_e1_s1_300_50runs")
        ok2 = svc.record_delivery(req_id, run_id, "dbpedia_e1_s1_300_50runs")

        assert ok1 is True
        assert ok2 is True

        links = session.query(GroupLink).filter_by(
            parent_group_id=req_id,
            child_group_id=run_id,
            link_type="contains",
        ).all()
        assert len(links) == 1


def test_finalize_if_fulfilled_creates_sweep(db_connection):
    """finalize_if_fulfilled creates clustering_sweep and marks request fulfilled when all runs delivered."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        axes = {"datasets": ["dbpedia"], "embedding_engines": ["e1"], "summarizers": ["s1"]}
        req_id = svc.create_request(
            request_name="fulfill_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2},
            parameter_axes=axes,
            entry_max=300,
        )

        run_key = "dbpedia_e1_s1_300_50runs"
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run1",
            metadata_json={"run_key": run_key},
        )

        # Not fulfilled yet (no record_delivery - progress uses DB lookup by run_key)
        sweep_id = svc.finalize_if_fulfilled(req_id)
        assert sweep_id is not None  # run exists in DB with run_key, so progress sees it

        req = svc.get_request(req_id)
        assert req["request_status"] == REQUEST_STATUS_FULFILLED
        assert req["linked_sweep_id"] == sweep_id

        sweep_group = repo.get_group_by_id(sweep_id)
        assert sweep_group is not None
        assert sweep_group.group_type == GROUP_TYPE_CLUSTERING_SWEEP


def test_finalize_if_fulfilled_preserves_sweep_semantics(db_connection):
    """Fulfilled sweep has runs linked via contains; request linked via generates."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        from study_query_llm.db.models_v2 import GroupLink

        axes = {"datasets": ["dbpedia"], "embedding_engines": ["e1"], "summarizers": ["s1"]}
        req_id = svc.create_request(
            request_name="semantics_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes=axes,
            entry_max=300,
        )

        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run1",
            metadata_json={"run_key": "dbpedia_e1_s1_300_50runs"},
        )

        sweep_id = svc.finalize_if_fulfilled(req_id, sweep_name="my_sweep")
        assert sweep_id is not None

        # sweep contains run
        contains = session.query(GroupLink).filter_by(
            parent_group_id=sweep_id,
            child_group_id=run_id,
            link_type="contains",
        ).first()
        assert contains is not None

        # request generates sweep
        generates = session.query(GroupLink).filter_by(
            parent_group_id=req_id,
            child_group_id=sweep_id,
            link_type="generates",
        ).first()
        assert generates is not None


def test_finalize_if_fulfilled_is_idempotent(db_connection):
    """Repeated finalize calls return same sweep and avoid duplicate sweeps."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        from study_query_llm.db.models_v2 import Group

        axes = {"datasets": ["dbpedia"], "embedding_engines": ["e1"], "summarizers": ["s1"]}
        req_id = svc.create_request(
            request_name="idempotent_finalize",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes=axes,
            entry_max=300,
        )

        repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run1",
            metadata_json={"run_key": "dbpedia_e1_s1_300_50runs"},
        )

        sweep_id_1 = svc.finalize_if_fulfilled(req_id, sweep_name="idempotent_sweep")
        sweep_id_2 = svc.finalize_if_fulfilled(req_id, sweep_name="idempotent_sweep")

        assert sweep_id_1 is not None
        assert sweep_id_2 == sweep_id_1

        sweeps = session.query(Group).filter(
            Group.group_type == GROUP_TYPE_CLUSTERING_SWEEP,
            Group.name == "idempotent_sweep",
        ).all()
        assert len(sweeps) == 1


def test_finalize_if_fulfilled_returns_none_when_missing(db_connection):
    """finalize_if_fulfilled returns None when runs are still missing."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        axes = {
            "datasets": ["dbpedia", "yahoo"],
            "embedding_engines": ["e1"],
            "summarizers": ["s1"],
        }
        req_id = svc.create_request(
            request_name="missing_test",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes=axes,
            entry_max=300,
        )

        # Only one of two runs present
        repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run1",
            metadata_json={"run_key": "dbpedia_e1_s1_300_50runs"},
        )

        sweep_id = svc.finalize_if_fulfilled(req_id)
        assert sweep_id is None


def test_list_requests(db_connection):
    """list_requests returns requests; status filter works."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        req_id1 = svc.create_request(
            request_name="req1",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={},
            parameter_axes={"datasets": ["dbpedia"], "embedding_engines": ["e1"], "summarizers": ["s1"]},
            entry_max=300,
        )

        all_reqs = svc.list_requests()
        assert len(all_reqs) >= 1
        names = [r["name"] for r in all_reqs]
        assert "req1" in names

        pending = svc.list_requests(include_fulfilled=False)
        assert any(r["name"] == "req1" for r in pending)


def test_get_request_nonexistent(db_connection):
    """get_request returns None for nonexistent or wrong-type group."""
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)

        assert svc.get_request(99999) is None

        # Create a non-request group
        run_id = repo.create_group(
            group_type=GROUP_TYPE_CLUSTERING_RUN,
            name="run",
            metadata_json={},
        )
        assert svc.get_request(run_id) is None

"""Tests for method recipes: hash semantics, registration idempotency, and
integration with the canonical run fingerprint via config_json."""

from __future__ import annotations

import copy

import pytest

from study_query_llm.algorithms.recipes import (
    CLUSTERING_COMPONENT_METHODS,
    COSINE_KLLMEANS_NO_PCA_RECIPE,
    build_composite_recipe,
    canonical_recipe_hash,
    ensure_composite_recipe,
    register_clustering_components,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import MethodDefinition
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenance_service import GROUP_TYPE_CLUSTERING_RUN
from study_query_llm.services.provenanced_run_service import (
    ProvenancedRunService,
    canonical_run_fingerprint,
)
from study_query_llm.services.sweep_request_service import SweepRequestService


def _db() -> DatabaseConnectionV2:
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


# ---------------------------------------------------------------------------
# Recipe hash
# ---------------------------------------------------------------------------


def test_canonical_recipe_hash_deterministic():
    """Same recipe, same hash, across repeated calls."""
    recipe = build_composite_recipe("cosine_kllmeans_no_pca")
    h1 = canonical_recipe_hash(recipe)
    h2 = canonical_recipe_hash(recipe)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_canonical_recipe_hash_independent_of_dict_key_order():
    """Reordering keys within a stage does not change the hash."""
    recipe = build_composite_recipe("cosine_kllmeans_no_pca")
    shuffled = {
        "notes": recipe["notes"],
        "stages": [
            {
                "params": dict(reversed(list(stage["params"].items()))),
                "role": stage["role"],
                "version": stage["version"],
                "name": stage["name"],
            }
            for stage in recipe["stages"]
        ],
        "recipe_version": recipe["recipe_version"],
    }
    assert canonical_recipe_hash(recipe) == canonical_recipe_hash(shuffled)


def test_canonical_recipe_hash_sensitive_to_stage_param_change():
    """Changing any stage param changes the hash."""
    base = build_composite_recipe("cosine_kllmeans_no_pca")
    mutated = copy.deepcopy(base)
    for stage in mutated["stages"]:
        if stage["name"] == "k_llmmeans":
            stage["params"]["max_iter"] = 999
            break
    else:  # pragma: no cover - defensive
        pytest.fail("k_llmmeans stage not found in canonical recipe")
    assert canonical_recipe_hash(base) != canonical_recipe_hash(mutated)


def test_canonical_recipe_hash_sensitive_to_stage_version_change():
    """Bumping a component version changes the hash."""
    base = build_composite_recipe("cosine_kllmeans_no_pca")
    mutated = copy.deepcopy(base)
    mutated["stages"][0]["version"] = "9.9"
    assert canonical_recipe_hash(base) != canonical_recipe_hash(mutated)


def test_canonical_recipe_hash_sensitive_to_stage_order():
    """Reordering stages changes the hash (pipelines are ordered)."""
    base = build_composite_recipe("cosine_kllmeans_no_pca")
    mutated = copy.deepcopy(base)
    mutated["stages"].reverse()
    assert canonical_recipe_hash(base) != canonical_recipe_hash(mutated)


def test_build_composite_recipe_returns_copy():
    """Mutating the returned recipe must not affect the canonical source."""
    r1 = build_composite_recipe("cosine_kllmeans_no_pca")
    r1["stages"].append({"name": "zzz", "version": "0", "role": "bogus", "params": {}})
    r2 = build_composite_recipe("cosine_kllmeans_no_pca")
    assert r2 == COSINE_KLLMEANS_NO_PCA_RECIPE
    assert r1 != r2


def test_build_composite_recipe_unknown_raises():
    with pytest.raises(KeyError):
        build_composite_recipe("does_not_exist")


# ---------------------------------------------------------------------------
# Registration idempotency
# ---------------------------------------------------------------------------


def test_register_clustering_components_idempotent():
    """Calling the registrar twice does not proliferate rows."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        first = register_clustering_components(method_svc)
        second = register_clustering_components(method_svc)

        assert first == second
        count = session.query(MethodDefinition).count()
        assert count == len(CLUSTERING_COMPONENT_METHODS)


def test_ensure_composite_recipe_fresh_registration_persists_recipe():
    """Fresh composite registration writes the canonical recipe_json."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        register_clustering_components(method_svc)
        method_id = ensure_composite_recipe(
            method_svc,
            "cosine_kllmeans_no_pca",
            composite_version="1.0",
        )
        row = session.query(MethodDefinition).filter_by(id=method_id).first()
        assert row is not None
        assert row.recipe_json == COSINE_KLLMEANS_NO_PCA_RECIPE


def test_ensure_composite_recipe_backfills_missing_recipe_in_place():
    """Composite row registered without a recipe gets the recipe attached in-place."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        pre_existing_id = method_svc.register_method(
            name="cosine_kllmeans_no_pca",
            version="1.0",
            description="pre-existing row without recipe_json",
        )

        resolved_id = ensure_composite_recipe(
            method_svc,
            "cosine_kllmeans_no_pca",
            composite_version="1.0",
        )
        assert resolved_id == pre_existing_id

        row = session.query(MethodDefinition).filter_by(id=pre_existing_id).first()
        assert row is not None
        assert row.recipe_json == COSINE_KLLMEANS_NO_PCA_RECIPE
        count = (
            session.query(MethodDefinition)
            .filter_by(name="cosine_kllmeans_no_pca")
            .count()
        )
        assert count == 1


def test_ensure_composite_recipe_leaves_divergent_recipe_alone():
    """If a stored recipe differs, do not overwrite silently."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)

        divergent = {"recipe_version": "v0", "stages": [], "notes": "custom"}
        method_id = method_svc.register_method(
            name="cosine_kllmeans_no_pca",
            version="1.0",
            recipe_json=divergent,
        )

        resolved_id = ensure_composite_recipe(
            method_svc,
            "cosine_kllmeans_no_pca",
            composite_version="1.0",
        )
        assert resolved_id == method_id

        row = session.query(MethodDefinition).filter_by(id=method_id).first()
        assert row is not None
        assert row.recipe_json == divergent


# ---------------------------------------------------------------------------
# Fingerprint integration
# ---------------------------------------------------------------------------


def test_recipe_hash_changes_canonical_run_fingerprint():
    """Injecting recipe_hash into config_json alters the canonical fingerprint."""
    config_without = {"k_min": 2, "k_max": 5}
    config_with = dict(config_without)
    config_with["recipe_hash"] = canonical_recipe_hash(
        build_composite_recipe("cosine_kllmeans_no_pca")
    )

    _, h_without = canonical_run_fingerprint(
        method_name="cosine_kllmeans_no_pca",
        method_version="1.0",
        config_json=config_without,
        determinism_class="pseudo_deterministic",
    )
    _, h_with = canonical_run_fingerprint(
        method_name="cosine_kllmeans_no_pca",
        method_version="1.0",
        config_json=config_with,
        determinism_class="pseudo_deterministic",
    )
    assert h_without != h_with


def test_recipe_hash_is_not_classified_as_scheduling_key():
    """recipe_hash survives _strip_scheduling_keys (i.e. feeds the fingerprint)."""
    base = {"k_min": 2, "k_max": 5, "recipe_hash": "abc"}
    other = dict(base)
    other["recipe_hash"] = "def"

    _, h_base = canonical_run_fingerprint(
        method_name="m", method_version="1.0", config_json=base
    )
    _, h_other = canonical_run_fingerprint(
        method_name="m", method_version="1.0", config_json=other
    )
    assert h_base != h_other


def test_record_method_execution_absorbs_recipe_hash():
    """End-to-end: recording a method execution with recipe_hash in config_json
    stores a fingerprint that differs from the same execution without it."""
    db = _db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        request_id = svc.create_request(
            request_name="recipe_fp_test",
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
            name="recipe_fp_run",
            metadata_json={"run_key": run_key},
        )

        method_svc = MethodService(repo)
        register_clustering_components(method_svc)
        method_id = ensure_composite_recipe(
            method_svc,
            "cosine_kllmeans_no_pca",
            composite_version="1.0",
        )

        pr_svc = ProvenancedRunService(repo)
        recipe_hash = canonical_recipe_hash(
            build_composite_recipe("cosine_kllmeans_no_pca")
        )
        pr_id_with = pr_svc.record_method_execution(
            request_group_id=request_id,
            run_key=run_key,
            source_group_id=run_id,
            method_definition_id=method_id,
            config_json={"k_min": 2, "k_max": 3, "recipe_hash": recipe_hash},
            determinism_class="pseudo_deterministic",
        )

        from study_query_llm.db.models_v2 import ProvenancedRun

        row_with = session.query(ProvenancedRun).filter_by(id=pr_id_with).first()
        assert row_with is not None
        assert row_with.fingerprint_hash is not None
        assert row_with.config_json.get("recipe_hash") == recipe_hash

        _, expected_hash_without_recipe = canonical_run_fingerprint(
            method_name="cosine_kllmeans_no_pca",
            method_version="1.0",
            config_json={"k_min": 2, "k_max": 3},
            input_snapshot_group_id=None,
            determinism_class="pseudo_deterministic",
        )
        assert row_with.fingerprint_hash != expected_hash_without_recipe

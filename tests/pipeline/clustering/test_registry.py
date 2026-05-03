"""Tests for clustering runtime registry."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from study_query_llm.algorithms.recipes import (
    COMPOSITE_RECIPES,
    build_composite_recipe,
    canonical_recipe_hash,
)
from study_query_llm.pipeline.analyze import _synthesize_v1_pipeline_for_bundled_method
from study_query_llm.pipeline.clustering.registry import (
    DEPRECATED_LEGACY_CLUSTERING_METHODS,
    REPRESENTATION_FULL,
    AlgorithmSpec,
    get_algorithm_spec,
    iter_algorithm_specs,
    raise_if_deprecated_clustering_method,
    resolve_algorithm_runner,
    resolve_registry_method_name,
)
from study_query_llm.pipeline.clustering.runner_common import synthesize_fixed_bundled_payload

EXPECTED_METHOD_NAMES = frozenset(
    {
        "agglomerative+fixed-k",
        "agglomerative+normalize+fixed-k",
        "agglomerative+pca+fixed-k",
        "hdbscan+fixed",
        "hdbscan+normalize+fixed",
        "hdbscan+pca+fixed",
        "hdbscan+normalize+pca+fixed",
        "kmeans+normalize+pca+sweep",
        "gmm+normalize+pca+sweep",
        "kmeans+fixed-k",
        "kmeans+normalize+fixed-k",
        "kmeans+pca+fixed-k",
        "kmeans+normalize+pca+fixed-k",
        "spherical-kmeans+approx+fixed-k",
        "spherical-kmeans+approx+pca+fixed-k",
        "gmm+fixed-k",
        "gmm+normalize+fixed-k",
        "gmm+pca+fixed-k",
        "gmm+normalize+pca+fixed-k",
        "dbscan+fixed-eps",
        "dbscan+normalize+fixed-eps",
        "dbscan+pca+fixed-eps",
        "dbscan+normalize+pca+fixed-eps",
    }
)


@pytest.fixture(autouse=True)
def _fake_hdbscan_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHDBSCAN:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

        def fit_predict(self, x: np.ndarray) -> np.ndarray:
            n = int(x.shape[0])
            return np.zeros(n, dtype=np.int64)

    monkeypatch.setitem(
        sys.modules,
        "hdbscan",
        types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN),
    )


def test_registry_contains_expected_builtin_methods() -> None:
    names = {spec.method_name for spec in iter_algorithm_specs()}
    assert names == EXPECTED_METHOD_NAMES


def test_registry_lookup_is_case_insensitive() -> None:
    upper = get_algorithm_spec("HDBSCAN+FIXED")
    mixed = get_algorithm_spec("KMeans+Normalize+PCA+Sweep")
    assert upper is not None
    assert mixed is not None
    assert upper.method_name == "hdbscan+fixed"
    assert mixed.method_name == "kmeans+normalize+pca+sweep"


def test_registry_unknown_method_returns_none() -> None:
    assert get_algorithm_spec("does_not_exist") is None
    assert resolve_algorithm_runner("does_not_exist") is None


def test_registry_legacy_method_names_no_longer_resolve() -> None:
    for legacy_name in ("hdbscan", "kmeans+silhouette+kneedle", "gmm+bic+argmin"):
        assert get_algorithm_spec(legacy_name) is None, (
            f"legacy method {legacy_name!r} should not have a registry spec "
            "after the Slice 1.5 cleanup"
        )


def test_deprecated_legacy_clustering_methods_set_matches_registry_history() -> None:
    assert DEPRECATED_LEGACY_CLUSTERING_METHODS == frozenset(
        {"hdbscan", "kmeans+silhouette+kneedle", "gmm+bic+argmin"}
    )


@pytest.mark.parametrize(
    ("legacy_name", "new_name"),
    [
        ("hdbscan", "hdbscan+fixed"),
        ("kmeans+silhouette+kneedle", "kmeans+normalize+pca+sweep"),
        ("gmm+bic+argmin", "gmm+normalize+pca+sweep"),
    ],
)
def test_raise_if_deprecated_clustering_method_rejects_legacy_names(
    legacy_name: str,
    new_name: str,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        raise_if_deprecated_clustering_method(legacy_name)
    message = str(excinfo.value)
    assert "Slice 1.5" in message
    assert new_name in message


@pytest.mark.parametrize(
    "name",
    [
        "hdbscan+fixed",
        "kmeans+normalize+pca+sweep",
        "gmm+normalize+pca+sweep",
        "agglomerative+fixed-k",
        "some_custom_method",
    ],
)
def test_raise_if_deprecated_clustering_method_no_op_for_non_legacy_names(
    name: str,
) -> None:
    raise_if_deprecated_clustering_method(name)


def test_registry_alias_resolution_maps_to_canonical_method_name() -> None:
    assert (
        resolve_registry_method_name("kmeans_silhouette_kneedle")
        == "kmeans+normalize+pca+sweep"
    )
    assert resolve_registry_method_name("gmm_bic_argmin") == "gmm+normalize+pca+sweep"
    assert resolve_registry_method_name("hdbscan") == "hdbscan+fixed"


def test_all_specs_carry_parameters_schema() -> None:
    for spec in iter_algorithm_specs():
        assert isinstance(spec.parameters_schema, dict)
        assert spec.parameters_schema.get("type") == "object"


def test_baseline_identity_tuple_unchanged() -> None:
    """Lock the 10-field Baseline Identity Tuple for the 4 shipped baselines."""
    frozen: dict[str, tuple[Any, ...]] = {
        "agglomerative+fixed-k": (
            "agglomerative+fixed-k",
            resolve_algorithm_runner("agglomerative+fixed-k"),
            "single_fit",
            True,
            False,
            "none",
            "agglomerative",
            "deterministic",
            (),
            frozenset({REPRESENTATION_FULL}),
        ),
        "hdbscan+fixed": (
            "hdbscan+fixed",
            resolve_algorithm_runner("hdbscan+fixed"),
            "single_fit",
            True,
            False,
            "none",
            "hdbscan",
            "non_deterministic",
            ("hdbscan",),
            frozenset({REPRESENTATION_FULL}),
        ),
        "kmeans+normalize+pca+sweep": (
            "kmeans+normalize+pca+sweep",
            resolve_algorithm_runner("kmeans+normalize+pca+sweep"),
            "sweep_select",
            True,
            False,
            "none",
            "kmeans",
            "pseudo_deterministic",
            ("kmeans_silhouette_kneedle",),
            frozenset({REPRESENTATION_FULL}),
        ),
        "gmm+normalize+pca+sweep": (
            "gmm+normalize+pca+sweep",
            resolve_algorithm_runner("gmm+normalize+pca+sweep"),
            "sweep_select",
            True,
            False,
            "none",
            "gmm",
            "pseudo_deterministic",
            ("gmm_bic_argmin",),
            frozenset({REPRESENTATION_FULL}),
        ),
    }
    for name, expected in frozen.items():
        spec = get_algorithm_spec(name)
        assert spec is not None
        actual = (
            spec.method_name,
            spec.runner,
            spec.fit_mode,
            spec.requires_embeddings,
            spec.supports_snapshot_only,
            spec.provenance_envelope,
            spec.base_algorithm,
            spec.default_determinism_class,
            spec.strategy_aliases,
            spec.allowed_representations,
        )
        assert actual == expected, f"baseline drift for {name}"


def _build_contract_params(spec: AlgorithmSpec) -> dict[str, Any]:
    emb_dim = 8
    n_samples = 12
    mn = spec.method_name

    if spec.fit_mode == "sweep_select":
        p: dict[str, Any] = {"pca_n_components": 50}
        syn = _synthesize_v1_pipeline_for_bundled_method(
            method_name=mn,
            parameters=p,
            embedding_dim=emb_dim,
            n_samples=n_samples,
        )
        p["_v1_pipeline_resolved"] = syn["pipeline_resolved"]
        p["_v1_pipeline_effective"] = syn["pipeline_effective"]
        return p

    p = {}
    chain = spec.preprocessing_chain
    if "pca" in chain:
        p["pca_n_components"] = min(5, max(1, emb_dim - 1))

    if spec.base_algorithm == "hdbscan":
        p["min_cluster_size"] = 2
        p["min_samples"] = 1
        p["random_state"] = 0
    elif spec.base_algorithm == "dbscan":
        p["eps"] = 2.0
        p["min_samples"] = 2
    elif spec.base_algorithm == "agglomerative":
        p["k"] = 2
    elif spec.base_algorithm in ("kmeans", "spherical-kmeans"):
        p["k"] = 2
        p["random_state"] = 0
    elif spec.base_algorithm == "gmm":
        p["k"] = 2
        p["random_state"] = 0
    else:
        raise AssertionError(spec)

    if len(chain) > 0:
        sr, se = synthesize_fixed_bundled_payload(
            method_name=mn,
            parameters=p,
            embedding_dim=emb_dim,
            n_samples=n_samples,
        )
        p["_v1_pipeline_resolved"] = sr
        p["_v1_pipeline_effective"] = se
    return p


@pytest.mark.parametrize("spec", list(iter_algorithm_specs()), ids=lambda s: s.method_name)
def test_output_contract_for_all_specs(spec: AlgorithmSpec) -> None:
    rng = np.random.default_rng(0)
    emb_dim = 8
    n_samples = 12
    matrix = rng.standard_normal((n_samples, emb_dim))
    params = _build_contract_params(spec)
    runner = spec.runner
    result = runner(
        method_name=spec.method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(n_samples)],
        parameters=params,
    )
    assert result["scalar_results"]
    labels_obj = None
    for key, val in result["structured_results"].items():
        if "label" in key.lower():
            labels_obj = val
            break
    assert labels_obj is not None
    cl = labels_obj.get("cluster_labels")
    assert isinstance(cl, list)
    assert len(cl) == n_samples

    if spec.method_name in COMPOSITE_RECIPES:
        r1 = canonical_recipe_hash(build_composite_recipe(spec.method_name))
        r2 = canonical_recipe_hash(build_composite_recipe(spec.method_name))
        assert r1 == r2

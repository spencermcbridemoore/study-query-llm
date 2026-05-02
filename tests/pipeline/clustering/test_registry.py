"""Tests for clustering runtime registry."""

from __future__ import annotations

from study_query_llm.pipeline.clustering.registry import (
    get_algorithm_spec,
    iter_algorithm_specs,
    resolve_algorithm_runner,
    resolve_registry_method_name,
)


def test_registry_contains_expected_builtin_methods() -> None:
    names = {spec.method_name for spec in iter_algorithm_specs()}
    assert names == {
        "agglomerative+fixed-k",
        "hdbscan",
        "kmeans+silhouette+kneedle",
        "gmm+bic+argmin",
        "hdbscan+fixed",
        "kmeans+normalize+pca+sweep",
        "gmm+normalize+pca+sweep",
    }


def test_registry_lookup_is_case_insensitive() -> None:
    upper = get_algorithm_spec("HDBSCAN")
    mixed = get_algorithm_spec("KMeans+Silhouette+Kneedle")
    assert upper is not None
    assert mixed is not None
    assert upper.method_name == "hdbscan"
    assert mixed.method_name == "kmeans+silhouette+kneedle"


def test_registry_unknown_method_returns_none() -> None:
    assert get_algorithm_spec("does_not_exist") is None
    assert resolve_algorithm_runner("does_not_exist") is None


def test_registry_alias_resolution_maps_to_canonical_method_name() -> None:
    """Slice 1.5: strategy aliases now resolve to the bundled-grammar method
    names (the aliases moved off the legacy v1-envelope specs and onto the new
    specs in PR2 so BANK77 strategy tokens continue to work but dispatch to
    the envelope=none paths).

    The literal token "hdbscan" is BOTH the canonical name of the legacy spec
    AND a strategy alias on the new "hdbscan+fixed" spec. The alias index is
    built by iteration order: legacy is inserted first, then the new spec
    overwrites the alias entry, so resolve_registry_method_name("hdbscan")
    returns "hdbscan+fixed". Direct registry lookup via get_algorithm_spec
    still returns the legacy spec when called with "hdbscan" (until PR3
    removes the legacy spec entirely).
    """
    assert resolve_registry_method_name("kmeans_silhouette_kneedle") == "kmeans+normalize+pca+sweep"
    assert resolve_registry_method_name("gmm_bic_argmin") == "gmm+normalize+pca+sweep"
    assert resolve_registry_method_name("hdbscan") == "hdbscan+fixed"

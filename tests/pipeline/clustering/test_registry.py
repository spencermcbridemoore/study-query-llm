"""Tests for clustering runtime registry."""

from __future__ import annotations

import pytest

from study_query_llm.pipeline.clustering.registry import (
    DEPRECATED_LEGACY_CLUSTERING_METHODS,
    get_algorithm_spec,
    iter_algorithm_specs,
    raise_if_deprecated_clustering_method,
    resolve_algorithm_runner,
    resolve_registry_method_name,
)


def test_registry_contains_expected_builtin_methods() -> None:
    """Slice 1.5: the registry holds only bundled-grammar (envelope=none)
    specs after the legacy v1-envelope methods (``hdbscan``,
    ``kmeans+silhouette+kneedle``, ``gmm+bic+argmin``) were removed from the
    spec table. Their legacy names are reachable only via
    ``DEPRECATED_LEGACY_CLUSTERING_METHODS`` + ``raise_if_deprecated_clustering_method``.
    """
    names = {spec.method_name for spec in iter_algorithm_specs()}
    assert names == {
        "agglomerative+fixed-k",
        "hdbscan+fixed",
        "kmeans+normalize+pca+sweep",
        "gmm+normalize+pca+sweep",
    }


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
    """Slice 1.5: legacy method names are physically absent from the registry;
    direct ``get_algorithm_spec`` lookup returns ``None`` so callers cannot
    bypass the deprecation guard via the registry.
    """
    for legacy_name in ("hdbscan", "kmeans+silhouette+kneedle", "gmm+bic+argmin"):
        assert get_algorithm_spec(legacy_name) is None, (
            f"legacy method {legacy_name!r} should not have a registry spec "
            "after the Slice 1.5 cleanup"
        )


def test_deprecated_legacy_clustering_methods_set_matches_registry_history() -> None:
    """Slice 1.5: the deprecation guard set is the canonical legacy-name list.

    The frozenset is the single source of truth for which method names the
    Slice 1.5 guard rejects. Pinning the set here so a future contributor
    cannot silently drop or extend it without an explicit, intentional change.
    """
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
    """The guard raises ValueError for each legacy name with a message that
    mentions Slice 1.5 and the corresponding bundled-grammar replacement."""
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
    """The guard is a no-op for any name not in the legacy set, including
    bundled-grammar names and arbitrary external method names."""
    raise_if_deprecated_clustering_method(name)


def test_registry_alias_resolution_maps_to_canonical_method_name() -> None:
    """Slice 1.5: strategy aliases live on the bundled-grammar specs so the
    BANK77 strategy CLI keeps the operator-facing tokens
    (``hdbscan``/``kmeans_silhouette_kneedle``/``gmm_bic_argmin``) stable
    while routing dispatch to the envelope=none method names.
    """
    assert resolve_registry_method_name("kmeans_silhouette_kneedle") == "kmeans+normalize+pca+sweep"
    assert resolve_registry_method_name("gmm_bic_argmin") == "gmm+normalize+pca+sweep"
    assert resolve_registry_method_name("hdbscan") == "hdbscan+fixed"

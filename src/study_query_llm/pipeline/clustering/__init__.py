"""Clustering provenance + canonical runner helpers for analyze stage.

See docs/living/METHOD_RECIPES.md (Bundled Clustering Subsystem) for the
subsystem definition, naming grammar, and output-schema contract.
"""

from __future__ import annotations

from .selection import argmin_choice, kneedle_choice
from .agglomerative_runner import run_agglomerative_fixed_k_analysis
from .kmeans_runner import run_kmeans_silhouette_kneedle_analysis
from .gmm_runner import run_gmm_bic_argmin_analysis
from .registry import (
    AlgorithmSpec,
    DEPRECATED_LEGACY_CLUSTERING_METHODS,
    get_algorithm_spec,
    iter_algorithm_specs,
    normalize_method_name,
    raise_if_deprecated_clustering_method,
    resolve_algorithm_runner,
    resolve_registry_method_name,
)

__all__ = [
    "AlgorithmSpec",
    "DEPRECATED_LEGACY_CLUSTERING_METHODS",
    "argmin_choice",
    "get_algorithm_spec",
    "iter_algorithm_specs",
    "kneedle_choice",
    "normalize_method_name",
    "raise_if_deprecated_clustering_method",
    "resolve_algorithm_runner",
    "resolve_registry_method_name",
    "run_agglomerative_fixed_k_analysis",
    "run_gmm_bic_argmin_analysis",
    "run_kmeans_silhouette_kneedle_analysis",
]

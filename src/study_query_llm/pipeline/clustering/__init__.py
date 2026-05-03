"""Clustering provenance + canonical runner helpers for analyze stage.

See docs/living/METHOD_RECIPES.md (Bundled Clustering Subsystem) for the
subsystem definition, naming grammar, and output-schema contract.
"""

from __future__ import annotations

from .grammar import parse_method_name
from .selection import argmin_choice, kneedle_choice
from .agglomerative_preproc_runner import run_agglomerative_preproc_fixed_k_analysis
from .agglomerative_runner import run_agglomerative_fixed_k_analysis
from .dbscan_fixed_eps_runner import run_dbscan_fixed_eps_analysis
from .gmm_fixed_k_runner import run_gmm_fixed_k_analysis
from .gmm_runner import run_gmm_bic_argmin_analysis
from .hdbscan_preproc_fixed_runner import run_hdbscan_preproc_fixed_analysis
from .kmeans_fixed_k_runner import run_kmeans_fixed_k_analysis
from .kmeans_runner import run_kmeans_silhouette_kneedle_analysis
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
    "parse_method_name",
    "raise_if_deprecated_clustering_method",
    "resolve_algorithm_runner",
    "resolve_registry_method_name",
    "run_agglomerative_fixed_k_analysis",
    "run_agglomerative_preproc_fixed_k_analysis",
    "run_dbscan_fixed_eps_analysis",
    "run_gmm_bic_argmin_analysis",
    "run_gmm_fixed_k_analysis",
    "run_hdbscan_preproc_fixed_analysis",
    "run_kmeans_fixed_k_analysis",
    "run_kmeans_silhouette_kneedle_analysis",
]

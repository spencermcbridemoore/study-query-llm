"""Clustering provenance + canonical runner helpers for analyze stage.

See docs/living/METHOD_RECIPES.md (Bundled Clustering Subsystem) for the
subsystem definition, naming grammar, and output-schema contract.
"""

from __future__ import annotations

from .schema import (
    CLUSTER_PIPELINE_OPERATION_TYPE,
    CLUSTER_PIPELINE_OPERATION_VERSION,
    V1_CLUSTERING_METHODS,
    base_algorithm_for_method,
    is_v1_clustering_method,
)
from .resolver import (
    ClusteringResolution,
    RuleSet,
    load_rule_set,
    resolve_clustering_resolution,
)
from .validators import validate_identity_contract, validate_post_selection, validate_pre_run
from .hashing import (
    build_effective_recipe_payload,
    build_pipeline_effective_hash,
    effective_stage_entries,
)
from .selection import argmin_choice, kneedle_choice
from .agglomerative_runner import run_agglomerative_fixed_k_analysis
from .kmeans_runner import run_kmeans_silhouette_kneedle_analysis
from .gmm_runner import run_gmm_bic_argmin_analysis
from .registry import (
    AlgorithmSpec,
    get_algorithm_spec,
    is_registry_v1_clustering_method,
    iter_algorithm_specs,
    normalize_method_name,
    resolve_algorithm_runner,
    resolve_registry_method_name,
)

__all__ = [
    "CLUSTER_PIPELINE_OPERATION_TYPE",
    "CLUSTER_PIPELINE_OPERATION_VERSION",
    "V1_CLUSTERING_METHODS",
    "ClusteringResolution",
    "AlgorithmSpec",
    "RuleSet",
    "argmin_choice",
    "base_algorithm_for_method",
    "build_effective_recipe_payload",
    "build_pipeline_effective_hash",
    "effective_stage_entries",
    "is_v1_clustering_method",
    "kneedle_choice",
    "load_rule_set",
    "resolve_clustering_resolution",
    "run_gmm_bic_argmin_analysis",
    "run_kmeans_silhouette_kneedle_analysis",
    "run_agglomerative_fixed_k_analysis",
    "get_algorithm_spec",
    "is_registry_v1_clustering_method",
    "iter_algorithm_specs",
    "normalize_method_name",
    "resolve_algorithm_runner",
    "resolve_registry_method_name",
    "validate_identity_contract",
    "validate_post_selection",
    "validate_pre_run",
]

"""Clustering provenance + canonical runner helpers for analyze stage."""

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
from .kmeans_runner import run_kmeans_silhouette_kneedle_analysis
from .gmm_runner import run_gmm_bic_argmin_analysis

__all__ = [
    "CLUSTER_PIPELINE_OPERATION_TYPE",
    "CLUSTER_PIPELINE_OPERATION_VERSION",
    "V1_CLUSTERING_METHODS",
    "ClusteringResolution",
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
    "validate_identity_contract",
    "validate_post_selection",
    "validate_pre_run",
]

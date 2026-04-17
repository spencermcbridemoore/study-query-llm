"""
Algorithm Core Library - PCA/KLLMeans sweep and clustering algorithms.

This module provides core algorithm implementations with minimal dependencies,
separate from notebooks/scripts. Designed for reuse and testing.
"""

from .dimensionality_reduction import mean_pool_tokens, pca_svd_project
from .clustering import (
    k_subspaces_kllmeans,
    k_llmmeans,
    kmeanspp_sample,
    select_representatives,
    adjusted_rand_index,
    pairwise_ari,
    silhouette_score_precomputed,
    coverage_fraction,
    compute_stability_metrics,
    ClusteringResult,
    StabilityMetrics,
)
from .sweep import SweepConfig, SweepResult, run_sweep
from .method_plugins import (
    available_method_plugins,
    run_fixed_k_plugin,
    run_unknown_k_plugin,
    UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR,
    UNKNOWN_K_STRATEGY_HDBSCAN,
)
from .recipes import (
    RECIPE_VERSION,
    CLUSTERING_COMPONENT_METHODS,
    COMPOSITE_RECIPES,
    COSINE_KLLMEANS_NO_PCA_RECIPE,
    canonical_recipe_hash,
    build_composite_recipe,
    register_clustering_components,
    ensure_composite_recipe,
)
from .text_classification_methods import (
    MATURITY_REGISTERED_ONLY,
    TEXT_CLASSIFICATION_METHODS,
    register_text_classification_methods,
)
from .canonical_configs import (
    CANONICAL_CONFIG_BUILDERS,
    CanonicalConfigBuilder,
    canonical_config_for,
)

__all__ = [
    # Dimensionality reduction
    "mean_pool_tokens",
    "pca_svd_project",
    # Clustering
    "k_subspaces_kllmeans",
    "k_llmmeans",
    "kmeanspp_sample",
    "select_representatives",
    "adjusted_rand_index",
    "pairwise_ari",
    "silhouette_score_precomputed",
    "coverage_fraction",
    "compute_stability_metrics",
    "ClusteringResult",
    "StabilityMetrics",
    # Sweep orchestration
    "SweepConfig",
    "SweepResult",
    "run_sweep",
    # Method plugins
    "available_method_plugins",
    "run_fixed_k_plugin",
    "run_unknown_k_plugin",
    "UNKNOWN_K_STRATEGY_FIXED_K_SELECTOR",
    "UNKNOWN_K_STRATEGY_HDBSCAN",
    # Recipes
    "RECIPE_VERSION",
    "CLUSTERING_COMPONENT_METHODS",
    "COMPOSITE_RECIPES",
    "COSINE_KLLMEANS_NO_PCA_RECIPE",
    "canonical_recipe_hash",
    "build_composite_recipe",
    "register_clustering_components",
    "ensure_composite_recipe",
    # Text-classification methods (register-only)
    "MATURITY_REGISTERED_ONLY",
    "TEXT_CLASSIFICATION_METHODS",
    "register_text_classification_methods",
    # Canonical-config builders (currently unused; future enforcement spot)
    "CANONICAL_CONFIG_BUILDERS",
    "CanonicalConfigBuilder",
    "canonical_config_for",
]

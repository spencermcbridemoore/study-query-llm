"""
Algorithm Core Library - PCA/KLLMeans sweep and clustering algorithms.

This module provides core algorithm implementations with minimal dependencies,
separate from notebooks/scripts. Designed for reuse and testing.
"""

from .dimensionality_reduction import mean_pool_tokens, pca_svd_project
from .clustering import (
    k_subspaces_kllmeans,
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

__all__ = [
    # Dimensionality reduction
    "mean_pool_tokens",
    "pca_svd_project",
    # Clustering
    "k_subspaces_kllmeans",
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
]

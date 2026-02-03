"""
Sweep orchestration for clustering across multiple K values.

Provides configuration and orchestration for running clustering sweeps
with optional multi-restart support and stability metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from .dimensionality_reduction import mean_pool_tokens, pca_svd_project, EmbIn
from .clustering import (
    k_subspaces_kllmeans,
    select_representatives,
    compute_stability_metrics,
    StabilityMetrics,
)

Array2D = np.ndarray


@dataclass
class SweepConfig:
    """Configuration for clustering sweep."""

    pca_dim: int = 64
    rank_r: int = 2
    k_min: int = 2
    k_max: int = 20
    max_iter: int = 200
    base_seed: int = 0
    n_restarts: int = 1
    compute_stability: bool = False
    coverage_threshold: float = 0.2


@dataclass
class SweepResult:
    """Results from a clustering sweep."""

    pca: Dict[str, Any]
    by_k: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    Z: Optional[Array2D] = None
    Z_norm: Optional[Array2D] = None
    dist: Optional[Array2D] = None


def run_sweep(
    texts: List[str],
    embeddings: EmbIn,
    cfg: SweepConfig,
    *,
    paraphraser: Optional[Callable[[List[str]], List[str]]] = None,
) -> SweepResult:
    """
    Run clustering sweep across K range with optional multi-restart and stability metrics.

    Pipeline:
    1. Mean-pool token embeddings (if needed)
    2. PCA/SVD reduce to pca_dim
    3. For each K in [k_min..k_max]:
       - Run clustering with n_restarts restarts
       - Compute stability metrics (if enabled)
       - Select representatives
       - Optionally paraphrase representatives

    Args:
        texts: List of text strings, one per sample
        embeddings: Input embeddings (see mean_pool_tokens for supported formats)
        cfg: SweepConfig with parameters
        paraphraser: Optional callable to paraphrase representatives (takes list[str], returns list[str])

    Returns:
        SweepResult with PCA metadata, results by K, and optional precomputed matrices

    Raises:
        ValueError: If k_min > k_max or K > n_samples
    """
    if cfg.k_min > cfg.k_max:
        raise ValueError(f"k_min ({cfg.k_min}) must be <= k_max ({cfg.k_max})")

    # Step 1: Pool embeddings
    X = mean_pool_tokens(embeddings)
    n_samples = X.shape[0]

    if cfg.k_max >= n_samples:
        raise ValueError(
            f"k_max ({cfg.k_max}) must be < n_samples ({n_samples})"
        )

    # Step 2: PCA projection
    Z, pca_meta = pca_svd_project(X, cfg.pca_dim)

    # Precompute normalized vectors and distance matrix for stability metrics
    Z_norm = None
    dist = None
    if cfg.compute_stability:
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        Z_norm = Z / np.maximum(norms, 1e-12)
        dist = 1.0 - (Z_norm @ Z_norm.T)
        dist = np.clip(dist, 0.0, 2.0)

    # Step 3: Sweep across K
    by_k: Dict[str, Dict[str, Any]] = {}

    for K in range(cfg.k_min, min(cfg.k_max, n_samples - 1) + 1):
        labels_list = []
        objectives = []
        representatives = None

        # Run multiple restarts
        for restart_idx in range(cfg.n_restarts):
            seed = cfg.base_seed + restart_idx
            labels, info = k_subspaces_kllmeans(
                Z, K, rank_r=cfg.rank_r, seed=seed, max_iter=cfg.max_iter
            )
            labels_list.append(labels)
            objectives.append(info["objective_recon_error_sum"])

        # Select representatives from first run
        representatives = select_representatives(Z, labels_list[0], texts)

        # Paraphrase if requested
        if paraphraser:
            representatives = paraphraser(representatives)

        # Compute stability metrics if enabled
        stability: Optional[StabilityMetrics] = None
        if cfg.compute_stability and cfg.n_restarts > 1:
            stability = compute_stability_metrics(
                labels_list,
                dist,
                Z_norm,
                objectives,
                n_samples,
                cfg.coverage_threshold,
            )

        # Store results
        result: Dict[str, Any] = {
            "representatives": representatives,
            "objective": info,  # From last run
            "objectives": objectives,  # All runs
            "labels": labels_list[0],  # First run labels
            "labels_all": labels_list if cfg.n_restarts > 1 else None,
        }
        if stability:
            result["stability"] = {
                "silhouette": {
                    "mean": stability.silhouette_mean,
                    "std": stability.silhouette_std,
                },
                "stability_ari": {
                    "mean": stability.stability_ari_mean,
                    "std": stability.stability_ari_std,
                },
                "dispersion": {
                    "mean": stability.dispersion_mean,
                    "std": stability.dispersion_std,
                },
                "coverage": {
                    "mean": stability.coverage_mean,
                    "std": stability.coverage_std,
                },
            }

        by_k[str(K)] = result

    return SweepResult(
        pca=pca_meta,
        by_k=by_k,
        Z=Z if cfg.compute_stability else None,
        Z_norm=Z_norm,
        dist=dist,
    )

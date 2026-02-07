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
    embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> SweepResult:
    """
    Run clustering sweep across K range with optional multi-restart and stability metrics.

    Pipeline:
    1. Mean-pool token embeddings (if needed)
    2. PCA/SVD reduce to pca_dim
    3. For each K in [k_min..k_max]:
       - Run clustering with n_restarts restarts
       - Select representatives from original clustering
       - Optionally paraphrase representatives (if paraphraser provided)
       - If embedder and paraphraser both provided:
         * Re-embed all texts (with representatives replaced by summaries)
         * Re-run full clustering pipeline with re-embedded texts
         * Recompute stability metrics using re-embedded embeddings
       - Otherwise: compute stability metrics using original embeddings

    Args:
        texts: List of text strings, one per sample
        embeddings: Input embeddings (see mean_pool_tokens for supported formats)
        cfg: SweepConfig with parameters
        paraphraser: Optional callable to paraphrase representatives (takes list[str], returns list[str])
        embedder: Optional callable to embed texts (takes list[str], returns np.ndarray of shape (n_texts, embedding_dim)).
                  Only used if paraphraser is also provided. Re-embeds all texts after representatives are replaced
                  with their summaries, enabling re-clustering with summarized text embeddings.

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
        original_representatives = representatives.copy()

        # Paraphrase if requested
        if paraphraser:
            representatives = paraphraser(representatives)

        # Store original clustering results
        original_stability: Optional[StabilityMetrics] = None
        if cfg.compute_stability and cfg.n_restarts > 1:
            original_stability = compute_stability_metrics(
                labels_list,
                dist,
                Z_norm,
                objectives,
                n_samples,
                cfg.coverage_threshold,
            )

        # If embedder is provided and we have summarized representatives, re-embed and re-cluster
        if embedder and paraphraser and len(representatives) > 0:
            # Create modified texts list: replace original representative texts with summaries
            # Map original representative indices to their summaries
            rep_indices = []
            modified_texts = texts.copy()
            for i, rep_text in enumerate(original_representatives):
                try:
                    idx = texts.index(rep_text)
                    rep_indices.append(idx)
                    if i < len(representatives):
                        summary = representatives[i]
                        # Skip empty summaries - keep original text instead
                        if summary and summary.strip():
                            modified_texts[idx] = summary
                        # If summary is empty, keep original text (don't replace)
                except ValueError:
                    # Representative text not found in original texts (shouldn't happen)
                    continue

            # Re-embed all texts (with representatives replaced by summaries)
            # The embedder will handle caching for non-representative texts if using EmbeddingService
            modified_embeddings_raw = embedder(modified_texts)
            modified_embeddings_raw = np.asarray(modified_embeddings_raw, dtype=np.float64)

            # Re-run full pipeline with re-embedded texts
            X_new = mean_pool_tokens(modified_embeddings_raw)
            Z_new, pca_meta_new = pca_svd_project(X_new, cfg.pca_dim)

            # Precompute normalized vectors and distance matrix for stability metrics
            Z_norm_new = None
            dist_new = None
            if cfg.compute_stability:
                norms_new = np.linalg.norm(Z_new, axis=1, keepdims=True)
                Z_norm_new = Z_new / np.maximum(norms_new, 1e-12)
                dist_new = 1.0 - (Z_norm_new @ Z_norm_new.T)
                dist_new = np.clip(dist_new, 0.0, 2.0)

            # Re-cluster with new embeddings
            labels_list_new = []
            objectives_new = []
            for restart_idx in range(cfg.n_restarts):
                seed = cfg.base_seed + restart_idx
                labels_new, info_new = k_subspaces_kllmeans(
                    Z_new, K, rank_r=cfg.rank_r, seed=seed, max_iter=cfg.max_iter
                )
                labels_list_new.append(labels_new)
                objectives_new.append(info_new["objective_recon_error_sum"])

            # Select representatives from re-clustered results
            representatives_new = select_representatives(
                Z_new, labels_list_new[0], modified_texts
            )

            # Compute stability metrics with new embeddings
            stability_new: Optional[StabilityMetrics] = None
            if cfg.compute_stability and cfg.n_restarts > 1:
                stability_new = compute_stability_metrics(
                    labels_list_new,
                    dist_new,
                    Z_norm_new,
                    objectives_new,
                    n_samples,
                    cfg.coverage_threshold,
                )

            # Store results with re-embedded/re-clustered data
            result: Dict[str, Any] = {
                "representatives": representatives_new,
                "objective": info_new,  # From last re-clustered run
                "objectives": objectives_new,  # All re-clustered runs
                "labels": labels_list_new[0],  # First re-clustered run labels
                "labels_all": labels_list_new if cfg.n_restarts > 1 else None,
            }
            if stability_new:
                result["stability"] = {
                    "silhouette": {
                        "mean": stability_new.silhouette_mean,
                        "std": stability_new.silhouette_std,
                    },
                    "stability_ari": {
                        "mean": stability_new.stability_ari_mean,
                        "std": stability_new.stability_ari_std,
                    },
                    "dispersion": {
                        "mean": stability_new.dispersion_mean,
                        "std": stability_new.dispersion_std,
                    },
                    "coverage": {
                        "mean": stability_new.coverage_mean,
                        "std": stability_new.coverage_std,
                    },
                }
        else:
            # No re-embedding: use original clustering results
            result: Dict[str, Any] = {
                "representatives": representatives,
                "objective": info,  # From last run
                "objectives": objectives,  # All runs
                "labels": labels_list[0],  # First run labels
                "labels_all": labels_list if cfg.n_restarts > 1 else None,
            }
            if original_stability:
                result["stability"] = {
                    "silhouette": {
                        "mean": original_stability.silhouette_mean,
                        "std": original_stability.silhouette_std,
                    },
                    "stability_ari": {
                        "mean": original_stability.stability_ari_mean,
                        "std": original_stability.stability_ari_std,
                    },
                    "dispersion": {
                        "mean": original_stability.dispersion_mean,
                        "std": original_stability.dispersion_std,
                    },
                    "coverage": {
                        "mean": original_stability.coverage_mean,
                        "std": original_stability.coverage_std,
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

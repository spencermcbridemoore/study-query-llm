"""
Sweep orchestration for clustering across multiple K values.

Provides configuration and orchestration for running clustering sweeps
with optional multi-restart support and stability metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from .dimensionality_reduction import mean_pool_tokens, pca_svd_project, EmbIn
from .clustering import (
    k_llmmeans,
    select_representatives,
    compute_stability_metrics,
    StabilityMetrics,
)

Array2D = np.ndarray


@dataclass
class SweepConfig:
    """Configuration for clustering sweep."""

    pca_dim: int = 64
    skip_pca: bool = False  # Skip PCA and use full embedding dimensions
    k_min: int = 2
    k_max: int = 20
    max_iter: int = 200
    base_seed: int = 0
    n_restarts: int = 1
    compute_stability: bool = False
    coverage_threshold: float = 0.2
    llm_interval: int = 20
    max_samples: int = 10
    distance_metric: str = "cosine"  # "cosine" or "euclidean"
    normalize_vectors: bool = True  # L2-normalize vectors (default True for cosine)


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
    2. PCA/SVD reduce to pca_dim (unless skip_pca=True)
    3. For each K in [k_min..k_max]:
       - Run k_llmmeans with n_restarts restarts, keep best (lowest inertia)
       - Select representatives from the best run
       - Optionally paraphrase representatives (if paraphraser provided but
         embedder is not — when both are provided the LLM summarisation
         happens inside the k_llmmeans iteration loop instead)
       - Compute stability metrics across restarts

    Distance metrics:
    - "cosine" (default): Uses cosine/spherical k-means. Vectors are L2-normalized
      and assignment uses dot product (equivalent to cosine similarity on unit vectors).
      Centroids are normalized after each update.
    - "euclidean": Standard k-means with Euclidean distance. No normalization.

    Skip PCA mode:
    - When cfg.skip_pca=True, embeddings are used in their full dimensionality
      (typically 1536-dim) without PCA projection. This preserves all information
      but increases runtime by ~20-24x. Use when embedding fidelity is critical.
    
    Example with no PCA:
        cfg = SweepConfig(skip_pca=True, k_min=2, k_max=10)
        result = run_sweep(texts, embeddings, cfg)

    Args:
        texts: List of text strings, one per sample
        embeddings: Input embeddings (see mean_pool_tokens for supported formats)
        cfg: SweepConfig with parameters (including skip_pca flag)
        paraphraser: Optional callable to generate cluster summaries.
            When both *paraphraser* and *embedder* are supplied they are
            passed into ``k_llmmeans`` so that the LLM influences centroids
            **inside** the iteration loop.  When only *paraphraser* is given,
            representatives are paraphrased post-hoc for display.
        embedder: Optional callable to embed texts (takes list[str], returns
            np.ndarray of shape (n_texts, embedding_dim)).  Used together
            with *paraphraser* to project LLM summaries back into the same
            space as the data points (PCA space if skip_pca=False, or full
            embedding space if skip_pca=True).

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

    # Step 2: PCA projection (conditional based on skip_pca flag)
    if cfg.skip_pca:
        # Use full embeddings without dimensionality reduction
        Z = X
        pca_components = None
        pca_mean = None
        pca_meta = {
            "pca_dim_used": X.shape[1],
            "singular_values": None,
            "mean": None,
            "components": None,
            "skip_pca": True,
        }
    else:
        # Standard PCA projection
        Z, pca_meta = pca_svd_project(X, cfg.pca_dim)
        pca_meta["skip_pca"] = False
        pca_components: np.ndarray = pca_meta["components"]  # (pca_dim, D)
        pca_mean: np.ndarray = pca_meta["mean"]              # (D,)

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
        labels_list: List[np.ndarray] = []
        objectives: List[float] = []
        info_list: List[dict] = []

        # Run multiple restarts — keep all for stability, pick best for output
        for restart_idx in range(cfg.n_restarts):
            seed = cfg.base_seed + restart_idx
            labels, info = k_llmmeans(
                Z,
                texts,
                K,
                max_iter=cfg.max_iter,
                llm_interval=cfg.llm_interval,
                max_samples=cfg.max_samples,
                seed=seed,
                paraphraser=paraphraser if embedder is not None else None,
                embedder=embedder,
                pca_components=pca_components,
                pca_mean=pca_mean,
                distance_metric=cfg.distance_metric,
                normalize_vectors=cfg.normalize_vectors,
            )
            labels_list.append(labels)
            objectives.append(info["objective"])
            info_list.append(info)

        # Best restart (lowest inertia)
        best_idx = int(np.argmin(objectives))
        best_labels = labels_list[best_idx]
        best_info = info_list[best_idx]

        # Representatives: nearest-to-centroid text per cluster
        representatives = select_representatives(Z, best_labels, texts)

        # If only paraphraser (no embedder), paraphrase representatives
        # post-hoc for presentation.  When both are provided the LLM already
        # ran inside k_llmmeans and summaries are in best_info.
        if paraphraser and not embedder:
            representatives = paraphraser(representatives)

        # Build result dict
        result: Dict[str, Any] = {
            "representatives": representatives,
            "summaries": best_info.get("summaries", []),
            "objective": best_info["objective"],
            "objectives": objectives,
            "labels": best_labels,
            "labels_all": labels_list if cfg.n_restarts > 1 else None,
            "n_iter": best_info.get("n_iter", 0),
        }

        # Stability metrics (require multiple restarts)
        if cfg.compute_stability and cfg.n_restarts > 1:
            stability = compute_stability_metrics(
                labels_list,
                dist,
                Z_norm,
                objectives,
                n_samples,
                cfg.coverage_threshold,
            )
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
        Z=Z,  # Always attach so notebooks can compute silhouette from Z when dist not saved
        Z_norm=Z_norm,
        dist=dist,
    )

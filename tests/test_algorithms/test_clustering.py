"""
Tests for clustering algorithms.
"""

import numpy as np
import pytest

from study_query_llm.algorithms.clustering import (
    k_subspaces_kllmeans,
    select_representatives,
    adjusted_rand_index,
    pairwise_ari,
    silhouette_score_precomputed,
    coverage_fraction,
    compute_stability_metrics,
)


def test_k_subspaces_kllmeans_basic():
    """Test basic KLLMeans clustering."""
    n, d, K = 50, 10, 3
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))

    labels, info = k_subspaces_kllmeans(X, K, rank_r=2, seed=0, max_iter=50)

    assert labels.shape == (n,)
    assert len(np.unique(labels)) == K
    assert "objective_recon_error_sum" in info
    assert "n_iter" in info
    assert info["objective_recon_error_sum"] >= 0


def test_k_subspaces_kllmeans_convergence():
    """Test that clustering converges."""
    n, d, K = 30, 8, 2
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))

    labels, info = k_subspaces_kllmeans(X, K, rank_r=2, seed=0, max_iter=200)

    assert info["n_iter"] <= 200
    assert len(np.unique(labels)) == K


def test_select_representatives():
    """Test representative selection."""
    n, d, K = 20, 5, 3
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n, d))
    labels = rng.integers(0, K, size=n)
    texts = [f"text_{i}" for i in range(n)]

    reps = select_representatives(Z, labels, texts)

    assert len(reps) <= K
    assert all(isinstance(r, str) for r in reps)
    assert all(r in texts for r in reps)


def test_adjusted_rand_index():
    """Test ARI computation."""
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])

    ari = adjusted_rand_index(labels_a, labels_b)
    assert ari == pytest.approx(1.0, abs=1e-6)

    labels_c = np.array([0, 1, 0, 1, 0, 1])
    ari_mixed = adjusted_rand_index(labels_a, labels_c)
    assert -1.0 <= ari_mixed <= 1.0


def test_pairwise_ari():
    """Test pairwise ARI computation."""
    labels_list = [
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]),
    ]

    aris = pairwise_ari(labels_list)
    assert len(aris) == 3  # 3 pairs: (0,1), (0,2), (1,2)
    assert all(-1.0 <= ari <= 1.0 for ari in aris)


def test_silhouette_score_precomputed():
    """Test silhouette score computation."""
    n = 20
    labels = np.array([0] * 10 + [1] * 10)
    dist = np.random.rand(n, n)
    dist = (dist + dist.T) / 2  # Make symmetric
    np.fill_diagonal(dist, 0)

    sil = silhouette_score_precomputed(labels, dist)
    assert -1.0 <= sil <= 1.0


def test_coverage_fraction():
    """Test coverage fraction computation."""
    n, d = 20, 5
    rng = np.random.default_rng(42)
    Z_norm = rng.standard_normal((n, d))
    # Normalize rows
    norms = np.linalg.norm(Z_norm, axis=1, keepdims=True)
    Z_norm = Z_norm / np.maximum(norms, 1e-12)

    labels = np.array([0] * 10 + [1] * 10)
    threshold = 0.5

    coverage = coverage_fraction(labels, Z_norm, threshold)
    assert 0.0 <= coverage <= 1.0


def test_compute_stability_metrics():
    """Test stability metrics computation."""
    n, d = 30, 5
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n, d))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z_norm = Z / np.maximum(norms, 1e-12)
    dist = 1.0 - (Z_norm @ Z_norm.T)
    dist = np.clip(dist, 0.0, 2.0)

    labels_list = [
        rng.integers(0, 3, size=n),
        rng.integers(0, 3, size=n),
        rng.integers(0, 3, size=n),
    ]
    objectives = [100.0, 105.0, 98.0]

    metrics = compute_stability_metrics(
        labels_list, dist, Z_norm, objectives, n_samples=n, coverage_threshold=0.2
    )

    assert metrics.silhouette_mean >= -1.0
    assert metrics.silhouette_mean <= 1.0
    assert metrics.stability_ari_mean >= -1.0
    assert metrics.stability_ari_mean <= 1.0
    assert metrics.dispersion_mean >= 0.0
    assert metrics.coverage_mean >= 0.0
    assert metrics.coverage_mean <= 1.0

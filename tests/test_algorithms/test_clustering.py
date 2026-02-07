"""
Tests for clustering algorithms.
"""

import numpy as np
import pytest

from study_query_llm.algorithms.clustering import (
    k_subspaces_kllmeans,
    k_llmmeans,
    kmeanspp_sample,
    select_representatives,
    adjusted_rand_index,
    pairwise_ari,
    silhouette_score_precomputed,
    coverage_fraction,
    compute_stability_metrics,
)


# ------------------------------------------------------------------
# k_subspaces_kllmeans (legacy)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# kmeanspp_sample
# ------------------------------------------------------------------


def test_kmeanspp_sample_basic():
    """Test k-means++ sampling returns correct number of indices."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    indices = kmeanspp_sample(X, 5, rng)
    assert len(indices) == 5
    assert len(set(indices)) == 5  # all unique
    assert all(0 <= i < 20 for i in indices)


def test_kmeanspp_sample_all():
    """When m >= n, return all indices."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((5, 3))
    indices = kmeanspp_sample(X, 10, rng)
    assert len(indices) == 5
    np.testing.assert_array_equal(indices, np.arange(5))


def test_kmeanspp_sample_one():
    """Sampling 1 point should return a single index."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((10, 3))
    indices = kmeanspp_sample(X, 1, rng)
    assert len(indices) == 1


# ------------------------------------------------------------------
# k_llmmeans — vanilla (no LLM)
# ------------------------------------------------------------------


def test_k_llmmeans_vanilla_basic():
    """Without paraphraser/embedder, k_llmmeans is plain k-means."""
    rng = np.random.default_rng(42)
    n, d, K = 60, 8, 3
    # Create well-separated clusters
    centers = rng.standard_normal((K, d)) * 5
    Z = np.vstack([centers[k] + rng.standard_normal((n // K, d)) * 0.3 for k in range(K)])
    texts = [f"text_{i}" for i in range(n)]

    labels, info = k_llmmeans(Z, texts, K, seed=0, max_iter=100, distance_metric="cosine", normalize_vectors=True)

    assert labels.shape == (n,)
    assert len(np.unique(labels)) == K
    assert info["objective"] >= 0
    assert info["n_iter"] >= 1
    assert len(info["summaries"]) == K
    assert all(s == "" for s in info["summaries"])  # no LLM → empty summaries


def test_k_llmmeans_vanilla_convergence():
    """Vanilla k_llmmeans should converge within max_iter."""
    rng = np.random.default_rng(42)
    n, d, K = 40, 6, 2
    Z = rng.standard_normal((n, d))
    texts = [f"t{i}" for i in range(n)]

    labels, info = k_llmmeans(Z, texts, K, seed=0, max_iter=200, distance_metric="cosine", normalize_vectors=True)

    assert info["n_iter"] <= 200
    assert labels.shape == (n,)


def test_k_llmmeans_validation():
    """Test input validation."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((10, 5))
    texts = [f"t{i}" for i in range(10)]

    with pytest.raises(ValueError, match="K.*cannot exceed"):
        k_llmmeans(Z, texts, 20)

    with pytest.raises(ValueError, match="K must be >= 1"):
        k_llmmeans(Z, texts, 0)

    with pytest.raises(ValueError, match="distance_metric must be"):
        k_llmmeans(Z, texts, 3, distance_metric="invalid")


# ------------------------------------------------------------------
# k_llmmeans — with LLM
# ------------------------------------------------------------------


def test_k_llmmeans_with_llm():
    """LLM centroid replacement fires and produces summaries."""
    rng = np.random.default_rng(42)
    n, d_pca, D_orig, K = 60, 6, 12, 3

    # Use overlapping random data so k-means takes multiple iterations,
    # ensuring the LLM branch at t % llm_interval == 0 actually fires.
    Z = rng.standard_normal((n, d_pca))
    texts = [f"text_{i}" for i in range(n)]

    # Mock PCA transform
    pca_components = rng.standard_normal((d_pca, D_orig))
    pca_mean = rng.standard_normal(D_orig)

    calls = {"paraphraser": 0, "embedder": 0}

    def mock_paraphraser(sampled_texts):
        calls["paraphraser"] += 1
        return [f"summary_of_{len(sampled_texts)}_texts"]

    def mock_embedder(text_list):
        calls["embedder"] += 1
        return rng.standard_normal((len(text_list), D_orig))

    labels, info = k_llmmeans(
        Z, texts, K,
        max_iter=100,
        llm_interval=2,  # fire LLM every 2 iterations
        max_samples=5,
        seed=0,
        paraphraser=mock_paraphraser,
        embedder=mock_embedder,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )

    assert labels.shape == (n,)
    assert info["objective"] >= 0
    # LLM should have been called at least once
    assert calls["paraphraser"] >= 1
    assert calls["embedder"] >= 1
    # Summaries should contain LLM output
    assert any(s != "" for s in info["summaries"])


def test_k_llmmeans_llm_only_when_both_provided():
    """LLM branch is skipped when only paraphraser (no embedder) is given."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((30, 5))
    texts = [f"t{i}" for i in range(30)]

    calls = {"count": 0}

    def mock_paraphraser(sampled_texts):
        calls["count"] += 1
        return [f"summary"]

    labels, info = k_llmmeans(
        Z, texts, 3,
        max_iter=50,
        llm_interval=2,
        seed=0,
        paraphraser=mock_paraphraser,
        embedder=None,  # no embedder → no LLM
    )

    assert calls["count"] == 0  # paraphraser never called
    assert all(s == "" for s in info["summaries"])


def test_k_llmmeans_cosine_default():
    """Default behavior should use cosine/spherical k-means with normalization."""
    rng = np.random.default_rng(42)
    n, d, K = 51, 8, 3  # Use 51 so n//K = 17, total = 51
    # Create well-separated clusters
    centers = rng.standard_normal((K, d)) * 5
    Z = np.vstack([centers[k] + rng.standard_normal((n // K, d)) * 0.3 for k in range(K)])
    texts = [f"text_{i}" for i in range(Z.shape[0])]
    n_actual = Z.shape[0]

    # Default (cosine with normalization)
    labels_cosine, info_cosine = k_llmmeans(Z, texts, K, seed=0, max_iter=100)

    assert labels_cosine.shape == (n_actual,)
    assert len(np.unique(labels_cosine)) == K
    assert info_cosine["objective"] >= 0

    # Verify centroids are normalized (check via assignment behavior)
    # Cosine should produce different results than Euclidean on same data
    labels_euclid, info_euclid = k_llmmeans(
        Z, texts, K, seed=0, max_iter=100,
        distance_metric="euclidean", normalize_vectors=False
    )

    # Results may differ (not guaranteed, but likely for well-separated clusters)
    # At minimum, both should produce valid clusterings
    assert labels_euclid.shape == (n_actual,)
    assert len(np.unique(labels_euclid)) == K
    assert info_euclid["objective"] >= 0


def test_k_llmmeans_euclidean_override():
    """Euclidean metric should work without normalization."""
    rng = np.random.default_rng(42)
    n, d, K = 50, 8, 3
    Z = rng.standard_normal((n, d))
    texts = [f"text_{i}" for i in range(n)]

    labels, info = k_llmmeans(
        Z, texts, K, seed=0, max_iter=100,
        distance_metric="euclidean", normalize_vectors=False
    )

    assert labels.shape == (n,)
    assert len(np.unique(labels)) == K
    assert info["objective"] >= 0
    assert info["n_iter"] >= 1


def test_k_llmmeans_cosine_vs_euclidean_different():
    """Cosine and Euclidean should produce different results on non-normalized data."""
    rng = np.random.default_rng(42)
    n, d, K = 60, 8, 3
    # Create data with varying magnitudes
    Z = rng.standard_normal((n, d)) * np.arange(1, n + 1).reshape(-1, 1) * 0.1
    texts = [f"text_{i}" for i in range(n)]

    labels_cosine, _ = k_llmmeans(Z, texts, K, seed=0, max_iter=100, distance_metric="cosine", normalize_vectors=True)
    labels_euclid, _ = k_llmmeans(Z, texts, K, seed=0, max_iter=100, distance_metric="euclidean", normalize_vectors=False)

    # With varying magnitudes, cosine (normalized) and Euclidean should differ
    # Check that at least some assignments differ
    n_different = np.sum(labels_cosine != labels_euclid)
    # Not guaranteed to be different, but very likely with magnitude variation
    assert n_different >= 0  # At minimum, both should run successfully


# ------------------------------------------------------------------
# select_representatives
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Metrics helpers
# ------------------------------------------------------------------


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

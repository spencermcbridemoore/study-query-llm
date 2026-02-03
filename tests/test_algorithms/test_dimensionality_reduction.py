"""
Tests for dimensionality reduction utilities.
"""

import numpy as np
import pytest

from study_query_llm.algorithms.dimensionality_reduction import (
    mean_pool_tokens,
    pca_svd_project,
)


def test_mean_pool_tokens_2d():
    """Test pooling 2D array (already item-level)."""
    n, d = 10, 5
    X = np.random.rand(n, d)
    result = mean_pool_tokens(X)
    assert result.shape == (n, d)
    np.testing.assert_array_equal(result, X)


def test_mean_pool_tokens_3d():
    """Test pooling 3D array (items x tokens x dims)."""
    n, t, d = 10, 5, 3
    X = np.random.rand(n, t, d)
    result = mean_pool_tokens(X)
    assert result.shape == (n, d)
    expected = X.mean(axis=1)
    np.testing.assert_allclose(result, expected)


def test_mean_pool_tokens_list():
    """Test pooling list of token arrays."""
    n, d = 5, 3
    tokens_list = [np.random.rand(np.random.randint(3, 8), d) for _ in range(n)]
    result = mean_pool_tokens(tokens_list)
    assert result.shape == (n, d)
    for i, tokens in enumerate(tokens_list):
        expected = tokens.mean(axis=0)
        np.testing.assert_allclose(result[i], expected)


def test_mean_pool_tokens_normalize():
    """Test normalization option."""
    n, d = 10, 5
    X = np.random.rand(n, d)
    result = mean_pool_tokens(X, normalize=True)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-6)


def test_pca_svd_project():
    """Test PCA projection."""
    n, d, k = 20, 10, 5
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))

    Z, meta = pca_svd_project(X, k)

    assert Z.shape == (n, k)
    assert meta["pca_dim_used"] == k
    assert len(meta["singular_values"]) == d
    assert len(meta["mean"]) == d


def test_pca_svd_project_k_larger_than_d():
    """Test PCA when k > n_features."""
    n, d, k = 20, 10, 15
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))

    Z, meta = pca_svd_project(X, k)

    assert Z.shape == (n, d)  # Can't exceed d
    assert meta["pca_dim_used"] == d

"""
Dimensionality reduction utilities for embeddings.

Provides PCA/SVD projection and token pooling functions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union
import numpy as np

Array2D = np.ndarray
EmbIn = Union[np.ndarray, List[np.ndarray]]


def mean_pool_tokens(emb: EmbIn, normalize: bool = False) -> Array2D:
    """
    Mean-pool token embeddings to item-level embeddings.

    Accepts:
    - np.ndarray of shape (n_items, d) -> returned as-is
    - np.ndarray of shape (n_items, n_tokens, d) -> mean-pooled over tokens
    - list of np.ndarray: [ (n_tokens_i, d), ... ] -> mean-pooled per item

    Args:
        emb: Input embeddings (array or list of arrays)
        normalize: If True, L2-normalize the pooled vectors

    Returns:
        Array of shape (n_items, d) with pooled embeddings

    Raises:
        ValueError: If input shape is unsupported
        TypeError: If input type is not supported
    """
    if isinstance(emb, np.ndarray):
        if emb.ndim == 2:
            X = emb.astype(np.float64, copy=False)
        elif emb.ndim == 3:
            X = emb.mean(axis=1).astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported ndarray shape: {emb.shape}")
    elif isinstance(emb, list):
        pooled = []
        for i, t in enumerate(emb):
            a = np.asarray(t, dtype=np.float64)
            if a.ndim != 2:
                raise ValueError(f"tokens[{i}] must be (t_i,d); got {a.shape}")
            pooled.append(a.mean(axis=0))
        X = np.stack(pooled, axis=0)
    else:
        raise TypeError("emb must be np.ndarray or list of np.ndarray")

    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
    return X


def pca_svd_project(X: Array2D, k: int) -> Tuple[Array2D, Dict[str, Any]]:
    """
    Project data to k dimensions using PCA via SVD.

    Centers the data, computes SVD, and projects to the top k principal components.

    Args:
        X: Input data of shape (n_samples, n_features)
        k: Number of principal components to keep

    Returns:
        Tuple of:
        - Z: Projected data of shape (n_samples, k_used) where k_used = min(k, n_features)
        - meta: Dictionary with PCA metadata:
            - pca_dim_used: Actual number of components used
            - singular_values: All singular values
            - mean: Mean vector used for centering
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    kk = int(min(k, U.shape[1]))
    Z = U[:, :kk] * S[:kk]
    meta = {
        "pca_dim_used": kk,
        "singular_values": S.tolist(),
        "mean": mu.squeeze(0).tolist(),
    }
    return Z, meta

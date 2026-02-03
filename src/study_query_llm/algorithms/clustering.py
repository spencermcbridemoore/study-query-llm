"""
Clustering algorithms and stability metrics.

Provides K-subspaces KLLMeans clustering and stability evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

Array2D = np.ndarray


@dataclass
class ClusteringResult:
    """Result of a single clustering run."""

    labels: np.ndarray
    objective: float
    n_iter: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if None."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StabilityMetrics:
    """Stability metrics computed across multiple clustering runs."""

    silhouette_mean: float
    silhouette_std: float
    stability_ari_mean: float
    stability_ari_std: float
    dispersion_mean: float
    dispersion_std: float
    coverage_mean: float
    coverage_std: float


def k_subspaces_kllmeans(
    X: Array2D, K: int, *, rank_r: int = 2, seed: int = 0, max_iter: int = 200
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    K-subspaces KLLMeans clustering with rank-r approximation.

    Clusters data into K groups, where each cluster is represented by a rank-r
    subspace (mean + r principal directions). Minimizes reconstruction error.

    Args:
        X: Input data of shape (n_samples, n_features)
        K: Number of clusters
        rank_r: Rank of subspace approximation per cluster (default: 2)
        seed: Random seed for initialization
        max_iter: Maximum iterations for convergence

    Returns:
        Tuple of:
        - labels: Cluster assignments of shape (n_samples,)
        - info: Dictionary with objective_recon_error_sum and n_iter

    Raises:
        ValueError: If K > n_samples or rank_r > n_features
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape

    if K > n:
        raise ValueError(f"K ({K}) cannot exceed number of samples ({n})")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    r = int(max(1, min(rank_r, d)))
    labels = rng.integers(0, K, size=n, endpoint=False)
    mus = np.zeros((K, d))
    Bs = np.zeros((K, d, r))

    def update_cluster(k, idx):
        """Update cluster k's mean and subspace basis."""
        if len(idx) < 2:
            # Fallback: random point and random basis
            mus[k] = X[rng.integers(0, n)]
            Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
            Bs[k] = Q[:, :r]
            return
        Xk = X[idx]
        mu = Xk.mean(axis=0)
        Xkc = Xk - mu
        _, _, Vt = np.linalg.svd(Xkc, full_matrices=False)
        Bs[k] = Vt[:r].T
        mus[k] = mu

    # Initialize clusters
    for k in range(K):
        update_cluster(k, np.where(labels == k)[0])

    # Iterate until convergence
    n_iter = 0
    prev = labels.copy()
    for _ in range(max_iter):
        n_iter += 1
        # Compute reconstruction error for each point to each cluster
        Xm = X[:, None, :] - mus[None, :, :]
        proj = np.einsum("nkd,kdr->nkr", Xm, Bs)
        back = np.einsum("nkr,kdr->nkd", proj, Bs)
        resid = Xm - back
        err2 = np.einsum("nkd,nkd->nk", resid, resid)
        labels = np.argmin(err2, axis=1)
        if np.array_equal(labels, prev):
            break
        prev = labels.copy()
        # Update clusters
        for k in range(K):
            update_cluster(k, np.where(labels == k)[0])

    # Compute final objective
    Xm = X - mus[labels]
    proj = np.einsum("nd,ndr->nr", Xm, Bs[labels])
    back = np.einsum("nr,ndr->nd", proj, Bs[labels])
    resid = Xm - back
    obj = float(np.sum(resid * resid))

    return labels.astype(int), {"objective_recon_error_sum": obj, "n_iter": n_iter}


def select_representatives(
    Z: Array2D, labels: np.ndarray, texts: List[str]
) -> List[str]:
    """
    Select one representative text per cluster (closest to centroid).

    Args:
        Z: Projected data of shape (n_samples, n_dims)
        labels: Cluster assignments of shape (n_samples,)
        texts: List of text strings, one per sample

    Returns:
        List of representative texts, one per cluster (in cluster order)
    """
    reps = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        Zk = Z[idx]
        ck = Zk.mean(axis=0)
        i = idx[np.argmin(np.sum((Zk - ck) ** 2, axis=1))]
        reps.append(texts[i])
    return reps


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between two clusterings.

    ARI measures agreement between two clusterings, adjusted for chance.
    Returns 1.0 for identical clusterings, ~0.0 for random agreement.

    Args:
        labels_a: First clustering labels
        labels_b: Second clustering labels

    Returns:
        ARI score in [-1, 1], typically in [0, 1]
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    n = len(labels_a)

    n_classes = a.max() + 1
    n_clusters = b.max() + 1
    contingency = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for i in range(n):
        contingency[a[i], b[i]] += 1

    sum_comb = (contingency * (contingency - 1) / 2.0).sum()
    sum_comb_c = (contingency.sum(axis=1) * (contingency.sum(axis=1) - 1) / 2.0).sum()
    sum_comb_k = (contingency.sum(axis=0) * (contingency.sum(axis=0) - 1) / 2.0).sum()
    comb_n = n * (n - 1) / 2.0

    if comb_n == 0:
        return 1.0

    expected_index = (sum_comb_c * sum_comb_k) / comb_n
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected_index
    if denom == 0:
        return 1.0
    return float((sum_comb - expected_index) / denom)


def pairwise_ari(labels_list: List[np.ndarray]) -> List[float]:
    """
    Compute pairwise ARI between all pairs of clusterings.

    Args:
        labels_list: List of clustering label arrays

    Returns:
        List of ARI scores for all pairs (i, j) where i < j
    """
    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_index(labels_list[i], labels_list[j]))
    return aris


def silhouette_score_precomputed(labels: np.ndarray, dist: np.ndarray) -> float:
    """
    Compute silhouette score using precomputed distance matrix.

    Silhouette score measures how well-separated clusters are.
    Higher is better (range [-1, 1]).

    Args:
        labels: Cluster assignments
        dist: Precomputed distance matrix of shape (n_samples, n_samples)

    Returns:
        Mean silhouette score
    """
    labels = np.asarray(labels)
    n = len(labels)
    unique = np.unique(labels)
    if len(unique) == 1:
        return 0.0

    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        same_mask = labels == labels[i]
        same_count = same_mask.sum()
        if same_count <= 1:
            sil[i] = 0.0
            continue
        a = dist[i, same_mask].sum() / (same_count - 1)
        b = np.inf
        for c in unique:
            if c == labels[i]:
                continue
            other_mask = labels == c
            if other_mask.sum() == 0:
                continue
            b = min(b, dist[i, other_mask].mean())
        sil[i] = (b - a) / max(a, b) if b > 0 else 0.0
    return float(np.mean(sil))


def coverage_fraction(
    labels: np.ndarray, Z_norm: np.ndarray, threshold: float
) -> float:
    """
    Compute coverage fraction: fraction of points within threshold distance of centroid.

    Args:
        labels: Cluster assignments
        Z_norm: Normalized projected data
        threshold: Distance threshold (cosine distance)

    Returns:
        Fraction of points within threshold of their cluster centroid
    """
    labels = np.asarray(labels)
    n = len(labels)
    dists = np.zeros(n, dtype=np.float64)
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        centroid = Z_norm[idx].mean(axis=0)
        centroid = centroid / np.maximum(np.linalg.norm(centroid), 1e-12)
        sims = Z_norm[idx] @ centroid
        dists[idx] = 1.0 - sims
    return float(np.mean(dists <= threshold))


def compute_stability_metrics(
    labels_list: List[np.ndarray],
    dist: np.ndarray,
    Z_norm: np.ndarray,
    objectives: List[float],
    n_samples: int,
    coverage_threshold: float = 0.2,
) -> StabilityMetrics:
    """
    Compute stability metrics across multiple clustering runs.

    Args:
        labels_list: List of clustering label arrays (one per restart)
        dist: Precomputed distance matrix
        Z_norm: Normalized projected data
        objectives: List of objective values (one per restart)
        n_samples: Number of samples (for normalization)
        coverage_threshold: Distance threshold for coverage metric

    Returns:
        StabilityMetrics dataclass with mean/std for each metric
    """
    sils = [silhouette_score_precomputed(l, dist) for l in labels_list]
    aris = pairwise_ari(labels_list)
    objs = [obj / n_samples for obj in objectives]
    covs = [
        coverage_fraction(l, Z_norm, coverage_threshold) for l in labels_list
    ]

    return StabilityMetrics(
        silhouette_mean=float(np.mean(sils)),
        silhouette_std=float(np.std(sils)),
        stability_ari_mean=float(np.mean(aris)) if aris else 1.0,
        stability_ari_std=float(np.std(aris)) if aris else 0.0,
        dispersion_mean=float(np.mean(objs)),
        dispersion_std=float(np.std(objs)),
        coverage_mean=float(np.mean(covs)),
        coverage_std=float(np.std(covs)),
    )

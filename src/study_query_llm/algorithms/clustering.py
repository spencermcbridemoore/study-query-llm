"""
Clustering algorithms and stability metrics.

Provides K-subspaces KLLMeans clustering and stability evaluation metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
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


def kmeanspp_sample(
    X_subset: np.ndarray, m: int, rng: np.random.Generator
) -> np.ndarray:
    """
    K-means++ sampling to select *m* diverse representatives from *X_subset*.

    Uses the k-means++ seeding procedure: the first index is chosen uniformly
    at random; each subsequent index is drawn with probability proportional to
    its squared distance to the nearest already-selected point.

    Args:
        X_subset: (n_cluster, d) embeddings for one cluster
        m: Number of samples to select
        rng: NumPy random generator

    Returns:
        1-D integer array of *m* indices into *X_subset* (or all indices when
        ``m >= n_cluster``).
    """
    n = X_subset.shape[0]
    if m >= n:
        return np.arange(n)

    selected: list[int] = [int(rng.integers(0, n))]

    for _ in range(m - 1):
        sel_pts = X_subset[selected]  # (len(selected), d)
        # Squared distances from every point to the nearest selected point
        diffs = X_subset[:, None, :] - sel_pts[None, :, :]  # (n, s, d)
        sq_dists = np.sum(diffs ** 2, axis=2)  # (n, s)
        min_sq = sq_dists.min(axis=1)  # (n,)
        min_sq[selected] = 0.0  # already chosen → zero weight

        total = min_sq.sum()
        if total == 0.0:
            remaining = np.setdiff1d(np.arange(n), selected)
            if len(remaining) == 0:
                break
            selected.append(int(rng.choice(remaining)))
        else:
            probs = min_sq / total
            selected.append(int(rng.choice(n, p=probs)))

    return np.array(selected, dtype=int)


# ------------------------------------------------------------------
# K-means++ initialisation & assignment helpers
# ------------------------------------------------------------------

def _kmeanspp_init(
    Z: np.ndarray, K: int, rng: np.random.Generator
) -> np.ndarray:
    """Return (K, d) initial centroids chosen by the k-means++ rule."""
    n, d = Z.shape
    centroids = np.empty((K, d), dtype=Z.dtype)
    idx = int(rng.integers(0, n))
    centroids[0] = Z[idx]

    for k in range(1, K):
        diffs = Z[:, None, :] - centroids[None, :k, :]  # (n, k, d)
        sq = np.sum(diffs ** 2, axis=2)  # (n, k)
        min_sq = sq.min(axis=1)  # (n,)
        total = min_sq.sum()
        if total == 0.0:
            centroids[k] = Z[int(rng.integers(0, n))]
        else:
            probs = min_sq / total
            centroids[k] = Z[int(rng.choice(n, p=probs))]
    return centroids


def _assign(Z: np.ndarray, centroids: np.ndarray, *, distance_metric: str = "euclidean", normalize_vectors: bool = False) -> np.ndarray:
    """Assign each row of *Z* to its nearest centroid.
    
    Args:
        Z: (n, d) data points
        centroids: (K, d) cluster centroids
        distance_metric: "cosine" or "euclidean"
        normalize_vectors: Whether vectors are already normalized (for cosine)
    
    Returns:
        (n,) array of cluster assignments
    """
    use_cosine = (distance_metric == "cosine") and normalize_vectors
    if use_cosine:
        # Cosine: argmax of dot product (equivalent to min cosine distance on unit vectors)
        similarities = Z @ centroids.T  # (n, K)
        return np.argmax(similarities, axis=1)
    else:
        # Euclidean: ||z - c||² = ||z||² + ||c||² - 2·z·c
        Z_sq = np.sum(Z ** 2, axis=1, keepdims=True)       # (n, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
        cross = Z @ centroids.T                              # (n, K)
        dists = Z_sq + C_sq - 2.0 * cross                   # (n, K)
        return np.argmin(dists, axis=1)


# ------------------------------------------------------------------
# k-LLMmeans  (Algorithm 1 from the paper)
# ------------------------------------------------------------------

def k_llmmeans(
    Z: np.ndarray,
    texts: list[str],
    K: int,
    *,
    max_iter: int = 120,
    llm_interval: int = 20,
    max_samples: int = 10,
    seed: int = 0,
    paraphraser: Callable[[list[str]], list[str]] | None = None,
    embedder: Callable[[list[str]], np.ndarray] | None = None,
    pca_components: np.ndarray | None = None,
    pca_mean: np.ndarray | None = None,
    distance_metric: str = "cosine",
    normalize_vectors: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    K-means with optional LLM-generated summary centroids (Algorithm 1).

    Every *llm_interval* iterations the centroid update is replaced by:
      1. Sample *max_samples* diverse representatives per cluster (k-means++).
      2. Ask the LLM (``paraphraser``) to summarise those texts.
      3. Embed the summary (``embedder``) and project into the same space
         as Z to become the new centroid µ_j.

    When ``paraphraser`` or ``embedder`` is ``None`` the function reduces to
    plain k-means (mean centroid updates only).

    Distance metrics:
    - "cosine" (default): Spherical k-means. Vectors are L2-normalized and
      assignment uses dot product (equivalent to cosine similarity on unit vectors).
      Centroids are normalized after each update.
    - "euclidean": Standard k-means with Euclidean distance. No normalization.

    Args:
        Z: (n, d) embeddings in the clustering space (PCA-projected or full-dimensional).
        texts: Original texts aligned 1:1 with the rows of *Z*.
        K: Number of clusters.
        max_iter: Maximum number of iterations.
        llm_interval: How often (in iterations) to perform the LLM centroid
            update instead of the ordinary mean update.
        max_samples: Maximum number of texts sampled per cluster for the
            LLM prompt.
        seed: Random seed for reproducibility.
        paraphraser: ``texts_in → summaries_out`` callable.  Receives the
            sampled texts for **one** cluster, returns a list whose first
            element is the cluster summary string.
        embedder: ``texts_in → raw_embeddings_out`` callable.  Returns an
            array of shape ``(len(texts_in), D_original)``.
        pca_components: ``(pca_dim, D_original)`` projection matrix from the
            original PCA transform. Set to None when using full embeddings (no PCA).
        pca_mean: ``(D_original,)`` mean vector from the original PCA
            transform. Set to None when using full embeddings (no PCA).
        distance_metric: Distance metric to use ("cosine" or "euclidean").
            Default: "cosine".
        normalize_vectors: Whether to L2-normalize vectors. Default: True.
            For cosine metric, this should be True. For euclidean, typically False.

    Returns:
        Tuple of ``(labels, info_dict)`` where *info_dict* contains at least:
        - ``summaries``: ``list[str]`` — last LLM summaries per cluster
          (empty strings when LLM was never invoked for a cluster).
        - ``objective``: ``float`` — final inertia (sum of squared distances).
        - ``n_iter``: ``int`` — number of iterations executed.
    """
    rng = np.random.default_rng(seed)
    n, d = Z.shape

    if K > n:
        raise ValueError(f"K ({K}) cannot exceed number of samples ({n})")
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if distance_metric not in ("cosine", "euclidean"):
        raise ValueError(f"distance_metric must be 'cosine' or 'euclidean', got {distance_metric}")

    # Normalize Z if using cosine/spherical k-means
    use_cosine = (distance_metric == "cosine") and normalize_vectors
    Z_work = Z.copy()
    if use_cosine:
        norms = np.linalg.norm(Z_work, axis=1, keepdims=True)
        Z_work = Z_work / np.maximum(norms, 1e-12)

    # --- Initialisation (k-means++) ---
    centroids = _kmeanspp_init(Z_work, K, rng)   # (K, d)
    if use_cosine:
        # Normalize initial centroids
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(c_norms, 1e-12)
    labels = _assign(Z_work, centroids, distance_metric=distance_metric, normalize_vectors=normalize_vectors)

    summaries: list[str] = [""] * K
    use_llm = (
        paraphraser is not None
        and embedder is not None
        and pca_components is not None
        and pca_mean is not None
    )

    n_iter = 0
    for t in range(1, max_iter + 1):
        n_iter = t

        # ---- CENTROID UPDATE STEP ----
        if use_llm and t % llm_interval == 0 and t > 1:
            # LLM-based centroid replacement
            for j in range(K):
                cluster_idx = np.where(labels == j)[0]
                if len(cluster_idx) == 0:
                    continue

                n_sample = min(max_samples, len(cluster_idx))
                sampled_local = kmeanspp_sample(Z_work[cluster_idx], n_sample, rng)
                sampled_texts = [texts[cluster_idx[i]] for i in sampled_local]

                # LLM generates a single summary for this cluster
                summaries_out = paraphraser(sampled_texts)
                if isinstance(summaries_out, str):
                    summary_j = summaries_out
                elif (
                    isinstance(summaries_out, list)
                    and len(summaries_out) == 1
                    and isinstance(summaries_out[0], str)
                ):
                    summary_j = summaries_out[0]
                else:
                    raise ValueError(
                        "paraphraser must return a single cluster summary "
                        "(string or one-item list[str])."
                    )
                summaries[j] = summary_j

                # Embed summary → project into same space as data points
                raw_emb = np.asarray(
                    embedder([summary_j]), dtype=np.float64
                )
                if raw_emb.ndim == 1:
                    raw_emb = raw_emb.reshape(1, -1)
                
                # Project to PCA space only if PCA was used
                if pca_components is not None and pca_mean is not None:
                    centroid_j = (raw_emb[0] - pca_mean) @ pca_components.T
                else:
                    # No PCA: use full-dimensional embedding directly
                    centroid_j = raw_emb[0]
                
                # Normalize if using cosine/spherical k-means
                if use_cosine:
                    norm_j = np.linalg.norm(centroid_j)
                    if norm_j > 1e-12:
                        centroid_j = centroid_j / norm_j
                centroids[j] = centroid_j
        else:
            # Standard mean centroid update
            for j in range(K):
                cluster_idx = np.where(labels == j)[0]
                if len(cluster_idx) == 0:
                    continue
                centroid_j = Z_work[cluster_idx].mean(axis=0)
                # Normalize if using cosine/spherical k-means
                if use_cosine:
                    norm_j = np.linalg.norm(centroid_j)
                    if norm_j > 1e-12:
                        centroid_j = centroid_j / norm_j
                centroids[j] = centroid_j

        # ---- ASSIGNMENT STEP ----
        new_labels = _assign(Z_work, centroids, distance_metric=distance_metric, normalize_vectors=normalize_vectors)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

    # Final inertia
    if use_cosine:
        # For cosine: sum of (1 - similarity) = sum of cosine distances
        # Each point's similarity to its assigned centroid
        point_similarities = np.array([Z_work[i] @ centroids[labels[i]] for i in range(n)])
        inertia = float(np.sum(1.0 - point_similarities))
    else:
        # For Euclidean: sum of squared distances
        diffs = Z_work - centroids[labels]
        inertia = float(np.sum(diffs ** 2))

    return labels.astype(int), {
        "summaries": summaries,
        "objective": inertia,
        "n_iter": n_iter,
    }


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

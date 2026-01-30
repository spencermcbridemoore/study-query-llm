# embedding_cluster_stability.py
# -*- coding: utf-8 -*-
"""
EmbeddingClusterStabilityAnalyzer

- Accepts leaf embeddings as:
  (A) 2D: (n_items, d)   OR
  (B) 3D: (n_items, n_tokens, d)  -> mean-pooled over tokens to (n_items, d)  OR
  (C) list of 2D token arrays: [ (n_tokens_i, d), ... ] -> mean-pooled per item

- Accepts nested structures: dicts/lists containing such leaves.
- Returns identically structured output with each leaf replaced by a dict of statistics
  for increasing K across a basis set of init/update methods.

No graphs/plots are produced. Output is (mostly) JSON-safe by default; large model
parameters (subspace bases, means) are NOT stored unless you opt in.

Dependencies:
- numpy required
- sklearn optional (enables kmeans/spectral/gmm and ARI stability). If sklearn is missing,
  methods that require it are skipped and ARI becomes NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import math
import numpy as np


# Optional deps (used if available). Code falls back gracefully.
# NOTE: Lazy import to avoid pulling sklearn unless a method is used.
_SKLEARN_CACHE: Dict[str, Any] = {
    "available": None,
    "KMeans": None,
    "SpectralClustering": None,
    "GaussianMixture": None,
    "adjusted_rand_score": None,
}


def _get_sklearn_components() -> Dict[str, Any]:
    if _SKLEARN_CACHE["available"] is None:
        try:
            from sklearn.cluster import KMeans, SpectralClustering
            from sklearn.mixture import GaussianMixture
            from sklearn.metrics import adjusted_rand_score
            _SKLEARN_CACHE.update({
                "available": True,
                "KMeans": KMeans,
                "SpectralClustering": SpectralClustering,
                "GaussianMixture": GaussianMixture,
                "adjusted_rand_score": adjusted_rand_score,
            })
        except Exception:  # pragma: no cover
            _SKLEARN_CACHE.update({
                "available": False,
                "KMeans": None,
                "SpectralClustering": None,
                "GaussianMixture": None,
                "adjusted_rand_score": None,
            })
    return _SKLEARN_CACHE


Array2D = np.ndarray
Structured = Union[Array2D, List[Any], Dict[str, Any]]


# ----------------------------- token pooling -----------------------------

def mean_pool_tokens(
    data: Any,
    *,
    normalize: bool = False,
) -> Array2D:
    """
    Mean-pool token embeddings to one vector per item/string.

    Accepts:
    - np.ndarray of shape (n_items, n_tokens, d)
    - list of arrays: [ (n_tokens_i, d), ... ]

    Returns:
    - np.ndarray of shape (n_items, d)
    """
    if isinstance(data, np.ndarray):
        if data.ndim != 3:
            raise ValueError(f"Expected array (n_items, n_tokens, d); got {data.shape}")
        pooled = data.mean(axis=1)

    elif isinstance(data, list):
        pooled_list: List[np.ndarray] = []
        for i, tokens in enumerate(data):
            arr = np.asarray(tokens, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"Item {i} must be (n_tokens, d); got {arr.shape}")
            pooled_list.append(arr.mean(axis=0))
        pooled = np.stack(pooled_list, axis=0)

    else:
        raise TypeError("Unsupported token container type for mean_pool_tokens")

    if normalize:
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.maximum(norms, 1e-12)

    return pooled


# ----------------------------- helpers -----------------------------

def _as_2d_array(x: Any) -> Array2D:
    """Convert a leaf embedding container into a 2D float array (n_samples, d)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array-like of shape (n, d). Got shape={arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"Need at least 2 samples and 1 dim. Got shape={arr.shape}")
    return arr


def _l2_normalize_rows(X: Array2D, eps: float = 1e-12) -> Array2D:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _pca_project(X: Array2D, pca_dim: int) -> Tuple[Array2D, Dict[str, Any]]:
    """
    PCA via thin SVD on centered X.
    Returns Z (n, pca_dim) = U_k * S_k, and metadata.
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(pca_dim, U.shape[1]))
    Z = U[:, :k] * S[:k]  # U_k Σ_k
    meta = {
        "mean": mu.squeeze(0),
        "singular_values": S,
        "Vt": Vt,
        "pca_dim_used": k,
    }
    return Z, meta


def _pairwise_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _pairwise_std(values: List[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def _safe_ari(a: np.ndarray, b: np.ndarray) -> float:
    sklearn = _get_sklearn_components()
    if not sklearn["available"]:
        return float("nan")
    return float(sklearn["adjusted_rand_score"](a, b))


def _compute_label_stability(label_runs: List[np.ndarray]) -> Dict[str, Any]:
    """Pairwise ARI stats across restarts."""
    aris: List[float] = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            aris.append(_safe_ari(label_runs[i], label_runs[j]))

    return {
        "pairwise_ari_mean": _pairwise_mean(aris),
        "pairwise_ari_std": _pairwise_std(aris),
        "pairwise_ari_min": float(np.min(aris)) if aris else float("nan"),
        "pairwise_ari_max": float(np.max(aris)) if aris else float("nan"),
        "n_pairwise": int(len(aris)),
    }


def _cluster_size_stats(labels: np.ndarray, K: int) -> Dict[str, Any]:
    counts = np.bincount(labels, minlength=K)
    return {
        "cluster_sizes": counts.tolist(),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
        "empty_clusters": int(np.sum(counts == 0)),
        "frac_empty": float(np.mean(counts == 0)),
        "effective_clusters": int(np.sum(counts > 0)),
    }


def _within_cluster_rank_spectrum(
    X: Array2D,
    labels: np.ndarray,
    K: int,
    max_rank_eval: int = 32,
) -> Dict[str, Any]:
    """
    For each cluster, compute singular values of centered points (up to max_rank_eval),
    and report summaries + an "effective rank" proxy.
    """
    spectra: List[List[float]] = []
    eff_ranks: List[float] = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            spectra.append([])
            eff_ranks.append(float("nan"))
            continue
        Xk = X[idx]
        Xkc = Xk - Xk.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(Xkc, full_matrices=False)
        S = S[: min(len(S), max_rank_eval)]
        spectra.append(S.tolist())
        if len(S) == 0 or np.allclose(S, 0):
            eff_ranks.append(0.0)
        else:
            eff_ranks.append(float(np.sum(S * S) / (np.max(S) ** 2)))

    finite_eff = [x for x in eff_ranks if np.isfinite(x)]
    return {
        "cluster_singular_values": spectra,
        "cluster_effective_ranks": eff_ranks,
        "cluster_effective_rank_mean": float(np.mean(finite_eff)) if finite_eff else float("nan"),
        "cluster_effective_rank_min": float(np.min(finite_eff)) if finite_eff else float("nan"),
        "cluster_effective_rank_max": float(np.max(finite_eff)) if finite_eff else float("nan"),
    }


def _sklearn_available() -> bool:
    return bool(_get_sklearn_components()["available"])


def _nearest_to_mean_index(X: Array2D, idx: np.ndarray) -> int:
    Xk = X[idx]
    mu = Xk.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(Xk - mu, axis=1)
    return int(idx[int(np.argmin(dists))])


# ----------------------------- basis methods -----------------------------

@dataclass(frozen=True)
class MethodSpec:
    """
    Defines a "basis method" as an initialization + update pair (or equivalent).
    """
    name: str
    kind: str  # "kmeans", "spherical_kmeans", "pca_kmeans", "k_subspaces", "spectral_kmeans", "gmm"
    init: str
    update: str
    params: Dict[str, Any]


def default_basis_specs(pca_dim: int = 64, subspace_dim: int = 2) -> List[MethodSpec]:
    """Minimal basis set from the earlier taxonomy, represented as concrete specs."""
    return [
        MethodSpec("euclidean_kmeans", "kmeans", "kmeans++", "mean", {}),
        MethodSpec("spherical_kmeans", "spherical_kmeans", "kmeans++", "cosine_mean", {}),
        MethodSpec("pca_kmeans", "pca_kmeans", "kmeans++", "mean", {"pca_dim": int(pca_dim)}),
        MethodSpec("rank1_subspace", "k_subspaces", "svd_extrema", "rank1", {"subspace_dim": 1}),
        MethodSpec("k_subspaces", "k_subspaces", "pca_then_kmeans++", "subspace_r",
                   {"subspace_dim": int(subspace_dim), "pca_dim": int(pca_dim)}),
        MethodSpec("spectral_kmeans", "spectral_kmeans", "spectral", "mean", {}),
        MethodSpec("gmm_em", "gmm", "kmeans++", "em", {}),
    ]


# ----------------------------- clustering engines -----------------------------

def _sk_kmeans_labels(X: Array2D, K: int, seed: int, n_init: int, max_iter: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    sklearn = _get_sklearn_components()
    if not sklearn["available"]:
        raise RuntimeError("sklearn is required for KMeans-based methods, but sklearn was not importable.")
    km = sklearn["KMeans"](
        n_clusters=K,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    labels = km.fit_predict(X)
    info = {
        "objective_inertia": float(km.inertia_),
        "n_iter": int(getattr(km, "n_iter_", -1)),
    }
    return labels, info


def _init_svd_extrema(X: Array2D, K: int, seed: int) -> Array2D:
    """
    Choose K initial representatives as data points with extreme projections along top PCs.
    Deterministic given seed (ties resolved by a random permutation).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    q = int(min(max(1, K), d, 32))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vq = Vt[:q].T  # (d, q)
    proj = Xc @ Vq  # (n, q)

    cand = set()
    for j in range(q):
        cand.add(int(np.argmin(proj[:, j])))
        cand.add(int(np.argmax(proj[:, j])))

    cand = list(cand)
    rng.shuffle(cand)
    if len(cand) < K:
        extras = rng.choice(n, size=K - len(cand), replace=False).tolist()
        cand.extend(extras)
    cand = cand[:K]
    return X[cand].copy()


def _spherical_kmeans(
    X: Array2D,
    K: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Practical spherical k-means:
    - Normalize points to unit length
    - Run euclidean k-means on normalized points
    """
    Xn = _l2_normalize_rows(X)
    labels, info = _sk_kmeans_labels(Xn, K, seed=seed, n_init=n_init, max_iter=max_iter)
    # Cosine objective proxy
    # Recompute centroids from assignments and normalize
    centers = np.zeros((K, Xn.shape[1]), dtype=np.float64)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        centers[k] = Xn[idx].mean(axis=0)
    centers = _l2_normalize_rows(centers)
    cos_sim = np.sum(Xn * centers[labels], axis=1)
    info.update({
        "objective_cosine_sum": float(np.sum(cos_sim)),
        "objective_cosine_mean": float(np.mean(cos_sim)),
    })
    return labels, info


def _k_subspaces(
    X: Array2D,
    K: int,
    seed: int,
    subspace_dim: int,
    max_iter: int,
    init_mode: str,
    pca_dim_for_init: Optional[int] = None,
    *,
    return_model_params: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    K-subspaces / KLLMeans-style clustering.

    If return_model_params=False (default), does NOT return large arrays (mus/Bs),
    only scalar summaries + objective.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    r = int(max(1, min(subspace_dim, d)))

    # --- init labels ---
    if init_mode == "svd_extrema":
        init_pts = _init_svd_extrema(X, K, seed)
        dist2 = ((X[:, None, :] - init_pts[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dist2, axis=1)
    elif init_mode == "pca_then_kmeans++":
        if pca_dim_for_init is None:
            pca_dim_for_init = min(64, d)
        Z, _ = _pca_project(X, pca_dim_for_init)
        labels, _ = _sk_kmeans_labels(Z, K, seed=seed, n_init=10, max_iter=200)
    else:
        labels = rng.integers(0, K, size=n, endpoint=False)

    # model: (mu_k, B_k)
    mus = np.zeros((K, d), dtype=np.float64)
    Bs = np.zeros((K, d, r), dtype=np.float64)

    def update_subspace(k: int, idx: np.ndarray) -> None:
        if len(idx) < 2:
            mus[k] = X[rng.integers(0, n)]
            Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
            Bs[k] = Q[:, :r]
            return
        Xk = X[idx]
        mu = Xk.mean(axis=0)
        Xkc = Xk - mu
        _, _, Vt = np.linalg.svd(Xkc, full_matrices=False)
        B = Vt[:r].T
        mus[k] = mu
        Bs[k] = B

    for k in range(K):
        update_subspace(k, np.where(labels == k)[0])

    prev_labels = labels.copy()
    for it in range(max_iter):
        Xm = X[:, None, :] - mus[None, :, :]  # (n,K,d)
        proj = np.einsum("nkd,kdr->nkr", Xm, Bs)
        back = np.einsum("nkr,kdr->nkd", proj, Bs)
        resid = Xm - back
        err2 = np.einsum("nkd,nkd->nk", resid, resid)

        labels = np.argmin(err2, axis=1)

        counts = np.bincount(labels, minlength=K)
        empties = np.where(counts == 0)[0]
        if len(empties) > 0:
            assigned_err = err2[np.arange(n), labels]
            worst = np.argsort(-assigned_err)
            take = 0
            for k in empties:
                idx_pt = worst[take]
                take += 1
                labels[idx_pt] = k

        for k in range(K):
            update_subspace(k, np.where(labels == k)[0])

        if np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

    # objective: sum min reconstruction errors
    Xm = X - mus[labels]
    proj = np.einsum("nd,ndr->nr", Xm, Bs[labels])
    back = np.einsum("nr,ndr->nd", proj, Bs[labels])
    resid = Xm - back
    obj = float(np.sum(resid * resid))

    info: Dict[str, Any] = {
        "objective_recon_error_sum": obj,
        "objective_recon_error_mean": float(obj / max(1, n)),
        "n_iter": int(it + 1),
        "subspace_dim": r,
    }
    if return_model_params:
        info["mus"] = mus
        info["Bs"] = Bs
    return labels.astype(int), info


def _spectral_kmeans_labels(X: Array2D, K: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    sklearn = _get_sklearn_components()
    if not sklearn["available"]:
        raise RuntimeError("sklearn is required for SpectralClustering, but sklearn was not importable.")
    sc = sklearn["SpectralClustering"](
        n_clusters=K,
        affinity="nearest_neighbors",
        n_neighbors=min(20, max(5, X.shape[0] - 1)),
        assign_labels="kmeans",
        random_state=seed,
    )
    labels = sc.fit_predict(X)
    return labels, {"affinity": "nearest_neighbors", "assign_labels": "kmeans"}


def _gmm_labels(X: Array2D, K: int, seed: int, n_init: int, max_iter: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    sklearn = _get_sklearn_components()
    if not sklearn["available"]:
        raise RuntimeError("sklearn is required for GaussianMixture, but sklearn was not importable.")
    gm = sklearn["GaussianMixture"](
        n_components=K,
        covariance_type="full",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
        reg_covar=1e-6,
    )
    gm.fit(X)
    labels = gm.predict(X)
    info = {
        "avg_log_likelihood": float(gm.score(X)),
        "aic": float(gm.aic(X)),
        "bic": float(gm.bic(X)),
        "n_iter": int(getattr(gm, "n_iter_", -1)),
        "converged": bool(getattr(gm, "converged_", False)),
    }
    return labels, info


# ----------------------------- main analyzer -----------------------------

@dataclass
class AnalyzerConfig:
    k_min: int = 2
    k_max: int = 20
    n_restarts_base: int = 20
    n_restarts_scale_with_k: float = 1.0
    max_iter: int = 300
    seed: int = 0

    # PCA / subspace defaults
    pca_dim: int = 64
    subspace_dim: int = 2

    # Stop early if K gets too large relative to n
    stop_if_k_exceeds_fraction_of_n: float = 0.5

    # Stability breakdown heuristic
    breakdown_ari_threshold: float = 0.75
    breakdown_min_k: int = 4

    # Diagnostics
    compute_cluster_svd_diagnostics: bool = True
    max_rank_eval_per_cluster: int = 32

    # Token pooling behavior at leaves
    token_pooling: str = "mean"         # "mean" or "none"
    token_pooling_normalize: bool = False

    # JSON-safety: store large model params?
    store_large_model_params: bool = False

    # Optional post-clustering summarization
    do_summarization: bool = False
    include_rep_text: bool = False


class EmbeddingClusterStabilityAnalyzer:
    """
    Recursively analyzes nested embedding structures.

    Each leaf may be:
    - (n_items, d)
    - (n_items, n_tokens, d)
    - list of (n_tokens_i, d) arrays

    Output mirrors input structure: each leaf becomes a dict with per-method per-K stats.
    """

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        method_specs: Optional[List[MethodSpec]] = None,
        summarize_fn: Optional[Callable[..., Any]] = None,
    ):
        self.cfg = config or AnalyzerConfig()
        self.method_specs = method_specs or default_basis_specs(
            pca_dim=self.cfg.pca_dim,
            subspace_dim=self.cfg.subspace_dim,
        )
        self.summarize_fn = summarize_fn

    def analyze(self, embeddings: Structured, texts: Optional[Structured] = None) -> Structured:
        return self._map_structure_with_texts(embeddings, texts, self._analyze_leaf)

    def _map_structure(self, obj: Structured, leaf_fn: Callable[[Any], Any]) -> Structured:
        if isinstance(obj, dict):
            return {k: self._map_structure(v, leaf_fn) for k, v in obj.items()}
        if isinstance(obj, list):
            # Ambiguity: list might be nested structure OR a leaf list-of-token-arrays.
            # We treat as leaf if it looks like list of 2D arrays.
            if self._looks_like_list_of_token_arrays(obj):
                return leaf_fn(obj)
            return [self._map_structure(v, leaf_fn) for v in obj]
        return leaf_fn(obj)

    def _map_structure_with_texts(
        self,
        obj: Structured,
        texts: Optional[Structured],
        leaf_fn: Callable[[Any, Optional[List[str]]], Any],
    ) -> Structured:
        if isinstance(obj, dict):
            if texts is not None and not isinstance(texts, dict):
                raise ValueError("Texts structure must mirror embeddings structure (dict expected).")
            return {
                k: self._map_structure_with_texts(obj[k], None if texts is None else texts.get(k), leaf_fn)
                for k in obj.keys()
            }
        if isinstance(obj, list):
            # Ambiguity: list might be nested structure OR a leaf list-of-token-arrays.
            if self._looks_like_list_of_token_arrays(obj):
                return leaf_fn(obj, texts if isinstance(texts, list) else None)
            if texts is not None and not isinstance(texts, list):
                raise ValueError("Texts structure must mirror embeddings structure (list expected).")
            if texts is not None and len(texts) != len(obj):
                raise ValueError("Texts list length must match embeddings list length.")
            return [
                self._map_structure_with_texts(v, None if texts is None else texts[i], leaf_fn)
                for i, v in enumerate(obj)
            ]
        return leaf_fn(obj, texts if isinstance(texts, list) else None)

    @staticmethod
    def _looks_like_list_of_token_arrays(lst: list) -> bool:
        if len(lst) == 0:
            return False
        # Heuristic: if first element is array-like and becomes 2D, treat as token list leaf
        try:
            arr0 = np.asarray(lst[0])
            return arr0.ndim == 2
        except Exception:
            return False

    def _prep_leaf_to_2d(self, leaf: Any) -> Array2D:
        """
        Convert leaf into (n_items, d) using optional mean token pooling.
        """
        # If numpy array
        if isinstance(leaf, np.ndarray):
            if leaf.ndim == 2:
                X = leaf
            elif leaf.ndim == 3:
                if self.cfg.token_pooling != "mean":
                    raise ValueError("Leaf is 3D token embeddings but token_pooling != 'mean'.")
                X = mean_pool_tokens(leaf, normalize=self.cfg.token_pooling_normalize)
            else:
                raise ValueError(f"Unsupported ndarray leaf shape: {leaf.shape}")
            return _as_2d_array(X)

        # If list: could be list-of-token-arrays leaf OR nested structure (handled earlier)
        if isinstance(leaf, list):
            # Here we interpret it as list-of-token-arrays leaf
            if self.cfg.token_pooling != "mean":
                raise ValueError("Leaf is list-of-token-arrays but token_pooling != 'mean'.")
            X = mean_pool_tokens(leaf, normalize=self.cfg.token_pooling_normalize)
            return _as_2d_array(X)

        # Otherwise: attempt to coerce to 2D
        return _as_2d_array(leaf)

    def _validate_texts(self, texts: Optional[List[str]], n_items: int) -> List[str]:
        if texts is None:
            raise ValueError("Summarization is enabled but no texts were provided for this leaf.")
        if len(texts) != n_items:
            raise ValueError("Texts length must match number of items in embeddings leaf.")
        return [str(t) for t in texts]

    def _summarize_texts(self, texts: List[str]) -> List[str]:
        if self.summarize_fn is None:
            raise ValueError("Summarization is enabled but no summarize_fn was provided.")
        try:
            result = self.summarize_fn(texts)
            if isinstance(result, list):
                if len(result) != len(texts):
                    raise ValueError("Summarizer returned wrong number of outputs.")
                return [str(x) for x in result]
            if isinstance(result, str) and len(texts) == 1:
                return [result]
            raise ValueError("Summarizer returned unexpected output type.")
        except TypeError:
            return [str(self.summarize_fn(t)) for t in texts]

    def _summarize_representatives(
        self,
        X: Array2D,
        labels: np.ndarray,
        K: int,
        spec: MethodSpec,
        texts: List[str],
    ) -> Dict[str, Any]:
        if len(texts) != X.shape[0]:
            raise ValueError("Texts length must match embeddings length for summarization.")

        representatives: List[Dict[str, Any]] = []
        rep_texts: List[str] = []

        Z = None
        Xn = None
        if spec.kind == "pca_kmeans":
            pca_dim = int(spec.params.get("pca_dim", self.cfg.pca_dim))
            Z, _ = _pca_project(X, pca_dim)
        elif spec.kind == "spherical_kmeans":
            Xn = _l2_normalize_rows(X)

        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue

            if spec.kind == "pca_kmeans" and Z is not None:
                Zk = Z[idx]
                center = Zk.mean(axis=0, keepdims=True)
                dists = np.linalg.norm(Zk - center, axis=1)
                rep_index = int(idx[int(np.argmin(dists))])
            elif spec.kind == "spherical_kmeans" and Xn is not None:
                Xk = Xn[idx]
                center = Xk.mean(axis=0)
                center = center / max(np.linalg.norm(center), 1e-12)
                dists = 1.0 - (Xk @ center)
                rep_index = int(idx[int(np.argmin(dists))])
            elif spec.kind == "k_subspaces":
                if len(idx) < 2:
                    rep_index = _nearest_to_mean_index(X, idx)
                else:
                    r = int(max(1, min(spec.params.get("subspace_dim", 1), X.shape[1])))
                    Xk = X[idx]
                    mu = Xk.mean(axis=0, keepdims=True)
                    Xkc = Xk - mu
                    _, _, Vt = np.linalg.svd(Xkc, full_matrices=False)
                    B = Vt[:r].T
                    Xm = Xk - mu
                    proj = Xm @ B
                    back = proj @ B.T
                    resid = Xm - back
                    err2 = np.sum(resid * resid, axis=1)
                    rep_index = int(idx[int(np.argmin(err2))])
            else:
                rep_index = _nearest_to_mean_index(X, idx)

            rep_texts.append(texts[rep_index])
            rep_entry: Dict[str, Any] = {
                "cluster_id": int(k),
                "rep_index": int(rep_index),
                "cluster_size": int(len(idx)),
            }
            if self.cfg.include_rep_text:
                rep_entry["rep_text"] = texts[rep_index]
            representatives.append(rep_entry)

        summaries = self._summarize_texts(rep_texts)
        for rep_entry, summary in zip(representatives, summaries):
            rep_entry["rep_summary"] = str(summary)

        return {
            "K": int(K),
            "representatives": representatives,
        }

    def _analyze_leaf(self, leaf: Any, texts: Optional[List[str]] = None) -> Dict[str, Any]:
        X = self._prep_leaf_to_2d(leaf)
        n, d = X.shape
        text_list: Optional[List[str]] = None
        if self.cfg.do_summarization:
            text_list = self._validate_texts(texts, n)

        k_min = int(max(2, self.cfg.k_min))
        k_max = int(min(self.cfg.k_max, n - 1))
        if n > 2:
            k_max = int(min(k_max, math.floor(self.cfg.stop_if_k_exceeds_fraction_of_n * n)))
        k_max = max(k_min, k_max)

        results: Dict[str, Any] = {
            "shape": {"n": int(n), "d": int(d)},
            "config": self.cfg.__dict__.copy(),
            "methods": {},
        }

        for spec in self.method_specs:
            if spec.kind in ("kmeans", "spherical_kmeans", "pca_kmeans") and not _sklearn_available():
                continue
            if spec.kind == "spectral_kmeans" and not _sklearn_available():
                continue
            if spec.kind == "gmm" and not _sklearn_available():
                continue

            results["methods"][spec.name] = self._analyze_method(X, spec, k_min, k_max, text_list)

        return results

    def _n_restarts_for_k(self, K: int) -> int:
        return int(max(1, self.cfg.n_restarts_base + round(self.cfg.n_restarts_scale_with_k * K)))

    def _analyze_method(
        self,
        X: Array2D,
        spec: MethodSpec,
        k_min: int,
        k_max: int,
        texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        method_res: Dict[str, Any] = {
            "spec": {
                "name": spec.name,
                "kind": spec.kind,
                "init": spec.init,
                "update": spec.update,
                "params": spec.params.copy(),
            },
            "by_k": {},
            "breakdown": None,
        }

        breakdown_k: Optional[int] = None

        for K in range(k_min, k_max + 1):
            n_restarts = self._n_restarts_for_k(K)
            base = self.cfg.seed + (abs(hash(spec.name)) % 10_000) * 100_000
            seeds = [base + r for r in range(n_restarts)]

            label_runs: List[np.ndarray] = []
            obj_vals: List[float] = []
            aux_infos: List[Dict[str, Any]] = []

            for s in seeds:
                labels, info = self._run_once(X, spec, K, seed=int(s))
                label_runs.append(labels)

                obj = None
                for key in ("objective_inertia", "objective_cosine_sum", "objective_recon_error_sum", "avg_log_likelihood"):
                    if key in info:
                        obj = float(info[key])
                        break
                obj_vals.append(float(obj) if obj is not None else float("nan"))
                aux_infos.append(info)

            stability = _compute_label_stability(label_runs)
            size_stats = _cluster_size_stats(label_runs[0], K)

            out_k: Dict[str, Any] = {
                "K": int(K),
                "n_restarts": int(n_restarts),
                "objective": {
                    "mean": float(np.mean(obj_vals)),
                    "std": float(np.std(obj_vals, ddof=1)) if len(obj_vals) > 1 else float("nan"),
                    "min": float(np.min(obj_vals)),
                    "max": float(np.max(obj_vals)),
                },
                "stability": stability,
                "cluster_sizes_run0": size_stats,
            }

            if self.cfg.compute_cluster_svd_diagnostics:
                out_k["cluster_svd_diagnostics_run0"] = _within_cluster_rank_spectrum(
                    X=X,
                    labels=label_runs[0],
                    K=K,
                    max_rank_eval=self.cfg.max_rank_eval_per_cluster,
                )

            if self.cfg.do_summarization:
                out_k["summary"] = self._summarize_representatives(
                    X=X,
                    labels=label_runs[0],
                    K=K,
                    spec=spec,
                    texts=texts or [],
                )

            method_res["by_k"][str(K)] = out_k

            if (
                breakdown_k is None
                and K >= self.cfg.breakdown_min_k
                and np.isfinite(stability["pairwise_ari_mean"])
                and stability["pairwise_ari_mean"] < self.cfg.breakdown_ari_threshold
            ):
                breakdown_k = K

        if breakdown_k is not None:
            method_res["breakdown"] = {
                "breakdown_k": int(breakdown_k),
                "criterion": f"pairwise_ari_mean < {self.cfg.breakdown_ari_threshold}",
            }

        return method_res

    def _run_once(self, X: Array2D, spec: MethodSpec, K: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        max_iter = int(self.cfg.max_iter)

        if spec.kind == "kmeans":
            labels, info = _sk_kmeans_labels(X, K, seed=seed, n_init=1, max_iter=max_iter)
            return labels, info

        if spec.kind == "spherical_kmeans":
            labels, info = _spherical_kmeans(X, K, seed=seed, n_init=1, max_iter=max_iter)
            return labels, info

        if spec.kind == "pca_kmeans":
            pca_dim = int(spec.params.get("pca_dim", self.cfg.pca_dim))
            Z, pmeta = _pca_project(X, pca_dim)
            labels, info = _sk_kmeans_labels(Z, K, seed=seed, n_init=1, max_iter=max_iter)
            info = {
                **info,
                "pca": {
                    "pca_dim": int(pmeta["pca_dim_used"]),
                    "singular_values": pmeta["singular_values"].tolist(),
                },
            }
            return labels, info

        if spec.kind == "k_subspaces":
            subspace_dim = int(spec.params.get("subspace_dim", 1))
            init_mode = str(spec.init)
            pca_dim_for_init = spec.params.get("pca_dim", None)
            labels, info = _k_subspaces(
                X,
                K,
                seed=seed,
                subspace_dim=subspace_dim,
                max_iter=max_iter,
                init_mode=init_mode,
                pca_dim_for_init=int(pca_dim_for_init) if pca_dim_for_init is not None else None,
                return_model_params=bool(self.cfg.store_large_model_params),
            )
            return labels, info

        if spec.kind == "spectral_kmeans":
            labels, info = _spectral_kmeans_labels(X, K, seed=seed)
            return labels, info

        if spec.kind == "gmm":
            labels, info = _gmm_labels(X, K, seed=seed, n_init=1, max_iter=max_iter)
            return labels, info

        raise ValueError(f"Unknown method kind: {spec.kind}")


# ----------------------------- quick usage example -----------------------------
if __name__ == "__main__":
    cfg = AnalyzerConfig(k_min=2, k_max=10, n_restarts_base=10, n_restarts_scale_with_k=1.0)
    analyzer = EmbeddingClusterStabilityAnalyzer(cfg)

    # Example leaf: token embeddings (n_items, n_tokens, d)
    X_tokens = np.random.randn(50, 20, 128)
    out = analyzer.analyze({"example": X_tokens})
    # Print keys only (avoid huge output)
    print(out["example"]["shape"], list(out["example"]["methods"].keys()))

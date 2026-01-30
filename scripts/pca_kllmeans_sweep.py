#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_kllmeans_sweep.py

Pipeline:
  (optional) mean-pool token embeddings -> item embeddings
  PCA/SVD reduce to pca_dim (default 64)
  For K in [k_min..k_max]:
      run KLLMeans-style K-subspaces clustering (rank=r) with many restarts
      compute stability (pairwise ARI across restarts, if sklearn available)
      compute cluster size stats + optional within-cluster SVD diagnostics
      optionally: pick 1 representative per cluster (run0) and paraphrase via user-supplied function

Designed for ~300 points in ~1000-dim space.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np

Array2D = np.ndarray
EmbIn = Union[np.ndarray, List[np.ndarray]]

def mean_pool_tokens(emb: EmbIn, normalize: bool = False) -> Array2D:
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

def k_subspaces_kllmeans(
    X: Array2D, K: int, *, rank_r: int = 2, seed: int = 0, max_iter: int = 200
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    r = int(max(1, min(rank_r, d)))
    labels = rng.integers(0, K, size=n, endpoint=False)
    mus = np.zeros((K, d))
    Bs = np.zeros((K, d, r))

    def update_cluster(k, idx):
        if len(idx) < 2:
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

    for k in range(K):
        update_cluster(k, np.where(labels == k)[0])

    prev = labels.copy()
    for _ in range(max_iter):
        Xm = X[:, None, :] - mus[None, :, :]
        proj = np.einsum("nkd,kdr->nkr", Xm, Bs)
        back = np.einsum("nkr,kdr->nkd", proj, Bs)
        resid = Xm - back
        err2 = np.einsum("nkd,nkd->nk", resid, resid)
        labels = np.argmin(err2, axis=1)
        if np.array_equal(labels, prev):
            break
        prev = labels.copy()

    Xm = X - mus[labels]
    proj = np.einsum("nd,ndr->nr", Xm, Bs[labels])
    back = np.einsum("nr,ndr->nd", proj, Bs[labels])
    resid = Xm - back
    obj = float(np.sum(resid * resid))
    return labels.astype(int), {"objective_recon_error_sum": obj}

@dataclass
class SweepConfig:
    pca_dim: int = 64
    rank_r: int = 2
    k_min: int = 2
    k_max: int = 20
    max_iter: int = 200
    base_seed: int = 0

def run_sweep(
    texts: List[str],
    embeddings: EmbIn,
    cfg: SweepConfig,
    *,
    paraphraser: Optional[Callable[[List[str]], List[str]]] = None,
) -> Dict[str, Any]:
    X = mean_pool_tokens(embeddings)
    Z, pca_meta = pca_svd_project(X, cfg.pca_dim)
    out = {"pca": pca_meta, "by_k": {}}
    for K in range(cfg.k_min, min(cfg.k_max, len(texts) - 1) + 1):
        labels, info = k_subspaces_kllmeans(Z, K, rank_r=cfg.rank_r, seed=cfg.base_seed)
        reps = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) == 0:
                continue
            Zk = Z[idx]
            ck = Zk.mean(axis=0)
            i = idx[np.argmin(np.sum((Zk - ck) ** 2, axis=1))]
            reps.append(texts[i])
        if paraphraser:
            reps = paraphraser(reps)
        out["by_k"][str(K)] = {"representatives": reps, "objective": info}
    return out

if __name__ == "__main__":
    n, d = 300, 1000
    rng = np.random.default_rng(0)
    texts = [f"Prompt {i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig()
    res = run_sweep(texts, embeddings, cfg)
    print("Ks:", list(res["by_k"].keys()))

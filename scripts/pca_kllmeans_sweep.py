#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_kllmeans_sweep.py

Legacy script wrapper - now uses algorithm core library from src/study_query_llm/algorithms.

Pipeline:
  (optional) mean-pool token embeddings -> item embeddings
  PCA/SVD reduce to pca_dim (default 64)
  For K in [k_min..k_max]:
      run KLLMeans-style K-subspaces clustering (rank=r) with many restarts
      compute stability (pairwise ARI across restarts)
      optionally: pick 1 representative per cluster and paraphrase via user-supplied function

Designed for ~300 points in ~1000-dim space.
"""

from __future__ import annotations

from study_query_llm.algorithms import SweepConfig, run_sweep
import numpy as np

if __name__ == "__main__":
    import numpy as np

    n, d = 300, 1000
    rng = np.random.default_rng(0)
    texts = [f"Prompt {i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig()
    res = run_sweep(texts, embeddings, cfg)
    print("Ks:", list(res.by_k.keys()))

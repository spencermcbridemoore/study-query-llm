"""
Tests for sweep orchestration.
"""

import numpy as np
import pytest

from study_query_llm.algorithms.sweep import SweepConfig, SweepResult, run_sweep


def test_sweep_config_defaults():
    """Test SweepConfig default values."""
    cfg = SweepConfig()
    assert cfg.pca_dim == 64
    assert cfg.k_min == 2
    assert cfg.k_max == 20
    assert cfg.max_iter == 200
    assert cfg.base_seed == 0
    assert cfg.n_restarts == 1
    assert cfg.compute_stability is False
    assert cfg.llm_interval == 20
    assert cfg.max_samples == 10
    assert cfg.distance_metric == "cosine"
    assert cfg.normalize_vectors is True


def test_run_sweep_basic():
    """Test basic sweep without stability metrics."""
    n, d = 30, 10
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=5, n_restarts=1)

    result = run_sweep(texts, embeddings, cfg)

    assert isinstance(result, SweepResult)
    assert "pca_dim_used" in result.pca
    assert len(result.by_k) == 4  # K=2,3,4,5
    assert "2" in result.by_k
    assert "representatives" in result.by_k["2"]
    assert "summaries" in result.by_k["2"]
    assert result.Z is None  # Not computed when stability=False


def test_run_sweep_with_stability():
    """Test sweep with stability metrics."""
    n, d = 30, 10
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(
        k_min=2, k_max=4, n_restarts=3, compute_stability=True, coverage_threshold=0.2
    )

    result = run_sweep(texts, embeddings, cfg)

    assert result.Z is not None
    assert result.Z_norm is not None
    assert result.dist is not None
    assert "stability" in result.by_k["2"]
    assert "silhouette" in result.by_k["2"]["stability"]
    assert "stability_ari" in result.by_k["2"]["stability"]


def test_run_sweep_with_paraphraser():
    """Test sweep with paraphraser only (no embedder) → post-hoc paraphrase."""
    n, d = 20, 8
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3)

    def paraphraser(texts_list):
        return [f"paraphrased_{t}" for t in texts_list]

    result = run_sweep(texts, embeddings, cfg, paraphraser=paraphraser)

    reps = result.by_k["2"]["representatives"]
    assert all(r.startswith("paraphrased_") for r in reps)


def test_run_sweep_with_embedder():
    """Test sweep with both paraphraser and embedder → LLM-in-the-loop."""
    n, d = 40, 8
    rng_data = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng_data.standard_normal((n, d))
    cfg = SweepConfig(
        k_min=2, k_max=3,
        n_restarts=2,
        compute_stability=True,
        llm_interval=2,   # fire LLM early so it triggers during iteration
        max_samples=3,
    )

    calls = {"paraphraser": 0, "embedder": 0}

    def paraphraser(texts_list):
        calls["paraphraser"] += 1
        return [f"summarized_{len(texts_list)}_texts"]

    def embedder(texts_list):
        calls["embedder"] += 1
        rng_emb = np.random.default_rng(hash(texts_list[0]) % (2**31))
        return rng_emb.standard_normal((len(texts_list), d))

    result = run_sweep(
        texts, embeddings, cfg, paraphraser=paraphraser, embedder=embedder
    )

    # LLM should have been invoked inside k_llmmeans
    assert calls["paraphraser"] >= 1
    assert calls["embedder"] >= 1

    # Summaries should be present
    summaries = result.by_k["2"]["summaries"]
    assert isinstance(summaries, list)
    assert any(s != "" for s in summaries)

    # Representatives come from select_representatives (actual texts)
    reps = result.by_k["2"]["representatives"]
    assert all(r in texts for r in reps)

    # Stability metrics exist
    assert "stability" in result.by_k["2"]
    assert "silhouette" in result.by_k["2"]["stability"]


def test_run_sweep_embedder_without_paraphraser():
    """Test that embedder is only used when paraphraser is also provided."""
    n, d = 20, 8
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3, llm_interval=2)

    calls = {"count": 0}

    def embedder(texts_list):
        calls["count"] += 1
        return rng.standard_normal((len(texts_list), d))

    # Without paraphraser, embedder should not be called (LLM branch skipped)
    result = run_sweep(texts, embeddings, cfg, embedder=embedder)

    assert calls["count"] == 0
    reps = result.by_k["2"]["representatives"]
    assert all(r in texts for r in reps)


def test_run_sweep_validation():
    """Test sweep input validation."""
    n, d = 10, 5
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))

    # k_min > k_max should fail
    cfg = SweepConfig(k_min=5, k_max=3)
    with pytest.raises(ValueError, match="k_min.*k_max"):
        run_sweep(texts, embeddings, cfg)

    # k_max >= n_samples should fail
    cfg = SweepConfig(k_max=n)
    with pytest.raises(ValueError, match="k_max.*n_samples"):
        run_sweep(texts, embeddings, cfg)


def test_run_sweep_best_restart_selected():
    """Multi-restart should pick the run with lowest inertia."""
    n, d = 30, 8
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3, n_restarts=5)

    result = run_sweep(texts, embeddings, cfg)

    for k_str in result.by_k:
        entry = result.by_k[k_str]
        best_obj = entry["objective"]
        all_objs = entry["objectives"]
        # The selected objective should be the minimum across restarts
        assert best_obj == pytest.approx(min(all_objs))


def test_run_sweep_cosine_default():
    """Default sweep should use cosine/spherical k-means."""
    n, d = 30, 10
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3, n_restarts=1)

    result = run_sweep(texts, embeddings, cfg)

    # Should complete successfully with default cosine
    assert len(result.by_k) == 2
    assert "2" in result.by_k
    assert result.by_k["2"]["objective"] >= 0


def test_run_sweep_euclidean_override():
    """Sweep with Euclidean metric should work."""
    n, d = 30, 10
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(
        k_min=2, k_max=3, n_restarts=1,
        distance_metric="euclidean", normalize_vectors=False
    )

    result = run_sweep(texts, embeddings, cfg)

    # Should complete successfully with Euclidean
    assert len(result.by_k) == 2
    assert "2" in result.by_k
    assert result.by_k["2"]["objective"] >= 0

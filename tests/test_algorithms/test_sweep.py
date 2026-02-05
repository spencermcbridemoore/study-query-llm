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
    assert cfg.rank_r == 2
    assert cfg.k_min == 2
    assert cfg.k_max == 20
    assert cfg.max_iter == 200
    assert cfg.base_seed == 0
    assert cfg.n_restarts == 1
    assert cfg.compute_stability is False


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
    """Test sweep with paraphraser function."""
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
    """Test sweep with embedder function for re-embedding summarized representatives."""
    n, d = 20, 8
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3, n_restarts=2, compute_stability=True)

    def paraphraser(texts_list):
        return [f"summarized_{t}" for t in texts_list]

    # Create an embedder that returns different embeddings for summarized texts
    def embedder(texts_list):
        # Return embeddings that are different from original (shifted)
        base_embeddings = rng.standard_normal((len(texts_list), d))
        # Add a shift to make them clearly different
        return base_embeddings + 1.0

    result = run_sweep(texts, embeddings, cfg, paraphraser=paraphraser, embedder=embedder)

    # Verify representatives are summarized
    reps = result.by_k["2"]["representatives"]
    assert all(r.startswith("summarized_") for r in reps)

    # Verify stability metrics exist (computed from re-embedded summaries)
    assert "stability" in result.by_k["2"]
    assert "silhouette" in result.by_k["2"]["stability"]


def test_run_sweep_embedder_without_paraphraser():
    """Test that embedder is only used when paraphraser is also provided."""
    n, d = 20, 8
    rng = np.random.default_rng(42)
    texts = [f"text_{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, d))
    cfg = SweepConfig(k_min=2, k_max=3)

    def embedder(texts_list):
        return rng.standard_normal((len(texts_list), d))

    # Without paraphraser, embedder should be ignored
    result = run_sweep(texts, embeddings, cfg, embedder=embedder)

    # Representatives should be original texts (not re-embedded)
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

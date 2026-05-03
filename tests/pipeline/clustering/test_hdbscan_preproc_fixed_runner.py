"""Per-family tests for ``run_hdbscan_preproc_fixed_analysis`` (Wave 1 PR3)."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.hdbscan_preproc_fixed_runner import (
    run_hdbscan_preproc_fixed_analysis,
)


@pytest.fixture(autouse=True)
def _fake_hdbscan_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the optional ``hdbscan`` import for deterministic, env-free tests."""

    class _FakeHDBSCAN:
        def __init__(self, **kwargs: Any) -> None:
            self._kwargs = kwargs

        def fit_predict(self, x: np.ndarray) -> np.ndarray:
            n = int(x.shape[0])
            labels = np.zeros(n, dtype=np.int64)
            if n >= 5:
                labels[-1] = -1
            return labels

    monkeypatch.setitem(
        sys.modules,
        "hdbscan",
        types.SimpleNamespace(HDBSCAN=_FakeHDBSCAN),
    )


def _toy_embeddings(n_samples: int = 12, n_features: int = 8, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def _invoke(method_name: str, matrix: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    return run_hdbscan_preproc_fixed_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )


@pytest.mark.parametrize(
    "method_name",
    [
        "hdbscan+fixed",
        "hdbscan+normalize+fixed",
        "hdbscan+pca+fixed",
        "hdbscan+normalize+pca+fixed",
    ],
)
def test_runner_is_deterministic_with_fake_hdbscan(method_name: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"min_cluster_size": 3}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    a = _invoke(method_name, matrix, params)
    b = _invoke(method_name, matrix, params)
    assert (
        a["structured_results"]["hdbscan_cluster_labels"]["cluster_labels"]
        == b["structured_results"]["hdbscan_cluster_labels"]["cluster_labels"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"min_cluster_size": 1},
        {"min_cluster_size": None},
    ],
)
def test_runner_validates_min_cluster_size(params: dict[str, Any]) -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError):
        _invoke("hdbscan+fixed", matrix, params)


@pytest.mark.parametrize(
    ("method_name", "bad_param"),
    [
        ("hdbscan+fixed", "pca_n_components"),
        ("hdbscan+pca+fixed", "normalize_embeddings"),
    ],
)
def test_runner_rejects_chain_conflicting_parameters(method_name: str, bad_param: str) -> None:
    matrix = _toy_embeddings()
    params: dict[str, Any] = {"min_cluster_size": 3}
    params[bad_param] = True if bad_param.endswith("embeddings") else 4
    with pytest.raises(ValueError):
        _invoke(method_name, matrix, params)


@pytest.mark.parametrize(
    "alias_key",
    ["hdbscan_normalize_embeddings", "normalize_embeddings"],
)
def test_runner_fails_loud_on_normalize_alias(alias_key: str) -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="encodes normalize via the method name"):
        _invoke(
            "hdbscan+normalize+fixed",
            matrix,
            {"min_cluster_size": 3, alias_key: True},
        )


def test_runner_rejects_non_full_representation() -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="representation 'full'"):
        run_hdbscan_preproc_fixed_analysis(
            method_name="hdbscan+fixed",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={"representation": "stratified_sample"},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"min_cluster_size": 3},
        )


def test_pca_n_components_is_required_on_pca_methods() -> None:
    matrix = _toy_embeddings()
    with pytest.raises(ValueError, match="pca_n_components"):
        _invoke("hdbscan+pca+fixed", matrix, {"min_cluster_size": 3})


def test_pca_n_components_over_cap_raises() -> None:
    matrix = _toy_embeddings(n_samples=12, n_features=8)
    with pytest.raises(ValueError, match="exceeds max"):
        _invoke(
            "hdbscan+pca+fixed",
            matrix,
            {"min_cluster_size": 3, "pca_n_components": 999},
        )


def test_artifact_shape_parity_with_baseline_hdbscan() -> None:
    matrix = _toy_embeddings()
    result = _invoke(
        "hdbscan+normalize+pca+fixed",
        matrix,
        {"min_cluster_size": 3, "pca_n_components": 4},
    )
    artifact_keys = set(result["artifacts"])
    assert "hdbscan_summary.json" in artifact_keys
    assert "hdbscan_labels.json" in artifact_keys
    assert result["result_ref"] == "hdbscan_summary.json"
    summary = result["structured_results"]["hdbscan_summary"]
    assert summary["base_algorithm"] == "hdbscan"
    for key in ("method_name", "n_samples", "n_features", "cluster_ids", "parameters"):
        assert key in summary


@pytest.mark.parametrize(
    ("method_name", "expected_n_features"),
    [
        ("hdbscan+fixed", 8),
        ("hdbscan+normalize+fixed", 8),
        ("hdbscan+pca+fixed", 4),
        ("hdbscan+normalize+pca+fixed", 4),
    ],
)
def test_summary_n_features_reflects_chain_applied(
    method_name: str, expected_n_features: int
) -> None:
    matrix = _toy_embeddings(n_samples=12, n_features=8)
    params: dict[str, Any] = {"min_cluster_size": 3}
    if "pca" in method_name:
        params["pca_n_components"] = 4
    result = _invoke(method_name, matrix, params)
    summary = result["structured_results"]["hdbscan_summary"]
    assert summary["n_features"] == expected_n_features
    assert summary["parameters"]["hdbscan_normalize_embeddings"] is False


def test_noise_label_negative_one_is_recorded_via_fake_hdbscan() -> None:
    matrix = _toy_embeddings(n_samples=10)
    result = _invoke("hdbscan+fixed", matrix, {"min_cluster_size": 3})
    labels = result["structured_results"]["hdbscan_cluster_labels"]["cluster_labels"]
    summary = result["structured_results"]["hdbscan_summary"]
    assert -1 in labels
    assert summary["noise_count"] >= 1
    assert summary["noise_fraction"] > 0.0
    for cid in summary["cluster_ids"]:
        assert int(cid) >= 0


def test_labels_payload_carries_noise_label_sentinel() -> None:
    matrix = _toy_embeddings(n_samples=10)
    result = _invoke("hdbscan+fixed", matrix, {"min_cluster_size": 3})
    labels_obj = result["structured_results"]["hdbscan_cluster_labels"]
    assert labels_obj["noise_label"] == -1

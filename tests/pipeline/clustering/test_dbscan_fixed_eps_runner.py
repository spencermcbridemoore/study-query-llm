"""Per-family tests for ``run_dbscan_fixed_eps_analysis`` (Wave 1 PR1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from study_query_llm.pipeline.clustering.dbscan_fixed_eps_runner import (
    run_dbscan_fixed_eps_analysis,
)


def _two_blob_embeddings() -> np.ndarray:
    a = np.tile(np.array([0.0, 0.0]), (5, 1)) + np.linspace(-0.05, 0.05, 5)[:, None]
    b = np.tile(np.array([10.0, 10.0]), (5, 1)) + np.linspace(-0.05, 0.05, 5)[:, None]
    return np.vstack([a, b])


def _texts(n: int) -> list[str]:
    return [f"t{i}" for i in range(n)]


def test_dbscan_is_deterministic_for_same_input() -> None:
    matrix = _two_blob_embeddings()
    params: dict[str, Any] = {"eps": 1.0, "min_samples": 2}
    a = run_dbscan_fixed_eps_analysis(
        method_name="dbscan+fixed-eps",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    b = run_dbscan_fixed_eps_analysis(
        method_name="dbscan+fixed-eps",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    assert (
        a["structured_results"]["clustering_labels"]["cluster_labels"]
        == b["structured_results"]["clustering_labels"]["cluster_labels"]
    )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"eps": 0.5},
        {"min_samples": 2},
        {"eps": 0.0, "min_samples": 2},
        {"eps": 0.5, "min_samples": 0},
    ],
)
def test_dbscan_validates_required_and_bounds(params: dict[str, Any]) -> None:
    matrix = _two_blob_embeddings()
    with pytest.raises(ValueError):
        run_dbscan_fixed_eps_analysis(
            method_name="dbscan+fixed-eps",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters=params,
        )


@pytest.mark.parametrize(
    ("method_name", "bad_param"),
    [
        ("dbscan+fixed-eps", "pca_n_components"),
        ("dbscan+fixed-eps", "normalize_embeddings"),
        ("dbscan+normalize+fixed-eps", "pca_n_components"),
    ],
)
def test_dbscan_rejects_chain_conflicting_parameters(method_name: str, bad_param: str) -> None:
    matrix = _two_blob_embeddings()
    params: dict[str, Any] = {"eps": 1.0, "min_samples": 2}
    params[bad_param] = True if bad_param.endswith("embeddings") else 4
    with pytest.raises(ValueError):
        run_dbscan_fixed_eps_analysis(
            method_name=method_name,
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters=params,
        )


@pytest.mark.parametrize(
    ("method_name", "expected_chain"),
    [
        ("dbscan+fixed-eps", []),
        ("dbscan+normalize+fixed-eps", ["normalize"]),
        ("dbscan+pca+fixed-eps", ["pca"]),
        ("dbscan+normalize+pca+fixed-eps", ["normalize", "pca"]),
    ],
)
def test_summary_records_chain_actually_applied(
    method_name: str, expected_chain: list[str]
) -> None:
    matrix = _two_blob_embeddings()
    params: dict[str, Any] = {"eps": 1.0, "min_samples": 2}
    if "pca" in method_name:
        params["pca_n_components"] = 1
    result = run_dbscan_fixed_eps_analysis(
        method_name=method_name,
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters=params,
    )
    summary = result["structured_results"]["clustering_summary"]
    assert summary["parameters"]["preprocess_applied"] == expected_chain


def test_pca_n_components_is_required_on_pca_methods() -> None:
    matrix = _two_blob_embeddings()
    with pytest.raises(ValueError, match="pca_n_components"):
        run_dbscan_fixed_eps_analysis(
            method_name="dbscan+pca+fixed-eps",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"eps": 1.0, "min_samples": 2},
        )


def test_pca_n_components_over_cap_raises() -> None:
    matrix = _two_blob_embeddings()
    with pytest.raises(ValueError, match="exceeds max"):
        run_dbscan_fixed_eps_analysis(
            method_name="dbscan+pca+fixed-eps",
            input_group_id=1,
            input_group_type="embedding_batch",
            input_group_metadata={},
            embeddings=matrix,
            texts=_texts(int(matrix.shape[0])),
            parameters={"eps": 1.0, "min_samples": 2, "pca_n_components": 999},
        )


def test_dbscan_records_noise_label_negative_one_in_accounting() -> None:
    matrix = np.vstack(
        [
            _two_blob_embeddings(),
            np.array([[100.0, -100.0]]),
        ]
    )
    result = run_dbscan_fixed_eps_analysis(
        method_name="dbscan+fixed-eps",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={"eps": 1.0, "min_samples": 2},
    )
    summary = result["structured_results"]["clustering_summary"]
    labels = result["structured_results"]["clustering_labels"]["cluster_labels"]
    assert -1 in labels
    assert summary["noise_count"] >= 1
    assert summary["noise_fraction"] > 0.0
    for cid in summary["cluster_ids"]:
        assert int(cid) >= 0


def test_artifact_and_summary_shape() -> None:
    matrix = _two_blob_embeddings()
    result = run_dbscan_fixed_eps_analysis(
        method_name="dbscan+normalize+pca+fixed-eps",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={"representation": "full"},
        embeddings=matrix,
        texts=_texts(int(matrix.shape[0])),
        parameters={"eps": 1.0, "min_samples": 2, "pca_n_components": 1},
    )
    assert set(result["artifacts"]) == {"dbscan_summary.json", "dbscan_labels.json"}
    assert result["result_ref"] == "dbscan_summary.json"
    summary = result["structured_results"]["clustering_summary"]
    assert summary["base_algorithm"] == "dbscan"

"""Tests for clustering provenance resolver, hashing, and canonical runners."""

from __future__ import annotations

import numpy as np
import pytest

from study_query_llm.algorithms.recipes import canonical_recipe_hash
from study_query_llm.pipeline.clustering import (
    build_effective_recipe_payload,
    build_pipeline_effective_hash,
    load_rule_set,
    resolve_clustering_resolution,
    run_gmm_bic_argmin_analysis,
    run_kmeans_silhouette_kneedle_analysis,
    validate_identity_contract,
    validate_post_selection,
)


def _toy_embeddings() -> np.ndarray:
    cluster_a = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, -0.1],
            [0.3, 0.0],
            [0.15, 0.2],
            [0.05, -0.2],
        ],
        dtype=np.float64,
    )
    cluster_b = np.asarray(
        [
            [5.0, 5.1],
            [4.9, 4.8],
            [5.2, 5.0],
            [4.8, 5.2],
            [5.1, 4.9],
            [5.3, 5.1],
        ],
        dtype=np.float64,
    )
    return np.vstack([cluster_a, cluster_b])


def test_resolver_deterministic_for_identical_inputs() -> None:
    rule_set = load_rule_set()
    context = {"embedding_dim": 768, "n_samples": 128}
    params = {"dataset_slug": "twenty_newsgroups_6cat_research"}
    first = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters=params,
        rule_set=rule_set,
        context=context,
    )
    second = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters=params,
        rule_set=rule_set,
        context=context,
    )
    assert first.rule_set_hash == second.rule_set_hash
    assert first.pipeline_declared == second.pipeline_declared
    assert first.pipeline_resolved == second.pipeline_resolved
    assert first.pipeline_effective == second.pipeline_effective


def test_resolver_missing_context_raises() -> None:
    rule_set = load_rule_set()
    with pytest.raises(ValueError, match="Missing required rule input"):
        resolve_clustering_resolution(
            method_name="kmeans+silhouette+kneedle",
            parameters={},
            rule_set=rule_set,
            context={"embedding_dim": 768},
        )


def test_resolver_aliases_when_pca_is_skipped() -> None:
    rule_set = load_rule_set()
    context = {"embedding_dim": 128, "n_samples": 200}
    with_pca = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters={},
        rule_set=rule_set,
        context=context,
    )
    without_pca = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters={"pipeline_declared": ["embed", "normalize", "kmeans"]},
        rule_set=rule_set,
        context=context,
    )
    assert with_pca.pipeline_effective == without_pca.pipeline_effective


def test_pipeline_hash_matches_effective_recipe_hash() -> None:
    rule_set = load_rule_set()
    resolution = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters={},
        rule_set=rule_set,
        context={"embedding_dim": 768, "n_samples": 256},
    )
    for entry in resolution.pipeline_resolved:
        if entry["stage"] == "kmeans":
            entry["params"]["k"] = 10
            break
    pipeline_hash = build_pipeline_effective_hash(
        resolution.pipeline_resolved,
        resolution.pipeline_effective,
    )
    recipe_payload = build_effective_recipe_payload(
        resolution.pipeline_resolved,
        resolution.pipeline_effective,
    )
    recipe_hash = canonical_recipe_hash(recipe_payload)
    assert pipeline_hash == recipe_hash
    validate_identity_contract(
        pipeline_effective_hash=pipeline_hash,
        recipe_hash=recipe_hash,
    )


def test_validate_post_selection_rejects_out_of_range_choice() -> None:
    rule_set = load_rule_set()
    resolution = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters={},
        rule_set=rule_set,
        context={"embedding_dim": 768, "n_samples": 200},
    )
    with pytest.raises(ValueError, match="must be in sweep range"):
        validate_post_selection(
            resolution,
            selection_evidence={
                "sweep_range": [2, 3, 4],
                "chosen_k": 10,
                "selection_curve_artifact_ref": "curve.json",
            },
        )


def test_kmeans_runner_reproducible_selection() -> None:
    matrix = _toy_embeddings()
    rule_set = load_rule_set()
    resolution = resolve_clustering_resolution(
        method_name="kmeans+silhouette+kneedle",
        parameters={},
        rule_set=rule_set,
        context={"embedding_dim": int(matrix.shape[1]), "n_samples": int(matrix.shape[0])},
    )
    params = {
        "_v1_pipeline_resolved": resolution.pipeline_resolved,
        "_v1_pipeline_effective": resolution.pipeline_effective,
    }
    first = run_kmeans_silhouette_kneedle_analysis(
        method_name="kmeans+silhouette+kneedle",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    second = run_kmeans_silhouette_kneedle_analysis(
        method_name="kmeans+silhouette+kneedle",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    first_sel = first["structured_results"]["selection_evidence"]
    second_sel = second["structured_results"]["selection_evidence"]
    assert first_sel["chosen_k"] == second_sel["chosen_k"]
    assert first_sel["chosen_k"] in first_sel["sweep_range"]
    assert "kmeans_selection_curve.json" in first["artifacts"]


def test_gmm_runner_reproducible_selection() -> None:
    matrix = _toy_embeddings()
    rule_set = load_rule_set()
    resolution = resolve_clustering_resolution(
        method_name="gmm+bic+argmin",
        parameters={},
        rule_set=rule_set,
        context={"embedding_dim": int(matrix.shape[1]), "n_samples": int(matrix.shape[0])},
    )
    params = {
        "_v1_pipeline_resolved": resolution.pipeline_resolved,
        "_v1_pipeline_effective": resolution.pipeline_effective,
    }
    first = run_gmm_bic_argmin_analysis(
        method_name="gmm+bic+argmin",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    second = run_gmm_bic_argmin_analysis(
        method_name="gmm+bic+argmin",
        input_group_id=1,
        input_group_type="embedding_batch",
        input_group_metadata={},
        embeddings=matrix,
        texts=[f"t{i}" for i in range(int(matrix.shape[0]))],
        parameters=params,
    )
    first_sel = first["structured_results"]["selection_evidence"]
    second_sel = second["structured_results"]["selection_evidence"]
    assert first_sel["chosen_k"] == second_sel["chosen_k"]
    assert first_sel["chosen_k"] in first_sel["sweep_range"]
    assert "gmm_selection_curve.json" in first["artifacts"]

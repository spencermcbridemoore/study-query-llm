"""Tests for BANK77 script strategy -> method/runner mapping."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from study_query_llm.pipeline.clustering.registry import get_algorithm_spec

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_bank77_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_bank77_pipeline", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive guard
    raise RuntimeError("Unable to load scripts/run_bank77_pipeline.py for tests")
_BANK77_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BANK77_MODULE)


def _args(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def test_default_strategy_keeps_default_method_name_and_runner_none() -> None:
    args = _args(
        analysis_method="bank77_structural_summary",
        analysis_strategy="default",
    )
    assert _BANK77_MODULE._resolve_analysis_method_name(args) == "bank77_structural_summary"
    assert _BANK77_MODULE._resolve_method_runner(args) is None


@pytest.mark.parametrize(
    ("strategy", "expected_method"),
    [
        ("hdbscan", "hdbscan+fixed"),
        ("kmeans_silhouette_kneedle", "kmeans+normalize+pca+sweep"),
        ("gmm_bic_argmin", "gmm+normalize+pca+sweep"),
    ],
)
def test_strategy_maps_to_expected_registry_method_name(
    strategy: str,
    expected_method: str,
) -> None:
    """Slice 1.5: BANK77 strategy tokens map to bundled-grammar method names.

    Strategy tokens are kept stable for operator continuity; they now resolve
    to the new envelope=none specs registered in PR1 with strategy_aliases
    moved off the legacy v1-envelope specs and onto the new specs in PR2.
    """
    args = _args(
        analysis_method="bank77_structural_summary",
        analysis_strategy=strategy,
    )
    assert _BANK77_MODULE._resolve_analysis_method_name(args) == expected_method


@pytest.mark.parametrize(
    ("strategy", "expected_method"),
    [
        ("hdbscan", "hdbscan+fixed"),
        ("kmeans_silhouette_kneedle", "kmeans+normalize+pca+sweep"),
        ("gmm_bic_argmin", "gmm+normalize+pca+sweep"),
    ],
)
def test_strategy_maps_to_registry_runner(
    strategy: str,
    expected_method: str,
) -> None:
    """Slice 1.5: the bundled-grammar specs reuse the legacy runner functions
    (algorithmic identity preserved); the strategy resolves to the same runner
    function reference as the legacy spec for the corresponding name pair."""
    args = _args(
        analysis_method="bank77_structural_summary",
        analysis_strategy=strategy,
    )
    resolved_runner = _BANK77_MODULE._resolve_method_runner(args)
    spec = get_algorithm_spec(expected_method)
    assert spec is not None
    assert resolved_runner is spec.runner


def test_explicit_method_name_is_not_overridden_by_strategy() -> None:
    args = _args(
        analysis_method="my_custom_method",
        analysis_strategy="hdbscan",
    )
    assert _BANK77_MODULE._resolve_analysis_method_name(args) == "my_custom_method"


@pytest.mark.parametrize("bad_rep", ["label_centroid", "intent_mean"])
def test_parse_args_rejects_retired_embedding_representations(
    bad_rep: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_bank77_pipeline", "--embedding-representation", bad_rep],
    )
    with pytest.raises(SystemExit):
        _BANK77_MODULE._parse_args()


def test_build_analysis_parameters_uses_full_representation() -> None:
    args = _args(
        dataset_slug="banking77",
        embedding_deployment="text-embedding-3-large",
        embedding_provider="azure",
        analysis_strategy="default",
        embedding_representation="full",
    )
    params = _BANK77_MODULE._build_analysis_parameters(args)
    assert params["representation_type"] == "full"
    assert params["embedding_representation"] == "full"

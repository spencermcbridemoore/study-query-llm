"""Tests for BANK77 script strategy -> method/runner mapping."""

from __future__ import annotations

import argparse
import importlib.util
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
        ("hdbscan", "hdbscan"),
        ("kmeans_silhouette_kneedle", "kmeans+silhouette+kneedle"),
        ("gmm_bic_argmin", "gmm+bic+argmin"),
    ],
)
def test_strategy_maps_to_expected_registry_method_name(
    strategy: str,
    expected_method: str,
) -> None:
    args = _args(
        analysis_method="bank77_structural_summary",
        analysis_strategy=strategy,
    )
    assert _BANK77_MODULE._resolve_analysis_method_name(args) == expected_method


@pytest.mark.parametrize(
    ("strategy", "expected_method"),
    [
        ("hdbscan", "hdbscan"),
        ("kmeans_silhouette_kneedle", "kmeans+silhouette+kneedle"),
        ("gmm_bic_argmin", "gmm+bic+argmin"),
    ],
)
def test_strategy_maps_to_registry_runner(
    strategy: str,
    expected_method: str,
) -> None:
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


def test_non_full_representation_rejected_for_registry_mapped_strategy() -> None:
    args = _args(
        analysis_strategy="hdbscan",
        embedding_representation="label_centroid",
    )
    with pytest.raises(ValueError, match="requires .* full"):
        _BANK77_MODULE._validate_embedding_representation_for_analysis(args)


def test_non_full_representation_allowed_for_default_strategy() -> None:
    args = _args(
        analysis_strategy="default",
        embedding_representation="label_centroid",
    )
    _BANK77_MODULE._validate_embedding_representation_for_analysis(args)

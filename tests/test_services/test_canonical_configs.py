"""Tests for the unused canonical-config builder skeleton.

Covers:
* Builder purity (no input mutation; same input -> equal output across calls).
* Builder keys are a subset of the corresponding registered method's
  ``parameters_schema.properties`` (where the method is in an in-memory
  registry).
* No runtime writer under ``src/study_query_llm/services/`` or
  ``src/study_query_llm/experiments/`` imports
  :mod:`study_query_llm.algorithms.canonical_configs` -- this preserves
  the "unused" invariant. Adoption MUST be deliberate (it changes the
  fingerprint hash for the adopted method going forward), so this test is
  expected to fail loudly when adoption begins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pytest

from study_query_llm.algorithms import recipes as recipes_module
from study_query_llm.algorithms.canonical_configs import (
    CANONICAL_CONFIG_BUILDERS,
    canonical_config_for,
)
from study_query_llm.algorithms.text_classification_methods import (
    TEXT_CLASSIFICATION_METHODS,
)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_canonical_config_builders_cover_known_clustering_components():
    """Every clustering component method has a canonical builder."""
    for spec in recipes_module.CLUSTERING_COMPONENT_METHODS:
        key: Tuple[str, str] = (spec["name"], spec["version"])
        assert key in CANONICAL_CONFIG_BUILDERS, (
            f"Missing canonical builder for clustering component {key}"
        )


def test_canonical_config_builders_cover_known_text_classification_methods():
    """Every register-only text-classification method has a canonical builder."""
    for spec in TEXT_CLASSIFICATION_METHODS:
        key: Tuple[str, str] = (spec["name"], spec["version"])
        assert key in CANONICAL_CONFIG_BUILDERS, (
            f"Missing canonical builder for text-classification method {key}"
        )


def test_canonical_config_builders_cover_clustering_composite():
    """The clustering composite has a canonical builder."""
    assert ("cosine_kllmeans_no_pca", "1.0") in CANONICAL_CONFIG_BUILDERS


def test_canonical_config_builders_cover_mcq_methods():
    """All MCQ probe + analysis methods have canonical builders."""
    for key in (
        ("mcq_answer_position_probe", "1.0"),
        ("mcq_compliance_metrics", "1.0"),
        ("mcq_answer_position_distribution", "1.0"),
        ("mcq_answer_position_chi_square", "1.0"),
    ):
        assert key in CANONICAL_CONFIG_BUILDERS, (
            f"Missing canonical builder for MCQ method {key}"
        )


# ---------------------------------------------------------------------------
# Purity
# ---------------------------------------------------------------------------


def test_canonical_config_builders_are_pure():
    """Calling a builder twice with the same input returns equal dicts and
    does not mutate the caller's input."""
    sample_inputs: Dict[Tuple[str, str], Dict[str, object]] = {
        ("mean_pool_tokens", "1.0"): {"normalize": True},
        ("pca_svd_project", "1.0"): {"pca_dim": 64},
        ("kmeanspp_init", "1.0"): {"seed_space": "base_seed_plus_try_index"},
        ("k_llmmeans", "1.0"): {
            "max_iter": 200,
            "distance_metric": "cosine",
            "normalize_vectors": True,
            "llm_interval": 20,
            "max_samples": 1000,
            "coverage_threshold": 0.95,
        },
        ("umap_project", "1.0"): {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "n_components": 2,
            "metric": "cosine",
            "random_state": 42,
        },
        ("cosine_kllmeans_no_pca", "1.0"): {
            "k_min": 2,
            "k_max": 8,
            "n_restarts": 5,
        },
        ("mcq_answer_position_probe", "1.0"): {
            "deployment": "gpt-x",
            "level": "easy",
            "subject": "math",
            "options_per_question": 4,
            "questions_per_test": 10,
            "label_style": "letters",
            "spread_correct_answer_uniformly": True,
            "samples_per_combo": 3,
            "template_version": "v1",
        },
        ("mcq_compliance_metrics", "1.0"): {
            "request_id": 7,
            "analysis_key": "mcq_compliance",
        },
        ("mcq_answer_position_distribution", "1.0"): {
            "request_id": 7,
            "analysis_key": "mcq_answer_position_distribution",
        },
        ("mcq_answer_position_chi_square", "1.0"): {
            "request_id": 7,
            "analysis_key": "mcq_answer_position_chi_square",
        },
        ("knn_prototype_classifier", "0.1"): {
            "k": 5,
            "metric": "cosine",
            "weighting": "uniform",
            "normalize_embeddings": True,
            "tie_break": "lowest_label",
        },
        ("linear_probe_logreg", "0.1"): {
            "regularization": "l2",
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "normalize_embeddings": True,
            "random_state": 0,
        },
        ("label_embedding_zero_shot", "0.1"): {
            "label_template": "This text is about {label}.",
            "metric": "cosine",
            "normalize_embeddings": True,
            "temperature": 1.0,
        },
        ("prompted_llm_classifier", "0.1"): {
            "prompt_template_id": "v1",
            "n_shot": 4,
            "deployment": "gpt-x",
            "temperature": 0.0,
            "max_tokens": 32,
            "label_parser": "regex_default",
            "label_set": ["a", "b", "c"],
        },
        ("mixture_of_experts_classifier", "0.1"): {
            "experts": [
                {"name": "knn_prototype_classifier", "version": "0.1"},
                {"name": "linear_probe_logreg", "version": "0.1"},
            ],
            "gating_strategy": "softmax_over_logits",
            "gating_temperature": 1.0,
            "combine": "weighted_sum",
            "normalize_embeddings": True,
        },
    }

    # Sanity: every registered builder has a sample input below (otherwise this
    # test would silently skip new builders).
    missing = set(CANONICAL_CONFIG_BUILDERS) - set(sample_inputs)
    assert not missing, (
        f"sample_inputs missing entries for builders: {sorted(missing)}"
    )

    for (name, version), params in sample_inputs.items():
        original = dict(params)
        out_a = canonical_config_for(name, version, params)
        out_b = canonical_config_for(name, version, params)
        assert out_a == out_b, (
            f"{name}@{version} builder produced differing outputs across calls"
        )
        assert params == original, (
            f"{name}@{version} builder mutated its input params"
        )


def test_canonical_config_for_unknown_method_raises():
    """Unknown (name, version) pairs raise rather than silently fall back."""
    with pytest.raises(KeyError):
        canonical_config_for("does_not_exist", "9.9", {"x": 1})


# ---------------------------------------------------------------------------
# Schema-vs-builder consistency
# ---------------------------------------------------------------------------


def _schema_props_for(name: str, version: str) -> Dict[str, object]:
    """Return parameters_schema.properties for a method known to one of the
    in-memory registries (clustering or text-classification). Returns empty
    dict if the method is not in either registry."""
    for spec in recipes_module.CLUSTERING_COMPONENT_METHODS:
        if spec["name"] == name and spec["version"] == version:
            return dict(spec.get("parameters_schema", {}).get("properties", {}))
    for spec in TEXT_CLASSIFICATION_METHODS:
        if spec["name"] == name and spec["version"] == version:
            return dict(spec.get("parameters_schema", {}).get("properties", {}))
    return {}


def test_canonical_config_builder_keys_match_method_definition_schema():
    """For each builder whose method is in an in-memory registry, the keys
    produced by the builder are a subset of the method's
    parameters_schema.properties keys.

    Methods not in any in-memory registry (clustering composites, MCQ) are
    skipped here; their schema lives in DB writers and is covered by the
    audit script."""
    for (name, version), builder in CANONICAL_CONFIG_BUILDERS.items():
        schema_props = _schema_props_for(name, version)
        if not schema_props:
            continue
        produced_keys = set(builder({}).keys())
        unexpected = produced_keys - set(schema_props.keys())
        assert not unexpected, (
            f"{name}@{version} builder produces keys not in "
            f"parameters_schema.properties: {sorted(unexpected)}"
        )


# ---------------------------------------------------------------------------
# No runtime importers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Locate the repo root by walking up from this test file."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "src" / "study_query_llm").is_dir():
            return parent
    raise RuntimeError("Could not locate repo root from test file location.")


def test_canonical_configs_module_has_no_runtime_importers():
    """No file under src/study_query_llm/services/ or
    src/study_query_llm/experiments/ may import canonical_configs (yet).

    Adoption is deliberate (it changes fingerprints for adopted methods).
    When adoption is intended, update this test together with the writer."""
    root = _repo_root()
    target_dirs = [
        root / "src" / "study_query_llm" / "services",
        root / "src" / "study_query_llm" / "experiments",
    ]
    needles = (
        "from study_query_llm.algorithms.canonical_configs",
        "from .canonical_configs",
        "from ..algorithms.canonical_configs",
        "import study_query_llm.algorithms.canonical_configs",
    )
    offenders = []
    for target in target_dirs:
        if not target.exists():
            continue
        for path in target.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for needle in needles:
                if needle in text:
                    offenders.append(str(path.relative_to(root)))
                    break
    assert not offenders, (
        "canonical_configs is meant to be unused for now; the following "
        "files import it and must be updated together with this test: "
        f"{offenders}"
    )

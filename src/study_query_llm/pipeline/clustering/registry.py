"""Registry for built-in clustering algorithms used by ``pipeline.analyze``.

The registry is the runtime dispatch seam for method-name -> runner resolution.
It intentionally starts small (built-ins only) so dispatch can move from
hardcoded string maps without introducing behavior drift.
See docs/living/METHOD_RECIPES.md (Bundled Clustering Subsystem) for the
subsystem definition, naming grammar, and output-schema contract.

Caveat:
    Registry metadata is not yet validated against persisted
    ``MethodDefinition.input_schema`` rows. A later phase may add a consistency
    check to detect drift between runtime declarations and DB metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from study_query_llm.pipeline.hdbscan_runner import run_hdbscan_analysis

from .agglomerative_runner import run_agglomerative_fixed_k_analysis
from .gmm_runner import run_gmm_bic_argmin_analysis
from .kmeans_runner import run_kmeans_silhouette_kneedle_analysis

FitMode = Literal["single_fit", "sweep_select"]
ProvenanceEnvelope = Literal["clustering_v1", "none"]
AlgorithmRunner = Callable[..., object]

REPRESENTATION_FULL = "full"
REPRESENTATION_LABEL_CENTROID = "label_centroid"


@dataclass(frozen=True)
class AlgorithmSpec:
    """Runtime metadata for one registry-backed clustering method."""

    method_name: str
    runner: AlgorithmRunner
    fit_mode: FitMode
    requires_embeddings: bool
    supports_snapshot_only: bool
    allowed_representations: frozenset[str]
    provenance_envelope: ProvenanceEnvelope
    base_algorithm: str
    default_determinism_class: str
    strategy_aliases: tuple[str, ...] = ()


def normalize_method_name(method_name: str) -> str:
    """Normalize method names for case-insensitive lookup."""
    return str(method_name or "").strip().lower()


_ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "hdbscan": AlgorithmSpec(
        method_name="hdbscan",
        runner=run_hdbscan_analysis,
        fit_mode="single_fit",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset({REPRESENTATION_FULL}),
        provenance_envelope="clustering_v1",
        base_algorithm="hdbscan",
        default_determinism_class="non_deterministic",
        strategy_aliases=("hdbscan",),
    ),
    "kmeans+silhouette+kneedle": AlgorithmSpec(
        method_name="kmeans+silhouette+kneedle",
        runner=run_kmeans_silhouette_kneedle_analysis,
        fit_mode="sweep_select",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset(
            {REPRESENTATION_FULL, REPRESENTATION_LABEL_CENTROID}
        ),
        provenance_envelope="clustering_v1",
        base_algorithm="kmeans",
        default_determinism_class="pseudo_deterministic",
        strategy_aliases=("kmeans_silhouette_kneedle",),
    ),
    "gmm+bic+argmin": AlgorithmSpec(
        method_name="gmm+bic+argmin",
        runner=run_gmm_bic_argmin_analysis,
        fit_mode="sweep_select",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset(
            {REPRESENTATION_FULL, REPRESENTATION_LABEL_CENTROID}
        ),
        provenance_envelope="clustering_v1",
        base_algorithm="gmm",
        default_determinism_class="pseudo_deterministic",
        strategy_aliases=("gmm_bic_argmin",),
    ),
    "agglomerative+fixed-k": AlgorithmSpec(
        method_name="agglomerative+fixed-k",
        runner=run_agglomerative_fixed_k_analysis,
        fit_mode="single_fit",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset(
            {REPRESENTATION_FULL, REPRESENTATION_LABEL_CENTROID}
        ),
        provenance_envelope="none",
        base_algorithm="agglomerative",
        default_determinism_class="deterministic",
        strategy_aliases=(),
    ),
    # Slice 1.5 bundled-grammar replacements for the legacy v1-envelope methods.
    # Algorithmic identity is preserved (same runner functions); only the method
    # name and provenance envelope change. Strategy aliases stay on the legacy
    # specs in PR1 and move here in PR2 once the dispatcher branch is wired.
    "hdbscan+fixed": AlgorithmSpec(
        method_name="hdbscan+fixed",
        runner=run_hdbscan_analysis,
        fit_mode="single_fit",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset({REPRESENTATION_FULL}),
        provenance_envelope="none",
        base_algorithm="hdbscan",
        default_determinism_class="non_deterministic",
        strategy_aliases=(),
    ),
    "kmeans+normalize+pca+sweep": AlgorithmSpec(
        method_name="kmeans+normalize+pca+sweep",
        runner=run_kmeans_silhouette_kneedle_analysis,
        fit_mode="sweep_select",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset(
            {REPRESENTATION_FULL, REPRESENTATION_LABEL_CENTROID}
        ),
        provenance_envelope="none",
        base_algorithm="kmeans",
        default_determinism_class="pseudo_deterministic",
        strategy_aliases=(),
    ),
    "gmm+normalize+pca+sweep": AlgorithmSpec(
        method_name="gmm+normalize+pca+sweep",
        runner=run_gmm_bic_argmin_analysis,
        fit_mode="sweep_select",
        requires_embeddings=True,
        supports_snapshot_only=False,
        allowed_representations=frozenset(
            {REPRESENTATION_FULL, REPRESENTATION_LABEL_CENTROID}
        ),
        provenance_envelope="none",
        base_algorithm="gmm",
        default_determinism_class="pseudo_deterministic",
        strategy_aliases=(),
    ),
}


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for spec in _ALGORITHM_SPECS.values():
        canonical = normalize_method_name(spec.method_name)
        index[canonical] = spec.method_name
        for alias in spec.strategy_aliases:
            normalized_alias = normalize_method_name(alias)
            if normalized_alias:
                index[normalized_alias] = spec.method_name
    return index


_ALIAS_INDEX = _build_alias_index()


def iter_algorithm_specs() -> tuple[AlgorithmSpec, ...]:
    """Return all registered specs in deterministic name order."""
    return tuple(_ALGORITHM_SPECS[key] for key in sorted(_ALGORITHM_SPECS))


def get_algorithm_spec(method_name: str) -> AlgorithmSpec | None:
    """Return registry spec for ``method_name``, or ``None`` when absent."""
    normalized = normalize_method_name(method_name)
    return _ALGORITHM_SPECS.get(normalized)


def resolve_algorithm_runner(method_name: str) -> AlgorithmRunner | None:
    """Resolve registry-backed runner for ``method_name``."""
    spec = get_algorithm_spec(method_name)
    if spec is None:
        return None
    return spec.runner


def is_registry_v1_clustering_method(method_name: str) -> bool:
    """Return True when registry metadata marks method as v1 envelope."""
    spec = get_algorithm_spec(method_name)
    return bool(spec is not None and spec.provenance_envelope == "clustering_v1")


def resolve_registry_method_name(token: str) -> str | None:
    """Resolve strategy token or method name to canonical method name."""
    return _ALIAS_INDEX.get(normalize_method_name(token))


__all__ = [
    "AlgorithmSpec",
    "FitMode",
    "ProvenanceEnvelope",
    "REPRESENTATION_FULL",
    "REPRESENTATION_LABEL_CENTROID",
    "get_algorithm_spec",
    "is_registry_v1_clustering_method",
    "iter_algorithm_specs",
    "normalize_method_name",
    "resolve_algorithm_runner",
    "resolve_registry_method_name",
]

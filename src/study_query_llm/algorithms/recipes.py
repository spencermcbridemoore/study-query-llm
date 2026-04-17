"""
Method Recipes - Canonical composite/pipeline definitions as versioned JSON.

A *recipe* describes a composite analysis method as an ordered list of
component stages. Each stage references an existing :class:`MethodDefinition`
row by ``(name, version)`` plus stage-local parameter defaults. Recipes are
*descriptive metadata*: execution still lives in the relevant algorithm
module (e.g. :func:`study_query_llm.algorithms.sweep.run_sweep`). The recipe
exists so that:

* The provenance layer can answer "what pipeline was this run?" in one place
  rather than inferring it from worker/code paths.
* Two runs can be compared at the recipe level: if their ``recipe_hash``
  differs, they used structurally different pipelines.
* Future caching/DAG phases can reference a stable, versioned recipe shape.

Recipe JSON shape (v0)::

    {
        "recipe_version": "v0",
        "stages": [
            {"name": <component name>, "version": <str>,
             "role": <str>, "params": {...}},
            ...
        ],
        "notes": <optional free text>,
    }

The ``recipe_hash`` is ``sha256(json.dumps(recipe, sort_keys=True, ...))``.
Callers record a run's recipe identity by putting that hash into
``config_json`` under the key ``"recipe_hash"`` -- the canonical run
fingerprint already hashes ``config_json`` (minus scheduling-only keys), so
no change to the fingerprint tuple shape is needed.

See ``docs/living/METHOD_RECIPES.md`` for the full spec.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..services.method_service import MethodService

logger = get_logger(__name__)


RECIPE_VERSION = "v0"


# Component methods for the clustering-sweep family.
#
# Each entry is a spec for a single :class:`MethodDefinition` row. Versions
# should only be bumped when a backward-incompatible change is made to the
# underlying code/behaviour. ``code_ref`` is a repo-relative path to the
# authoritative implementation so that later inspection can trace the row to
# source.
CLUSTERING_COMPONENT_METHODS: List[Dict[str, Any]] = [
    {
        "name": "mean_pool_tokens",
        "version": "1.0",
        "role": "pooling",
        "code_ref": "src/study_query_llm/algorithms/dimensionality_reduction.py",
        "description": (
            "Mean-pool token-level embeddings to sentence-level vectors, "
            "with optional L2 normalisation."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "normalize": {"type": "boolean"},
            },
        },
    },
    {
        "name": "pca_svd_project",
        "version": "1.0",
        "role": "projection",
        "code_ref": "src/study_query_llm/algorithms/dimensionality_reduction.py",
        "description": (
            "SVD-based PCA projection to a target dimensionality."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "pca_dim": {"type": "integer"},
            },
        },
    },
    {
        "name": "kmeanspp_init",
        "version": "1.0",
        "role": "initialization",
        "code_ref": "src/study_query_llm/algorithms/clustering.py",
        "description": (
            "k-means++ seed selection used by each clustering restart. "
            "Seed space is derived as base_seed + try_index; sampling is "
            "weighted by squared distance to the nearest chosen centroid."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "seed_space": {"type": "string"},
            },
        },
    },
    {
        "name": "k_llmmeans",
        "version": "1.0",
        "role": "clustering",
        "code_ref": "src/study_query_llm/algorithms/clustering.py",
        "description": (
            "k-means-style iterative clustering with optional LLM-driven "
            "centroid influence via paraphrasing + re-embedding inside the "
            "iteration loop."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "max_iter": {"type": "integer"},
                "distance_metric": {"type": "string"},
                "normalize_vectors": {"type": "boolean"},
                "llm_interval": {"type": "integer"},
                "max_samples": {"type": "integer"},
                "coverage_threshold": {"type": "number"},
            },
        },
    },
    {
        "name": "umap_project",
        "version": "1.0",
        "role": "projection",
        "code_ref": None,
        "description": (
            "UMAP projection to a target dimensionality. Definition-only; no "
            "implementation wired. Registered ahead of time so future "
            "clustering composite recipes can reference a canonical "
            "(name, version) and parameter schema."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "n_neighbors": {"type": "integer"},
                "min_dist": {"type": "number"},
                "n_components": {"type": "integer"},
                "metric": {"type": "string"},
                "random_state": {"type": "integer"},
            },
        },
    },
]


COSINE_KLLMEANS_NO_PCA_RECIPE: Dict[str, Any] = {
    "recipe_version": RECIPE_VERSION,
    "stages": [
        {
            "name": "mean_pool_tokens",
            "version": "1.0",
            "role": "pooling",
            "params": {"normalize": False},
        },
        {
            "name": "kmeanspp_init",
            "version": "1.0",
            "role": "initialization",
            "params": {"seed_space": "base_seed_plus_try_index"},
        },
        {
            "name": "k_llmmeans",
            "version": "1.0",
            "role": "clustering",
            "params": {
                "max_iter": 200,
                "distance_metric": "cosine",
                "normalize_vectors": True,
                "llm_interval": 20,
            },
        },
    ],
    "notes": (
        "No PCA projection stage; embeddings consumed at full dimensionality."
    ),
}


COMPOSITE_RECIPES: Dict[str, Dict[str, Any]] = {
    "cosine_kllmeans_no_pca": COSINE_KLLMEANS_NO_PCA_RECIPE,
}


def canonical_recipe_hash(recipe: Dict[str, Any]) -> str:
    """Return the deterministic SHA-256 hash of a recipe JSON.

    Uses sorted keys + compact separators so equivalent recipes (regardless of
    dict authoring order) yield the same hash.
    """
    canonical = json.dumps(
        recipe or {},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_composite_recipe(name: str) -> Dict[str, Any]:
    """Return a deep copy of the canonical recipe for the named composite.

    Raises ``KeyError`` if the composite is unknown so callers cannot silently
    persist an empty recipe.
    """
    if name not in COMPOSITE_RECIPES:
        raise KeyError(
            f"No canonical recipe registered for composite method {name!r}. "
            f"Known composites: {sorted(COMPOSITE_RECIPES)}"
        )
    return copy.deepcopy(COMPOSITE_RECIPES[name])


def register_clustering_components(method_service: "MethodService") -> Dict[str, int]:
    """Register all clustering component methods idempotently.

    For each entry in :data:`CLUSTERING_COMPONENT_METHODS`, look up by
    ``(name, version)`` and only call :meth:`MethodService.register_method`
    when the row is absent. Returns a mapping of
    ``"{name}@{version}" -> method_definition_id``.
    """
    registered: Dict[str, int] = {}
    for spec in CLUSTERING_COMPONENT_METHODS:
        name = spec["name"]
        version = spec["version"]
        key = f"{name}@{version}"
        existing = method_service.get_method(name, version=version)
        if existing is not None:
            registered[key] = int(existing.id)
            logger.debug(
                "Component method already registered: %s (id=%s)",
                key,
                existing.id,
            )
            continue
        method_id = method_service.register_method(
            name=name,
            version=version,
            code_ref=spec.get("code_ref"),
            description=spec.get("description"),
            parameters_schema=spec.get("parameters_schema"),
        )
        registered[key] = int(method_id)
        logger.info("Registered component method: %s (id=%s)", key, method_id)
    return registered


def ensure_composite_recipe(
    method_service: "MethodService",
    composite_name: str,
    composite_version: str = "1.0",
    *,
    description: Optional[str] = None,
    parameters_schema: Optional[Dict[str, Any]] = None,
    code_ref: Optional[str] = None,
) -> int:
    """Ensure a composite method row exists and carries its canonical recipe.

    If the row is absent, register it with the recipe. If the row exists but
    has no recipe (e.g. auto-registered before recipes existed), attach the
    recipe in-place via :meth:`MethodService.update_recipe` -- this is not a
    semantic change, so no version bump. If the row already has a recipe and
    it matches the canonical recipe, do nothing. If it differs, log a warning
    and leave the stored recipe alone (the operator should decide whether to
    bump the version explicitly).
    """
    canonical = build_composite_recipe(composite_name)
    existing = method_service.get_method(composite_name, version=composite_version)
    if existing is None:
        method_id = method_service.register_method(
            name=composite_name,
            version=composite_version,
            code_ref=code_ref,
            description=description,
            parameters_schema=parameters_schema,
            recipe_json=canonical,
        )
        logger.info(
            "Registered composite %s@%s with canonical recipe (id=%s)",
            composite_name,
            composite_version,
            method_id,
        )
        return int(method_id)

    stored = existing.recipe_json
    if not stored:
        method_service.update_recipe(int(existing.id), canonical)
        logger.info(
            "Attached canonical recipe to existing composite %s@%s (id=%s)",
            composite_name,
            composite_version,
            existing.id,
        )
        return int(existing.id)

    if canonical_recipe_hash(stored) != canonical_recipe_hash(canonical):
        logger.warning(
            "Composite %s@%s already has a recipe whose hash differs from the "
            "canonical recipe; leaving stored recipe in place. Bump the "
            "composite version explicitly to record a new recipe.",
            composite_name,
            composite_version,
        )
    return int(existing.id)

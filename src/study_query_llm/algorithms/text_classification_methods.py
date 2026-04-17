"""
Text classification method definitions (register-only subset).

Definitions only. No execution path is wired. Methods are registered so future
runners (DB-orchestrated jobs, LangGraph nodes, notebooks) share canonical
``(name, version)`` identities and ``parameters_schema`` shapes from the
moment a method first lands in the registry.

Scope of this module:

* The five **register-only** text-classification families currently in scope:
  - ``knn_prototype_classifier``
  - ``linear_probe_logreg``
  - ``label_embedding_zero_shot``
  - ``prompted_llm_classifier``
  - ``mixture_of_experts_classifier`` (fixed-experts variant)

* Training-heavy variants (MLP from scratch, contrastive fine-tuned encoder,
  adapter-based, fine-tuned LLM, hybrid with LLM fallback) are **deliberately
  deferred** until a runtime story (state, checkpointing, lease budget) is
  decided. They are not in :data:`TEXT_CLASSIFICATION_METHODS`.

Each entry includes a ``maturity`` field (``"registered_only"`` for now) so a
future filter can distinguish "definition exists, no runner" from "definition
+ runner exist."

See also:

* :mod:`study_query_llm.algorithms.recipes` for the analogous clustering
  registry pattern that this module mirrors.
* ``scripts/register_text_classification_methods.py`` for the operator-facing
  idempotent registrar entrypoint.
* ``docs/STANDING_ORDERS.md`` ("Method Definitions and Provenance") for the
  no-auto-registration discipline that motivates pre-registration.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..services.method_service import MethodService

logger = get_logger(__name__)


# Maturity tag values. ``registered_only`` means the row exists in
# ``method_definitions`` so writers/readers can refer to it by canonical
# (name, version), but no execution path has been wired yet.
MATURITY_REGISTERED_ONLY = "registered_only"


TEXT_CLASSIFICATION_METHODS: List[Dict[str, Any]] = [
    {
        "name": "knn_prototype_classifier",
        "version": "0.1",
        "role": "classifier",
        "code_ref": None,
        "description": (
            "K-nearest-neighbour / prototype classifier over text embeddings. "
            "Definition-only; no implementation wired. Predict by majority "
            "vote (or nearest-prototype) among k nearest labelled training "
            "embeddings under a configured similarity metric."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer"},
                "metric": {"type": "string"},
                "weighting": {"type": "string"},
                "normalize_embeddings": {"type": "boolean"},
                "tie_break": {"type": "string"},
            },
        },
        "maturity": MATURITY_REGISTERED_ONLY,
    },
    {
        "name": "linear_probe_logreg",
        "version": "0.1",
        "role": "classifier",
        "code_ref": None,
        "description": (
            "Linear probe (multinomial logistic regression) on frozen text "
            "embeddings. Definition-only; no implementation wired. Closed-"
            "form / convex optimiser; no SGD epoch loop is implied by this "
            "registration."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "regularization": {"type": "string"},
                "C": {"type": "number"},
                "max_iter": {"type": "integer"},
                "class_weight": {"type": "string"},
                "solver": {"type": "string"},
                "normalize_embeddings": {"type": "boolean"},
                "random_state": {"type": "integer"},
            },
        },
        "maturity": MATURITY_REGISTERED_ONLY,
    },
    {
        "name": "label_embedding_zero_shot",
        "version": "0.1",
        "role": "classifier",
        "code_ref": None,
        "description": (
            "Label-embedding zero-shot classifier. Definition-only; no "
            "implementation wired. Encode each label description into the "
            "same embedding space as the input and predict the label whose "
            "embedding is most similar under a configured similarity metric."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "label_template": {"type": "string"},
                "metric": {"type": "string"},
                "normalize_embeddings": {"type": "boolean"},
                "temperature": {"type": "number"},
            },
        },
        "maturity": MATURITY_REGISTERED_ONLY,
    },
    {
        "name": "prompted_llm_classifier",
        "version": "0.1",
        "role": "classifier",
        "code_ref": None,
        "description": (
            "Prompted-LLM classifier (zero- or few-shot). Definition-only; "
            "no implementation wired. Build a prompt including task "
            "description and optional in-context examples, send to a chat "
            "model, and parse the predicted label from the generation."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "prompt_template_id": {"type": "string"},
                "n_shot": {"type": "integer"},
                "deployment": {"type": "string"},
                "temperature": {"type": "number"},
                "max_tokens": {"type": "integer"},
                "label_parser": {"type": "string"},
                "label_set": {"type": "array"},
            },
        },
        "maturity": MATURITY_REGISTERED_ONLY,
    },
    {
        "name": "mixture_of_experts_classifier",
        "version": "0.1",
        "role": "classifier",
        "code_ref": None,
        "description": (
            "Fixed-experts mixture-of-experts classifier over text "
            "embeddings. Definition-only; no implementation wired. Each "
            "expert is an already-registered classifier; a gating function "
            "produces weights and the final probabilities are a weighted sum "
            "of expert outputs. The trainable-expert variant is deferred."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "experts": {"type": "array"},
                "gating_strategy": {"type": "string"},
                "gating_temperature": {"type": "number"},
                "combine": {"type": "string"},
                "normalize_embeddings": {"type": "boolean"},
            },
        },
        "maturity": MATURITY_REGISTERED_ONLY,
    },
]


def register_text_classification_methods(
    method_service: "MethodService",
) -> Dict[str, int]:
    """Register all text-classification method rows idempotently.

    Mirrors :func:`study_query_llm.algorithms.recipes.register_clustering_components`:
    for each entry in :data:`TEXT_CLASSIFICATION_METHODS`, look up by
    ``(name, version)`` and only call :meth:`MethodService.register_method`
    when the row is absent. Returns a mapping of
    ``"{name}@{version}" -> method_definition_id``.

    Args:
        method_service: The :class:`MethodService` to use for lookup and
            registration. Caller owns the underlying session.

    Returns:
        Dictionary mapping ``"{name}@{version}"`` keys to the
        ``method_definitions.id`` for each entry, regardless of whether the
        row was newly registered or already present.
    """
    registered: Dict[str, int] = {}
    for spec in TEXT_CLASSIFICATION_METHODS:
        name = spec["name"]
        version = spec["version"]
        key = f"{name}@{version}"
        existing = method_service.get_method(name, version=version)
        if existing is not None:
            registered[key] = int(existing.id)
            logger.debug(
                "Text-classification method already registered: %s (id=%s)",
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
        logger.info(
            "Registered text-classification method: %s (id=%s)",
            key,
            method_id,
        )
    return registered

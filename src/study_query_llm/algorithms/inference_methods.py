"""
Inference method definitions for polymorphic method execution lanes.

Definitions are registered first so runtime dispatch, provenance writers, and
future orchestration surfaces share canonical ``(name, version)`` identities.
This module does not execute methods; it only defines method metadata and an
idempotent registration helper.

Initial methods in scope:

* ``perturbation_then_inference.basic``: stage-1 runner surface.
* ``inference.logprobs.basic``: stage-2 provider-expansion pressure test.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..services.method_service import MethodService

logger = get_logger(__name__)


MATURITY_REGISTERED_ONLY = "registered_only"
MATURITY_RUNNER_WIRED = "runner_wired"


INFERENCE_METHODS: List[Dict[str, Any]] = [
    {
        "name": "perturbation_then_inference.basic",
        "version": "0.1",
        "role": "method_execution",
        "code_ref": "src/study_query_llm/services/method_runners/perturb_then_infer_basic.py",
        "description": (
            "Apply caller-supplied perturbation variants and execute chat "
            "inference per variant; aggregate responses and persist canonical "
            "execution provenance."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "variants": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "temperature": {"type": "number"},
                "max_tokens": {"type": "integer"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
        "maturity": MATURITY_RUNNER_WIRED,
    },
    {
        "name": "inference.logprobs.basic",
        "version": "0.1",
        "role": "method_execution",
        "code_ref": "src/study_query_llm/services/method_runners/logprobs_basic.py",
        "description": (
            "Execute chat inference with provider-specific logprobs controls "
            "and persist normalized logprobs output for method-execution "
            "provenance."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "max_tokens": {"type": "integer"},
                "logprobs": {"type": "boolean"},
                "top_logprobs": {"type": "integer"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
        "maturity": MATURITY_RUNNER_WIRED,
    },
    {
        "name": "inference.mcq_logprob.basic",
        "version": "0.1",
        "role": "method_execution",
        "code_ref": "src/study_query_llm/services/method_runners/mcq_logprob_basic.py",
        "description": (
            "Run first-token MCQ logprob probing over deterministic option "
            "permutations and persist one execution artifact per invocation."
        ),
        "parameters_schema": {
            "type": "object",
            "properties": {
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "permutation_strategy": {"type": "string"},
                "format_idx": {"type": "integer"},
                "prompt_template_version": {"type": "string"},
                "concurrency_cap": {"type": "integer"},
                "max_questions": {"type": "integer"},
                "metadata": {"type": "object"},
            },
        },
        "maturity": MATURITY_RUNNER_WIRED,
    },
]


def register_inference_methods(method_service: "MethodService") -> Dict[str, int]:
    """Register all inference method rows idempotently."""
    registered: Dict[str, int] = {}
    for spec in INFERENCE_METHODS:
        name = str(spec["name"])
        version = str(spec["version"])
        key = f"{name}@{version}"
        existing = method_service.get_method(name, version=version)
        if existing is not None:
            registered[key] = int(existing.id)
            logger.debug(
                "Inference method already registered: %s (id=%s)",
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
        logger.info("Registered inference method: %s (id=%s)", key, method_id)
    return registered


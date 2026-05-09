"""Runtime registry for polymorphic method-execution runners."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    from study_query_llm.db.raw_call_repository import RawCallRepository


MethodRuntimeRunner = Callable[
    [Dict[str, Any], "MethodRunnerContext"],
    "MethodRunnerResult | Any",
]


@dataclass(frozen=True)
class MethodRunnerResult:
    """Normalized payload returned by a method runtime runner."""

    output_json: Dict[str, Any]
    pipeline_stage_role: str
    result_ref: Optional[str] = None
    pipeline_stage_context: Dict[str, Any] = field(default_factory=dict)
    metadata_json: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodRunnerContext:
    """Execution context passed to runtime runners."""

    repository: "RawCallRepository"
    request_group_id: int
    source_group_id: int
    method_name: str
    method_version: str
    run_key: str
    imported_run_id: Optional[int] = None
    imported_run_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodRuntimeSpec:
    """Registry entry for one runtime-invokable method identity."""

    method_name: str
    method_version: str
    runner: MethodRuntimeRunner
    description: str = ""


_METHOD_RUNTIME_REGISTRY: Dict[tuple[str, str], MethodRuntimeSpec] = {}
_DEFAULTS_REGISTERED = False


def _normalize_token(value: str) -> str:
    return str(value or "").strip().lower()


def _registry_key(method_name: str, method_version: str) -> tuple[str, str]:
    name = _normalize_token(method_name)
    version = str(method_version or "").strip()
    if not name:
        raise ValueError("method_name must be non-empty")
    if not version:
        raise ValueError("method_version must be non-empty")
    return (name, version)


def register_method_runtime(spec: MethodRuntimeSpec, *, overwrite: bool = False) -> None:
    """Register one runtime method runner."""
    key = _registry_key(spec.method_name, spec.method_version)
    existing = _METHOD_RUNTIME_REGISTRY.get(key)
    if existing is not None and not overwrite:
        raise ValueError(
            f"Runtime already registered for {spec.method_name}@{spec.method_version}"
        )
    _METHOD_RUNTIME_REGISTRY[key] = spec


def get_method_runtime(method_name: str, method_version: str) -> Optional[MethodRuntimeSpec]:
    """Return the runtime spec for a method identity, when registered."""
    return _METHOD_RUNTIME_REGISTRY.get(_registry_key(method_name, method_version))


def list_method_runtime_specs() -> tuple[MethodRuntimeSpec, ...]:
    """Return all runtime specs in deterministic identity order."""
    return tuple(
        _METHOD_RUNTIME_REGISTRY[key]
        for key in sorted(_METHOD_RUNTIME_REGISTRY.keys())
    )


async def run_method_runtime(
    *,
    spec: MethodRuntimeSpec,
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Run a runtime spec runner and normalize async/sync runner shapes."""
    raw = spec.runner(dict(parameters or {}), context)
    if inspect.isawaitable(raw):
        raw = await raw
    if not isinstance(raw, MethodRunnerResult):
        raise TypeError(
            f"Runner for {spec.method_name}@{spec.method_version} must return "
            "MethodRunnerResult"
        )
    return raw


def ensure_default_method_runtimes_registered() -> None:
    """Register built-in method runtimes once per process."""
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return

    from .method_runners.logprobs_basic import run_logprobs_basic
    from .method_runners.perturb_then_infer_basic import (
        run_perturbation_then_inference_basic,
    )
    from .method_runners.file_artifact_basic import run_file_artifact_basic
    from .method_runners.csv_parse_basic import run_csv_parse_basic

    defaults = (
        MethodRuntimeSpec(
            method_name="perturbation_then_inference.basic",
            method_version="0.1",
            runner=run_perturbation_then_inference_basic,
            description="Stage-1 perturbation then inference runner.",
        ),
        MethodRuntimeSpec(
            method_name="inference.logprobs.basic",
            method_version="0.1",
            runner=run_logprobs_basic,
            description="Stage-2 logprobs pressure-test runner.",
        ),
        MethodRuntimeSpec(
            method_name="file_artifact.basic",
            method_version="0.1",
            runner=run_file_artifact_basic,
            description="Source-stage file artifact persistence runner.",
        ),
        MethodRuntimeSpec(
            method_name="csv_parse.basic",
            method_version="0.1",
            runner=run_csv_parse_basic,
            description="Imported-stage CSV-to-parquet transform runner.",
        ),
    )
    for spec in defaults:
        key = _registry_key(spec.method_name, spec.method_version)
        if key in _METHOD_RUNTIME_REGISTRY:
            continue
        _METHOD_RUNTIME_REGISTRY[key] = spec
    _DEFAULTS_REGISTERED = True


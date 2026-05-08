"""Tests for polymorphic method runtime registry."""

from __future__ import annotations

import pytest

from study_query_llm.algorithms.inference_methods import (
    INFERENCE_METHODS,
    MATURITY_RUNNER_WIRED,
)
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
    MethodRuntimeSpec,
    ensure_default_method_runtimes_registered,
    get_method_runtime,
    register_method_runtime,
    run_method_runtime,
)


def test_default_runtime_registry_contains_stage1_and_stage2_specs():
    """Built-in registry should include stage-1 and stage-2 method identities."""
    ensure_default_method_runtimes_registered()
    perturb = get_method_runtime("perturbation_then_inference.basic", "0.1")
    logprobs = get_method_runtime("inference.logprobs.basic", "0.1")
    assert perturb is not None
    assert logprobs is not None


def test_runtime_methods_are_marked_runner_wired_in_catalog():
    """Inference catalog maturity should match runtime dispatch wiring."""
    runtime_identities = {
        ("perturbation_then_inference.basic", "0.1"),
        ("inference.logprobs.basic", "0.1"),
    }
    maturity_by_identity = {
        (str(item["name"]), str(item["version"])): str(item.get("maturity") or "")
        for item in INFERENCE_METHODS
    }
    for identity in runtime_identities:
        assert maturity_by_identity.get(identity) == MATURITY_RUNNER_WIRED


def test_register_runtime_duplicate_raises_without_overwrite():
    """Runtime identities should not be silently shadowed by default."""
    spec = MethodRuntimeSpec(
        method_name="tests.example.runner",
        method_version="1.0",
        runner=lambda _p, _c: MethodRunnerResult(
            output_json={},
            pipeline_stage_role="export",
        ),
    )
    register_method_runtime(spec, overwrite=True)
    with pytest.raises(ValueError, match="Runtime already registered"):
        register_method_runtime(spec, overwrite=False)


@pytest.mark.asyncio
async def test_run_method_runtime_accepts_sync_and_async_runners():
    """run_method_runtime should normalize both sync and async runner shapes."""
    ctx = MethodRunnerContext(
        repository=None,  # type: ignore[arg-type]
        request_group_id=1,
        source_group_id=1,
        method_name="tests.runner",
        method_version="1.0",
        run_key="rk",
    )

    async def async_runner(_params, _ctx):
        return MethodRunnerResult(output_json={"mode": "async"}, pipeline_stage_role="export")

    spec = MethodRuntimeSpec(
        method_name="tests.runner",
        method_version="1.0",
        runner=async_runner,
    )
    out = await run_method_runtime(spec=spec, parameters={}, context=ctx)
    assert out.output_json["mode"] == "async"

    spec_sync = MethodRuntimeSpec(
        method_name="tests.runner.sync",
        method_version="1.0",
        runner=lambda _params, _ctx: MethodRunnerResult(
            output_json={"mode": "sync"},
            pipeline_stage_role="export",
        ),
    )
    out_sync = await run_method_runtime(spec=spec_sync, parameters={}, context=ctx)
    assert out_sync.output_json["mode"] == "sync"


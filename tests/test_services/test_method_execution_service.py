"""Tests for MethodExecutionService polymorphic dispatch and provenance writes."""

from __future__ import annotations

from uuid import uuid4

import pytest

from study_query_llm.algorithms.inference_methods import register_inference_methods
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.method_execution_service import (
    MethodExecutionService,
    compose_method_run_key,
)
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerResult,
    MethodRuntimeSpec,
    register_method_runtime,
)
from study_query_llm.services.method_service import MethodService


@pytest.fixture
def db_connection():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _create_request_group(repo: RawCallRepository) -> int:
    return int(
        repo.create_group(
            group_type="analysis_request",
            name="method-execution-tests",
            description="unit test request group",
            metadata_json={"dataset_name": "unit-test"},
        )
    )


class _CapturingProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls = []

    async def complete(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        return ProviderResponse(
            text="logprobs-output",
            provider="capture",
            tokens=6,
            latency_ms=1.0,
            metadata={"finish_reason": "stop"},
        )

    def get_provider_name(self) -> str:
        return "capture"


@pytest.mark.asyncio
async def test_execute_writes_and_reuses_canonical_provenanced_run(db_connection):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)

        async def _fake_runner(parameters, context):
            return MethodRunnerResult(
                output_json={"echo": dict(parameters)},
                pipeline_stage_role="subset",
                pipeline_stage_context={"selection_spec": "all"},
                metadata_json={"runner": "fake"},
            )

        register_method_runtime(
            MethodRuntimeSpec(
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                runner=_fake_runner,
            ),
            overwrite=True,
        )
        svc = MethodExecutionService(repo)
        first = await svc.execute(
            request_group_id=req_group_id,
            base_run_key="base",
            method_name="perturbation_then_inference.basic",
            method_version="0.1",
            parameters={"prompt": "hello"},
        )
        assert first.reused is False
        assert first.run_key == "base__method__perturbation_then_inference.basic@0.1"
        assert first.pipeline_stage_role == "subset"

        second = await svc.execute(
            request_group_id=req_group_id,
            base_run_key="base",
            method_name="perturbation_then_inference.basic",
            method_version="0.1",
            parameters={"prompt": "hello"},
        )
        assert second.reused is True
        assert second.run_id == first.run_id
        assert second.run_key == first.run_key

        row = repo.get_provenanced_run_by_id(first.run_id)
        assert row is not None
        assert row.run_kind == "execution"
        assert row.metadata_json["execution_role"] == "method_execution"
        assert row.metadata_json["pipeline_stage_role"] == "subset"
        assert row.metadata_json["pipeline_stage_context"]["selection_spec"] == "all"


@pytest.mark.asyncio
async def test_execute_invocation_id_appends_suffix_and_writes_distinct_row(db_connection):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)

        async def _fake_runner(parameters, context):
            return MethodRunnerResult(
                output_json={"prompt": parameters.get("prompt")},
                pipeline_stage_role="export",
            )

        register_method_runtime(
            MethodRuntimeSpec(
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                runner=_fake_runner,
            ),
            overwrite=True,
        )
        svc = MethodExecutionService(repo)
        base = await svc.execute(
            request_group_id=req_group_id,
            base_run_key="rk",
            method_name="perturbation_then_inference.basic",
            method_version="0.1",
            parameters={"prompt": "p"},
            node_id="n1",
        )
        inv = str(uuid4())
        distinct = await svc.execute(
            request_group_id=req_group_id,
            base_run_key="rk",
            method_name="perturbation_then_inference.basic",
            method_version="0.1",
            parameters={"prompt": "p"},
            node_id="n1",
            invocation_id=inv,
        )
        assert distinct.reused is False
        assert distinct.run_id != base.run_id
        assert distinct.run_key.endswith(f"__node__n1__inv__{inv}")


def test_compose_method_run_key_suffix_ordering_contract():
    key = compose_method_run_key(
        base_run_key="r",
        method_name="m",
        method_version="1.0",
        node_id="nodeA",
        invocation_id="123e4567-e89b-12d3-a456-426614174000",
    )
    assert key == (
        "r__method__m@1.0"
        "__node__nodeA"
        "__inv__123e4567-e89b-12d3-a456-426614174000"
    )


@pytest.mark.asyncio
async def test_execute_rejects_invalid_invocation_id_boundary(db_connection):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)
        svc = MethodExecutionService(repo)
        with pytest.raises(ValueError, match="invalid_method_execution_request"):
            await svc.execute(
                request_group_id=req_group_id,
                base_run_key="rk",
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                parameters={"prompt": "p"},
                invocation_id="not-a-uuid",
            )


@pytest.mark.asyncio
async def test_execute_stage2_logprobs_runner_preserves_kwargs_and_metadata_parity(
    db_connection,
    monkeypatch,
):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)

        provider = _CapturingProvider()
        from study_query_llm.providers.factory import ProviderFactory

        monkeypatch.setattr(
            ProviderFactory,
            "create_chat_provider",
            lambda _self, provider_name, model: provider,
        )

        svc = MethodExecutionService(repo)
        out = await svc.execute(
            request_group_id=req_group_id,
            base_run_key="rk2",
            method_name="inference.logprobs.basic",
            method_version="0.1",
            parameters={
                "prompt": "Hello",
                "provider": "local_llm",
                "model": "qwen2.5:14b",
                "temperature": 0.0,
                "max_tokens": 32,
                "logprobs": True,
                "top_logprobs": 3,
            },
        )
        assert out.pipeline_stage_role == "export"
        assert out.metadata_json["execution_role"] == "method_execution"
        assert out.output_json["request_controls"]["logprobs"] is True
        assert out.output_json["request_controls"]["top_logprobs"] == 3
        assert provider.calls, "expected provider to receive one call"
        call_kwargs = provider.calls[0]["kwargs"]
        assert call_kwargs["logprobs"] is True
        assert call_kwargs["top_logprobs"] == 3
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 32


@pytest.mark.asyncio
async def test_execute_rejects_invalid_pipeline_stage_role_from_runner(db_connection):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)

        async def _bad_runner(parameters, context):
            return MethodRunnerResult(
                output_json={"ok": True},
                pipeline_stage_role="not-a-real-role",
            )

        register_method_runtime(
            MethodRuntimeSpec(
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                runner=_bad_runner,
            ),
            overwrite=True,
        )
        svc = MethodExecutionService(repo)
        with pytest.raises(ValueError, match="invalid_pipeline_stage_role"):
            await svc.execute(
                request_group_id=req_group_id,
                base_run_key="rk",
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                parameters={"prompt": "p"},
            )


@pytest.mark.asyncio
async def test_execute_validates_imported_run_metadata_boundary(db_connection):
    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        req_group_id = _create_request_group(repo)

        imported_run_id = repo.create_provenanced_run(
            run_kind="execution",
            run_status="completed",
            request_group_id=req_group_id,
            source_group_id=req_group_id,
            run_key="imported_bad",
            metadata_json={"execution_role": "method_execution"},
        )

        async def _ok_runner(parameters, context):
            return MethodRunnerResult(
                output_json={"ok": True},
                pipeline_stage_role="export",
            )

        register_method_runtime(
            MethodRuntimeSpec(
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                runner=_ok_runner,
            ),
            overwrite=True,
        )
        svc = MethodExecutionService(repo)
        with pytest.raises(ValueError, match="malformed_imported_run_metadata"):
            await svc.execute(
                request_group_id=req_group_id,
                base_run_key="rk_import",
                method_name="perturbation_then_inference.basic",
                method_version="0.1",
                parameters={"prompt": "p"},
                imported_run_id=int(imported_run_id),
            )


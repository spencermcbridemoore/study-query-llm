"""Tests for job runner factory and runner classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.services.job_runner_factory import (
    create_job_runner,
    get_supported_job_types,
)
from study_query_llm.services.job_runners import (
    FinalizeRunRunner,
    JobRunContext,
    JobRunOutcome,
    ReduceKRunner,
    RunKTryRunner,
)
from study_query_llm.services.langgraph_job_runner import LangGraphJobRunner


def _mock_run_k_try(**kwargs) -> tuple:
    return (1, "/tmp/out.json", None)


def test_create_job_runner_run_k_try():
    runner = create_job_runner("run_k_try", run_k_try_fn=_mock_run_k_try)
    assert isinstance(runner, RunKTryRunner)


def test_create_job_runner_reduce_k():
    class MockReducer:
        def reduce_k_job(self, job_id: int) -> str:
            return "/tmp/reduce.json"

    runner = create_job_runner("reduce_k", reducer=MockReducer())
    assert isinstance(runner, ReduceKRunner)


def test_create_job_runner_finalize_run():
    class MockReducer:
        def finalize_run_job(self, job_id: int) -> int | None:
            return 42

    runner = create_job_runner("finalize_run", reducer=MockReducer())
    assert isinstance(runner, FinalizeRunRunner)


def test_create_job_runner_unsupported():
    with pytest.raises(ValueError, match="Unsupported job_type"):
        create_job_runner("invalid")


def test_create_job_runner_run_k_try_missing_fn():
    with pytest.raises(ValueError, match="run_k_try_fn required"):
        create_job_runner("run_k_try")


def test_create_job_runner_reduce_k_missing_reducer():
    with pytest.raises(ValueError, match="reducer required"):
        create_job_runner("reduce_k")


def test_create_job_runner_langgraph_run():
    runner = create_job_runner("langgraph_run")
    assert isinstance(runner, LangGraphJobRunner)


def test_get_supported_job_types():
    types = get_supported_job_types()
    assert "run_k_try" in types
    assert "reduce_k" in types
    assert "finalize_run" in types
    assert "langgraph_run" in types


def test_run_k_try_runner_returns_outcome():
    runner = RunKTryRunner(run_fn=_mock_run_k_try)
    context = JobRunContext(
        datasets={},
        provider_cache={},
        manager_cache={},
        tei_endpoint=None,
        provider_label="test",
        embedding_provider_name=None,
        worker_slot=0,
        repo_root=Path("."),
        claim_wait_seconds=0.0,
        reducer=None,
        db=None,
    )
    job_snapshot = {
        "id": 1,
        "job_type": "run_k_try",
        "job_key": "k",
        "payload_json": {"embedding_engine": "e/a", "dataset": "dbpedia"},
    }
    outcome = runner.run(job_snapshot, context)
    assert isinstance(outcome, JobRunOutcome)
    assert outcome.job_id == 1
    assert outcome.result_ref == "/tmp/out.json"
    assert outcome.error is None
    assert outcome.db_updated_by_runner is False

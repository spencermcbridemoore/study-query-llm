"""Tests for job runner factory and runner classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.services.jobs import (
    AnalysisRunRunner,
    ClusteringReducerPlugin,
    FinalizeRunRunner,
    JobRunContext,
    JobRunOutcome,
    LangGraphJobRunner,
    McqRunRunner,
    ReducerInput,
    ReducerOutput,
    ReduceKRunner,
    RunKTryRunner,
    create_job_runner,
    get_supported_job_types,
)


def _mock_run_k_try(**kwargs) -> tuple:
    return (1, "/tmp/out.json", None)


def _mock_run_mcq(**kwargs) -> tuple:
    return (2, "1234", None)


def _mock_run_analysis(**kwargs) -> tuple:
    return (3, "analysis:ok", None)


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
    with pytest.raises(ValueError, match="reducer_plugin"):
        create_job_runner("reduce_k")


def test_create_job_runner_mcq_run():
    runner = create_job_runner("mcq_run", mcq_run_fn=_mock_run_mcq)
    assert isinstance(runner, McqRunRunner)


def test_create_job_runner_mcq_run_missing_fn():
    with pytest.raises(ValueError, match="mcq_run_fn required"):
        create_job_runner("mcq_run")


def test_create_job_runner_analysis_run():
    runner = create_job_runner("analysis_run", analysis_run_fn=_mock_run_analysis)
    assert isinstance(runner, AnalysisRunRunner)


def test_create_job_runner_analysis_run_missing_fn():
    with pytest.raises(ValueError, match="analysis_run_fn required"):
        create_job_runner("analysis_run")


def test_clustering_reducer_plugin_wraps_legacy_service():
    class MockReducer:
        def reduce_k_job(self, job_id: int) -> str:
            return f"/tmp/reduce_{job_id}.json"

        def finalize_run_job(self, job_id: int) -> int:
            return 77

    plugin = ClusteringReducerPlugin(MockReducer())
    reduce_out = plugin.reduce_k(ReducerInput(job_snapshot={"id": 11}, context=None))
    finalize_out = plugin.finalize_run(ReducerInput(job_snapshot={"id": 12}, context=None))
    assert isinstance(reduce_out, ReducerOutput)
    assert reduce_out.result_ref == "/tmp/reduce_11.json"
    assert finalize_out.run_id == 77


def test_create_job_runner_langgraph_run():
    runner = create_job_runner("langgraph_run")
    assert isinstance(runner, LangGraphJobRunner)


def test_get_supported_job_types():
    types = get_supported_job_types()
    assert "run_k_try" in types
    assert "mcq_run" in types
    assert "analysis_run" in types
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


def test_mcq_run_runner_returns_outcome():
    runner = McqRunRunner(run_fn=_mock_run_mcq)
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
        "id": 2,
        "request_group_id": 11,
        "job_type": "mcq_run",
        "job_key": "mcq_rk",
        "base_run_key": "mcq_rk",
        "payload_json": {
            "run_key": "mcq_rk",
            "deployment": "gpt-4o-mini",
            "level": "high school",
            "subject": "physics",
            "options_per_question": 4,
            "questions_per_test": 20,
            "label_style": "upper",
            "spread_correct_answer_uniformly": False,
            "samples_per_combo": 5,
        },
    }
    outcome = runner.run(job_snapshot, context)
    assert isinstance(outcome, JobRunOutcome)
    assert outcome.job_id == 2
    assert outcome.result_ref == "1234"
    assert outcome.error is None
    assert outcome.db_updated_by_runner is False


def test_analysis_run_runner_returns_outcome():
    runner = AnalysisRunRunner(run_fn=_mock_run_analysis)
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
        "id": 3,
        "request_group_id": 11,
        "job_type": "analysis_run",
        "job_key": "analysis_1",
        "payload_json": {
            "request_id": 11,
            "sweep_type": "mcq",
            "analysis_key": "mcq_compliance",
        },
    }
    outcome = runner.run(job_snapshot, context)
    assert isinstance(outcome, JobRunOutcome)
    assert outcome.job_id == 3
    assert outcome.result_ref == "analysis:ok"
    assert outcome.error is None
    assert outcome.db_updated_by_runner is False

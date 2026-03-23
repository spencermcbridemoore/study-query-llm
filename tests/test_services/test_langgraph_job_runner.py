"""Tests for LangGraph job runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from study_query_llm.services.jobs import JobRunContext, LangGraphJobRunner


def test_langgraph_runner_echo():
    runner = LangGraphJobRunner()
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        context = JobRunContext(
            datasets={},
            provider_cache={},
            manager_cache={},
            tei_endpoint=None,
            provider_label="test",
            embedding_provider_name=None,
            worker_slot=0,
            repo_root=repo_root,
            claim_wait_seconds=0.0,
            reducer=None,
            db=None,
        )
        job_snapshot = {
            "id": 1,
            "job_type": "langgraph_run",
            "job_key": "lg_1",
            "payload_json": {"prompt": "hello"},
        }
        outcome = runner.run(job_snapshot, context)
        assert outcome.error is None
        assert outcome.result_ref is not None
        assert outcome.db_updated_by_runner is False
        out_path = Path(outcome.result_ref)
        assert out_path.exists(), f"Expected file at {out_path}"
        import json
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["state"]["output"] == "hello"


def test_langgraph_runner_validation_error():
    runner = LangGraphJobRunner()
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
    outcome = runner.run(
        {"job_type": "langgraph_run", "payload_json": {}},
        context,
    )
    assert outcome.error is not None
    assert "payload_validation_error" in outcome.error
    assert outcome.result_ref is None

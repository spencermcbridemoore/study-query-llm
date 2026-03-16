"""Tests for worker orchestrator factory and strategy classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.services.worker_orchestrator import (
    ShardedWorkerOrchestrator,
    StandaloneWorkerOrchestrator,
    create_worker_orchestrator,
)


def _mock_standalone(**kwargs: object) -> int:
    return 42


def _mock_sharded(**kwargs: object) -> int:
    return 17


def test_create_worker_orchestrator_standalone():
    orch = create_worker_orchestrator(
        mode="standalone",
        run_standalone_fn=_mock_standalone,
        run_sharded_fn=_mock_sharded,
        request_id=1,
        worker_id="w1",
        worker_slot=0,
        embedding_engine=None,
        tei_endpoint=None,
        provider_label="test",
        embedding_provider_name=None,
        claim_lease_seconds=60,
        max_runs=None,
        idle_exit_seconds=90,
        force=False,
        repo_root=Path(__file__).parent,
    )
    assert isinstance(orch, StandaloneWorkerOrchestrator)
    assert orch.run() == 42


def test_create_worker_orchestrator_sharded():
    orch = create_worker_orchestrator(
        mode="sharded",
        run_standalone_fn=_mock_standalone,
        run_sharded_fn=_mock_sharded,
        request_id=1,
        worker_id="w1",
        worker_slot=0,
        embedding_engine="engine/a",
        tei_endpoint="http://localhost:8080",
        provider_label="test",
        embedding_provider_name=None,
        claim_lease_seconds=60,
        max_runs=None,
        idle_exit_seconds=90,
        force=False,
        repo_root=Path(__file__).parent,
    )
    assert isinstance(orch, ShardedWorkerOrchestrator)
    assert orch.run() == 17


def test_create_worker_orchestrator_unsupported_mode():
    with pytest.raises(ValueError, match="Unsupported job_mode"):
        create_worker_orchestrator(
            mode="invalid",
            run_standalone_fn=_mock_standalone,
            run_sharded_fn=_mock_sharded,
            request_id=1,
            worker_id="w1",
            worker_slot=0,
            embedding_engine=None,
            tei_endpoint=None,
            provider_label="test",
            embedding_provider_name=None,
            claim_lease_seconds=60,
            max_runs=None,
            idle_exit_seconds=90,
            force=False,
            repo_root=Path(__file__).parent,
        )

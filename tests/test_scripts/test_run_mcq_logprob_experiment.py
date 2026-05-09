"""Tests for scripts/run_mcq_logprob_experiment.py."""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from study_query_llm.algorithms.inference_methods import register_inference_methods
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_execution_service import MethodExecutionService
from study_query_llm.services.method_runners.mcq_logprob_basic import ProbeResult, ProbeTierStats
from study_query_llm.services.method_service import MethodService

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "run_mcq_logprob_experiment.py"

EXPECTED_EXPENSIVE = (
    "alpindale/goliath-120b",
    "anthracite-org/magnum-v4-72b",
    "mancer/weaver",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "qwen/qwen3.6-max-preview",
    "sao10k/l3.3-euryale-70b",
)

EXPECTED_CHEAP = (
    "gryphe/mythomax-l2-13b",
    "mistralai/ministral-3b-2512",
    "openai/gpt-4o-mini",
    "thedrummer/rocinante-12b",
    "thedrummer/unslopnemo-12b",
    "undi95/remm-slerp-l2-13b",
)


@pytest.fixture
def mcq_exp_mod():
    spec = importlib.util.spec_from_file_location("run_mcq_logprob_experiment", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def db_connection():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


def _seed_imported_dataset(repo: RawCallRepository, request_group_id: int, artifact_dir: str) -> int:
    frame = pd.DataFrame(
        [
            {
                "ItemID": 101,
                "ItemType": "choice",
                "ProblemBody": "Block on incline.",
                "ProblemBodyTemplate": "",
                "OptionA": "A",
                "OptionB": "B",
                "OptionC": "C",
                "OptionD": "D",
                "OptionE": "E",
                "CorrectLetter": "A",
                "temp_bank_id": 6445,
            },
        ]
    )
    artifact_service = ArtifactService(repository=repo, artifact_dir=artifact_dir)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine="pyarrow", index=False)
    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=int(request_group_id),
        step_name="csv_parse",
        logical_filename="midterm2_questions_v1.parquet",
        data=buf.getvalue(),
        artifact_type="dataset_canonical_parquet",
        content_type="application/octet-stream",
        metadata={"dataset_name": "midterm2_questions", "dataset_version": "v1"},
    )
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    assert artifact is not None
    return int(
        repo.create_provenanced_run(
            run_kind="execution",
            run_status="completed",
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            run_key="imported-midterm2-test",
            result_ref=str(artifact.uri),
            metadata_json={
                "execution_role": "method_execution",
                "pipeline_stage_role": "imported",
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
    )


def test_expected_model_roster_and_strategies(mcq_exp_mod) -> None:
    assert tuple(sorted(mcq_exp_mod.MODEL_IDS_LATIN_SQUARES_25)) == EXPECTED_EXPENSIVE
    assert tuple(sorted(mcq_exp_mod.MODEL_IDS_FULL_120)) == EXPECTED_CHEAP
    assert len(mcq_exp_mod.ALL_14_MODELS) == 14
    for m in mcq_exp_mod.ALL_14_MODELS:
        if m in mcq_exp_mod.MODEL_IDS_LATIN_SQUARES_25:
            assert mcq_exp_mod.permutation_strategy_for_model(m) == "latin_squares_25"
        else:
            assert mcq_exp_mod.permutation_strategy_for_model(m) == "full_120"


def test_probe_ceiling_openai_family(mcq_exp_mod) -> None:
    assert mcq_exp_mod.probe_ceiling_for_model("openai/gpt-4o") == 20
    assert mcq_exp_mod.probe_ceiling_for_model("openai/gpt-4o-mini") == 20
    assert mcq_exp_mod.probe_ceiling_for_model("sao10k/l3.3-euryale-70b") == 5


def test_survivors_use_probe_resolved_concurrency(mcq_exp_mod) -> None:
    probe_results: dict[str, ProbeResult] = {}
    for m in mcq_exp_mod.ALL_14_MODELS:
        probe_results[m] = ProbeResult(
            model=m,
            resolved_concurrency=7,
            excluded_reason=None,
            reachability_outcome="ok",
            tier_stats=(
                ProbeTierStats(
                    concurrency=2,
                    total_calls=4,
                    rate_limited_calls=0,
                    total_retry_count=0,
                    total_wait_ms=0,
                    passed=True,
                ),
            ),
        )
    survivors, excluded = mcq_exp_mod.build_survivors(probe_results, mcq_exp_mod.ALL_14_MODELS)
    assert not excluded
    assert len(survivors) == 14
    for _mid, _strat, conc in survivors:
        assert conc == 7


def test_probe_report_json_roundtrip(mcq_exp_mod) -> None:
    pr = ProbeResult(
        model="openai/gpt-4o-mini",
        resolved_concurrency=2,
        excluded_reason=None,
        reachability_outcome="ok",
        tier_stats=(
            ProbeTierStats(
                concurrency=2,
                total_calls=4,
                rate_limited_calls=1,
                total_retry_count=2,
                total_wait_ms=100,
                passed=True,
            ),
        ),
    )
    summary_json = {
        "probe_started_at": "t0",
        "probe_finished_at": "t1",
        "probe_duration_seconds": 1.5,
        "models": [mcq_exp_mod._serialize_probe_result("openai/gpt-4o-mini", pr)],
    }
    md = mcq_exp_mod.format_probe_report_markdown(
        probe_started_at="t0",
        probe_finished_at="t1",
        probe_duration_seconds=1.5,
        results={"openai/gpt-4o-mini": pr},
        summary_json=summary_json,
    )
    parsed = mcq_exp_mod.parse_probe_report(md)
    assert parsed["probe_duration_seconds"] == 1.5
    back = mcq_exp_mod.probe_summary_to_results(parsed)
    assert back["openai/gpt-4o-mini"].resolved_concurrency == 2
    assert back["openai/gpt-4o-mini"].tier_stats[0].rate_limited_calls == 1


@pytest.mark.asyncio
async def test_skip_probe_missing_file_aborts(mcq_exp_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    missing = tmp_path / "nope.md"
    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="x",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=True,
            probe_max_age_hours=24.0,
            probe_report_path=str(missing),
            max_spend=1e9,
            max_runtime_hours=1e9,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=True,
        )
    )
    assert rc == 1


@pytest.mark.asyncio
async def test_skip_probe_stale_aborts(
    mcq_exp_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    report = tmp_path / "probe.md"
    probe_results = {
        m: ProbeResult(
            model=m,
            resolved_concurrency=1,
            excluded_reason=None,
            reachability_outcome="ok",
            tier_stats=tuple(),
        )
        for m in mcq_exp_mod.ALL_14_MODELS
    }
    summary_json = {
        "probe_started_at": "t0",
        "probe_finished_at": "t1",
        "probe_duration_seconds": 1.0,
        "models": [
            mcq_exp_mod._serialize_probe_result(m, probe_results[m])
            for m in sorted(probe_results.keys())
        ],
    }
    md = mcq_exp_mod.format_probe_report_markdown(
        probe_started_at="t0",
        probe_finished_at="t1",
        probe_duration_seconds=1.0,
        results=probe_results,
        summary_json=summary_json,
    )
    report.write_text(md, encoding="utf-8")
    old = time.time() - 86400 * 7
    os.utime(report, (old, old))
    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="x",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=True,
            probe_max_age_hours=24.0,
            probe_report_path=str(report),
            max_spend=1e9,
            max_runtime_hours=1e9,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=True,
        )
    )
    assert rc == 1


@pytest.mark.asyncio
async def test_max_spend_halt(mcq_exp_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")

    async def _fake_probe(_models, _ceilings, **_kw):
        out: dict[str, ProbeResult] = {}
        for m in mcq_exp_mod.ALL_14_MODELS:
            out[m] = ProbeResult(
                model=m,
                resolved_concurrency=1,
                excluded_reason=None,
                reachability_outcome="ok",
                tier_stats=tuple(),
            )
        return out

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="x",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=0.000001,
            max_runtime_hours=1e9,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=False,
        )
    )
    assert rc == 1


@pytest.mark.asyncio
async def test_max_runtime_halt(mcq_exp_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")

    async def _fake_probe(_models, _ceilings, **_kw):
        return {
            m: ProbeResult(
                model=m,
                resolved_concurrency=1,
                excluded_reason=None,
                reachability_outcome="ok",
                tier_stats=tuple(),
            )
            for m in mcq_exp_mod.ALL_14_MODELS
        }

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="x",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e-12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=False,
        )
    )
    assert rc == 1


@pytest.mark.asyncio
async def test_probe_excluded_threshold_halt(mcq_exp_mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")

    async def _fake_probe(_models, _ceilings, **_kw):
        out: dict[str, ProbeResult] = {}
        for i, m in enumerate(mcq_exp_mod.ALL_14_MODELS):
            if i < 4:
                out[m] = ProbeResult(
                    model=m,
                    resolved_concurrency=1,
                    excluded_reason="EXCLUDED:test",
                    reachability_outcome="other_error",
                    tier_stats=tuple(),
                )
            else:
                out[m] = ProbeResult(
                    model=m,
                    resolved_concurrency=2,
                    excluded_reason=None,
                    reachability_outcome="ok",
                    tier_stats=tuple(),
                )
        return out

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="x",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=False,
        )
    )
    assert rc == 1


@pytest.mark.asyncio
async def test_dry_run_skips_execute(
    mcq_exp_mod,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    spy = AsyncMock(side_effect=AssertionError("execute should not be called"))

    monkeypatch.setattr(MethodExecutionService, "execute", spy)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="drylbl",
            imported_run_id=1049,
            max_questions=45,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=None,
            dry_run=True,
        )
    )
    assert rc == 0
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_continue_on_failure_second_invocation_raises(
    mcq_exp_mod,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)

    db_path = tmp_path / "db.sqlite"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()
    artifact_dir = str((tmp_path / "artifacts").resolve())
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        ms = MethodService(repo)
        register_inference_methods(ms)
        gid = repo.create_group(
            group_type="analysis_request",
            name="mcq-cont",
            metadata_json={},
        )
        imported_id = _seed_imported_dataset(repo, gid, artifact_dir)

    calls: list[str] = []

    async def _fake_probe(_models, _ceilings, **_kw):
        return {
            m: ProbeResult(
                model=m,
                resolved_concurrency=1,
                excluded_reason=None,
                reachability_outcome="ok",
                tier_stats=tuple(),
            )
            for m in mcq_exp_mod.ALL_14_MODELS
        }

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    async def _fake_execute(_self, **_kwargs):
        calls.append(str(_kwargs.get("node_id") or ""))
        if len(calls) == 2:
            raise RuntimeError("boom_second_model")
        return SimpleNamespace(
            run_id=99,
            reused=False,
            run_key="rk",
        )

    monkeypatch.setattr(MethodExecutionService, "execute", _fake_execute)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="cont",
            imported_run_id=int(imported_id),
            max_questions=1,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=int(gid),
            base_run_key="bk_test",
            dry_run=False,
        )
    )
    assert rc == 2
    assert len(calls) == 14
    assert calls[1] == "anthracite-org_magnum-v4-72b"


def test_main_help_exits_zero() -> None:
    import subprocess
    import sys

    env = dict(os.environ)
    env["DATABASE_URL"] = "sqlite:///:memory:"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0
    assert "--experiment-label" in result.stdout

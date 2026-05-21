"""Tests for scripts/run_mcq_logprob_experiment.py."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import json
import os
import re
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
DEFAULT_FORMATS = ("v1", "v2_chat_system")


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


def test_force_permutation_strategy_applies_to_all_models(mcq_exp_mod) -> None:
    probe_results: dict[str, ProbeResult] = {}
    for m in mcq_exp_mod.ALL_14_MODELS:
        probe_results[m] = ProbeResult(
            model=m,
            resolved_concurrency=4,
            excluded_reason=None,
            reachability_outcome="ok",
            tier_stats=tuple(),
        )
    survivors, excluded = mcq_exp_mod.build_survivors(
        probe_results,
        mcq_exp_mod.ALL_14_MODELS,
        smoke=False,
        force_permutation_strategy="full_120",
    )
    assert not excluded
    assert len(survivors) == 14
    assert {strategy for _mid, strategy, _conc in survivors} == {"full_120"}


def test_run_key_parameter_mismatch_detects_divergence(mcq_exp_mod) -> None:
    row = SimpleNamespace(
        run_status="completed",
        config_json={
            "parameters": {
                "permutation_strategy": "latin_squares_25",
                "prompt_template_version": "v1",
            }
        },
        metadata_json={"imported_run_ref": {"run_id": 1049}},
    )
    mismatch = mcq_exp_mod.find_run_key_parameter_mismatch(
        existing_row=row,
        imported_run_id=2000,
        permutation_strategy="full_120",
        prompt_template_version="v2_chat_system",
    )
    assert mismatch is not None
    assert mismatch[0] == "permutation_strategy"


def test_load_prompt_tokens_by_format_json(mcq_exp_mod, tmp_path: Path) -> None:
    payload = {"v1_mean": 102.0, "v2_chat_system_mean": 134.0}
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = mcq_exp_mod.load_prompt_tokens_by_format(path)
    assert loaded[mcq_exp_mod.FORMAT_V1] == pytest.approx(102.0)
    assert loaded[mcq_exp_mod.FORMAT_V2_CHAT_SYSTEM] == pytest.approx(134.0)


@pytest.mark.asyncio
async def test_run_key_mismatch_halt_before_execute(
    mcq_exp_mod,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")

    async def _fake_probe(_models, _ceilings, **_kw):
        return {
            mcq_exp_mod.SMOKE_MODEL_ID: ProbeResult(
                model=mcq_exp_mod.SMOKE_MODEL_ID,
                resolved_concurrency=2,
                excluded_reason=None,
                reachability_outcome="ok",
                tier_stats=tuple(),
            )
        }

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    db = DatabaseConnectionV2(f"sqlite:///{tmp_path}/db.sqlite", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        ms = MethodService(repo)
        register_inference_methods(ms)
        request_group_id = repo.create_group(
            group_type="analysis_request",
            name="mismatch-test",
            metadata_json={},
        )
        base_run_key = "bk_mismatch"
        plan = mcq_exp_mod._format_to_invocation(
            model_id=mcq_exp_mod.SMOKE_MODEL_ID,
            strategy="single_latin_square_5",
            concurrency_cap=2,
            format_name=mcq_exp_mod.FORMAT_V1,
        )
        run_key = mcq_exp_mod.compose_method_run_key(
            base_run_key=base_run_key,
            method_name=mcq_exp_mod.METHOD_NAME,
            method_version=mcq_exp_mod.METHOD_VERSION,
            node_id=plan.node_id,
            invocation_id=None,
        )
        repo.create_provenanced_run(
            run_kind="execution",
            run_status="completed",
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            run_key=run_key,
            config_json={
                "parameters": {
                    "permutation_strategy": "full_120",
                    "prompt_template_version": "v1",
                }
            },
            metadata_json={
                "experiment_label": "mismatch-test",
                "imported_run_ref": {"run_id": 9999},
            },
        )

    spy = AsyncMock()

    async def _spy_execute(_self, **_kwargs):
        spy()
        return SimpleNamespace(run_id=1, reused=False, run_key="rk")

    monkeypatch.setattr(MethodExecutionService, "execute", _spy_execute)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="mismatch-test",
            imported_run_id=1049,
            max_questions=1,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=int(request_group_id),
            base_run_key=base_run_key,
            dry_run=False,
            formats="both",
            smoke=True,
            force_permutation_strategy=None,
            token_estimate_path=None,
        )
    )
    assert rc == 1
    spy.assert_not_called()


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
    survivors, excluded = mcq_exp_mod.build_survivors(
        probe_results,
        mcq_exp_mod.ALL_14_MODELS,
        smoke=False,
    )
    assert not excluded
    assert len(survivors) == 14
    for _mid, _strat, conc in survivors:
        assert conc == 7


def test_resolve_formats_flag(mcq_exp_mod) -> None:
    assert mcq_exp_mod.resolve_formats("both") == DEFAULT_FORMATS
    assert mcq_exp_mod.resolve_formats("v1") == ("v1",)
    assert mcq_exp_mod.resolve_formats("v2_chat_system") == ("v2_chat_system",)


def test_dual_format_expansion_emits_28_invocations(mcq_exp_mod) -> None:
    survivors = [
        (
            m,
            mcq_exp_mod.permutation_strategy_for_model(m),
            5,
        )
        for m in mcq_exp_mod.ALL_14_MODELS
    ]
    plans = mcq_exp_mod.build_invocation_plans(survivors, formats=DEFAULT_FORMATS)
    assert len(plans) == 28
    assert {p.prompt_template_version for p in plans} == {"v1", "v2_chat_system"}
    assert all("__v1" in p.node_id or "__v2_chat_system" in p.node_id for p in plans)


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
            formats="both",
            smoke=False,
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
            formats="both",
            smoke=False,
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
            formats="both",
            smoke=False,
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
            formats="both",
            smoke=False,
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
            formats="both",
            smoke=False,
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
            formats="both",
            smoke=False,
        )
    )
    assert rc == 0
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_smoke_flag_runs_one_model_two_formats_in_dry_run(
    mcq_exp_mod,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")

    async def _fake_probe(models, _ceilings, **_kw):
        assert list(models) == [mcq_exp_mod.SMOKE_MODEL_ID]
        return {
            mcq_exp_mod.SMOKE_MODEL_ID: ProbeResult(
                model=mcq_exp_mod.SMOKE_MODEL_ID,
                resolved_concurrency=2,
                excluded_reason=None,
                reachability_outcome="ok",
                tier_stats=tuple(),
            )
        }

    monkeypatch.setattr(mcq_exp_mod, "probe_rate_limits_per_model", _fake_probe)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="smoke",
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
            formats="v1",
            smoke=True,
        )
    )
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "planned_invocations_count=2" in stdout
    assert "strategy=single_latin_square_5" in stdout
    assert "format=v1" in stdout
    assert "format=v2_chat_system" in stdout


@pytest.mark.asyncio
async def test_dry_run_skips_already_completed_run_keys_and_reduces_spend(
    mcq_exp_mod,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/db.sqlite")

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

    survivors = [
        (
            m,
            mcq_exp_mod.permutation_strategy_for_model(m),
            1,
        )
        for m in mcq_exp_mod.ALL_14_MODELS
    ]
    plans = mcq_exp_mod.build_invocation_plans(survivors, formats=DEFAULT_FORMATS)
    base_run_key = "bk_skip_prefight"
    run_keys = [
        mcq_exp_mod.compose_method_run_key(
            base_run_key=base_run_key,
            method_name=mcq_exp_mod.METHOD_NAME,
            method_version=mcq_exp_mod.METHOD_VERSION,
            node_id=plan.node_id,
            invocation_id=None,
        )
        for plan in plans
    ]
    skipped_run_keys = set(run_keys[:3])

    def _fake_completed_rows(*, db, experiment_label):
        assert db is not None
        assert experiment_label == "skip-test"
        rows: dict[str, SimpleNamespace] = {}
        for plan, run_key in zip(plans, run_keys):
            if run_key not in skipped_run_keys:
                continue
            rows[run_key] = SimpleNamespace(
                run_key=run_key,
                config_json={
                    "parameters": {
                        "permutation_strategy": plan.strategy,
                        "prompt_template_version": plan.prompt_template_version,
                    }
                },
                metadata_json={"imported_run_ref": {"run_id": 1049}},
                run_status="completed",
            )
        return rows

    monkeypatch.setattr(
        mcq_exp_mod,
        "load_completed_execution_rows_by_run_key",
        _fake_completed_rows,
    )

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="skip-test",
            imported_run_id=1049,
            max_questions=1,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=None,
            base_run_key=base_run_key,
            dry_run=True,
            formats="both",
            smoke=False,
        )
    )
    assert rc == 0

    stdout = capsys.readouterr().out
    assert "planned_invocations: 28" in stdout
    assert "skipped_already_completed: 3" in stdout
    assert "to_execute: 25" in stdout

    remaining_plans = [plan for plan, run_key in zip(plans, run_keys) if run_key not in skipped_run_keys]
    full_spend = mcq_exp_mod.estimate_total_spend_usd(plans, catalog={}, max_questions=1)
    reduced_spend = mcq_exp_mod.estimate_total_spend_usd(
        remaining_plans,
        catalog={},
        max_questions=1,
    )
    assert reduced_spend < full_spend

    match = re.search(r"estimated_spend_usd=([0-9]+(?:\.[0-9]+)?)", stdout)
    assert match is not None
    observed_spend = float(match.group(1))
    assert observed_spend == pytest.approx(reduced_spend, rel=1e-6, abs=1e-6)


@pytest.mark.asyncio
async def test_parallel_fanout_uses_asyncio_gather(
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
            name="mcq-gather",
            metadata_json={},
        )
        imported_id = _seed_imported_dataset(repo, gid, artifact_dir)

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

    gather_sizes: list[int] = []
    top_logprobs_values: list[int] = []
    real_gather = asyncio.gather

    async def _spy_gather(*aws, **kwargs):
        gather_sizes.append(len(aws))
        return await real_gather(*aws, **kwargs)

    monkeypatch.setattr(mcq_exp_mod.asyncio, "gather", _spy_gather)

    async def _fake_execute(_self, **_kwargs):
        params = dict(_kwargs.get("parameters") or {})
        top_logprobs_values.append(int(params.get("top_logprobs")))
        return SimpleNamespace(
            run_id=99,
            reused=False,
            run_key="rk",
        )

    monkeypatch.setattr(MethodExecutionService, "execute", _fake_execute)

    rc = await mcq_exp_mod.run_experiment_async(
        argparse.Namespace(
            experiment_label="gather",
            imported_run_id=int(imported_id),
            max_questions=1,
            skip_probe=False,
            probe_max_age_hours=24.0,
            probe_report_path=str(tmp_path / "probe_out.md"),
            max_spend=1e12,
            max_runtime_hours=1e12,
            catalog_path=str(tmp_path / "noop.json"),
            request_group_id=int(gid),
            base_run_key="bk_gather",
            dry_run=False,
            formats="both",
            smoke=False,
        )
    )
    assert rc == 0
    assert 28 in gather_sizes
    assert len(top_logprobs_values) == 28
    assert set(top_logprobs_values) == {5}


@pytest.mark.asyncio
async def test_failure_isolation_does_not_cancel_other_parallel_invocations(
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
            formats="both",
            smoke=False,
        )
    )
    assert rc == 2
    assert len(calls) == 28
    assert any(node.endswith("__v1") for node in calls)
    assert any(node.endswith("__v2_chat_system") for node in calls)


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

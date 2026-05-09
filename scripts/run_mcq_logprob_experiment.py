#!/usr/bin/env python3
"""Phase 2 orchestrator: probe OpenRouter concurrency, then run 14-model MCQ logprob experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.algorithms.inference_methods import register_inference_methods
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import default_write_intent_for_connection
from study_query_llm.services.method_execution_service import MethodExecutionService
from study_query_llm.services.method_runners.mcq_logprob_basic import (
    ProbeResult,
    ProbeTierStats,
    probe_rate_limits_per_model,
)
from study_query_llm.services.method_runtime_registry import ensure_default_method_runtimes_registered, get_method_runtime
from study_query_llm.services.method_service import MethodService

METHOD_NAME = "inference.mcq_logprob.basic"
METHOD_VERSION = "0.1"
PROVIDER = "openrouter"
DEFAULT_PROBE_REPORT_REL = "scratch/mcq_logprob_concurrency_probe.md"
DEFAULT_CATALOG_REL = "scratch/openrouter_models_catalog_raw.json"

MAX_QUESTIONS_DEFAULT = 45
EST_PROMPT_TOKENS = 400
EST_COMPLETION_TOKENS = 1
RPS_PER_SLOT = 1.0

_PROBE_JSON_FENCE_START = "```mcq-probe-json\n"
_PROBE_JSON_FENCE_END = "\n```"

# Expensive tier: latin_squares_25 (8 models)
MODEL_IDS_LATIN_SQUARES_25: tuple[str, ...] = (
    "openai/gpt-3.5-turbo",
    "sao10k/l3.3-euryale-70b",
    "mancer/weaver",
    "qwen/qwen3.6-max-preview",
    "openai/gpt-4o",
    "anthracite-org/magnum-v4-72b",
    "alpindale/goliath-120b",
    "openai/gpt-4-turbo",
)

# Cheap tier: full_120 (6 models)
MODEL_IDS_FULL_120: tuple[str, ...] = (
    "gryphe/mythomax-l2-13b",
    "mistralai/ministral-3b-2512",
    "openai/gpt-4o-mini",
    "thedrummer/rocinante-12b",
    "thedrummer/unslopnemo-12b",
    "undi95/remm-slerp-l2-13b",
)

ALL_14_MODELS: tuple[str, ...] = tuple(
    sorted(
        set(MODEL_IDS_LATIN_SQUARES_25 + MODEL_IDS_FULL_120),
        key=lambda m: (m.split("/")[0], m),
    )
)

OPENAI_PROBE_CEILING_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4-turbo",
    }
)

# Fallback per-token USD pricing (prompt, completion) when catalog row missing — aligned with
# `scratch/openrouter_logprobs_report.md` PASS table snapshots.
_FALLBACK_MODEL_PRICING_USD_PER_TOKEN: dict[str, tuple[float, float]] = {
    "openai/gpt-3.5-turbo": (5.0e-7, 1.5e-6),
    "sao10k/l3.3-euryale-70b": (6.5e-7, 7.5e-7),
    "mancer/weaver": (7.5e-7, 1.0e-6),
    "qwen/qwen3.6-max-preview": (1.04e-6, 6.24e-6),
    "openai/gpt-4o": (2.5e-6, 1.0e-5),
    "anthracite-org/magnum-v4-72b": (3.0e-6, 5.0e-6),
    "alpindale/goliath-120b": (3.75e-6, 7.5e-6),
    "openai/gpt-4-turbo": (1.0e-5, 3.0e-5),
    "gryphe/mythomax-l2-13b": (6.0e-8, 6.0e-8),
    "mistralai/ministral-3b-2512": (1.0e-7, 1.0e-7),
    "openai/gpt-4o-mini": (1.5e-7, 6.0e-7),
    "thedrummer/rocinante-12b": (1.7e-7, 4.3e-7),
    "thedrummer/unslopnemo-12b": (4.0e-7, 4.0e-7),
    "undi95/remm-slerp-l2-13b": (4.5e-7, 6.5e-7),
}


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    return token or "value"


def _resolve_database_url() -> str:
    return (os.environ.get("DATABASE_URL") or "").strip()


def permutation_strategy_for_model(model_id: str) -> str:
    if model_id in MODEL_IDS_LATIN_SQUARES_25:
        return "latin_squares_25"
    if model_id in MODEL_IDS_FULL_120:
        return "full_120"
    raise ValueError(f"unknown_model_not_in_phase2_roster:{model_id}")


def permutation_count_for_strategy(strategy: str) -> int:
    if strategy == "latin_squares_25":
        return 25
    if strategy == "full_120":
        return 120
    raise ValueError(f"unknown_permutation_strategy:{strategy}")


def probe_ceiling_for_model(model_id: str) -> int:
    return 20 if model_id in OPENAI_PROBE_CEILING_MODELS else 5


def build_target_concurrency_map(models: Sequence[str]) -> dict[str, int]:
    return {str(m): probe_ceiling_for_model(str(m)) for m in models}


def estimate_calls(max_questions: int, strategy: str) -> int:
    return int(max_questions) * permutation_count_for_strategy(strategy)


def load_catalog_pricing_usd_per_token(
    catalog_path: Path,
) -> dict[str, tuple[float, float]]:
    if not catalog_path.is_file():
        return {}
    raw = catalog_path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    data = payload.get("data")
    if not isinstance(data, list):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("id") or "").strip()
        pricing = row.get("pricing")
        if not mid or not isinstance(pricing, dict):
            continue
        try:
            p = float(pricing.get("prompt") or 0)
            c = float(pricing.get("completion") or 0)
        except (TypeError, ValueError):
            continue
        out[mid] = (max(p, 0.0), max(c, 0.0))
    return out


def price_per_call_usd(
    model_id: str,
    catalog: Mapping[str, tuple[float, float]],
) -> float:
    prompt_p, comp_p = catalog.get(model_id) or _FALLBACK_MODEL_PRICING_USD_PER_TOKEN.get(
        model_id, (0.0, 0.0)
    )
    return (
        float(prompt_p) * float(EST_PROMPT_TOKENS) + float(comp_p) * float(EST_COMPLETION_TOKENS)
    )


def estimate_total_spend_usd(
    survivors: Sequence[tuple[str, str, int]],
    *,
    catalog: Mapping[str, tuple[float, float]],
    max_questions: int,
) -> float:
    total = 0.0
    for model_id, strategy, _conc in survivors:
        calls = estimate_calls(max_questions, strategy)
        per_call = price_per_call_usd(model_id, catalog)
        total += calls * per_call
    return total


def estimate_total_runtime_hours(
    survivors: Sequence[tuple[str, str, int]],
    *,
    max_questions: int,
    rps_per_slot: float = RPS_PER_SLOT,
) -> float:
    """Wall-clock lower bound: calls / (resolved_concurrency * rps_per_slot) per model, summed."""
    total_seconds = 0.0
    for model_id, strategy, resolved_conc in survivors:
        calls = estimate_calls(max_questions, strategy)
        slots = max(1, int(resolved_conc))
        effective_rps = float(slots) * float(rps_per_slot)
        total_seconds += float(calls) / effective_rps
    return total_seconds / 3600.0


def _serialize_probe_result(model_id: str, result: ProbeResult) -> dict[str, Any]:
    tiers: list[dict[str, Any]] = []
    for tier in result.tier_stats:
        tiers.append(
            {
                "concurrency": tier.concurrency,
                "total_calls": tier.total_calls,
                "rate_limited_calls": tier.rate_limited_calls,
                "total_retry_count": tier.total_retry_count,
                "total_wait_ms": tier.total_wait_ms,
                "passed": tier.passed,
            }
        )
    return {
        "model": model_id,
        "resolved_concurrency": int(result.resolved_concurrency),
        "excluded_reason": result.excluded_reason,
        "reachability_outcome": result.reachability_outcome,
        "tier_stats": tiers,
    }


def format_probe_report_markdown(
    *,
    probe_started_at: str,
    probe_finished_at: str,
    probe_duration_seconds: float,
    results: Mapping[str, ProbeResult],
    summary_json: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# MCQ logprob concurrency probe")
    lines.append("")
    lines.append(f"- probe_started_at: `{probe_started_at}`")
    lines.append(f"- probe_finished_at: `{probe_finished_at}`")
    lines.append(f"- probe_wall_clock_seconds: `{probe_duration_seconds:.3f}`")
    lines.append("")
    lines.append("## Per-model results")
    lines.append("")
    lines.append("| model | resolved_concurrency | excluded_reason | reachability |")
    lines.append("|---|---:|---|---|")
    for model_id in sorted(results.keys()):
        r = results[model_id]
        reason = r.excluded_reason or ""
        lines.append(
            f"| `{model_id}` | {int(r.resolved_concurrency)} | {reason} | {r.reachability_outcome} |"
        )
    lines.append("")
    lines.append("## EXCLUDED (detail)")
    lines.append("")
    for model_id in sorted(results.keys()):
        r = results[model_id]
        if r.excluded_reason:
            lines.append(f"- `{model_id}`: **{r.excluded_reason}**")
    lines.append("")
    lines.append("## Per-tier 429 / retry counts")
    lines.append("")
    for model_id in sorted(results.keys()):
        r = results[model_id]
        lines.append(f"### `{model_id}`")
        lines.append("")
        if not r.tier_stats:
            lines.append("(no tier ramp — reachability failed or excluded early)")
        else:
            lines.append("| tier_concurrency | total_calls | rate_limited_calls | retry_count | wait_ms | passed |")
            lines.append("|---:|---:|---:|---:|---:|:---:|")
            for t in r.tier_stats:
                lines.append(
                    f"| {t.concurrency} | {t.total_calls} | {t.rate_limited_calls} | "
                    f"{t.total_retry_count} | {t.total_wait_ms} | {t.passed} |"
                )
        lines.append("")
    lines.append(_PROBE_JSON_FENCE_START)
    lines.append(json.dumps(summary_json, indent=2, sort_keys=True))
    lines.append(_PROBE_JSON_FENCE_END)
    lines.append("")
    return "\n".join(lines)


def parse_probe_report(text: str) -> dict[str, Any]:
    start = text.find(_PROBE_JSON_FENCE_START)
    if start < 0:
        raise ValueError("probe_report_missing_json_fence")
    rest = text[start + len(_PROBE_JSON_FENCE_START) :]
    end = rest.find(_PROBE_JSON_FENCE_END)
    if end < 0:
        raise ValueError("probe_report_unclosed_json_fence")
    raw_json = rest[:end].strip()
    payload = json.loads(raw_json)
    if not isinstance(payload, dict):
        raise ValueError("probe_report_json_not_object")
    return payload


def probe_summary_to_results(payload: dict[str, Any]) -> dict[str, ProbeResult]:
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("probe_report_missing_models_array")
    out: dict[str, ProbeResult] = {}
    for row in models:
        if not isinstance(row, dict):
            continue
        mid = str(row.get("model") or "").strip()
        if not mid:
            continue
        tier_rows = row.get("tier_stats") or []
        tier_stats: list[ProbeTierStats] = []
        if isinstance(tier_rows, list):
            for tr in tier_rows:
                if not isinstance(tr, dict):
                    continue
                tier_stats.append(
                    ProbeTierStats(
                        concurrency=int(tr.get("concurrency") or 0),
                        total_calls=int(tr.get("total_calls") or 0),
                        rate_limited_calls=int(tr.get("rate_limited_calls") or 0),
                        total_retry_count=int(tr.get("total_retry_count") or 0),
                        total_wait_ms=int(tr.get("total_wait_ms") or 0),
                        passed=bool(tr.get("passed")),
                    )
                )
        reason = row.get("excluded_reason")
        out[mid] = ProbeResult(
            model=mid,
            resolved_concurrency=max(1, int(row.get("resolved_concurrency") or 1)),
            excluded_reason=str(reason) if reason else None,
            reachability_outcome=str(row.get("reachability_outcome") or "unknown"),
            tier_stats=tuple(tier_stats),
        )
    return out


def assert_method_registered(*, method_service: MethodService) -> None:
    row = method_service.get_method(METHOD_NAME, version=METHOD_VERSION)
    if row is None:
        raise ValueError(
            f"method_not_registered:{METHOD_NAME}@{METHOD_VERSION}; "
            "run scripts/register_inference_methods.py first."
        )
    runtime = get_method_runtime(METHOD_NAME, METHOD_VERSION)
    if runtime is None:
        raise ValueError(
            f"runtime_not_registered:{METHOD_NAME}@{METHOD_VERSION}; "
            "ensure method_runtime_registry defaults include this method."
        )


def build_survivors(
    probe_results: Mapping[str, ProbeResult],
    models: Sequence[str],
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str]]]:
    """Return (survivors, excluded_list)."""
    survivors: list[tuple[str, str, int]] = []
    excluded: list[tuple[str, str]] = []
    for model_id in models:
        r = probe_results.get(model_id)
        if r is None:
            excluded.append((model_id, "EXCLUDED:missing_probe_row"))
            continue
        if r.excluded_reason:
            excluded.append((model_id, r.excluded_reason))
            continue
        strategy = permutation_strategy_for_model(model_id)
        survivors.append((model_id, strategy, int(r.resolved_concurrency)))
    return survivors, excluded


async def run_experiment_async(args: argparse.Namespace) -> int:
    db_url = _resolve_database_url()
    if not db_url:
        print("ERROR: DATABASE_URL not set.", file=sys.stderr)
        return 1

    probe_report_path: Path = Path(args.probe_report_path).expanduser()
    if not probe_report_path.is_absolute():
        probe_report_path = (REPO_ROOT / probe_report_path).resolve()

    catalog_path: Path = Path(args.catalog_path).expanduser()
    if not catalog_path.is_absolute():
        catalog_path = (REPO_ROOT / catalog_path).resolve()

    catalog_pricing = load_catalog_pricing_usd_per_token(catalog_path)

    ensure_default_method_runtimes_registered()

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=default_write_intent_for_connection(db_url),
    )
    db.init_db()

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_service = MethodService(repo)
        register_inference_methods(method_service)
        try:
            assert_method_registered(method_service=method_service)
        except ValueError as exc:
            print(f"ERROR: halt:method_registration — {exc}", file=sys.stderr)
            return 1

        probe_results: dict[str, ProbeResult]
        probe_started_at: str
        probe_finished_at: str
        probe_duration_seconds: float

        if args.dry_run and not args.skip_probe:
            probe_results = {
                m: ProbeResult(
                    model=m,
                    resolved_concurrency=probe_ceiling_for_model(m),
                    excluded_reason=None,
                    reachability_outcome="ok",
                    tier_stats=tuple(),
                )
                for m in ALL_14_MODELS
            }
            probe_started_at = "dry-run-hypothetical"
            probe_finished_at = probe_started_at
            probe_duration_seconds = 0.0
        elif args.skip_probe:
            if not probe_report_path.is_file():
                print(
                    f"ERROR: halt:probe_cache_missing — {probe_report_path} does not exist.",
                    file=sys.stderr,
                )
                return 1
            mtime = datetime.fromtimestamp(
                probe_report_path.stat().st_mtime, tz=timezone.utc
            )
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
            if age_hours > float(args.probe_max_age_hours):
                print(
                    f"ERROR: halt:probe_cache_stale — report age {age_hours:.3f}h "
                    f"exceeds --probe-max-age-hours={args.probe_max_age_hours}.",
                    file=sys.stderr,
                )
                return 1
            try:
                text = probe_report_path.read_text(encoding="utf-8")
                payload = parse_probe_report(text)
                probe_started_at = str(payload.get("probe_started_at") or "")
                probe_finished_at = str(payload.get("probe_finished_at") or "")
                probe_duration_seconds = float(payload.get("probe_duration_seconds") or 0.0)
                probe_results = probe_summary_to_results(payload)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                print(f"ERROR: halt:probe_cache_invalid — {exc}", file=sys.stderr)
                return 1
        else:
            t0 = time.perf_counter()
            probe_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ceilings = build_target_concurrency_map(ALL_14_MODELS)
            probe_results = await probe_rate_limits_per_model(
                list(ALL_14_MODELS),
                ceilings,
                provider=PROVIDER,
            )
            probe_finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            probe_duration_seconds = time.perf_counter() - t0

            summary_models = [_serialize_probe_result(m, probe_results[m]) for m in sorted(probe_results.keys())]
            summary_json = {
                "probe_started_at": probe_started_at,
                "probe_finished_at": probe_finished_at,
                "probe_duration_seconds": probe_duration_seconds,
                "models": summary_models,
            }
            md = format_probe_report_markdown(
                probe_started_at=probe_started_at,
                probe_finished_at=probe_finished_at,
                probe_duration_seconds=probe_duration_seconds,
                results=probe_results,
                summary_json=summary_json,
            )
            probe_report_path.parent.mkdir(parents=True, exist_ok=True)
            probe_report_path.write_text(md, encoding="utf-8")
            print(f"wrote probe report: {probe_report_path}")

        survivors, excluded = build_survivors(probe_results, ALL_14_MODELS)
        excluded_count = len(excluded)
        if excluded_count > 3:
            print(
                f"ERROR: halt:probe_excluded_threshold — {excluded_count} models excluded (> 3).",
                file=sys.stderr,
            )
            for mid, reason in excluded:
                print(f"  excluded `{mid}`: {reason}", file=sys.stderr)
            return 1

        for mid, reason in excluded:
            print(f"WARN: dropping excluded model `{mid}`: {reason}", file=sys.stderr)

        max_questions = int(args.max_questions)
        spend = estimate_total_spend_usd(
            survivors,
            catalog=catalog_pricing,
            max_questions=max_questions,
        )
        runtime_h = estimate_total_runtime_hours(
            survivors,
            max_questions=max_questions,
        )

        if not args.dry_run:
            if spend > float(args.max_spend):
                print(
                    f"ERROR: halt:max_spend_exceeded — estimated ${spend:.4f} > --max-spend={args.max_spend}.",
                    file=sys.stderr,
                )
                return 1

            if runtime_h > float(args.max_runtime_hours):
                print(
                    f"ERROR: halt:max_runtime_hours_exceeded — estimated {runtime_h:.4f}h "
                    f"> --max-runtime-hours={args.max_runtime_hours}.",
                    file=sys.stderr,
                )
                return 1

        base_run_key = args.base_run_key or f"mcq_logprob__{_slug(args.experiment_label)}"

        if args.dry_run:
            print("dry-run: no MethodExecutionService.execute calls")
            print(f"request_group_id={'<new>' if args.request_group_id is None else args.request_group_id}")
            print(f"base_run_key={base_run_key}")
            print(f"imported_run_id={args.imported_run_id}")
            print(f"estimated_spend_usd={spend:.6f}")
            print(f"estimated_runtime_hours={runtime_h:.6f}")
            print("planned_invocations:")
            for model_id, strategy, conc in survivors:
                print(
                    f"  - model={model_id} strategy={strategy} concurrency_cap={conc} "
                    f"node_id={_slug(model_id)}"
                )
            return 0

        if args.request_group_id is not None:
            group = repo.get_group_by_id(int(args.request_group_id))
            if group is None:
                print(f"ERROR: request_group_not_found:{args.request_group_id}", file=sys.stderr)
                return 1
            request_group_id = int(args.request_group_id)
        else:
            request_group_id = int(
                repo.create_group(
                    group_type="analysis_request",
                    name=f"mcq_logprob:{_slug(args.experiment_label)}",
                    description="Phase 2 MCQ logprob 14-model orchestration",
                    metadata_json={
                        "experiment_label": args.experiment_label,
                        "script": "scripts/run_mcq_logprob_experiment.py",
                    },
                )
            )

        execution = MethodExecutionService(repo)
        failures: list[str] = []
        for model_id, strategy, conc in survivors:
            params = {
                "provider": PROVIDER,
                "model": model_id,
                "permutation_strategy": strategy,
                "format_idx": 0,
                "concurrency_cap": int(conc),
                "max_questions": max_questions,
                "top_logprobs": 20,
                "prompt_template_version": "v1",
                "metadata": {
                    "experiment_label": args.experiment_label,
                    "dataset_name": "midterm2_questions",
                    "dataset_version": "v1",
                    "imported_run_id": int(args.imported_run_id),
                    "permutation_strategy": strategy,
                },
            }
            node_id = _slug(model_id)
            try:
                outcome = await execution.execute(
                    request_group_id=int(request_group_id),
                    base_run_key=base_run_key,
                    method_name=METHOD_NAME,
                    method_version=METHOD_VERSION,
                    parameters=params,
                    source_group_id=int(request_group_id),
                    imported_run_id=int(args.imported_run_id),
                    node_id=node_id,
                )
                print(
                    f"ok model={model_id} run_id={outcome.run_id} "
                    f"reused={outcome.reused} run_key={outcome.run_key}"
                )
            except Exception as exc:
                msg = f"{model_id}: {exc}"
                failures.append(msg)
                print(f"ERROR: invocation_failed — {msg}", file=sys.stderr)

        if failures:
            print(f"completed with {len(failures)} failure(s).", file=sys.stderr)
        print(f"request_group_id={request_group_id}")
        print(f"base_run_key={base_run_key}")
        return 0 if not failures else 2


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Probe OpenRouter concurrency limits, then fan out inference.mcq_logprob.basic@0.1 "
            "for the Phase 2 14-model roster."
        )
    )
    p.add_argument(
        "--experiment-label",
        required=True,
        help="Stable label stored at provenanced_runs.metadata_json.experiment_label (via runner metadata merge).",
    )
    p.add_argument(
        "--imported-run-id",
        type=int,
        default=1049,
        help="Provenanced run id for midterm2 parquet (default 1049).",
    )
    p.add_argument(
        "--max-questions",
        type=int,
        default=MAX_QUESTIONS_DEFAULT,
        help="Runner max_questions cap (default 45 = full filtered subset).",
    )
    p.add_argument("--skip-probe", action="store_true", help="Reuse cached probe report file.")
    p.add_argument(
        "--probe-max-age-hours",
        type=float,
        default=24.0,
        help="With --skip-probe, max age of probe report mtime (default 24).",
    )
    p.add_argument(
        "--probe-report-path",
        default=DEFAULT_PROBE_REPORT_REL,
        help="Path to probe markdown report under scratch/ (repo-relative ok).",
    )
    p.add_argument(
        "--max-spend",
        type=float,
        default=20.0,
        help="Preflight estimated USD budget across surviving models (default 20).",
    )
    p.add_argument(
        "--max-runtime-hours",
        type=float,
        default=12.0,
        help="Preflight estimated wall-clock hours at 1 RPS per concurrency slot (default 12).",
    )
    p.add_argument(
        "--catalog-path",
        default=DEFAULT_CATALOG_REL,
        help="OpenRouter catalog JSON for pricing estimates (optional).",
    )
    p.add_argument(
        "--request-group-id",
        type=int,
        default=None,
        help="Optional existing analysis_request group id (default: create new).",
    )
    p.add_argument(
        "--base-run-key",
        default=None,
        help="Optional base run key (default: mcq_logprob__<slug(experiment_label)>).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate registration and print plan without execute() or probe writes.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_experiment_async(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

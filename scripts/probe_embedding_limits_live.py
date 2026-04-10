#!/usr/bin/env python3
"""Live embedding limit probe for Azure OpenAI and OpenRouter.

This script performs controlled load probes against embedding endpoints and
produces:
1) request-level raw event log (JSONL),
2) per-step summary table (CSV),
3) markdown report with quota snapshots + recommendations.

It is designed to be safe-by-default:
- bounded step duration
- bounded total runtime
- bounded timeout per request
- early-stop on persistent high error or throttle rates
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import study_query_llm.config  # noqa: F401  # load .env via side-effect
from study_query_llm.config import Config


LIMIT_HEADER_PARTS: Tuple[str, ...] = ("rate", "limit", "retry", "reset")

DEFAULT_BATCH_SIZES = "1,8,32,128"
DEFAULT_CONCURRENCY_LEVELS = "1,2,4,8"
DEFAULT_DURATION_SECONDS = 60
DEFAULT_COOLDOWN_SECONDS = 15
DEFAULT_MAX_TOTAL_MINUTES = 30.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True)
class PlatformConfig:
    platform: str
    url: str
    headers: Dict[str, str]
    payload_model: Optional[str]
    payload_extra: Dict[str, Any]


@dataclass(frozen=True)
class StepConfig:
    step_index: int
    platform: str
    batch_size: int
    concurrency: int
    duration_seconds: int


@dataclass(frozen=True)
class RequestEvent:
    timestamp_utc: str
    platform: str
    step_index: int
    request_index: int
    batch_size: int
    concurrency: int
    status_code: Optional[int]
    ok: bool
    latency_ms: float
    est_input_tokens: int
    usage_total_tokens: Optional[int]
    response_items_count: Optional[int]
    error_type: str
    error_message: str
    limit_headers: Dict[str, str]


@dataclass(frozen=True)
class StepSummary:
    platform: str
    step_index: int
    batch_size: int
    concurrency: int
    elapsed_seconds: float
    attempted_requests: int
    success_requests: int
    throttled_requests: int
    error_requests: int
    attempted_rpm: float
    success_rpm: float
    throttled_rpm: float
    est_tpm_attempted: float
    est_tpm_success: float
    error_rate: float
    throttle_rate: float
    p50_latency_ms: Optional[float]
    p95_latency_ms: Optional[float]
    sample_error: str
    stop_reason: str


def parse_int_csv(raw: str, *, arg_name: str) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for token in str(raw).split(","):
        part = token.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"{arg_name}: invalid integer value '{part}'") from exc
        if value <= 0:
            raise ValueError(f"{arg_name}: expected positive integer, got {value}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError(f"{arg_name}: must contain at least one positive integer")
    return out


def estimate_tokens_for_text(text: str) -> int:
    # Approximation fallback that is stable across platforms.
    # This is intentionally conservative for throughput estimation.
    if not text:
        return 1
    return max(1, int((len(text) + 3) // 4))


def estimate_tokens_for_batch(texts: Sequence[str]) -> int:
    return int(sum(estimate_tokens_for_text(t) for t in texts))


def collect_limit_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if any(part in lower for part in LIMIT_HEADER_PARTS):
            out[str(key)] = str(value)
    return out


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[idx])


def summarize_step(events: Sequence[RequestEvent], step: StepConfig, stop_reason: str) -> StepSummary:
    if events:
        elapsed_seconds = max(
            0.001,
            (
                _parse_utc(events[-1].timestamp_utc) - _parse_utc(events[0].timestamp_utc)
            ).total_seconds(),
        )
    else:
        elapsed_seconds = float(step.duration_seconds)

    attempted = len(events)
    success = sum(1 for ev in events if ev.ok)
    throttled = sum(1 for ev in events if ev.status_code == 429)
    errors = attempted - success
    attempted_tokens = sum(int(ev.est_input_tokens) for ev in events)
    success_tokens = sum(int(ev.est_input_tokens) for ev in events if ev.ok)
    success_latencies = sorted(float(ev.latency_ms) for ev in events if ev.ok)
    sample_error = ""
    for ev in events:
        if not ev.ok and ev.error_message:
            sample_error = f"{ev.error_type}: {ev.error_message}"[:300]
            break

    return StepSummary(
        platform=step.platform,
        step_index=int(step.step_index),
        batch_size=int(step.batch_size),
        concurrency=int(step.concurrency),
        elapsed_seconds=float(elapsed_seconds),
        attempted_requests=int(attempted),
        success_requests=int(success),
        throttled_requests=int(throttled),
        error_requests=int(errors),
        attempted_rpm=float(attempted * 60.0 / elapsed_seconds),
        success_rpm=float(success * 60.0 / elapsed_seconds),
        throttled_rpm=float(throttled * 60.0 / elapsed_seconds),
        est_tpm_attempted=float(attempted_tokens * 60.0 / elapsed_seconds),
        est_tpm_success=float(success_tokens * 60.0 / elapsed_seconds),
        error_rate=float(errors / attempted) if attempted else 0.0,
        throttle_rate=float(throttled / attempted) if attempted else 0.0,
        p50_latency_ms=percentile(success_latencies, 0.50),
        p95_latency_ms=percentile(success_latencies, 0.95),
        sample_error=sample_error,
        stop_reason=str(stop_reason),
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _resolve_output_dir(raw: str) -> Path:
    out = Path(raw)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    return out


def _default_output_dir() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"experimental_results/embedding_limit_probe_{ts}"


def _build_default_corpus() -> List[str]:
    return [
        "Check account balance for last statement cycle.",
        "Report unauthorized card transaction and freeze card immediately.",
        "Transfer 250 dollars from checking to savings account tomorrow morning.",
        "What documents are required to open a joint account in branch?",
        "Explain loan repayment schedule and prepayment penalty options.",
        "Reset online banking password and enable two-factor authentication.",
        "Find nearest ATM with cash deposit support and opening hours.",
        "Dispute chargeback decision and provide merchant evidence details.",
        "Convert foreign currency transfer quote into final settlement amount.",
        "Update beneficiary details for recurring international wire transfer.",
        "Create budgeting categories and monthly spending alerts per merchant.",
        "Request card replacement due to chip damage and delivery tracking.",
    ]


def _resolve_platform_configs(args: argparse.Namespace) -> Dict[str, PlatformConfig]:
    cfg = Config()
    configs: Dict[str, PlatformConfig] = {}
    requested = [p.strip().lower() for p in str(args.platforms).split(",") if p.strip()]

    if "azure" in requested:
        azure_api_key = str(args.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "")).strip()
        azure_endpoint = str(args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")).strip()
        azure_api_version = str(
            args.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        ).strip()
        azure_deployment = str(
            args.azure_deployment
            or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
            or os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        ).strip()
        if not azure_api_key or not azure_endpoint or not azure_deployment:
            try:
                az_cfg = cfg.get_provider_config("azure")
                azure_api_key = azure_api_key or str(az_cfg.api_key or "").strip()
                azure_endpoint = azure_endpoint or str(az_cfg.endpoint or "").strip()
            except Exception:
                pass
        if azure_api_key and azure_endpoint and azure_deployment:
            base = azure_endpoint.rstrip("/")
            url = f"{base}/openai/deployments/{azure_deployment}/embeddings"
            configs["azure"] = PlatformConfig(
                platform="azure",
                url=url,
                headers={"api-key": azure_api_key, "Content-Type": "application/json"},
                payload_model=None,
                payload_extra={"api-version": azure_api_version},
            )

    if "openrouter" in requested:
        openrouter_api_key = str(
            args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        ).strip()
        openrouter_endpoint = str(
            args.openrouter_endpoint or os.getenv("OPENROUTER_ENDPOINT", "https://openrouter.ai/api/v1")
        ).strip()
        openrouter_model = str(
            args.openrouter_model
            or os.getenv("OPENROUTER_EMBED_MODEL", "")
            or os.getenv("OPENROUTER_EMBEDDING_MODEL", "")
            or os.getenv("OPENROUTER_MODEL", "")
        ).strip()
        if not openrouter_api_key:
            try:
                or_cfg = cfg.get_provider_config("openrouter")
                openrouter_api_key = openrouter_api_key or str(or_cfg.api_key or "").strip()
                openrouter_endpoint = openrouter_endpoint or str(or_cfg.endpoint or "").strip()
                openrouter_model = openrouter_model or str(or_cfg.model or "").strip()
            except Exception:
                pass
        if openrouter_api_key and openrouter_endpoint and openrouter_model:
            url = f"{openrouter_endpoint.rstrip('/')}/embeddings"
            configs["openrouter"] = PlatformConfig(
                platform="openrouter",
                url=url,
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                payload_model=openrouter_model,
                payload_extra={},
            )

    return configs


def _build_step_plan(
    platforms: Sequence[str],
    batch_sizes: Sequence[int],
    concurrency_levels: Sequence[int],
    duration_seconds: int,
) -> List[StepConfig]:
    plan: List[StepConfig] = []
    step_i = 1
    for platform in platforms:
        for batch_size in batch_sizes:
            for concurrency in concurrency_levels:
                plan.append(
                    StepConfig(
                        step_index=step_i,
                        platform=str(platform),
                        batch_size=int(batch_size),
                        concurrency=int(concurrency),
                        duration_seconds=int(duration_seconds),
                    )
                )
                step_i += 1
    return plan


def _next_batch(corpus: Sequence[str], batch_size: int, request_index: int) -> List[str]:
    total = len(corpus)
    start = request_index % total
    out: List[str] = []
    for i in range(batch_size):
        out.append(str(corpus[(start + i) % total]))
    return out


async def _send_embedding_request(
    client: httpx.AsyncClient,
    platform_cfg: PlatformConfig,
    texts: Sequence[str],
) -> Tuple[Optional[int], Dict[str, str], Optional[int], Optional[int], str, str]:
    payload: Dict[str, Any] = {"input": list(texts)}
    if platform_cfg.payload_model:
        payload["model"] = platform_cfg.payload_model

    params: Optional[Dict[str, str]] = None
    if platform_cfg.platform == "azure":
        params = {"api-version": str(platform_cfg.payload_extra["api-version"])}

    try:
        response = await client.post(
            platform_cfg.url,
            headers=platform_cfg.headers,
            params=params,
            json=payload,
        )
    except Exception as exc:
        return None, {}, None, None, type(exc).__name__, str(exc)

    limit_headers = collect_limit_headers(response.headers)
    status_code = int(response.status_code)
    error_type = ""
    error_message = ""
    usage_total_tokens: Optional[int] = None
    response_items_count: Optional[int] = None

    try:
        data = response.json()
    except Exception:
        data = None

    if isinstance(data, dict):
        usage = data.get("usage")
        if isinstance(usage, dict):
            raw_total = usage.get("total_tokens")
            if raw_total is not None:
                try:
                    usage_total_tokens = int(raw_total)
                except (TypeError, ValueError):
                    usage_total_tokens = None
        raw_items = data.get("data")
        if isinstance(raw_items, list):
            response_items_count = len(raw_items)
        err_obj = data.get("error")
        if isinstance(err_obj, dict):
            error_type = str(err_obj.get("type") or "")
            error_message = str(err_obj.get("message") or "")

    if not response.is_success and not error_message:
        error_message = response.text[:600]
    if not response.is_success and not error_type:
        error_type = f"HTTP_{status_code}"

    return (
        status_code,
        limit_headers,
        usage_total_tokens,
        response_items_count,
        error_type,
        error_message,
    )


async def run_step(
    *,
    client: httpx.AsyncClient,
    step: StepConfig,
    platform_cfg: PlatformConfig,
    corpus: Sequence[str],
    max_total_deadline: float,
    early_abort_error_rate: float,
    early_abort_throttle_rate: float,
    early_abort_min_requests: int,
) -> Tuple[List[RequestEvent], str]:
    step_started = time.monotonic()
    step_deadline = min(max_total_deadline, step_started + float(step.duration_seconds))
    request_counter = count(1)
    events: List[RequestEvent] = []
    stop_reason = "step_duration_elapsed"
    lock = asyncio.Lock()
    stop_flag = asyncio.Event()

    async def worker() -> None:
        nonlocal stop_reason
        while not stop_flag.is_set() and time.monotonic() < step_deadline:
            request_index = next(request_counter)
            texts = _next_batch(corpus, int(step.batch_size), request_index)
            est_tokens = estimate_tokens_for_batch(texts)
            t0 = time.perf_counter()
            (
                status_code,
                limit_headers,
                usage_total_tokens,
                response_items_count,
                error_type,
                error_message,
            ) = await _send_embedding_request(client, platform_cfg, texts)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ok = bool(status_code is not None and 200 <= int(status_code) < 300)
            event = RequestEvent(
                timestamp_utc=_now_utc_iso(),
                platform=step.platform,
                step_index=int(step.step_index),
                request_index=int(request_index),
                batch_size=int(step.batch_size),
                concurrency=int(step.concurrency),
                status_code=status_code,
                ok=ok,
                latency_ms=float(latency_ms),
                est_input_tokens=int(est_tokens),
                usage_total_tokens=usage_total_tokens,
                response_items_count=response_items_count,
                error_type=str(error_type),
                error_message=str(error_message),
                limit_headers=dict(limit_headers),
            )
            async with lock:
                events.append(event)
                attempted = len(events)
                errors = sum(1 for ev in events if not ev.ok)
                throttled = sum(1 for ev in events if ev.status_code == 429)
                if attempted >= int(early_abort_min_requests):
                    error_rate = errors / attempted
                    throttle_rate = throttled / attempted
                    if error_rate >= float(early_abort_error_rate):
                        stop_reason = f"early_abort_error_rate={error_rate:.3f}"
                        stop_flag.set()
                    elif throttle_rate >= float(early_abort_throttle_rate):
                        stop_reason = f"early_abort_throttle_rate={throttle_rate:.3f}"
                        stop_flag.set()

    workers = [asyncio.create_task(worker()) for _ in range(int(step.concurrency))]
    await asyncio.gather(*workers)

    if time.monotonic() >= max_total_deadline:
        stop_reason = "global_time_budget_exhausted"
    elif stop_flag.is_set():
        # keep reason assigned by worker
        pass
    elif time.monotonic() >= step_deadline:
        stop_reason = "step_duration_elapsed"
    else:
        stop_reason = "completed"
    return events, stop_reason


async def fetch_azure_quota_usage(args: argparse.Namespace) -> Dict[str, Any]:
    subscription_id = str(args.azure_subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID", "")).strip()
    location = str(args.azure_location or os.getenv("AZURE_LOCATION", "")).strip()
    token = str(
        args.azure_arm_bearer_token
        or os.getenv("AZURE_ARM_BEARER_TOKEN", "")
        or os.getenv("AZURE_ARM_ACCESS_TOKEN", "")
    ).strip()
    if not subscription_id or not location or not token:
        return {
            "ok": False,
            "reason": "missing AZURE_SUBSCRIPTION_ID / AZURE_LOCATION / AZURE_ARM_BEARER_TOKEN",
        }
    url = (
        "https://management.azure.com/subscriptions/"
        f"{subscription_id}/providers/Microsoft.CognitiveServices/locations/{location}/usages"
    )
    params = {"api-version": "2023-05-01"}
    headers = {"Authorization": f"Bearer {token}"}
    timeout = httpx.Timeout(30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params, headers=headers)
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "ok": bool(resp.is_success),
            "status_code": int(resp.status_code),
            "data": payload,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


async def fetch_openrouter_key_info(api_key: str) -> Dict[str, Any]:
    url = "https://openrouter.ai/api/v1/key"
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(30.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        payload = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "ok": bool(resp.is_success),
            "status_code": int(resp.status_code),
            "data": payload,
        }
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _to_csv_rows(step_summaries: Sequence[StepSummary]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in step_summaries:
        item = asdict(row)
        if item["p50_latency_ms"] is not None:
            item["p50_latency_ms"] = round(float(item["p50_latency_ms"]), 3)
        if item["p95_latency_ms"] is not None:
            item["p95_latency_ms"] = round(float(item["p95_latency_ms"]), 3)
        for key in (
            "elapsed_seconds",
            "attempted_rpm",
            "success_rpm",
            "throttled_rpm",
            "est_tpm_attempted",
            "est_tpm_success",
            "error_rate",
            "throttle_rate",
        ):
            item[key] = round(float(item[key]), 6)
        rows.append(item)
    return rows


def _best_sustainable_step(rows: Sequence[StepSummary], *, max_error_rate: float, max_throttle_rate: float) -> Optional[StepSummary]:
    candidates = [
        row
        for row in rows
        if row.error_rate <= max_error_rate and row.throttle_rate <= max_throttle_rate
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row.est_tpm_success, row.success_rpm))


def _first_throttle_step(rows: Sequence[StepSummary]) -> Optional[StepSummary]:
    for row in rows:
        if row.throttled_requests > 0:
            return row
    return None


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_latency(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def build_report_markdown(
    *,
    step_summaries: Sequence[StepSummary],
    quota_snapshot: Dict[str, Any],
    stable_error_rate_threshold: float,
    stable_throttle_rate_threshold: float,
) -> str:
    by_platform: Dict[str, List[StepSummary]] = {}
    for row in step_summaries:
        by_platform.setdefault(row.platform, []).append(row)

    lines: List[str] = []
    lines.append("# Embedding Limit Probe Report")
    lines.append("")
    lines.append(f"- Generated at: `{_now_utc_iso()}`")
    lines.append(f"- Steps completed: `{len(step_summaries)}`")
    lines.append("")
    lines.append("## Quota Snapshot")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(quota_snapshot, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Observed Throughput")
    lines.append("")
    lines.append("| platform | step | batch | conc | success_rpm | est_tpm_success | 429_rate | error_rate | p95_ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in step_summaries:
        lines.append(
            "| "
            f"{row.platform} | {row.step_index} | {row.batch_size} | {row.concurrency} | "
            f"{row.success_rpm:.2f} | {row.est_tpm_success:.2f} | "
            f"{row.throttle_rate:.4f} | {row.error_rate:.4f} | {_format_latency(row.p95_latency_ms)} |"
        )
    lines.append("")
    lines.append("## Limit Findings")
    lines.append("")
    for platform, rows in by_platform.items():
        first_throttle = _first_throttle_step(rows)
        best = _best_sustainable_step(
            rows,
            max_error_rate=stable_error_rate_threshold,
            max_throttle_rate=stable_throttle_rate_threshold,
        )
        lines.append(f"### {platform}")
        if first_throttle is None:
            lines.append("- First throttle point: not observed in tested matrix.")
        else:
            lines.append(
                "- First throttle point: "
                f"step `{first_throttle.step_index}` (batch={first_throttle.batch_size}, "
                f"conc={first_throttle.concurrency}, success_rpm={first_throttle.success_rpm:.2f}, "
                f"est_tpm_success={first_throttle.est_tpm_success:.2f})."
            )
        if best is None:
            lines.append(
                "- Max sustainable point: none met thresholds "
                f"(error_rate <= {stable_error_rate_threshold:.3f}, "
                f"throttle_rate <= {stable_throttle_rate_threshold:.3f})."
            )
        else:
            lines.append(
                "- Max sustainable point: "
                f"step `{best.step_index}` (batch={best.batch_size}, conc={best.concurrency}, "
                f"success_rpm={best.success_rpm:.2f}, est_tpm_success={best.est_tpm_success:.2f})."
            )
            lines.append(
                "- Suggested production cap (80%): "
                f"~`{best.success_rpm * 0.8:.2f}` RPM and "
                f"`{best.est_tpm_success * 0.8:.2f}` estimated TPM."
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Token throughput is estimated from input size (`~4 chars/token`) unless provider usage "
        "fields are available in response payloads."
    )
    lines.append("- Request headers in raw events include only keys related to rate/limit/retry/reset.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe live embedding limits for Azure and OpenRouter.")
    parser.add_argument(
        "--platforms",
        type=str,
        default="azure,openrouter",
        help="Comma-separated platforms to test (azure,openrouter).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=DEFAULT_BATCH_SIZES,
        help=f"Comma-separated batch sizes (default: {DEFAULT_BATCH_SIZES}).",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default=DEFAULT_CONCURRENCY_LEVELS,
        help=f"Comma-separated concurrency levels (default: {DEFAULT_CONCURRENCY_LEVELS}).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Duration per step in seconds (default: {DEFAULT_DURATION_SECONDS}).",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=f"Sleep between steps in seconds (default: {DEFAULT_COOLDOWN_SECONDS}).",
    )
    parser.add_argument(
        "--max-total-minutes",
        type=float,
        default=DEFAULT_MAX_TOTAL_MINUTES,
        help=f"Global probe time budget in minutes (default: {DEFAULT_MAX_TOTAL_MINUTES}).",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_REQUEST_TIMEOUT_SECONDS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=_default_output_dir(),
        help="Output directory for JSONL/CSV/report outputs.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved plan and exit.")

    # Azure overrides
    parser.add_argument("--azure-endpoint", type=str, default="")
    parser.add_argument("--azure-api-key", type=str, default="")
    parser.add_argument("--azure-api-version", type=str, default="")
    parser.add_argument("--azure-deployment", type=str, default="")
    parser.add_argument("--azure-subscription-id", type=str, default="")
    parser.add_argument("--azure-location", type=str, default="")
    parser.add_argument("--azure-arm-bearer-token", type=str, default="")

    # OpenRouter overrides
    parser.add_argument("--openrouter-endpoint", type=str, default="")
    parser.add_argument("--openrouter-api-key", type=str, default="")
    parser.add_argument("--openrouter-model", type=str, default="")

    # Safety / threshold controls
    parser.add_argument(
        "--early-abort-error-rate",
        type=float,
        default=0.20,
        help="Abort current step when observed error rate reaches this value.",
    )
    parser.add_argument(
        "--early-abort-throttle-rate",
        type=float,
        default=0.50,
        help="Abort current step when observed 429 rate reaches this value.",
    )
    parser.add_argument(
        "--early-abort-min-requests",
        type=int,
        default=20,
        help="Minimum requests observed before early-abort thresholds apply.",
    )
    parser.add_argument(
        "--stable-error-rate-threshold",
        type=float,
        default=0.01,
        help="Threshold used when selecting max sustainable point in final report.",
    )
    parser.add_argument(
        "--stable-throttle-rate-threshold",
        type=float,
        default=0.01,
        help="Threshold used when selecting max sustainable point in final report.",
    )

    return parser.parse_args()


async def amain(args: argparse.Namespace) -> int:
    random.seed(int(args.seed))
    batch_sizes = parse_int_csv(args.batch_sizes, arg_name="--batch-sizes")
    concurrency_levels = parse_int_csv(args.concurrency_levels, arg_name="--concurrency-levels")
    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    platform_cfgs = _resolve_platform_configs(args)
    requested_platforms = [p.strip().lower() for p in str(args.platforms).split(",") if p.strip()]
    selected_platforms = [p for p in requested_platforms if p in platform_cfgs]
    missing = [p for p in requested_platforms if p not in platform_cfgs]

    if not selected_platforms:
        print("[FATAL] No configured platforms resolved from --platforms and environment.", file=sys.stderr)
        if missing:
            print(f"Missing configs for: {', '.join(missing)}", file=sys.stderr)
        return 1

    if missing:
        print(f"[WARN] Skipping unconfigured platforms: {', '.join(missing)}", flush=True)

    step_plan = _build_step_plan(
        platforms=selected_platforms,
        batch_sizes=batch_sizes,
        concurrency_levels=concurrency_levels,
        duration_seconds=int(args.duration_seconds),
    )
    corpus = _build_default_corpus()

    print("[PLAN] Embedding limit probe", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    print(f"  platforms={selected_platforms}", flush=True)
    print(f"  batch_sizes={batch_sizes}", flush=True)
    print(f"  concurrency_levels={concurrency_levels}", flush=True)
    print(f"  duration_seconds={int(args.duration_seconds)}", flush=True)
    print(f"  cooldown_seconds={int(args.cooldown_seconds)}", flush=True)
    print(f"  max_total_minutes={float(args.max_total_minutes):.2f}", flush=True)
    print(f"  steps={len(step_plan)}", flush=True)
    if args.dry_run:
        return 0

    max_total_deadline = time.monotonic() + float(args.max_total_minutes) * 60.0
    timeout = httpx.Timeout(float(args.request_timeout_seconds))
    raw_events: List[RequestEvent] = []
    step_summaries: List[StepSummary] = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        for idx, step in enumerate(step_plan):
            if time.monotonic() >= max_total_deadline:
                print("[STOP] Global time budget exhausted before next step.", flush=True)
                break

            cfg = platform_cfgs[step.platform]
            print(
                f"[STEP {step.step_index}/{len(step_plan)}] platform={step.platform} "
                f"batch={step.batch_size} conc={step.concurrency} dur={step.duration_seconds}s",
                flush=True,
            )
            events, stop_reason = await run_step(
                client=client,
                step=step,
                platform_cfg=cfg,
                corpus=corpus,
                max_total_deadline=max_total_deadline,
                early_abort_error_rate=float(args.early_abort_error_rate),
                early_abort_throttle_rate=float(args.early_abort_throttle_rate),
                early_abort_min_requests=int(args.early_abort_min_requests),
            )
            raw_events.extend(events)
            summary = summarize_step(events, step, stop_reason)
            step_summaries.append(summary)
            print(
                f"  attempted={summary.attempted_requests} ok={summary.success_requests} "
                f"429={summary.throttled_requests} err_rate={summary.error_rate:.3f} "
                f"throttle_rate={summary.throttle_rate:.3f} "
                f"success_rpm={summary.success_rpm:.2f} est_tpm={summary.est_tpm_success:.2f} "
                f"stop={summary.stop_reason}",
                flush=True,
            )

            if idx < len(step_plan) - 1 and time.monotonic() < max_total_deadline:
                await asyncio.sleep(max(0, int(args.cooldown_seconds)))

    quota_snapshot: Dict[str, Any] = {}
    if "azure" in selected_platforms:
        quota_snapshot["azure"] = await fetch_azure_quota_usage(args)
    if "openrouter" in selected_platforms:
        openrouter_key = str(
            args.openrouter_api_key
            or os.getenv("OPENROUTER_API_KEY", "")
            or platform_cfgs.get("openrouter", PlatformConfig("", "", {}, None, {})).headers.get("Authorization", "").replace("Bearer ", "")
        ).strip()
        if openrouter_key:
            quota_snapshot["openrouter"] = await fetch_openrouter_key_info(openrouter_key)
        else:
            quota_snapshot["openrouter"] = {"ok": False, "reason": "missing OPENROUTER_API_KEY"}

    raw_events_path = output_dir / "raw_events.jsonl"
    summary_path = output_dir / "step_summary.csv"
    report_path = output_dir / "final_report.md"
    quota_path = output_dir / "quota_snapshot.json"

    _write_jsonl(raw_events_path, (asdict(ev) for ev in raw_events))
    _write_csv(summary_path, _to_csv_rows(step_summaries))
    _write_json(quota_path, quota_snapshot)

    report = build_report_markdown(
        step_summaries=step_summaries,
        quota_snapshot=quota_snapshot,
        stable_error_rate_threshold=float(args.stable_error_rate_threshold),
        stable_throttle_rate_threshold=float(args.stable_throttle_rate_threshold),
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print("[OK] Probe completed", flush=True)
    print(f"  raw_events={raw_events_path}", flush=True)
    print(f"  step_summary={summary_path}", flush=True)
    print(f"  quota_snapshot={quota_path}", flush=True)
    print(f"  final_report={report_path}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.", flush=True)
        return 130
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
"""Enumerate and validate OpenRouter chat models with logprobs support.

This script performs two stages:
1) Catalog enumeration from ``/models`` (no inference),
2) Empirical validation via one minimal chat completion per model.

Outputs:
- markdown report: ``scratch/openrouter_logprobs_report.md``
- raw catalog payload: ``scratch/openrouter_models_catalog_raw.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    import study_query_llm.config  # noqa: F401  # load .env via side effect
except Exception:
    pass

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_REPORT_PATH = "scratch/openrouter_logprobs_report.md"
DEFAULT_CATALOG_PATH = "scratch/openrouter_models_catalog_raw.json"
DEFAULT_PROMPT = "Answer with a single letter: A, B, C, or D. Q: 2+2=? A:"


@dataclass(frozen=True)
class CatalogModel:
    model_id: str
    name: str
    family: str
    context_length: Optional[int]
    prompt_price: Optional[Decimal]
    completion_price: Optional[Decimal]
    supported_parameters: tuple[str, ...]


@dataclass
class ProbeResult:
    model: CatalogModel
    classification: str
    response_excerpt: str
    observed_top_logprobs_cap: Optional[int]
    status_code: Optional[int]
    attempts: int


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_abs_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _normalize_supported_parameters(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return tuple()
    values: list[str] = []
    for item in raw:
        value = str(item or "").strip().lower()
        if not value:
            continue
        values.append(value)
    return tuple(values)


def _parse_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def _parse_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _family_from_model_id(model_id: str) -> str:
    cleaned = str(model_id or "").strip()
    if not cleaned:
        return "unknown"
    if "/" not in cleaned:
        return "unknown"
    prefix = cleaned.split("/", 1)[0].strip().lower()
    return prefix or "unknown"


def _safe_excerpt(value: str, max_chars: int = 240) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_error_text(payload: Any, fallback_text: str = "") -> str:
    if isinstance(payload, dict):
        error_obj = payload.get("error")
        if isinstance(error_obj, dict):
            message = str(error_obj.get("message") or "").strip()
            if message:
                return message
        message = str(payload.get("message") or "").strip()
        if message:
            return message
    return str(fallback_text or "").strip()


def _format_price(value: Optional[Decimal]) -> str:
    if value is None:
        return "n/a"
    try:
        return format(value, "f")
    except Exception:
        return str(value)


def _required_logprob_support(params: Iterable[str]) -> bool:
    values = {str(item or "").strip().lower() for item in params}
    return "logprobs" in values and "top_logprobs" in values


def _is_unsupported_logprob_error(message: str) -> bool:
    lowered = str(message or "").lower()
    if "logprob" not in lowered:
        return False
    unsupported_terms = (
        "unsupported",
        "not support",
        "not allowed",
        "invalid",
        "unknown parameter",
        "unrecognized",
    )
    return any(term in lowered for term in unsupported_terms)


def _estimate_probe_cost_usd(
    model: CatalogModel,
    estimated_prompt_tokens: int,
    max_completion_tokens: int,
) -> Decimal:
    prompt_price = model.prompt_price if model.prompt_price is not None else Decimal("0")
    completion_price = model.completion_price if model.completion_price is not None else Decimal("0")
    if prompt_price < Decimal("0"):
        prompt_price = Decimal("0")
    if completion_price < Decimal("0"):
        completion_price = Decimal("0")
    prompt_cost = prompt_price * Decimal(max(0, int(estimated_prompt_tokens)))
    completion_cost = completion_price * Decimal(max(0, int(max_completion_tokens)))
    return prompt_cost + completion_cost


async def _fetch_catalog(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    auth_headers: dict[str, str],
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models"
    response = await client.get(url, headers=auth_headers)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenRouter /models response is not a JSON object.")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("OpenRouter /models response missing 'data' list.")
    return payload


def _parse_catalog_models(catalog_payload: dict[str, Any]) -> list[CatalogModel]:
    data = catalog_payload.get("data")
    if not isinstance(data, list):
        return []
    seen_ids: set[str] = set()
    out: list[CatalogModel] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "").strip()
        if not model_id or model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        supported_parameters = _normalize_supported_parameters(
            row.get("supported_parameters")
        )
        if not _required_logprob_support(supported_parameters):
            continue
        pricing = row.get("pricing")
        pricing_obj = pricing if isinstance(pricing, dict) else {}
        out.append(
            CatalogModel(
                model_id=model_id,
                name=str(row.get("name") or "").strip(),
                family=_family_from_model_id(model_id),
                context_length=_parse_positive_int(row.get("context_length")),
                prompt_price=_parse_decimal(pricing_obj.get("prompt")),
                completion_price=_parse_decimal(pricing_obj.get("completion")),
                supported_parameters=tuple(supported_parameters),
            )
        )
    return out


def _classify_success_payload(payload: Any) -> tuple[str, str, Optional[int]]:
    if not isinstance(payload, dict):
        return ("EMPTY", "Successful response with non-object JSON payload.", None)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ("EMPTY", "Successful response with missing choices.", None)
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return ("EMPTY", "Successful response with malformed first choice.", None)
    logprobs_obj = choice0.get("logprobs")
    if not isinstance(logprobs_obj, dict):
        return ("EMPTY", "Successful response without logprobs object.", None)
    content = logprobs_obj.get("content")
    if not isinstance(content, list) or not content:
        return ("EMPTY", "Successful response without logprobs.content entries.", None)
    first = content[0]
    if not isinstance(first, dict):
        return ("EMPTY", "Successful response with malformed logprobs.content[0].", None)
    top_logprobs = first.get("top_logprobs")
    if not isinstance(top_logprobs, list) or not top_logprobs:
        return ("EMPTY", "Successful response without top_logprobs entries.", None)
    valid_entries = [
        item
        for item in top_logprobs
        if isinstance(item, dict)
        and "token" in item
        and "logprob" in item
    ]
    if not valid_entries:
        return (
            "EMPTY",
            "Successful response had top_logprobs, but entries lacked token/logprob.",
            len(top_logprobs),
        )
    return ("PASS", "logprobs payload present and well-formed.", len(top_logprobs))


async def _probe_one_model(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    auth_headers: dict[str, str],
    model: CatalogModel,
    request_timeout_seconds: float,
    prompt: str,
    max_tokens: int,
) -> ProbeResult:
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    timeout = httpx.Timeout(request_timeout_seconds)
    final_classification = "OTHER_ERROR"
    final_excerpt = ""
    final_status: Optional[int] = None
    final_cap: Optional[int] = None
    attempts = 0

    for attempt in (1, 2):
        attempts = attempt
        try:
            response = await client.post(
                url,
                headers=auth_headers,
                json=body,
                timeout=timeout,
            )
        except Exception as exc:
            final_classification = "OTHER_ERROR"
            final_excerpt = _safe_excerpt(f"{type(exc).__name__}: {exc}")
            final_status = None
            final_cap = None
            if attempt == 1:
                await asyncio.sleep(1.0)
                continue
            break

        payload: Any
        try:
            payload = response.json()
        except Exception:
            payload = None
        status = int(response.status_code)
        message = _extract_error_text(payload, fallback_text=response.text)
        final_status = status

        if 200 <= status < 300:
            classification, excerpt, cap = _classify_success_payload(payload)
            final_classification = classification
            final_excerpt = _safe_excerpt(excerpt)
            final_cap = cap
            break

        if 400 <= status < 500 and _is_unsupported_logprob_error(message):
            final_classification = "REJECTED"
            final_excerpt = _safe_excerpt(message or "4xx unsupported logprob parameter.")
            final_cap = None
            break

        final_classification = "OTHER_ERROR"
        final_excerpt = _safe_excerpt(message or f"HTTP {status}")
        final_cap = None
        if attempt == 1:
            await asyncio.sleep(1.0)
            continue
        break

    return ProbeResult(
        model=model,
        classification=final_classification,
        response_excerpt=final_excerpt,
        observed_top_logprobs_cap=final_cap,
        status_code=final_status,
        attempts=attempts,
    )


def _build_family_summary(
    pass_results: list[ProbeResult],
    non_pass_results: list[ProbeResult],
    skipped_due_cost: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}

    def bump(family: str, key: str) -> None:
        row = summary.setdefault(
            family,
            {
                "PASS": 0,
                "EMPTY": 0,
                "REJECTED": 0,
                "OTHER_ERROR": 0,
                "SKIPPED_DUE_COST": 0,
            },
        )
        row[key] = row.get(key, 0) + 1

    for item in pass_results:
        bump(item.model.family, "PASS")
    for item in non_pass_results:
        bump(item.model.family, item.classification)
    for item in skipped_due_cost:
        bump(str(item.get("family") or "unknown"), "SKIPPED_DUE_COST")
    return summary


def _render_report(
    *,
    pass_results: list[ProbeResult],
    non_pass_results: list[ProbeResult],
    skipped_due_cost: list[dict[str, Any]],
    family_summary: dict[str, dict[str, int]],
    started_at: str,
    finished_at: str,
) -> str:
    lines: list[str] = []
    lines.append("# OpenRouter Logprobs Validation Report")
    lines.append("")
    lines.append(f"- Run started: `{started_at}`")
    lines.append(f"- Run finished: `{finished_at}`")
    lines.append("")

    lines.append("## Section 1: PASS")
    lines.append("")
    if not pass_results:
        lines.append("_No PASS models._")
    else:
        lines.append(
            "| Model | Family | Context Length | Prompt Cost | Completion Cost | Observed top_logprobs cap |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for result in pass_results:
            lines.append(
                "| "
                f"{result.model.model_id} | "
                f"{result.model.family} | "
                f"{result.model.context_length if result.model.context_length is not None else 'n/a'} | "
                f"{_format_price(result.model.prompt_price)} | "
                f"{_format_price(result.model.completion_price)} | "
                f"{result.observed_top_logprobs_cap if result.observed_top_logprobs_cap is not None else 'n/a'} |"
            )
    lines.append("")

    lines.append("## Section 2: EMPTY/REJECTED")
    lines.append("")
    if not non_pass_results:
        lines.append("_No EMPTY/REJECTED/OTHER_ERROR results._")
    else:
        lines.append("| Model | Classification | Response Excerpt |")
        lines.append("|---|---|---|")
        for result in non_pass_results:
            lines.append(
                "| "
                f"{result.model.model_id} | "
                f"{result.classification} | "
                f"{result.response_excerpt.replace('|', '\\|')} |"
            )
    lines.append("")

    lines.append("## Section 3: SKIPPED-DUE-TO-COST")
    lines.append("")
    if not skipped_due_cost:
        lines.append("_No models skipped due to cost guard._")
    else:
        lines.append("| Model | Family | Prompt Cost | Completion Cost | Reason |")
        lines.append("|---|---|---:|---:|---|")
        for item in skipped_due_cost:
            lines.append(
                "| "
                f"{item['model_id']} | "
                f"{item['family']} | "
                f"{item['prompt_price']} | "
                f"{item['completion_price']} | "
                f"{item['reason']} |"
            )
    lines.append("")

    lines.append("## Section 4: Catalog Summary Stats")
    lines.append("")
    if not family_summary:
        lines.append("_No family summary data available._")
    else:
        lines.append(
            "| Family | PASS | EMPTY | REJECTED | OTHER_ERROR | SKIPPED-DUE-TO-COST |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for family in sorted(family_summary.keys()):
            row = family_summary[family]
            lines.append(
                "| "
                f"{family} | "
                f"{row.get('PASS', 0)} | "
                f"{row.get('EMPTY', 0)} | "
                f"{row.get('REJECTED', 0)} | "
                f"{row.get('OTHER_ERROR', 0)} | "
                f"{row.get('SKIPPED_DUE_COST', 0)} |"
            )
    lines.append("")

    lines.append(
        f"Validation executed at `{finished_at}` UTC; OpenRouter catalog and behavior may change over time."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate and empirically validate OpenRouter logprobs support."
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report output path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default=DEFAULT_CATALOG_PATH,
        help="Raw /models JSON output path (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--http-referer",
        type=str,
        default="https://github.com/spenc/study-query-llm",
        help="HTTP-Referer header required by OpenRouter.",
    )
    parser.add_argument(
        "--x-title",
        type=str,
        default="study-query-llm openrouter logprobs probe",
        help="X-Title header required by OpenRouter.",
    )
    parser.add_argument(
        "--cost-threshold-prompt",
        type=float,
        default=0.00001,
        help="Skip models with prompt cost above this USD/token threshold.",
    )
    parser.add_argument(
        "--include-expensive",
        action="store_true",
        help="Include models above --cost-threshold-prompt.",
    )
    parser.add_argument(
        "--max-total-usd",
        type=float,
        default=1.0,
        help="Estimated total spend guard across all probes.",
    )
    parser.add_argument(
        "--max-probe-models",
        type=int,
        default=200,
        help="Model-count guard for empirical probes.",
    )
    parser.add_argument(
        "--estimated-prompt-tokens",
        type=int,
        default=64,
        help="Prompt-token estimate used for spend guard.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=45.0,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run Stage 1 only, print selected model counts, and exit.",
    )
    return parser.parse_args()


async def amain(args: argparse.Namespace) -> int:
    api_key = str(os.getenv("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        print("[FATAL] OPENROUTER_API_KEY is not set.", file=sys.stderr)
        return 1

    report_path = _to_abs_path(args.report_path)
    catalog_path = _to_abs_path(args.catalog_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": str(args.http_referer),
        "X-Title": str(args.x_title),
        "Content-Type": "application/json",
    }

    started_at = _now_utc_iso()
    async with httpx.AsyncClient() as client:
        catalog_payload = await _fetch_catalog(
            client,
            base_url=str(args.base_url),
            auth_headers=auth_headers,
        )

    with open(catalog_path, "w", encoding="utf-8") as handle:
        json.dump(catalog_payload, handle, indent=2, ensure_ascii=False)

    catalog_models = _parse_catalog_models(catalog_payload)
    prompt_threshold = Decimal(str(args.cost_threshold_prompt))
    max_total_usd = Decimal(str(args.max_total_usd))

    skipped_due_cost: list[dict[str, Any]] = []
    eligible_models: list[CatalogModel] = []
    for model in catalog_models:
        if model.prompt_price is None:
            skipped_due_cost.append(
                {
                    "model_id": model.model_id,
                    "family": model.family,
                    "prompt_price": _format_price(model.prompt_price),
                    "completion_price": _format_price(model.completion_price),
                    "reason": "missing_or_non_numeric_prompt_price",
                }
            )
            continue
        if not args.include_expensive and model.prompt_price > prompt_threshold:
            skipped_due_cost.append(
                {
                    "model_id": model.model_id,
                    "family": model.family,
                    "prompt_price": _format_price(model.prompt_price),
                    "completion_price": _format_price(model.completion_price),
                    "reason": "prompt_price_above_threshold",
                }
            )
            continue
        eligible_models.append(model)

    print(
        "[INFO] Stage 1 complete: "
        f"catalog_rows={len(catalog_payload.get('data', []))} "
        f"logprob_declared={len(catalog_models)} "
        f"eligible={len(eligible_models)} "
        f"skipped_due_cost={len(skipped_due_cost)}",
        flush=True,
    )

    if args.dry_run:
        print("[INFO] Dry run enabled; Stage 2 not executed.", flush=True)
        return 0

    pass_results: list[ProbeResult] = []
    non_pass_results: list[ProbeResult] = []
    estimated_total_spend = Decimal("0")
    max_probe_models = max(0, int(args.max_probe_models))
    max_tokens = max(1, int(args.max_tokens))

    async with httpx.AsyncClient() as client:
        for idx, model in enumerate(eligible_models):
            if idx >= max_probe_models:
                for rem in eligible_models[idx:]:
                    skipped_due_cost.append(
                        {
                            "model_id": rem.model_id,
                            "family": rem.family,
                            "prompt_price": _format_price(rem.prompt_price),
                            "completion_price": _format_price(rem.completion_price),
                            "reason": "model_count_guard",
                        }
                    )
                    print(f"[SKIPPED-DUE-COST] {rem.model_id}", flush=True)
                break

            projected = _estimate_probe_cost_usd(
                model,
                estimated_prompt_tokens=int(args.estimated_prompt_tokens),
                max_completion_tokens=max_tokens,
            )
            if estimated_total_spend + projected > max_total_usd:
                for rem in eligible_models[idx:]:
                    skipped_due_cost.append(
                        {
                            "model_id": rem.model_id,
                            "family": rem.family,
                            "prompt_price": _format_price(rem.prompt_price),
                            "completion_price": _format_price(rem.completion_price),
                            "reason": "budget_guard",
                        }
                    )
                    print(f"[SKIPPED-DUE-COST] {rem.model_id}", flush=True)
                break

            result = await _probe_one_model(
                client,
                base_url=str(args.base_url),
                auth_headers=auth_headers,
                model=model,
                request_timeout_seconds=float(args.request_timeout_seconds),
                prompt=str(args.prompt),
                max_tokens=max_tokens,
            )
            estimated_total_spend += projected
            print(f"[{result.classification}] {model.model_id}", flush=True)
            if result.classification == "PASS":
                pass_results.append(result)
            else:
                non_pass_results.append(result)

    family_summary = _build_family_summary(
        pass_results=pass_results,
        non_pass_results=non_pass_results,
        skipped_due_cost=skipped_due_cost,
    )
    finished_at = _now_utc_iso()
    report_text = _render_report(
        pass_results=pass_results,
        non_pass_results=non_pass_results,
        skipped_due_cost=skipped_due_cost,
        family_summary=family_summary,
        started_at=started_at,
        finished_at=finished_at,
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print("[OK] Probe complete", flush=True)
    print(f"[OK] Report: {report_path}", flush=True)
    print(f"[OK] Raw catalog: {catalog_path}", flush=True)
    print(
        "[INFO] Totals: "
        f"PASS={len(pass_results)} "
        f"NON_PASS={len(non_pass_results)} "
        f"SKIPPED_DUE_COST={len(skipped_due_cost)} "
        f"EST_SPEND_USD={format(estimated_total_spend, 'f')}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user.", flush=True)
        return 130
    except httpx.HTTPStatusError as exc:
        body = _safe_excerpt(exc.response.text if exc.response is not None else str(exc))
        print(
            f"[FATAL] HTTPStatusError status={exc.response.status_code if exc.response is not None else 'n/a'} "
            f"message={body}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


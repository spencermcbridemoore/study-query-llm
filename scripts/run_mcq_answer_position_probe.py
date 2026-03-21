#!/usr/bin/env python3
"""Run an MCQ answer-position probe for one model/deployment.

This script repeatedly asks an LLM to create a multiple-choice test with an
answer key, parses the key, and reports answer-position distribution stats.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from study_query_llm.config import Config
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services._shared import deployment_override
from study_query_llm.services.inference_service import InferenceService


ANSWER_KEY_HEADING_RE = re.compile(r"(?i)answer\s*key")


def _answer_line_re(labels: List[str]) -> re.Pattern:
    """Build regex to match answer lines; label class from first to last letter (e.g. A-D, A-E)."""
    if not labels:
        return re.compile(r"(?im)^\s*(?:question\s*)?(\d{1,3})\s*[:\)\.-]\s*([A-E])\b")
    first, last = labels[0].upper(), labels[-1].upper()
    return re.compile(
        rf"(?im)^\s*(?:question\s*)?(\d{{1,3}})\s*[:\)\.-]\s*([{first}-{last}])\b"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", value).strip("_") or "value"


def build_prompt(subject: str, question_count: int, labels: List[str]) -> str:
    labels_inline = ",".join(labels)
    num_options = len(labels)
    label_range = f"{labels[0]}-{labels[-1]}" if labels else "A-E"
    options_block = "\n".join(f"{l}) <option>" for l in labels)
    return (
        f"Create a {subject} multiple-choice test with exactly {question_count} questions.\n\n"
        f"Requirements:\n"
        f"- Each question must have exactly {num_options} options labeled [{labels_inline}].\n"
        f"- Keep stems and options concise.\n"
        f"- Include an answer key at the end.\n"
        f"- Use this exact answer-key line format: `1: {labels[0]}`, `2: {labels[1]}`, ...\n\n"
        f"Output format:\n"
        f"Q1. <question>\n"
        f"{options_block}\n\n"
        f"... continue until Q{question_count} ...\n\n"
        f"Answer Key:\n"
        f"1: <{label_range}>\n"
        f"2: <{label_range}>\n"
        f"...\n"
        f"{question_count}: <{label_range}>\n"
    )


def extract_answer_key(
    text: str,
    question_count: int,
    valid_labels: set[str],
    labels_list: Optional[List[str]] = None,
) -> Tuple[Optional[List[str]], bool, Optional[str]]:
    """Parse answer-key labels for questions 1..question_count."""
    heading_match = ANSWER_KEY_HEADING_RE.search(text)
    heading_present = heading_match is not None
    labels_for_re = sorted(valid_labels) if labels_list is None else labels_list
    answer_line_re = _answer_line_re(labels_for_re)

    by_question: Dict[int, str] = {}
    for match in answer_line_re.finditer(text):
        q_num = int(match.group(1))
        label = match.group(2).upper()
        if 1 <= q_num <= question_count and label in valid_labels:
            by_question[q_num] = label

    if all(i in by_question for i in range(1, question_count + 1)):
        return [by_question[i] for i in range(1, question_count + 1)], heading_present, None

    # Fallback: look for raw label stream after "Answer Key" heading.
    if heading_match:
        tail = text[heading_match.end() :]
        first, last = labels_for_re[0], labels_for_re[-1]
        labels = [m.upper() for m in re.findall(rf"\b([{first}-{last}])\b", tail)]
        if len(labels) >= question_count and all(l in valid_labels for l in labels[:question_count]):
            return labels[:question_count], heading_present, "fallback_label_stream"

    return None, heading_present, "missing_or_invalid_answer_key"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean_val = float(sum(values) / len(values))
    if len(values) < 2:
        return mean_val, 0.0
    return mean_val, float(statistics.stdev(values))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    p = min(max(float(p), 0.0), 1.0)
    rank = (len(sorted_vals) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_vals[low])
    weight = rank - low
    return float(sorted_vals[low] * (1.0 - weight) + sorted_vals[high] * weight)


async def run_probe(
    deployment: str,
    subject: str,
    question_count: int,
    labels: List[str],
    samples: int,
    concurrency: int,
    temperature: float,
    max_tokens: Optional[int],
    progress_every: int,
) -> dict:
    valid_labels = {label.upper() for label in labels}
    labels_upper = [label.upper() for label in labels]
    prompt = build_prompt(subject=subject, question_count=question_count, labels=labels_upper)

    call_errors: List[dict] = []
    parse_failures: List[dict] = []
    fallback_parse_count = 0
    heading_present_count = 0
    valid_runs = 0
    successful_call_latencies_ms: List[float] = []

    pooled_counts: Counter = Counter()
    per_sample_props: Dict[str, List[float]] = {label: [] for label in labels_upper}
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    started_at = time.perf_counter()

    with deployment_override("AZURE_OPENAI_DEPLOYMENT", deployment):
        fresh_config = Config()
        if "azure" in fresh_config._provider_configs:
            del fresh_config._provider_configs["azure"]
        provider = ProviderFactory(fresh_config).create_chat_provider("azure", deployment)
        service = InferenceService(
            provider=provider,
            repository=None,
            require_db_persistence=False,
            max_retries=4,
            initial_wait=1.0,
            max_wait=8.0,
        )

        async def _run_one_sample(sample_idx: int) -> dict:
            async with semaphore:
                try:
                    result = await service.run_inference(
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    return {
                        "sample_idx": sample_idx,
                        "error": str(exc)[:500],
                        "response_text": "",
                        "latency_ms": None,
                    }

                metadata = result.get("metadata") if isinstance(result, dict) else {}
                latency_ms = None
                if isinstance(metadata, dict):
                    raw_latency = metadata.get("latency_ms")
                    if isinstance(raw_latency, (int, float)):
                        latency_ms = float(raw_latency)

                return {
                    "sample_idx": sample_idx,
                    "error": None,
                    "response_text": str(result.get("response") or ""),
                    "latency_ms": latency_ms,
                }

        tasks: List[asyncio.Task] = []
        try:
            tasks = [
                asyncio.create_task(_run_one_sample(sample_idx))
                for sample_idx in range(1, samples + 1)
            ]
            completed = 0
            for future in asyncio.as_completed(tasks):
                row = await future
                completed += 1
                sample_idx = int(row["sample_idx"])
                error = row["error"]
                response_text = str(row["response_text"] or "")
                latency_ms = row["latency_ms"]

                if error is not None:
                    call_errors.append(
                        {
                            "sample_idx": sample_idx,
                            "error": error,
                        }
                    )
                else:
                    if isinstance(latency_ms, (int, float)):
                        successful_call_latencies_ms.append(float(latency_ms))
                    answers, heading_present, parse_reason = extract_answer_key(
                        response_text,
                        question_count=question_count,
                        valid_labels=valid_labels,
                        labels_list=labels_upper,
                    )
                    if heading_present:
                        heading_present_count += 1

                    if answers is None:
                        parse_failures.append(
                            {
                                "sample_idx": sample_idx,
                                "reason": parse_reason,
                                "preview": response_text[:400],
                            }
                        )
                    else:
                        if parse_reason == "fallback_label_stream":
                            fallback_parse_count += 1

                        run_counts = Counter(answers)
                        pooled_counts.update(run_counts)
                        valid_runs += 1
                        for label in labels_upper:
                            per_sample_props[label].append(run_counts.get(label, 0) / float(question_count))

                if progress_every > 0 and completed % progress_every == 0:
                    print(
                        f"[progress] completed={completed}/{samples} "
                        f"valid={valid_runs} call_errors={len(call_errors)} parse_failures={len(parse_failures)}"
                    )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await service.close()

    total_answers = valid_runs * question_count
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    pooled_distribution = {
        label: {
            "count": int(pooled_counts.get(label, 0)),
            "pct": (
                (float(pooled_counts.get(label, 0)) / float(total_answers))
                if total_answers > 0
                else math.nan
            ),
        }
        for label in labels_upper
    }

    per_sample_stats = {}
    for label in labels_upper:
        mean_prop, std_prop = _mean_std(per_sample_props[label])
        per_sample_stats[label] = {
            "mean_prop": mean_prop,
            "std_prop": std_prop,
            "mean_pct": mean_prop * 100.0 if not math.isnan(mean_prop) else math.nan,
            "std_pct": std_prop * 100.0 if not math.isnan(std_prop) else math.nan,
        }

    chi_square = math.nan
    if total_answers > 0:
        expected = float(total_answers) / float(len(labels_upper))
        chi_square = float(
            sum(((pooled_counts.get(label, 0) - expected) ** 2) / expected for label in labels_upper)
        )

    summary = {
        "deployment": deployment,
        "subject": subject,
        "question_count": int(question_count),
        "labels": labels_upper,
        "samples_requested": int(samples),
        "samples_attempted": int(samples),
        "samples_with_successful_call": int(samples - len(call_errors)),
        "samples_with_valid_answer_key": int(valid_runs),
        "concurrency_requested": int(max(1, int(concurrency))),
        "elapsed_seconds": elapsed_seconds,
        "throughput_samples_per_second": (
            float(samples) / elapsed_seconds if elapsed_seconds > 0.0 else math.nan
        ),
        "call_error_count": int(len(call_errors)),
        "parse_failure_count": int(len(parse_failures)),
        "heading_present_count": int(heading_present_count),
        "fallback_parse_count": int(fallback_parse_count),
        "answer_count_total": int(total_answers),
        "successful_call_latency_ms_mean": (
            float(statistics.mean(successful_call_latencies_ms))
            if successful_call_latencies_ms
            else math.nan
        ),
        "successful_call_latency_ms_p95": _percentile(successful_call_latencies_ms, 0.95),
        "pooled_distribution": pooled_distribution,
        "per_sample_distribution_stats": per_sample_stats,
        "chi_square_vs_uniform": chi_square,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    details = {
        "summary": summary,
        "call_errors": call_errors,
        "parse_failures": parse_failures,
    }
    return details


def print_summary(summary: dict) -> None:
    labels = summary["labels"]
    print("\n=== MCQ Answer Position Probe ===")
    print(f"deployment: {summary['deployment']}")
    print(f"subject: {summary['subject']}")
    print(f"question_count: {summary['question_count']}")
    print(
        "samples: requested={requested}, successful_calls={ok_calls}, valid_keys={valid}".format(
            requested=summary["samples_requested"],
            ok_calls=summary["samples_with_successful_call"],
            valid=summary["samples_with_valid_answer_key"],
        )
    )
    print(
        "runtime: concurrency={concurrency}, elapsed={elapsed:.2f}s, throughput={throughput:.2f} samples/s".format(
            concurrency=summary.get("concurrency_requested", "?"),
            elapsed=summary.get("elapsed_seconds", math.nan),
            throughput=summary.get("throughput_samples_per_second", math.nan),
        )
    )
    print(
        "compliance: call_errors={call_errors}, parse_failures={parse_failures}, heading_present={heading_present}".format(
            call_errors=summary["call_error_count"],
            parse_failures=summary["parse_failure_count"],
            heading_present=summary["heading_present_count"],
        )
    )
    print(
        "latency: mean={mean:.2f}ms, p95={p95:.2f}ms".format(
            mean=summary.get("successful_call_latency_ms_mean", math.nan),
            p95=summary.get("successful_call_latency_ms_p95", math.nan),
        )
    )
    print(f"chi_square_vs_uniform: {summary['chi_square_vs_uniform']:.4f}")
    print("\nlabel stats:")
    for label in labels:
        pooled = summary["pooled_distribution"][label]
        per_sample = summary["per_sample_distribution_stats"][label]
        print(
            f"  {label}: pooled={pooled['count']} ({pooled['pct'] * 100.0:.2f}%), "
            f"mean={per_sample['mean_pct']:.2f}%, stdev={per_sample['std_pct']:.2f}%"
        )


async def _main_async(args: argparse.Namespace) -> int:
    labels = [label.strip().upper() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("labels must be non-empty (e.g. A,B,C,D or A,B,C,D,E)")

    details = await run_probe(
        deployment=args.deployment,
        subject=args.subject,
        question_count=args.question_count,
        labels=labels,
        samples=args.samples,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        progress_every=args.progress_every,
    )
    summary = details["summary"]
    print_summary(summary)

    out_dir = PROJECT_ROOT / "experimental_results" / "mcq_answer_position_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    num_choices = len(labels)
    out_name = (
        f"{_safe_name(args.deployment)}_{_safe_name(args.subject)}_"
        f"q{args.question_count}_c{num_choices}_n{args.samples}_{timestamp}.json"
    )
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results: {out_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCQ answer-position probe.")
    parser.add_argument("--deployment", type=str, required=True, help="Azure deployment name.")
    parser.add_argument("--subject", type=str, default="physics")
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument("--labels", type=str, default="A,B,C,D,E", help="Comma-separated labels (e.g. A,B,C,D or A,B,C,D,E)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

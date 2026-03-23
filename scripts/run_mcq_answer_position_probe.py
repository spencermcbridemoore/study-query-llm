#!/usr/bin/env python3
"""CLI wrapper for MCQ answer-position probe (library in study_query_llm.experiments)."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from study_query_llm.experiments.mcq_answer_position_probe import print_summary, run_probe


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", value).strip("_") or "value"


async def _main_async(args: argparse.Namespace) -> int:
    labels = [label.strip().upper() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("labels must be non-empty (e.g. A,B,C,D or A,B,C,D,E)")

    def _cb(completed, samples, valid_runs, call_errors, parse_failures):
        if args.progress_every <= 0:
            return
        print(
            f"[progress] completed={completed}/{samples} "
            f"valid={valid_runs} call_errors={call_errors} parse_failures={parse_failures}"
        )

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
        progress_callback=_cb if args.progress_every > 0 else None,
        level=args.level.strip() or None,
        spread_correct_answer_uniformly=args.spread_correct_answer_uniformly,
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
    parser.add_argument(
        "--level",
        type=str,
        default="",
        help='Optional difficulty (e.g. "high school"); included in the prompt.',
    )
    parser.add_argument("--subject", type=str, default="physics")
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument(
        "--labels",
        type=str,
        default="A,B,C,D,E",
        help="Comma-separated labels (e.g. A,B,C,D or A,B,C,D,E)",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--spread-correct-answer-uniformly",
        action="store_true",
        help="Ask the model to spread correct answers across option positions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

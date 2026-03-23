#!/usr/bin/env python3
"""Run an MCQ answer-position sweep across multiple parameter combinations and LLMs.

This script loads a sweep configuration, expands the parameter space, and runs
the MCQ probe for each (parameter_combo, LLM) pair.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from study_query_llm.experiments.mcq_answer_position_probe import run_probe
from study_query_llm.utils.mcq_template_loader import (
    load_config,
    load_sweep_config,
    expand_parameter_schema_filtered,
)
from study_query_llm.utils import mcq_template_loader


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", value).strip("_") or "value"


def _parse_subset(subset_str: str) -> tuple[int, int]:
    """Parse subset string like '0:5' into (start, end)."""
    if ":" not in subset_str:
        raise ValueError(f"subset must be in format 'start:end', got {subset_str!r}")
    parts = subset_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"subset must be in format 'start:end', got {subset_str!r}")
    return int(parts[0]), int(parts[1])


def _task_key(
    deployment: str,
    subject: str,
    question_count: int,
    labels: List[str],
    samples: int,
    *,
    level: str = "",
    spread: bool = False,
) -> Tuple[str, str, str, int, str, int, bool]:
    """Build a deterministic key for a sweep cell."""
    return (
        deployment.strip().lower(),
        (level or "").strip().lower(),
        subject.strip().lower(),
        int(question_count),
        ",".join(labels),
        int(samples),
        bool(spread),
    )


def _load_existing_completed_keys(
    out_dir: Path,
) -> Dict[Tuple[str, str, str, int, str, int, bool], Path]:
    """
    Return keys for cells already fully completed on disk.

    A result is considered complete when:
    - samples_with_successful_call == samples_requested
    """
    existing: Dict[Tuple[str, str, str, int, str, int, bool], Path] = {}
    if not out_dir.exists():
        return existing

    for json_path in out_dir.glob("*.json"):
        try:
            with open(json_path, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue

        deployment = summary.get("deployment")
        subject = summary.get("subject")
        level = summary.get("level")
        if not isinstance(level, str):
            level = ""
        spread_raw = summary.get("spread_correct_answer_uniformly", False)
        spread = bool(spread_raw) if isinstance(spread_raw, bool) else False
        question_count = summary.get("question_count")
        labels = summary.get("labels")
        samples_requested = summary.get("samples_requested")
        samples_successful = summary.get("samples_with_successful_call")

        if not isinstance(deployment, str) or not isinstance(subject, str):
            continue
        if not isinstance(question_count, int):
            continue
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            continue
        if not isinstance(samples_requested, int) or not isinstance(samples_successful, int):
            continue

        if samples_successful != samples_requested:
            continue

        key = _task_key(
            deployment=deployment,
            subject=subject,
            question_count=question_count,
            labels=labels,
            samples=samples_requested,
            level=level,
            spread=spread,
        )
        existing[key] = json_path

    return existing


async def run_sweep(
    sweep_config_path: Path,
    dry_run: bool = False,
    subset: Optional[str] = None,
    concurrency_override: Optional[int] = None,
    idempotent: bool = False,
) -> int:
    """
    Run the MCQ sweep as defined in the sweep config.
    
    Args:
        sweep_config_path: Path to sweep config JSON
        dry_run: If True, print all combos without executing
        subset: Optional subset string like "0:5" to run only first 5 combos
        concurrency_override: Optional override for concurrency (from sweep config)
        idempotent: Skip cells that are already complete on disk
    
    Returns:
        0 on success, 1 on error
    """
    # Load configs
    print(f"[INFO] Loading sweep config: {sweep_config_path}")
    sweep_config = load_sweep_config(sweep_config_path)
    template_config = load_config()
    
    sweep_name = sweep_config.get("name", "unknown")
    parameter_filter = sweep_config["parameter_filter"]
    llms = sweep_config["llms"]
    samples_per_combo = sweep_config["samples_per_combo"]
    concurrency = concurrency_override or sweep_config.get("concurrency", 20)
    temperature = sweep_config.get("temperature", 0.7)
    max_tokens = sweep_config.get("max_tokens", 2000)
    idempotent_enabled = idempotent or bool(sweep_config.get("idempotent", False))
    out_dir = PROJECT_ROOT / "experimental_results" / "mcq_answer_position_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Sweep: {sweep_name}")
    print(f"[INFO] LLMs: {len(llms)} ({', '.join(llms)})")
    print(f"[INFO] Samples per combo: {samples_per_combo}")
    print(f"[INFO] Concurrency: {concurrency}")
    print(f"[INFO] Idempotent mode: {idempotent_enabled}")
    
    # Expand parameter schema with filter
    print(f"[INFO] Expanding parameter schema with filter...")
    all_combos = list(
        expand_parameter_schema_filtered(
            template_config["parameter_schema"], parameter_filter
        )
    )
    print(f"[INFO] Parameter combinations: {len(all_combos)}")
    
    # Apply subset if specified
    if subset:
        start, end = _parse_subset(subset)
        all_combos = all_combos[start:end]
        print(f"[INFO] Subset applied: {start}:{end} -> {len(all_combos)} combos")
    
    # Generate all (params, deployment) pairs
    all_tasks = []
    for params in all_combos:
        for deployment in llms:
            all_tasks.append((params, deployment))
    
    total_tasks = len(all_tasks)
    print(f"[INFO] Total tasks: {total_tasks} (combos x LLMs)")
    existing_completed = (
        _load_existing_completed_keys(out_dir) if idempotent_enabled else {}
    )
    if idempotent_enabled:
        print(
            f"[INFO] Existing completed cells on disk: {len(existing_completed)} "
            f"(will skip matching cells)"
        )
    
    if dry_run:
        print("\n[DRY RUN] Would execute:")
        for i, (params, deployment) in enumerate(all_tasks):
            level = params["level"]
            subject = params["subject"]
            options = params["options_per_question"]
            questions = params["questions_per_test"]
            label_style = params["label_style"]
            spread = bool(params.get("spread_correct_answer_uniformly", False))
            subject_str = f"{level} {subject}"
            labels = mcq_template_loader._labels_for(options, label_style)
            key = _task_key(
                deployment=deployment,
                subject=subject,
                question_count=questions,
                labels=labels,
                samples=samples_per_combo,
                level=level if isinstance(level, str) else "",
                spread=spread,
            )
            dry_run_status = "SKIP" if idempotent_enabled and key in existing_completed else "RUN"
            print(
                f"  {i+1:4d}. [{dry_run_status}] {deployment:40s} | "
                f"{level:15s} {subject:12s} | q={questions:2d} "
                f"opts={options} samples={samples_per_combo}"
            )
        return 0
    
    # Execute tasks
    print(f"\n[INFO] Starting execution...")
    completed = 0
    skipped = 0
    errors = []
    
    for task_idx, (params, deployment) in enumerate(all_tasks, 1):
        level = params["level"]
        subject = params["subject"]
        options = params["options_per_question"]
        questions = params["questions_per_test"]
        label_style = params["label_style"]
        spread = bool(params.get("spread_correct_answer_uniformly", False))

        subject_str = f"{level} {subject}"
        labels = mcq_template_loader._labels_for(options, label_style)
        key = _task_key(
            deployment=deployment,
            subject=subject,
            question_count=questions,
            labels=labels,
            samples=samples_per_combo,
            level=level if isinstance(level, str) else "",
            spread=spread,
        )

        if idempotent_enabled and key in existing_completed:
            print(
                f"\n[{task_idx}/{total_tasks}] {deployment} | {subject_str} | "
                f"q={questions} opts={options} ({','.join(labels)})"
            )
            print(f"  [SKIP] Already completed in {existing_completed[key].name}")
            skipped += 1
            continue
        
        print(
            f"\n[{task_idx}/{total_tasks}] {deployment} | {subject_str} | "
            f"q={questions} opts={options} ({','.join(labels)})"
        )
        
        try:
            details = await run_probe(
                deployment=deployment,
                subject=subject,
                question_count=questions,
                labels=labels,
                samples=samples_per_combo,
                concurrency=concurrency,
                temperature=temperature,
                max_tokens=max_tokens,
                progress_every=10,
                level=level if isinstance(level, str) and level.strip() else None,
                spread_correct_answer_uniformly=spread,
            )
            
            # Save results
            summary = details["summary"]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            out_name = (
                f"{_safe_name(deployment)}_{_safe_name(level)}_{_safe_name(subject)}_"
                f"q{questions}_c{options}_n{samples_per_combo}_{timestamp}.json"
            )
            out_path = out_dir / out_name
            
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=2, ensure_ascii=False)
            existing_completed[key] = out_path
            
            valid = summary.get("samples_with_valid_answer_key", 0)
            errors_count = summary.get("call_error_count", 0)
            parse_failures = summary.get("parse_failure_count", 0)
            print(
                f"  [OK] Saved: {out_path.name} | "
                f"valid={valid}/{samples_per_combo} errors={errors_count} parse_failures={parse_failures}"
            )
            completed += 1
            
        except Exception as exc:
            error_msg = f"Task {task_idx} failed: {exc}"
            print(f"  [ERROR] {error_msg}")
            errors.append((task_idx, params, deployment, str(exc)))
    
    # Summary
    print(f"\n[SUMMARY]")
    print(f"  Completed: {completed}/{total_tasks}")
    if idempotent_enabled:
        print(f"  Skipped (idempotent): {skipped}")
    print(f"  Errors: {len(errors)}")
    
    if errors:
        print(f"\n[ERRORS]")
        for task_idx, params, deployment, error in errors:
            print(f"  {task_idx}. {deployment} | {params} | {error}")
        return 1
    
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCQ answer-position sweep across parameter combinations and LLMs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to sweep config JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print all tasks without executing",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Run subset of combos (e.g. '0:5' for first 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Override concurrency from config",
    )
    parser.add_argument(
        "--idempotent",
        action="store_true",
        help="Skip cells already completed on disk to avoid duplicate spend",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(
        run_sweep(
            sweep_config_path=args.config,
            dry_run=args.dry_run,
            subset=args.subset,
            concurrency_override=args.concurrency,
            idempotent=args.idempotent,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

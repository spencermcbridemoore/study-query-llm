#!/usr/bin/env python3
"""Export MCQ answer-position probe JSON summaries to CSV.

Reads files from experimental_results/mcq_answer_position_probe/ (or --input-dir).

Outputs:
  - Wide: one row per JSON (or per deduped combo), pooled A/B/... + mean/std across samples.
  - Long: one row per (row × option letter).
  - Pooled groups: consecutive samples of size ``--pooled-group-size`` (default 5);
    each row is % of correct answers A/B/C/D/E over ``group_size`` × question_count
    answers (e.g. 5 × 20 = 100). Requires ``per_sample_label_counts`` in summary
    (probe runs with enough samples).

Use ``--dedupe-combo`` when the directory contains multiple sweeps for the same
(deployment, level, subject, questions, options); keeps the newest file by mtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "experimental_results" / "mcq_answer_position_probe"
DEFAULT_WIDE = PROJECT_ROOT / "experimental_results" / "mcq_sweep_answer_label_stats.csv"
DEFAULT_LONG = PROJECT_ROOT / "experimental_results" / "mcq_sweep_answer_label_stats_long.csv"
DEFAULT_POOLED = (
    PROJECT_ROOT / "experimental_results" / "mcq_sweep_pooled_groups_of_5.csv"
)


def _combo_key(summary: Dict[str, Any]) -> Tuple[Any, ...]:
    labels = summary.get("labels")
    n = len(labels) if isinstance(labels, list) else 0
    return (
        str(summary.get("deployment") or ""),
        str(summary.get("level") or ""),
        str(summary.get("subject") or ""),
        int(summary.get("question_count") or 0),
        n,
    )


def _iter_summaries(
    input_dir: Path, *, dedupe_combo: bool
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    paths = list(input_dir.glob("*.json"))
    if dedupe_combo:
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        paths.sort(key=lambda p: p.name)
    seen: set[Tuple[Any, ...]] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        labels = summary.get("labels")
        if not isinstance(labels, list) or not labels:
            continue
        if dedupe_combo:
            k = _combo_key(summary)
            if k in seen:
                continue
            seen.add(k)
        yield path, summary


def _summary_to_wide_row(path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    labels = summary.get("labels")
    assert isinstance(labels, list)
    row: Dict[str, Any] = {
        "file": path.name,
        "deployment": summary.get("deployment"),
        "level": summary.get("level") or "",
        "subject": summary.get("subject"),
        "question_count": summary.get("question_count"),
        "num_options": len(labels),
        "spread_correct_answer_uniformly": summary.get(
            "spread_correct_answer_uniformly"
        ),
        "samples_requested": summary.get("samples_requested"),
        "samples_with_valid_answer_key": summary.get(
            "samples_with_valid_answer_key"
        ),
        "chi_square_vs_uniform": summary.get("chi_square_vs_uniform"),
    }
    pooled = summary.get("pooled_distribution") or {}
    pst = summary.get("per_sample_distribution_stats") or {}
    for letter in labels:
        if not isinstance(letter, str):
            continue
        pdist = pooled.get(letter) if isinstance(pooled, dict) else {}
        stats = pst.get(letter) if isinstance(pst, dict) else {}
        if not isinstance(pdist, dict):
            pdist = {}
        if not isinstance(stats, dict):
            stats = {}
        row[f"pooled_count_{letter}"] = pdist.get("count")
        row[f"pooled_pct_{letter}"] = pdist.get("pct")
        row[f"mean_pct_{letter}"] = stats.get("mean_pct")
        row[f"std_pct_{letter}"] = stats.get("std_pct")
        row[f"mean_prop_{letter}"] = stats.get("mean_prop")
        row[f"std_prop_{letter}"] = stats.get("std_prop")
    return row


def _load_wide_rows(
    input_dir: Path, *, dedupe_combo: bool
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path, summary in _iter_summaries(input_dir, dedupe_combo=dedupe_combo):
        rows.append(_summary_to_wide_row(path, summary))
    return rows


def _wide_to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long_rows: List[Dict[str, Any]] = []
    id_cols = [
        "file",
        "deployment",
        "level",
        "subject",
        "question_count",
        "num_options",
        "spread_correct_answer_uniformly",
    ]
    id_cols = [c for c in id_cols if c in wide.columns]
    for _, r in wide.iterrows():
        for col in wide.columns:
            if not col.startswith("pooled_pct_"):
                continue
            letter = col.removeprefix("pooled_pct_")
            if pd.isna(r[col]):
                continue
            base = {k: r[k] for k in id_cols}
            base["correct_option_label"] = letter
            base["pooled_pct"] = r.get(f"pooled_pct_{letter}")
            base["pooled_count"] = r.get(f"pooled_count_{letter}")
            base["mean_pct"] = r.get(f"mean_pct_{letter}")
            base["std_pct"] = r.get(f"std_pct_{letter}")
            base["mean_prop"] = r.get(f"mean_prop_{letter}")
            base["std_prop"] = r.get(f"std_prop_{letter}")
            long_rows.append(base)
    return pd.DataFrame(long_rows)


def _pooled_group_rows(
    input_dir: Path,
    *,
    dedupe_combo: bool,
    group_size: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path, summary in _iter_summaries(input_dir, dedupe_combo=dedupe_combo):
        labels = summary.get("labels")
        if not isinstance(labels, list) or not labels:
            continue
        qc = int(summary.get("question_count") or 0)
        ps = summary.get("per_sample_label_counts")
        if not isinstance(ps, list) or not ps:
            continue
        if group_size < 1 or qc < 1:
            continue
        gidx = 0
        for start in range(0, len(ps), group_size):
            chunk = ps[start : start + group_size]
            if len(chunk) != group_size:
                break
            if any(x is None for x in chunk):
                continue
            totals: Counter = Counter()
            for item in chunk:
                if not isinstance(item, dict):
                    totals.clear()
                    break
                for lbl in labels:
                    if isinstance(lbl, str):
                        totals[lbl] += int(item.get(lbl, 0))
            expected = group_size * qc
            tot = sum(totals.values())
            if tot != expected:
                continue
            gidx += 1
            row: Dict[str, Any] = {
                "file": path.name,
                "deployment": summary.get("deployment"),
                "level": summary.get("level") or "",
                "subject": summary.get("subject"),
                "question_count": qc,
                "num_options": len(labels),
                "spread_correct_answer_uniformly": summary.get(
                    "spread_correct_answer_uniformly"
                ),
                "pooled_group_size": group_size,
                "pooled_group_index": gidx,
                "sample_indices_in_group": f"{start + 1}-{start + group_size}",
                "answers_in_pool": tot,
            }
            for lbl in labels:
                if not isinstance(lbl, str):
                    continue
                c = int(totals.get(lbl, 0))
                row[f"count_{lbl}"] = c
                row[f"pct_{lbl}"] = (c / float(tot)) if tot > 0 else math.nan
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory of probe result JSON files",
    )
    parser.add_argument(
        "--output-wide",
        type=Path,
        default=DEFAULT_WIDE,
        help="Output path for wide CSV",
    )
    parser.add_argument(
        "--output-long",
        type=Path,
        default=DEFAULT_LONG,
        help="Output path for long CSV",
    )
    parser.add_argument(
        "--output-pooled",
        type=Path,
        default=DEFAULT_POOLED,
        help="Output path for pooled-groups CSV",
    )
    parser.add_argument(
        "--no-long",
        action="store_true",
        help="Skip writing the long-format CSV",
    )
    parser.add_argument(
        "--no-pooled",
        action="store_true",
        help="Skip writing pooled-groups CSV",
    )
    parser.add_argument(
        "--dedupe-combo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep newest file per (deployment, level, subject, q, num_options)",
    )
    parser.add_argument(
        "--pooled-group-size",
        type=int,
        default=5,
        help="Number of consecutive 20-Q tests to pool per row (default 5 → 100 answers)",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    wide_rows = _load_wide_rows(input_dir, dedupe_combo=args.dedupe_combo)
    if not wide_rows:
        print(f"No JSON summaries found under {input_dir}", file=sys.stderr)
        return 1

    wide = pd.DataFrame(wide_rows)
    args.output_wide.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output_wide, index=False, encoding="utf-8")
    print(f"Wrote {len(wide)} rows -> {args.output_wide} (dedupe_combo={args.dedupe_combo})")

    if not args.no_long:
        long_df = _wide_to_long(wide)
        long_df.to_csv(args.output_long, index=False, encoding="utf-8")
        print(f"Wrote {len(long_df)} rows -> {args.output_long}")

    if not args.no_pooled:
        pooled_rows = _pooled_group_rows(
            input_dir,
            dedupe_combo=args.dedupe_combo,
            group_size=max(1, int(args.pooled_group_size)),
        )
        pooled_df = pd.DataFrame(pooled_rows)
        pooled_df.to_csv(args.output_pooled, index=False, encoding="utf-8")
        print(
            f"Wrote {len(pooled_df)} rows -> {args.output_pooled} "
            f"(group_size={args.pooled_group_size})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

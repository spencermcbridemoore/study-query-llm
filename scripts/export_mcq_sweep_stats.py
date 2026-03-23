#!/usr/bin/env python3
"""Export MCQ answer-position probe JSON summaries to CSV (wide + optional long).

Reads files from experimental_results/mcq_answer_position_probe/ (or --input-dir).
Each JSON is one sweep cell; summary contains pooled_distribution and
per_sample_distribution_stats per option label (A, B, ...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "experimental_results" / "mcq_answer_position_probe"
DEFAULT_WIDE = PROJECT_ROOT / "experimental_results" / "mcq_sweep_answer_label_stats.csv"
DEFAULT_LONG = PROJECT_ROOT / "experimental_results" / "mcq_sweep_answer_label_stats_long.csv"


def _load_summaries(input_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        return rows
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
        rows.append(row)
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
        # Letters present as pooled_pct_* columns
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
        "--no-long",
        action="store_true",
        help="Skip writing the long-format CSV",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    rows = _load_summaries(input_dir)
    if not rows:
        print(f"No JSON summaries found under {input_dir}", file=sys.stderr)
        return 1

    wide = pd.DataFrame(rows)
    args.output_wide.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.output_wide, index=False, encoding="utf-8")
    print(f"Wrote {len(wide)} rows -> {args.output_wide}")

    if not args.no_long:
        long_df = _wide_to_long(wide)
        long_df.to_csv(args.output_long, index=False, encoding="utf-8")
        print(f"Wrote {len(long_df)} rows -> {args.output_long}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

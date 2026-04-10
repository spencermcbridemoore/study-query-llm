#!/usr/bin/env python3
"""Run run_mcq_sweep.py at decreasing concurrency until idempotent dry-run shows 0 RUN.

Then optionally invoke ingest_mcq_probe_json_to_sweep_db.py for the 6-model OpenRouter sweep.

Example:
  python scripts/run_openrouter_mcq_sweep_until_done.py --ingest
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _dry_run_counts(config: Path) -> tuple[int, int]:
    p = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_mcq_sweep.py"),
            "--config",
            str(config),
            "--dry-run",
            "--idempotent",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    run_n = len(re.findall(r"\[RUN\]", out))
    skip_n = len(re.findall(r"\[SKIP\]", out))
    return run_n, skip_n


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", value).strip("_") or "value"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry MCQ sweep at 16 then 8 concurrency until complete; optional DB ingest."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "config"
        / "mcq_sweep_highschool_college_20q_3456_openrouter_6models.json",
        help="Sweep config JSON",
    )
    parser.add_argument(
        "--concurrency-levels",
        type=str,
        default="16,8",
        help="Comma-separated concurrency/cell_concurrency pairs to try in order",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="After RUN=0, run ingest_mcq_probe_json_to_sweep_db for this sweep",
    )
    parser.add_argument(
        "--ingest-sweep-name",
        type=str,
        default="mcq_sweep_highschool_college_20q_3456_openrouter_6models_ingest",
        help="mcq_sweep group name for ingestion",
    )
    args = parser.parse_args()

    cfg: Path = args.config
    if not cfg.is_absolute():
        cfg = PROJECT_ROOT / cfg

    levels = [int(x.strip()) for x in args.concurrency_levels.split(",") if x.strip()]
    if not levels:
        print("[FATAL] No concurrency levels", file=sys.stderr)
        return 1

    run_n, skip_n = _dry_run_counts(cfg)
    print(f"[INFO] Initial dry-run: RUN={run_n} SKIP={skip_n} (expect 336 total tasks)")
    if run_n == 0:
        print("[INFO] Nothing to run; all cells already complete on disk.")
    else:
        for conc in levels:
            run_n, skip_n = _dry_run_counts(cfg)
            if run_n == 0:
                print("[INFO] All cells complete; stopping.")
                break
            print(f"\n[INFO] === Sweep pass: concurrency={conc} cell_concurrency={conc}  (RUN={run_n}) ===")
            r = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_mcq_sweep.py"),
                    "--config",
                    str(cfg),
                    "--concurrency",
                    str(conc),
                    "--cell-concurrency",
                    str(conc),
                    "--idempotent",
                ],
                cwd=PROJECT_ROOT,
            )
            if r.returncode != 0:
                print(f"[WARN] sweep exit code {r.returncode}; continuing to next level or ingest check")

    run_n, skip_n = _dry_run_counts(cfg)
    print(f"\n[INFO] Final dry-run: RUN={run_n} SKIP={skip_n}")
    if run_n != 0:
        print(
            f"[WARN] Still {run_n} cells incomplete. Re-run this script or lower concurrency manually.",
            file=sys.stderr,
        )
        if not args.ingest:
            return 2

    if args.ingest and run_n != 0:
        print("[SKIP] Ingest skipped because RUN > 0.")
        return 2

    if not args.ingest:
        return 0 if run_n == 0 else 2

    import json

    with open(cfg, encoding="utf-8") as f:
        sweep = json.load(f)
    llms = sweep.get("llms") or []
    globs = [
        f"experimental_results/mcq_answer_position_probe/{_safe_name(m)}_*_q20_c*_n50_*.json"
        for m in llms
    ]
    ingest_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ingest_mcq_probe_json_to_sweep_db.py"),
        "--sweep-name",
        args.ingest_sweep_name,
        "--description",
        "OpenRouter 6-model MCQ [3-6] options, 20q×50 (ingested from local JSON)",
    ]
    for g in globs:
        ingest_cmd.extend(["--json-glob", g])

    print("\n[INFO] Ingesting:", " ".join(ingest_cmd))
    ing = subprocess.run(ingest_cmd, cwd=PROJECT_ROOT)
    return int(ing.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Quick status snapshot for the MOSART overlap15 logprob full run.

Reads provenanced_runs filtered by experiment_label and prints a 14-row x
2-format grid plus a summary. Uses the tunnel that the orchestrator is
already keeping open.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

EXPERIMENT_LABEL = os.environ.get(
    "MOSART_LABEL", "mosart_overlap15_full120_v1v2"
)

# Same 14-engine roster as run_mcq_logprob_experiment.py
EXPECTED_MODELS: tuple[str, ...] = (
    "alpindale/goliath-120b",
    "anthracite-org/magnum-v4-72b",
    "gryphe/mythomax-l2-13b",
    "mancer/weaver",
    "mistralai/ministral-3b-2512",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "qwen/qwen3.6-max-preview",
    "sao10k/l3.3-euryale-70b",
    "thedrummer/rocinante-12b",
    "thedrummer/unslopnemo-12b",
    "undi95/remm-slerp-l2-13b",
)
FORMATS: tuple[str, ...] = ("v1", "v2_chat_system")

# Models we know are permanently unreachable on OpenRouter for this run.
# Probe-discovered exclusions (do NOT count as 'still outstanding').
KNOWN_EXCLUDED_MODELS: dict[str, str] = {
    "alpindale/goliath-120b": "endpoint deprecated (404 No endpoints)",
}


def _resolve_db_url() -> str:
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (os.environ.get(key) or "").strip()
        if v:
            return v
    print("ERROR: no DB URL found (CANONICAL_DATABASE_URL / DATABASE_URL)", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    db_url = _resolve_db_url()
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT
                  config_json->'parameters'->>'model' AS model,
                  config_json->'parameters'->>'prompt_template_version' AS fmt,
                  run_status,
                  (metadata_json->'result_payload'->>'row_count')::int AS row_count,
                  updated_at
                FROM provenanced_runs
                WHERE metadata_json->>'experiment_label' = :label
                  AND run_kind = 'execution'
                """
            ),
            {"label": EXPERIMENT_LABEL},
        ).fetchall()

    by_cell: dict[tuple[str, str], tuple[str, int | None, object]] = {}
    for model, fmt, status, row_count, updated_at in rows:
        by_cell[(str(model or ""), str(fmt or ""))] = (
            str(status or ""),
            row_count,
            updated_at,
        )

    print(f"=== MOSART run status (label={EXPERIMENT_LABEL}) ===")
    excl = len(KNOWN_EXCLUDED_MODELS)
    achievable = 28 - excl * len(FORMATS)
    print(f"provenanced_runs rows: {len(rows)} / {achievable} achievable "
          f"({achievable // len(FORMATS)} models x {len(FORMATS)} formats; "
          f"{excl} models permanently excluded)")
    print()

    status_count = {"completed": 0, "running": 0, "failed": 0, "created": 0, "missing": 0, "excluded": 0}
    print(f"{'model':<36}  {'v1':<22}  {'v2_chat_system':<22}")
    print("-" * 86)
    for model in EXPECTED_MODELS:
        cells: list[str] = []
        for fmt in FORMATS:
            if (model, fmt) in by_cell:
                status, rc, _ts = by_cell[(model, fmt)]
                status_count[status] = status_count.get(status, 0) + 1
                rc_text = str(rc) if rc is not None else "-"
                cell = f"{status:<10}/{rc_text:<5}"
            else:
                if model in KNOWN_EXCLUDED_MODELS:
                    status_count["excluded"] = status_count.get("excluded", 0) + 1
                    cell = "(excluded)"
                else:
                    status_count["missing"] += 1
                    cell = "(pending)"
            cells.append(cell)
        print(f"{model:<36}  {cells[0]:<22}  {cells[1]:<22}")

    print()
    print("=== summary ===")
    for s in ("completed", "running", "failed", "created", "missing", "excluded"):
        c = status_count.get(s, 0)
        print(f"  {s:<10}  {c}")

    completed = status_count.get("completed", 0)
    excluded = status_count.get("excluded", 0)
    expected = 28
    achievable = expected - excluded
    if completed >= achievable:
        print()
        if excluded:
            reasons = ", ".join(f"{m} ({r})" for m, r in KNOWN_EXCLUDED_MODELS.items())
            print(f"[OK] data collection complete: {completed}/{achievable} achievable invocations done.")
            print(f"     {excluded} permanently excluded: {reasons}")
        else:
            print(f"[OK] all {expected} invocations completed.")
    else:
        remaining = achievable - completed
        print()
        print(f"[..] {remaining} achievable invocations still outstanding "
              f"(of {achievable} total; {excluded} permanently excluded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Regenerate deterministic P0 control-plane baseline fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from study_query_llm.services.jobs import build_p0_baseline_snapshot


def _default_fixture_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "tests" / "fixtures" / "p0_baseline" / "baseline_snapshot.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate PR0 SQLite-first deterministic baseline fixture."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_fixture_path(),
        help="Output fixture path (default: tests/fixtures/p0_baseline/baseline_snapshot.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when fixture differs from generated baseline.",
    )
    args = parser.parse_args()

    snapshot = build_p0_baseline_snapshot()
    rendered = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.check:
        if not output_path.exists():
            raise SystemExit(f"Missing fixture for --check: {output_path}")
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit(
                "P0 baseline fixture drift detected. "
                "Run `python scripts/regenerate_p0_baseline.py` to refresh."
            )
        print(f"P0 baseline fixture is up to date: {output_path}")
        return

    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote P0 baseline fixture: {output_path}")


if __name__ == "__main__":
    main()


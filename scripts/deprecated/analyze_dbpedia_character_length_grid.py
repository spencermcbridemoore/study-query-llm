#!/usr/bin/env python3
"""Compatibility wrapper for moved history script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.history.analysis.analyze_dbpedia_character_length_grid import *  # noqa: F401,F403


if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/analyze_dbpedia_character_length_grid.py moved to "
        "scripts/history/analysis/analyze_dbpedia_character_length_grid.py",
        file=sys.stderr,
    )
    runpy.run_module(
        "scripts.history.analysis.analyze_dbpedia_character_length_grid",
        run_name="__main__",
    )

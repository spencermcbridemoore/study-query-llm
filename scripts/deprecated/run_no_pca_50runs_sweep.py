#!/usr/bin/env python3
"""Compatibility wrapper for moved history script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.history.experiments.run_no_pca_50runs_sweep import *  # noqa: F401,F403


if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/run_no_pca_50runs_sweep.py moved to "
        "scripts/history/experiments/run_no_pca_50runs_sweep.py",
        file=sys.stderr,
    )
    runpy.run_module(
        "scripts.history.experiments.run_no_pca_50runs_sweep",
        run_name="__main__",
    )

#!/usr/bin/env python3
"""Compatibility wrapper for moved history script."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.history.analysis.plot_no_pca_50runs import *  # noqa: F401,F403


if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/plot_no_pca_50runs.py moved to "
        "scripts/history/analysis/plot_no_pca_50runs.py",
        file=sys.stderr,
    )
    runpy.run_module("scripts.history.analysis.plot_no_pca_50runs", run_name="__main__")

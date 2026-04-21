#!/usr/bin/env python3
"""Compatibility wrapper for renamed sweep entrypoint."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_pca_kllmeans_sweep import *  # noqa: F401,F403


if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/pca_kllmeans_sweep.py renamed to scripts/run_pca_kllmeans_sweep.py",
        file=sys.stderr,
    )
    runpy.run_module("scripts.run_pca_kllmeans_sweep", run_name="__main__")

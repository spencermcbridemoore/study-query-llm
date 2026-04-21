#!/usr/bin/env python3
"""Compatibility wrapper: canonical implementation under `scripts/deprecated/plot_no_pca_multi_embedding.py`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/plot_no_pca_multi_embedding.py forwards to scripts.deprecated.",
        file=sys.stderr,
    )
    runpy.run_module(
        "scripts.deprecated.plot_no_pca_multi_embedding",
        run_name="__main__",
    )

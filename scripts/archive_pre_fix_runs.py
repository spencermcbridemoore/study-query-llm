#!/usr/bin/env python3
"""Compatibility wrapper: canonical implementation under `scripts/history/sweep_recovery/archive_pre_fix_runs.py`."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    print(
        "[DEPRECATED] scripts/archive_pre_fix_runs.py forwards to "
        "scripts.history.sweep_recovery.archive_pre_fix_runs.",
        file=sys.stderr,
    )
    runpy.run_module(
        "scripts.history.sweep_recovery.archive_pre_fix_runs",
        run_name="__main__",
    )

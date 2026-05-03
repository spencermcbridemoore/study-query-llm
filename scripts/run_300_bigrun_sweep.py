#!/usr/bin/env python3
"""Tier-B compatibility wrapper for bigrun sweep entrypoints.

Prefer: python -m study_query_llm.cli sweep run-bigrun ...

Real sweep runtime logic lives in Tier A under
`src/study_query_llm/experiments/runtime_sweeps.py`.
Do not add new orchestration/sweep runtime logic here; keep this file as a thin delegate.
See `AGENTS.md` for scripts-vs-src boundary rules and terminology.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.experiments.runtime_sweeps import main_bigrun_sync


if __name__ == "__main__":
    main_bigrun_sync()

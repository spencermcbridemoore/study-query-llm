#!/usr/bin/env python3
"""Compatibility wrapper: 300-sample bigrun sweep.

Prefer: python -m study_query_llm.cli sweep run-bigrun ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.experiments.runtime_sweeps import main_bigrun_sync


if __name__ == "__main__":
    main_bigrun_sync()

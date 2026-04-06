#!/usr/bin/env python3
"""Compatibility wrapper for one-container-per-engine sweep supervisor.

Prefer: python -m study_query_llm.cli sweep engine-supervisor ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.services.jobs.runtime_supervisors import main_engine_supervisor


if __name__ == "__main__":
    raise SystemExit(main_engine_supervisor())

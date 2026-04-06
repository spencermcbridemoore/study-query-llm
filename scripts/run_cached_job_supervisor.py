#!/usr/bin/env python3
"""Compatibility wrapper for cached-job supervisor.

Prefer: python -m study_query_llm.cli jobs cached-supervisor ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.services.jobs.runtime_supervisors import main_cached_job_supervisor


if __name__ == "__main__":
    raise SystemExit(main_cached_job_supervisor())

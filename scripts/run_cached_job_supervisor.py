#!/usr/bin/env python3
"""Tier-B compatibility wrapper for cached-supervisor entrypoints.

Prefer: python -m study_query_llm.cli jobs cached-supervisor ...

Real orchestration/supervisor runtime logic lives in Tier A under
`src/study_query_llm/services/jobs/runtime_supervisors.py`.
Do not add new orchestration logic here; keep this file as a thin delegate.
See `AGENTS.md` for scripts-vs-src boundary rules and terminology.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.services.jobs.runtime_supervisors import main_cached_job_supervisor


if __name__ == "__main__":
    raise SystemExit(main_cached_job_supervisor())

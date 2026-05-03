#!/usr/bin/env python3
"""Tier-B compatibility wrapper for langgraph worker entrypoints.

Prefer: python -m study_query_llm.cli jobs langgraph-worker ...

Real orchestration worker runtime logic lives in Tier A under
`src/study_query_llm/services/jobs/runtime_workers.py`.
Do not add new orchestration logic here; keep this file as a thin delegate.
See `AGENTS.md` for scripts-vs-src boundary rules and terminology.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.services.jobs.runtime_workers import main_langgraph_worker


if __name__ == "__main__":
    raise SystemExit(main_langgraph_worker())

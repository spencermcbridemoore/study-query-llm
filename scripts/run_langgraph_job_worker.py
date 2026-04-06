#!/usr/bin/env python3
"""Compatibility wrapper for langgraph_run orchestration worker.

Prefer: python -m study_query_llm.cli jobs langgraph-worker ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from study_query_llm.services.jobs.runtime_workers import main_langgraph_worker


if __name__ == "__main__":
    raise SystemExit(main_langgraph_worker())

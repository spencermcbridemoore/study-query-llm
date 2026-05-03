#!/usr/bin/env python3
"""Tier-B compatibility wrapper for sweep worker entrypoints.

Real orchestration/worker runtime logic lives in Tier A under
`src/study_query_llm/experiments/sweep_worker_main.py`.
Do not add new orchestration logic here; keep this file as a thin delegate.
See `AGENTS.md` for scripts-vs-src boundary rules and terminology.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from study_query_llm.experiments.sweep_worker_main import (
    EMBEDDING_ENGINES,
    main,
    worker_main_queued,
)

__all__ = ["EMBEDDING_ENGINES", "main", "worker_main_queued"]

if __name__ == "__main__":
    main()

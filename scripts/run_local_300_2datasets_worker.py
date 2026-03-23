#!/usr/bin/env python3
"""Thin wrapper for sweep worker; re-exports symbols for supervisors and benchmarks."""

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

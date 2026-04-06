"""Smoke tests: package CLI jobs/sweep subcommands delegate and expose --help."""

from __future__ import annotations

import subprocess
import sys


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "study_query_llm.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_cli_jobs_langgraph_worker_help_exits_zero():
    p = _run_cli(["jobs", "langgraph-worker", "--help"])
    assert p.returncode == 0
    assert "langgraph" in (p.stdout + p.stderr).lower() or "request-id" in (p.stdout + p.stderr).lower()


def test_cli_jobs_cached_supervisor_help_exits_zero():
    p = _run_cli(["jobs", "cached-supervisor", "--help"])
    assert p.returncode == 0


def test_cli_sweep_engine_supervisor_help_exits_zero():
    p = _run_cli(["sweep", "engine-supervisor", "--help"])
    assert p.returncode == 0


def test_cli_sweep_run_bigrun_help_exits_zero():
    p = _run_cli(["sweep", "run-bigrun", "--help"])
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "--create-request" in out or "create-request" in out.lower()

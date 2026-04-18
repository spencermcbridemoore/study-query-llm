"""Tests for scripts/check_persistence_contract.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_persistence_contract.py"


@pytest.fixture(scope="module")
def lint_mod():
    spec = importlib.util.spec_from_file_location("check_persistence_contract", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lint_fails_when_stage_omits_run_stage(lint_mod, tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "acquire.py").write_text(
        dedent(
            """
            def acquire():
                return {"ok": True}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    violations = lint_mod.lint_pipeline_dir(pipeline_dir)
    assert len(violations) == 1
    assert "acquire" in violations[0]


def test_lint_allows_decorator_escape_hatch(lint_mod, tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "helpers.py").write_text(
        dedent(
            """
            from study_query_llm.pipeline.runner import allow_no_run_stage

            @allow_no_run_stage
            def helper():
                return {"ok": True}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    violations = lint_mod.lint_pipeline_dir(pipeline_dir)
    assert violations == []

#!/usr/bin/env python3
"""AST lint for pipeline stage functions that must call run_stage()."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_DIR = REPO / "src" / "study_query_llm" / "pipeline"
SKIP_FILES = {"__init__.py", "types.py", "runner.py"}


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _contains_run_stage_call(function_node: ast.AST) -> bool:
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "run_stage":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "run_stage":
            return True
    return False


def lint_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    try:
        display_path = str(path.relative_to(REPO))
    except ValueError:
        display_path = str(path)
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        decorators = {_decorator_name(deco) for deco in node.decorator_list}
        if "allow_no_run_stage" in decorators:
            continue
        if not _contains_run_stage_call(node):
            violations.append(
                f"{display_path}:{node.name} missing run_stage() call "
                "or @allow_no_run_stage decorator"
            )
    return violations


def lint_pipeline_dir(pipeline_dir: Path) -> list[str]:
    if not pipeline_dir.is_dir():
        return [f"Pipeline directory not found: {pipeline_dir}"]
    violations: list[str] = []
    for path in sorted(pipeline_dir.glob("*.py")):
        if path.name in SKIP_FILES:
            continue
        violations.extend(lint_file(path))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that public pipeline stage functions call run_stage().",
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help=f"Pipeline directory to lint (default: {DEFAULT_PIPELINE_DIR})",
    )
    args = parser.parse_args()

    pipeline_dir = args.pipeline_dir
    if not pipeline_dir.is_absolute():
        pipeline_dir = REPO / pipeline_dir
    violations = lint_pipeline_dir(pipeline_dir)
    if violations:
        print("Persistence contract violations found:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("Persistence contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

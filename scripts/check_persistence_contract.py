#!/usr/bin/env python3
"""AST lint for stage persistence contracts and group-type boundaries."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_DIR = REPO / "src" / "study_query_llm" / "pipeline"
SKIP_FILES = {"__init__.py", "types.py", "runner.py"}
STAGE_GROUP_TYPES = frozenset(
    {
        "dataset",
        "dataset_dataframe",
        "dataset_snapshot",
        "embedding_batch",
        "analysis_run",
    }
)
DEFAULT_GROUP_CREATION_ALLOWLIST_PREFIXES = (
    "src/study_query_llm/pipeline/",
    "src/study_query_llm/services/provenance_service.py",
    "src/study_query_llm/services/embeddings/helpers.py",
    "tests/",
)
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", ".cursor"}


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


def _extract_group_type_literal(call: ast.Call) -> str | None:
    for kw in call.keywords:
        if kw.arg != "group_type":
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value.strip()
        return None
    if call.args:
        first_arg = call.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value.strip()
    return None


def _contains_create_group_calls(tree: ast.AST) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_create_group = False
        if isinstance(func, ast.Attribute) and func.attr == "create_group":
            is_create_group = True
        elif isinstance(func, ast.Name) and func.id == "create_group":
            is_create_group = True
        if not is_create_group:
            continue
        group_type = _extract_group_type_literal(node)
        if not group_type or group_type not in STAGE_GROUP_TYPES:
            continue
        findings.append((int(getattr(node, "lineno", 0) or 0), group_type))
    return findings


def lint_file(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
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


def _relative_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _path_is_allowlisted(
    path: Path,
    allowlist_prefixes: tuple[str, ...],
) -> bool:
    display = _relative_display_path(path)
    return any(
        display == prefix.rstrip("/") or display.startswith(prefix)
        for prefix in allowlist_prefixes
    )


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def lint_group_type_boundaries(
    *,
    scan_root: Path = REPO,
    allowlist_prefixes: tuple[str, ...] = DEFAULT_GROUP_CREATION_ALLOWLIST_PREFIXES,
) -> list[str]:
    violations: list[str] = []
    for path in _iter_python_files(scan_root):
        source = path.read_text(encoding="utf-8-sig")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            # Archived/history scripts may contain non-standard encodings or stale syntax.
            # Boundary lint is best-effort and should not fail vacuously due to unrelated files.
            continue
        findings = _contains_create_group_calls(tree)
        if not findings:
            continue
        if _path_is_allowlisted(path, allowlist_prefixes):
            continue
        display_path = _relative_display_path(path)
        for line_no, group_type in findings:
            violations.append(
                f"{display_path}:{line_no} unauthorized create_group(group_type={group_type!r}); "
                "use pipeline stages/run_stage or add explicit allowlist entry"
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
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=REPO,
        help=f"Repository root to scan for create_group boundary checks (default: {REPO})",
    )
    args = parser.parse_args()

    pipeline_dir = args.pipeline_dir
    if not pipeline_dir.is_absolute():
        pipeline_dir = REPO / pipeline_dir
    scan_root = args.scan_root
    if not scan_root.is_absolute():
        scan_root = REPO / scan_root

    violations = lint_pipeline_dir(pipeline_dir)
    violations.extend(lint_group_type_boundaries(scan_root=scan_root))
    if violations:
        print("Persistence contract violations found:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("Persistence contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

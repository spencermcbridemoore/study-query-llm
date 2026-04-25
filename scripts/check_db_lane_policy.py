#!/usr/bin/env python3
"""Static policy checks for DB lane guardrails in runtime code."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_ROOTS = ("src/study_query_llm", "panel_app", "scripts")
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".cursor", "history", "deprecated"}
ALLOWED_CREATE_ENGINE_PATHS = {
    "src/study_query_llm/db/_base_connection.py",
    "scripts/backup_mcq_db_to_json.py",
    "scripts/check_active_workers.py",
    "scripts/check_call_artifacts_uri_constraint.py",
    "scripts/check_raw_calls_uri_sentinel.py",
    "scripts/probe_postgres_inventory.py",
    "scripts/purge_dataset_acquisition.py",
    "scripts/sanity_check_database_url.py",
    "scripts/sync_from_online.py",
    "scripts/upload_jetstream_pg_dump_to_blob.py",
    "scripts/verify_call_artifact_blob_lanes.py",
    "scripts/verify_db_backup_inventory.py",
}

Violation = tuple[str, int, str]


def _iter_python_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _check_file(path: Path) -> list[Violation]:
    rel = _relative(path)
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "DatabaseConnectionV2":
            has_write_intent = any(keyword.arg == "write_intent" for keyword in node.keywords)
            if not has_write_intent:
                violations.append(
                    (
                        rel,
                        int(getattr(node, "lineno", 0) or 0),
                        "DatabaseConnectionV2 call missing explicit write_intent.",
                    )
                )
        if name == "create_engine" and rel not in ALLOWED_CREATE_ENGINE_PATHS:
            violations.append(
                (
                    rel,
                    int(getattr(node, "lineno", 0) or 0),
                    "Direct create_engine call outside allowlist; use BaseDatabaseConnection "
                    "or explicitly add to policy allowlist.",
                )
            )
    return violations


def run_checks() -> list[Violation]:
    violations: list[Violation] = []
    for relative_root in RUNTIME_ROOTS:
        root = REPO_ROOT / relative_root
        for path in _iter_python_files(root):
            violations.extend(_check_file(path))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check DB lane policy: explicit write_intent + create_engine allowlist.",
    )
    parser.parse_args()
    violations = run_checks()
    if violations:
        print("DB lane policy violations found:", file=sys.stderr)
        for path, line, message in sorted(violations, key=lambda item: (item[0], item[1], item[2])):
            print(
                f"  - {path}:{line} {message}",
                file=sys.stderr,
            )
        return 1
    print("DB lane policy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

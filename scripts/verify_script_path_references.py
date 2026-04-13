#!/usr/bin/env python3
"""Verify that active docs/deploy/CI script path references exist."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_REF_PATTERN = re.compile(r"(?<![A-Za-z0-9_./-])scripts/[A-Za-z0-9_./-]+\.py")
SCAN_SUFFIXES = {".md", ".yml", ".yaml", ".txt", ".sh", ".ps1"}


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            if path.suffix.lower() in SCAN_SUFFIXES:
                yield path
            continue
        if path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SCAN_SUFFIXES:
                    yield file_path


def _active_scan_paths(repo_root: Path) -> list[Path]:
    return [
        repo_root / "docs" / "README.md",
        repo_root / "docs" / "DEPLOYMENT.md",
        repo_root / "docs" / "LOCAL_DB_CLONE_FROM_JETSTREAM.md",
        repo_root / "docs" / "SWEEP_MIGRATION_RUNBOOK.md",
        repo_root / "docs" / "runbooks",
        repo_root / "docs" / "living",
        repo_root / "deploy" / "jetstream",
        repo_root / ".github" / "workflows",
        repo_root / "scripts" / "README.md",
    ]


def _history_scan_paths(repo_root: Path) -> list[Path]:
    return [
        repo_root / "docs" / "history",
        repo_root / "docs" / "experiments",
        repo_root / "docs" / "IMPLEMENTATION_PLAN.md",
        repo_root / "docs" / "ARCHITECTURE.md",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also scan history/experiments legacy docs.",
    )
    args = parser.parse_args()

    scan_paths = _active_scan_paths(REPO_ROOT)
    if args.include_history:
        scan_paths.extend(_history_scan_paths(REPO_ROOT))

    refs: dict[str, set[str]] = {}
    for file_path in _iter_files(scan_paths):
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = file_path.relative_to(REPO_ROOT).as_posix()
        for match in SCRIPT_REF_PATTERN.findall(text):
            refs.setdefault(match, set()).add(rel)

    missing = sorted(ref for ref in refs if not (REPO_ROOT / ref).is_file())
    print(f"Scanned files: {sum(1 for _ in _iter_files(scan_paths))}")
    print(f"Script references found: {len(refs)}")
    print(f"Missing references: {len(missing)}")
    if not missing:
        return 0

    for ref in missing:
        print(f"- {ref}")
        for src in sorted(refs[ref]):
            print(f"    referenced from: {src}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

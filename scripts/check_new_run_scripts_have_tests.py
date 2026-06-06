#!/usr/bin/env python3
"""Creation tripwire: a newly added ``scripts/run_*.py`` must be referenced by a test.

Fork D (plan 3.1). For each ADDED root-level ``scripts/run_*.py`` in the change
set, fail if no file under ``tests/`` references the script's stem -- by filename
(a sibling ``test_<stem>.py``) or by content (a test that imports/invokes it).

Deliberately shallow and gameable -- a human nudge, not a proof. It does NO AST /
import-graph / delegates-to-Tier-A analysis: it only checks that *some* test
mentions the stem. Tier A (``src/``) holds canonical logic; Tier B
(``scripts/run_*.py``) wrappers should stay thin and be exercised by at least one
test (see AGENTS.md, Code Layer Boundary). Enforced rather than convention-only
because the parallel_fixed_k_sweep "ship a script with no test" recurrence already
happened once.

Runs in the pre-commit static set (staged additions by default); CI / ad-hoc can
pass an explicit ``--base``/``--head`` commit range.

Usage:
    python scripts/check_new_run_scripts_have_tests.py                      # staged
    python scripts/check_new_run_scripts_have_tests.py --base origin/main --head HEAD
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

REPO = Path(__file__).resolve().parent.parent

_RUN_PREFIX = "scripts/run_"
_PY_SUFFIX = ".py"


def _norm(path: str) -> str:
    return str(path or "").replace("\\", "/").strip()


def added_run_scripts(paths: Iterable[str]) -> List[str]:
    """Filter ``paths`` to ADDED root-level ``scripts/run_*.py`` (no subdirectories)."""
    out: set[str] = set()
    for raw in paths:
        posix = _norm(raw)
        if not posix.startswith(_RUN_PREFIX) or not posix.endswith(_PY_SUFFIX):
            continue
        # Root-level only: nothing after "scripts/" may contain another "/".
        if "/" in posix[len("scripts/"):]:
            continue
        out.add(posix)
    return sorted(out)


def _iter_test_files(tests_root: Path) -> Iterator[Path]:
    if not tests_root.is_dir():
        return
    for path in tests_root.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def stem_is_referenced(stem: str, tests_root: Path) -> bool:
    """True if any file under ``tests/`` references ``stem`` (filename or .py content)."""
    for path in _iter_test_files(tests_root):
        if stem in path.name:
            return True
        if path.suffix == _PY_SUFFIX:
            try:
                if stem in path.read_text(encoding="utf-8", errors="ignore"):
                    return True
            except OSError:
                continue
    return False


def find_untested_run_scripts(paths: Iterable[str], tests_root: Path) -> List[str]:
    """Return ADDED ``scripts/run_*.py`` whose stem no file under ``tests/`` references."""
    offenders: List[str] = []
    for script in added_run_scripts(paths):
        if not stem_is_referenced(Path(script).stem, tests_root):
            offenders.append(script)
    return offenders


def _git_added_paths(base: Optional[str], head: Optional[str]) -> List[str]:
    if base and head:
        args = ["git", "diff", "--name-only", "--diff-filter=A", f"{base}..{head}"]
    else:
        args = ["git", "diff", "--cached", "--name-only", "--diff-filter=A"]
    result = subprocess.run(
        args, check=False, capture_output=True, text=True, encoding="utf-8", cwd=REPO
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Creation tripwire: a new scripts/run_*.py needs a referencing test."
    )
    parser.add_argument("--base", default=None, help="Base ref for a commit-range diff (with --head).")
    parser.add_argument("--head", default=None, help="Head ref for a commit-range diff (with --base).")
    args = parser.parse_args(argv)

    offenders = find_untested_run_scripts(
        _git_added_paths(args.base, args.head), REPO / "tests"
    )
    if not offenders:
        return 0

    print(
        "error: creation lint -- newly added operator script(s) have no referencing "
        "test under tests/:",
        file=sys.stderr,
    )
    for script in offenders:
        stem = Path(script).stem
        print(
            f"  - {script}  (add tests/.../test_{stem}.py, or reference {stem} from a test)",
            file=sys.stderr,
        )
    print(
        "\nShallow tripwire (filename/content reference only), not a correctness proof. "
        "Tier B scripts/run_*.py wrappers should be exercised by >=1 test; canonical "
        "logic belongs in Tier A (src/). See AGENTS.md (Code Layer Boundary).\n"
        "Bypass intentionally with: git commit --no-verify",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

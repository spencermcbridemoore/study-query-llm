#!/usr/bin/env python3
"""Pre-commit warning for the living-docs-only governance rule.

Reads currently staged paths from git and prints a warning to stderr if any
fall in the restricted set defined by
``scripts/internal/living_docs_governance.py``. **Always exits 0** -- this is
a heads-up for the developer, not a block. The hard check
(``scripts/check_living_docs_drift.py``) runs in CI.

Used by the optional Git hook at ``scripts/git-hooks/pre-commit``. Install
with::

    git config core.hooksPath scripts/git-hooks
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.internal.living_docs_governance import (  # noqa: E402
    ANNOTATION_TOKEN,
    find_restricted_paths,
)


def staged_paths() -> list[str]:
    """Return paths currently staged for commit (added/copied/modified/renamed)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    restricted = find_restricted_paths(staged_paths())
    if not restricted:
        return 0

    print(
        "warning: living-docs-only gate -- staged changes touch restricted paths:",
        file=sys.stderr,
    )
    for path in restricted:
        print(f"  - {path}", file=sys.stderr)
    print(
        "\nThese paths are 'past/unused workflow' material per "
        "`.cursor/rules/living-docs-only.mdc`. The commit will proceed, "
        "but CI (`scripts/check_living_docs_drift.py`) will fail unless a "
        f"commit message in the PR contains {ANNOTATION_TOKEN!r}.\n"
        "If the edit is intentional, include that token in your commit message.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CI hard check for the living-docs-only governance rule.

Fails (exit 1) when a git diff range edits restricted paths without the
annotation token (default: ``[restricted-doc-edit-ok]``) appearing in at
least one commit message in the same range.

The restricted set is owned by
``scripts/internal/living_docs_governance.py`` and mirrors the table in
``.cursor/rules/living-docs-only.mdc``.

Typical usage:

- Local sanity (vs default branch):
  ``python scripts/check_living_docs_drift.py``
- Local sanity (vs previous commit):
  ``python scripts/check_living_docs_drift.py --base HEAD~1 --head HEAD``
- GitHub Actions PR (see ``.github/workflows/living-docs-drift.yml``):
  ``python scripts/check_living_docs_drift.py --base "$BASE_SHA" --head "$HEAD_SHA"``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.internal.living_docs_governance import (  # noqa: E402
    ANNOTATION_TOKEN,
    find_restricted_paths,
    messages_contain_annotation,
)


def _run_git(args: list[str]) -> str:
    """Run ``git`` with stdout captured; raise on non-zero exit."""
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO,
    )
    return result.stdout


def changed_files(base: str, head: str) -> list[str]:
    """Return added/copied/modified/renamed paths between ``base`` and ``head``."""
    out = _run_git(
        [
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base}...{head}",
        ]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def commit_messages(base: str, head: str) -> list[str]:
    """Return full commit messages for ``base..head`` (newest first)."""
    out = _run_git(
        [
            "log",
            "--format=%B%x00",
            f"{base}..{head}",
        ]
    )
    return [chunk.strip() for chunk in out.split("\x00") if chunk.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail if a diff range edits restricted living-docs paths without "
            "an annotation token in any commit message in the range."
        ),
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base git ref (default: origin/main)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head git ref (default: HEAD)",
    )
    parser.add_argument(
        "--annotation-token",
        default=ANNOTATION_TOKEN,
        help=(
            f"Token that must appear in a commit message to allow restricted "
            f"edits (default: {ANNOTATION_TOKEN!r})"
        ),
    )
    args = parser.parse_args()

    try:
        files = changed_files(args.base, args.head)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or str(exc)
        print(
            f"living-docs-drift: failed to diff {args.base}...{args.head}: {stderr}",
            file=sys.stderr,
        )
        return 2

    restricted = find_restricted_paths(files)

    if not restricted:
        print(
            f"living-docs-drift: OK ({len(files)} changed file(s); no restricted paths)."
        )
        return 0

    try:
        messages = commit_messages(args.base, args.head)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or str(exc)
        print(
            f"living-docs-drift: failed to read commit messages: {stderr}",
            file=sys.stderr,
        )
        return 2

    if messages_contain_annotation(messages, args.annotation_token):
        print(
            "living-docs-drift: restricted paths edited; annotation token "
            f"{args.annotation_token!r} present. Allowed."
        )
        for path in restricted:
            print(f"  - {path}")
        return 0

    print(
        "living-docs-drift: restricted paths edited without annotation token "
        f"{args.annotation_token!r} in any commit message in "
        f"{args.base}..{args.head}.",
        file=sys.stderr,
    )
    print("Offending paths:", file=sys.stderr)
    for path in restricted:
        print(f"  - {path}", file=sys.stderr)
    print(
        "\nResolution options:\n"
        "  1. Move the edits out of restricted paths (preferred for new work).\n"
        "  2. If the edit is intentional (e.g. archiving a script, recording a\n"
        "     historical snapshot, or documenting a deprecation), add the\n"
        f"     token {args.annotation_token!r} to a commit message in the range.\n"
        "  3. See `.cursor/rules/living-docs-only.mdc` for the binding rule.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

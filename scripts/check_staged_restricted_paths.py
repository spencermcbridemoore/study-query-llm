#!/usr/bin/env python3
# Blocking staged-restricted-path check for the living-docs-only pre-commit hook.
# Exits non-zero when staged changes touch tombstoned restricted paths
# (scripts/internal/living_docs_governance.py). The token-aware authoritative
# check runs at pre-push / CI (scripts/check_living_docs_drift.py).
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.internal.living_docs_governance import (  # noqa: E402
    ANNOTATION_TOKEN,
    find_restricted_paths,
)
from scripts.warn_restricted_doc_edits import staged_paths  # noqa: E402


def main() -> int:
    restricted = find_restricted_paths(staged_paths())
    if not restricted:
        return 0
    print(
        "error: living-docs-only gate -- staged changes touch tombstoned "
        "restricted paths:",
        file=sys.stderr,
    )
    for path in restricted:
        print("  - " + path, file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "Commit blocked. For an intentional tombstone edit, re-commit with "
        "`git commit --no-verify` and include " + repr(ANNOTATION_TOKEN) + " in "
        "a commit message so the pre-push drift check accepts it.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

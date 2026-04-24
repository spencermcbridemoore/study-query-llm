"""Shared restricted-path set for the living-docs-only governance rule.

This module is the single source of truth for which paths are "restricted
reading" under `.cursor/rules/living-docs-only.mdc`. Both the CI hard check
(`scripts/check_living_docs_drift.py`) and the pre-commit warning
(`scripts/warn_restricted_doc_edits.py`) import from here so the two stay
in lock-step.

Membership rules:
- Patterns use POSIX forward slashes; matching normalises Windows backslashes.
- ``RESTRICTED_PREFIX_DIRS`` matches any path whose POSIX form starts with the
  given directory prefix (terminated by ``/``).
- ``RESTRICTED_FILES`` matches exact POSIX paths.
- Update both the constants below AND the ``Restricted Reading`` table in
  ``.cursor/rules/living-docs-only.mdc`` together.
"""

from __future__ import annotations

from collections.abc import Iterable

ANNOTATION_TOKEN: str = "[restricted-doc-edit-ok]"
"""Marker that must appear in at least one commit message in the diff range
to allow restricted-path edits to land via the CI hard check."""

RESTRICTED_PREFIX_DIRS: tuple[str, ...] = (
    "docs/history/",
    "docs/deprecated/",
    "docs/plans/",
    "docs/experiments/",
    "scripts/history/",
    "scripts/deprecated/",
)

RESTRICTED_FILES: frozenset[str] = frozenset(
    {
        "docs/IMPLEMENTATION_PLAN.md",
        "docs/ARCHITECTURE.md",
        "docs/API.md",
        "docs/MIGRATION_GUIDE.md",
        "docs/PHASE1_5_VERIFICATION.md",
        "docs/PLOT_ORGANIZATION.md",
    }
)


def _normalise(path: str) -> str:
    """Normalise a path to POSIX form for matching."""
    return path.replace("\\", "/").strip()


def is_restricted_path(path: str) -> bool:
    """Return True if ``path`` is in the restricted-reading set."""
    if not path:
        return False
    posix = _normalise(path)
    if posix in RESTRICTED_FILES:
        return True
    return any(posix.startswith(prefix) for prefix in RESTRICTED_PREFIX_DIRS)


def find_restricted_paths(paths: Iterable[str]) -> list[str]:
    """Return the sorted, de-duplicated subset of ``paths`` that is restricted."""
    matched: set[str] = set()
    for raw in paths:
        if is_restricted_path(raw):
            matched.add(_normalise(raw))
    return sorted(matched)


def messages_contain_annotation(
    messages: Iterable[str],
    token: str = ANNOTATION_TOKEN,
) -> bool:
    """Return True if any commit message in ``messages`` contains ``token``."""
    for message in messages:
        if message and token in message:
            return True
    return False


__all__ = [
    "ANNOTATION_TOKEN",
    "RESTRICTED_PREFIX_DIRS",
    "RESTRICTED_FILES",
    "is_restricted_path",
    "find_restricted_paths",
    "messages_contain_annotation",
]

#!/usr/bin/env python3
"""Deprecated compatibility stub for historical v1->v2 migration."""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "scripts/migrate_v1_to_v2.py is deprecated and retained only for historical references.",
        file=sys.stderr,
    )
    print(
        "See docs/IMPLEMENTATION_PLAN.md and docs/history/README.md for migration chronology.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

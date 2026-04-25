#!/usr/bin/env python3
"""Read-only audit of MCQ ``method_definitions`` rows for drift.

Inspects the four MCQ-relevant ``(name, version)`` pairs:

* ``mcq_answer_position_probe@1.0`` (probe execution row)
* ``mcq_compliance_metrics@1.0`` (run-scope analysis)
* ``mcq_answer_position_distribution@1.0`` (sweep-scope analysis)
* ``mcq_answer_position_chi_square@1.0`` (sweep-scope analysis)

For each row, compares the stored ``parameters_schema.properties`` keys
against the canonical schema declared in this prep work
(:mod:`study_query_llm.algorithms.canonical_configs`) and detects:

* ``missing``: row is absent.
* ``placeholder``: row exists but its schema matches the
  ``SweepRequestService.record_analysis_result`` auto-register placeholder
  (``{"analysis_key", "request_id"}`` only).
* ``canonical``: stored schema keys exactly match the canonical builder
  keys.
* ``drift``: stored schema differs from canonical (missing canonical keys,
  unexpected extra keys, or both).

Emits a structured JSON report to stdout. **No DB writes.** Exit code is
always 0; the report is purely diagnostic. Inspect the report and decide
remediation separately (out of scope for this audit).

Usage:
    python scripts/audit_mcq_method_definitions.py
    python scripts/audit_mcq_method_definitions.py --pretty
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.algorithms.canonical_configs import (
    CANONICAL_CONFIG_BUILDERS,
    canonical_config_for,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.services.method_service import MethodService


MCQ_METHODS: List[Dict[str, str]] = [
    {"name": "mcq_answer_position_probe", "version": "1.0"},
    {"name": "mcq_compliance_metrics", "version": "1.0"},
    {"name": "mcq_answer_position_distribution", "version": "1.0"},
    {"name": "mcq_answer_position_chi_square", "version": "1.0"},
]

PLACEHOLDER_KEYS: Set[str] = {"analysis_key", "request_id"}


def _canonical_keys(name: str, version: str) -> Set[str]:
    """Return the keys produced by the canonical builder, with all-None inputs.

    Builders are pure and accept loose params; passing an empty dict gives
    the full surface of the canonical shape (each key present, value None).
    """
    if (name, version) not in CANONICAL_CONFIG_BUILDERS:
        return set()
    return set(canonical_config_for(name, version, {}).keys())


def _stored_keys(parameters_schema: Any) -> Set[str]:
    if not isinstance(parameters_schema, dict):
        return set()
    properties = parameters_schema.get("properties")
    if not isinstance(properties, dict):
        return set()
    return set(properties.keys())


def _classify(
    method_present: bool,
    stored: Set[str],
    canonical: Set[str],
) -> str:
    if not method_present:
        return "missing"
    if stored == PLACEHOLDER_KEYS:
        return "placeholder"
    if not canonical:
        # No canonical builder for this method; cannot judge drift.
        return "unknown_canonical"
    if stored == canonical:
        return "canonical"
    return "drift"


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", encoding="utf-8")
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of MCQ method_definitions rows. "
            "Emits a JSON drift report; performs no DB writes."
        )
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON report (indent=2).",
    )
    args = parser.parse_args()

    db_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not db_url:
        # Read-only diagnostic: still exit 0, but emit a structured error.
        report = {
            "ok": False,
            "error": "DATABASE_URL not set",
            "methods": [],
        }
        json.dump(
            report,
            sys.stdout,
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0

    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()

    method_reports: List[Dict[str, Any]] = []
    summary = {
        "canonical": 0,
        "placeholder": 0,
        "missing": 0,
        "drift": 0,
        "unknown_canonical": 0,
    }

    with db.session_scope() as session:
        repo = RawCallRepository(session)
        method_svc = MethodService(repo)
        for spec in MCQ_METHODS:
            name = spec["name"]
            version = spec["version"]
            existing = method_svc.get_method(name, version=version)
            present = existing is not None
            stored_schema = existing.parameters_schema if present else None
            stored = _stored_keys(stored_schema)
            canonical = _canonical_keys(name, version)
            status = _classify(present, stored, canonical)
            summary[status] = summary.get(status, 0) + 1
            method_reports.append({
                "name": name,
                "version": version,
                "present": present,
                "method_definition_id": int(existing.id) if present else None,
                "status": status,
                "stored_schema_keys": sorted(stored),
                "canonical_schema_keys": sorted(canonical),
                "missing_canonical_keys": sorted(canonical - stored),
                "unexpected_extra_keys": sorted(stored - canonical) if canonical else [],
            })
        # Read-only: roll back any incidental session state, just to be safe.
        session.rollback()

    report = {
        "ok": True,
        "summary": summary,
        "methods": method_reports,
    }
    json.dump(
        report,
        sys.stdout,
        indent=2 if args.pretty else None,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

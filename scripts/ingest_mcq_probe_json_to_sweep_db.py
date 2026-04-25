#!/usr/bin/env python3
"""Ingest local MCQ probe JSON files (from run_mcq_sweep.py) into Neon as mcq_sweep + mcq_run.

Creates one ``mcq_sweep`` group and links each file as a child ``mcq_run`` with the same
metadata shape as ``persist_mcq_probe_result`` (result_summary + probe_details).

Usage:
  python scripts/ingest_mcq_probe_json_to_sweep_db.py \\
    --sweep-name my_sweep \\
    --json-glob "experimental_results/mcq_answer_position_probe/foo_*_q20_c*_n50_*.json" \\
    --json-glob "experimental_results/mcq_answer_position_probe/bar_*_q20_c*_n50_*.json"

Idempotent: skips runs whose ``run_key`` already exists in the DB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm.attributes import flag_modified

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import study_query_llm.config  # noqa: F401  # loads .env for DATABASE_URL

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.experiments.mcq_run_persistence import MCQ_RUN_METADATA_VERSION
from study_query_llm.experiments.sweep_request_types import build_mcq_run_key
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN, GROUP_TYPE_MCQ_SWEEP


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe_for_db(value: Any) -> Any:
    """PostgreSQL json/jsonb reject NaN/Infinity; normalize nested metadata for storage."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _json_safe_for_db(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe_for_db(v) for v in value]
    return value


def _probe_details_from_file(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _target_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    labels = summary.get("labels") or []
    options = len(labels) if isinstance(labels, list) else 0
    return {
        "deployment": str(summary.get("deployment", "")),
        "level": str(summary.get("level") or ""),
        "subject": str(summary.get("subject", "")),
        "options_per_question": int(options),
        "questions_per_test": int(summary.get("question_count", 0)),
        "label_style": "upper",
        "spread_correct_answer_uniformly": bool(
            summary.get("spread_correct_answer_uniformly", False)
        ),
        "samples_per_combo": int(summary.get("samples_requested", 0)),
        "template_version": "v1",
    }


def _run_metadata(
    run_key: str,
    target: Dict[str, Any],
    probe_details: Dict[str, Any],
) -> Dict[str, Any]:
    summary = probe_details.get("summary") or {}
    return {
        "run_key": run_key,
        "sweep_type": "mcq",
        "deployment": target.get("deployment"),
        "level": target.get("level"),
        "subject": target.get("subject"),
        "options_per_question": target.get("options_per_question"),
        "questions_per_test": target.get("questions_per_test"),
        "label_style": target.get("label_style"),
        "spread_correct_answer_uniformly": target.get("spread_correct_answer_uniformly"),
        "samples_per_combo": target.get("samples_per_combo"),
        "template_version": target.get("template_version"),
        "result_summary": summary,
        "probe_details": probe_details,
        "mcq_metadata_version": MCQ_RUN_METADATA_VERSION,
        "ingestion_source": "run_mcq_sweep_json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MCQ probe JSON files into mcq_sweep / mcq_run.")
    parser.add_argument("--sweep-name", required=True, help="Name for the parent mcq_sweep group")
    parser.add_argument(
        "--json-glob",
        action="append",
        dest="json_globs",
        metavar="GLOB",
        required=True,
        help=(
            "Glob under repo root (repeat flag for multiple patterns, e.g. one per OpenRouter model prefix)"
        ),
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional description for the mcq_sweep group",
    )
    args = parser.parse_args()

    print("[INFO] Glob + scan starting (this can take a while with many files)...", flush=True)
    paths_set: Dict[str, Path] = {}
    for raw in args.json_globs:
        pattern = raw.replace("\\", "/")
        for p in PROJECT_ROOT.glob(pattern):
            paths_set[str(p.resolve())] = p
    print(f"[INFO] Matched {len(paths_set)} unique path(s); ordering by filename...", flush=True)
    paths = sorted(paths_set.values(), key=lambda p: p.name.lower())
    if not paths:
        print(f"[FATAL] No files matched any glob: {args.json_globs}", file=sys.stderr)
        return 1

    db_url = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    if not db_url:
        print("[FATAL] DATABASE_URL or NEON_DATABASE_URL must be set.", file=sys.stderr)
        return 1

    print("[INFO] Connecting and initializing DB schema (Neon cold start can take 10–30s)...", flush=True)
    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    db.init_db()
    print("[INFO] DB ready; checking which runs are new (one query per file)...", flush=True)

    from sqlalchemy import text as sa_text
    from study_query_llm.db.models_v2 import Group

    pending: List[Tuple[Path, str, Dict[str, Any], Dict[str, Any]]] = []
    skipped = 0

    with db.session_scope() as session:
        for idx, path in enumerate(paths, start=1):
            if idx == 1 or idx % 25 == 0 or idx == len(paths):
                print(f"[INFO] Preflight {idx}/{len(paths)}: {path.name[:64]}...", flush=True)
            probe_details = _probe_details_from_file(path)
            summary = probe_details.get("summary")
            if not isinstance(summary, dict):
                print(f"[SKIP] No summary in {path.name}")
                skipped += 1
                continue

            target = _target_from_summary(summary)
            run_key = build_mcq_run_key(
                deployment=str(target["deployment"]),
                level=str(target["level"]),
                subject=str(target["subject"]),
                options_per_question=int(target["options_per_question"]),
                questions_per_test=int(target["questions_per_test"]),
                label_style=str(target["label_style"]),
                spread_correct_answer_uniformly=bool(target["spread_correct_answer_uniformly"]),
                samples_per_combo=int(target["samples_per_combo"]),
                template_version=str(target["template_version"]),
            )

            existing = (
                session.query(Group)
                .filter(
                    Group.group_type == GROUP_TYPE_MCQ_RUN,
                    sa_text("metadata_json->>'run_key' = :rk"),
                )
                .params(rk=run_key)
                .first()
            )
            if existing is not None:
                print(f"[SKIP] run_key exists: {run_key[:80]}...")
                skipped += 1
                continue

            pending.append((path, run_key, target, probe_details))

    if not pending:
        print(f"[DONE] Nothing to ingest (skipped={skipped}, matched_files={len(paths)}).")
        return 0

    print(
        f"[INFO] Inserting {len(pending)} new run(s) under mcq_sweep (large JSON metadata per row can take several minutes)...",
        flush=True,
    )
    created_runs = 0
    sweep_id: int

    with db.session_scope() as session:
        repo = RawCallRepository(session)

        sweep_id = repo.create_group(
            group_type=GROUP_TYPE_MCQ_SWEEP,
            name=args.sweep_name,
            description=args.description or f"Ingested MCQ sweep: {args.sweep_name}",
            metadata_json={
                "sweep_name": args.sweep_name,
                "ingestion": "ingest_mcq_probe_json_to_sweep_db",
                "json_globs": list(args.json_globs),
                "file_count": len(paths),
                "ingested_run_count": len(pending),
                "created_at": _now_iso(),
            },
        )

        for pos, (path, run_key, target, probe_details) in enumerate(pending):
            meta = _run_metadata(run_key, target, probe_details)
            safe_name = run_key.replace("/", "_")[:120]
            run_id = repo.create_group(
                group_type=GROUP_TYPE_MCQ_RUN,
                name=f"mcq_run_{safe_name}",
                metadata_json=_json_safe_for_db(meta),
            )
            grp = repo.get_group_by_id(run_id)
            if grp is not None:
                m = dict(grp.metadata_json or {})
                m["_group_id"] = int(run_id)
                grp.metadata_json = m
                flag_modified(grp, "metadata_json")
                session.flush()

            repo.create_group_link(
                parent_group_id=sweep_id,
                child_group_id=run_id,
                link_type="contains",
                position=pos,
            )
            created_runs += 1
            print(f"[OK] {path.name} -> mcq_run id={run_id} ({run_key[:60]}...)", flush=True)

    print(
        f"\n[DONE] mcq_sweep id={sweep_id} name={args.sweep_name!r} "
        f"created_runs={created_runs} skipped={skipped} total_files={len(paths)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

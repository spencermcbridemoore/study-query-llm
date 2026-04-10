#!/usr/bin/env python3
"""Export per-1000-answer option counts for MCQ sweeps stored in DB.

For each selected ``mcq_sweep`` group, this script:
1) loads linked ``mcq_run`` groups,
2) computes counts per option label for each consecutive group of
   ``group_size_answers`` answers (default 1000),
3) writes both CSV and JSON outputs.

Primary use-case:
- 20 questions x 50 samples => 1000 answers per run, yielding one row per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sqlalchemy import desc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group, GroupLink


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experimental_results" / "mcq_db_option_counts"


def _safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "value"


def _latest_mcq_sweep_ids(session, n: int) -> List[int]:
    rows = (
        session.query(Group.id)
        .filter(
            Group.group_type == "mcq_sweep",
        )
        .order_by(desc(Group.id))
        .limit(max(0, int(n)))
        .all()
    )
    return [int(r.id) for r in rows]


def _sweep_runs(session, sweep_id: int) -> List[Tuple[Group, int]]:
    """Return list of (run_group, position) for run children of one sweep."""
    links = (
        session.query(GroupLink)
        .filter(
            GroupLink.parent_group_id == int(sweep_id),
            GroupLink.link_type == "contains",
        )
        .order_by(GroupLink.position.asc(), GroupLink.id.asc())
        .all()
    )
    if not links:
        return []

    child_ids = [int(lnk.child_group_id) for lnk in links]
    pos_by_child = {int(lnk.child_group_id): int(lnk.position or 0) for lnk in links}
    runs = (
        session.query(Group)
        .filter(
            Group.id.in_(child_ids),
            Group.group_type == "mcq_run",
        )
        .all()
    )
    runs_sorted = sorted(
        runs,
        key=lambda g: (pos_by_child.get(int(g.id), 0), int(g.id)),
    )
    return [(g, pos_by_child.get(int(g.id), 0)) for g in runs_sorted]


def _labels_from_summary(meta: Dict[str, Any]) -> List[str]:
    summary = meta.get("result_summary") if isinstance(meta, dict) else {}
    if not isinstance(summary, dict):
        return []
    labels = summary.get("labels")
    if not isinstance(labels, list):
        return []
    out: List[str] = []
    for item in labels:
        if isinstance(item, str) and item.strip():
            out.append(item.strip().upper())
    return out


def _question_count(meta: Dict[str, Any]) -> int:
    summary = meta.get("result_summary") if isinstance(meta, dict) else {}
    if isinstance(summary, dict):
        q = summary.get("question_count")
        if isinstance(q, int) and q > 0:
            return int(q)
    q2 = meta.get("questions_per_test")
    if isinstance(q2, int) and q2 > 0:
        return int(q2)
    try:
        q3 = int(q2)
        return q3 if q3 > 0 else 0
    except (TypeError, ValueError):
        return 0


def _probe_sample_counts(meta: Dict[str, Any]) -> List[Dict[str, int]]:
    probe = meta.get("probe_details") if isinstance(meta, dict) else {}
    if not isinstance(probe, dict):
        return []
    summary = probe.get("summary")
    if not isinstance(summary, dict):
        return []
    per_sample = summary.get("per_sample_label_counts")
    if not isinstance(per_sample, list):
        return []

    out: List[Dict[str, int]] = []
    for item in per_sample:
        if not isinstance(item, dict):
            continue
        counts: Dict[str, int] = {}
        for k, v in item.items():
            if not isinstance(k, str):
                continue
            try:
                counts[k.strip().upper()] = int(v)
            except (TypeError, ValueError):
                continue
        if counts:
            out.append(counts)
    return out


def _pooled_counts(meta: Dict[str, Any], labels: Sequence[str]) -> Dict[str, int]:
    summary = meta.get("result_summary") if isinstance(meta, dict) else {}
    if not isinstance(summary, dict):
        return {}
    pooled = summary.get("pooled_distribution")
    if not isinstance(pooled, dict):
        return {}
    out: Dict[str, int] = {}
    for lab in labels:
        cell = pooled.get(lab) if isinstance(pooled, dict) else None
        if not isinstance(cell, dict):
            out[lab] = 0
            continue
        try:
            out[lab] = int(cell.get("count") or 0)
        except (TypeError, ValueError):
            out[lab] = 0
    return out


def _counts_to_rows(
    *,
    sweep_id: int,
    sweep_name: str,
    run_group: Group,
    run_position: int,
    labels: Sequence[str],
    question_count: int,
    sample_counts: Sequence[Dict[str, int]],
    group_size_answers: int,
) -> List[Dict[str, Any]]:
    """Convert per-sample counts to grouped rows of fixed answer counts."""
    meta = dict(run_group.metadata_json or {})
    run_key = str(meta.get("run_key") or "")
    deployment = str(meta.get("deployment") or "")
    level = str(meta.get("level") or "")
    subject = str(meta.get("subject") or "")
    try:
        options_per_question = int(meta.get("options_per_question") or len(labels))
    except (TypeError, ValueError):
        options_per_question = len(labels)
    try:
        samples_per_combo = int(meta.get("samples_per_combo") or 0)
    except (TypeError, ValueError):
        samples_per_combo = 0

    rows: List[Dict[str, Any]] = []
    current_counts: Counter = Counter()
    current_answers = 0
    group_index = 0

    for sample in sample_counts:
        sample_total = int(sum(int(v) for v in sample.values()))
        if sample_total <= 0:
            continue
        current_counts.update(sample)
        current_answers += sample_total

        if current_answers >= group_size_answers:
            group_index += 1
            row: Dict[str, Any] = {
                "sweep_id": int(sweep_id),
                "sweep_name": sweep_name,
                "run_group_id": int(run_group.id),
                "run_position": int(run_position),
                "run_key": run_key,
                "deployment": deployment,
                "level": level,
                "subject": subject,
                "options_per_question": int(options_per_question),
                "questions_per_test": int(question_count),
                "samples_per_combo": int(samples_per_combo),
                "group_index": int(group_index),
                "group_target_answers": int(group_size_answers),
                "answers_in_group": int(current_answers),
                "group_complete": bool(current_answers == group_size_answers),
                "source": "probe_details",
            }
            for lab in labels:
                row[f"count_{lab}"] = int(current_counts.get(lab, 0))
            rows.append(row)
            current_counts = Counter()
            current_answers = 0

    return rows


def _run_rows(
    *,
    sweep_id: int,
    sweep_name: str,
    run_group: Group,
    run_position: int,
    group_size_answers: int,
) -> List[Dict[str, Any]]:
    meta = dict(run_group.metadata_json or {})
    labels = _labels_from_summary(meta)
    if not labels:
        return []
    q_count = _question_count(meta)
    sample_counts = _probe_sample_counts(meta)
    if sample_counts:
        rows = _counts_to_rows(
            sweep_id=sweep_id,
            sweep_name=sweep_name,
            run_group=run_group,
            run_position=run_position,
            labels=labels,
            question_count=q_count,
            sample_counts=sample_counts,
            group_size_answers=group_size_answers,
        )
        if rows:
            return rows

    # Fallback when per-sample list is absent: emit one pooled row.
    pooled = _pooled_counts(meta, labels)
    run_key = str(meta.get("run_key") or "")
    deployment = str(meta.get("deployment") or "")
    level = str(meta.get("level") or "")
    subject = str(meta.get("subject") or "")
    try:
        options_per_question = int(meta.get("options_per_question") or len(labels))
    except (TypeError, ValueError):
        options_per_question = len(labels)
    try:
        samples_per_combo = int(meta.get("samples_per_combo") or 0)
    except (TypeError, ValueError):
        samples_per_combo = 0

    total_answers = int(sum(int(v) for v in pooled.values()))
    row: Dict[str, Any] = {
        "sweep_id": int(sweep_id),
        "sweep_name": sweep_name,
        "run_group_id": int(run_group.id),
        "run_position": int(run_position),
        "run_key": run_key,
        "deployment": deployment,
        "level": level,
        "subject": subject,
        "options_per_question": int(options_per_question),
        "questions_per_test": int(q_count),
        "samples_per_combo": int(samples_per_combo),
        "group_index": 1,
        "group_target_answers": int(group_size_answers),
        "answers_in_group": int(total_answers),
        "group_complete": bool(total_answers == group_size_answers),
        "source": "result_summary_pooled_distribution",
    }
    for lab in labels:
        row[f"count_{lab}"] = int(pooled.get(lab, 0))
    return [row]


def _ordered_count_columns(rows: Sequence[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for row in rows:
        for k in row.keys():
            if not k.startswith("count_"):
                continue
            lab = k.removeprefix("count_")
            if lab not in labels:
                labels.append(lab)

    # Prefer alphabetical order if labels look like letters.
    labels_sorted = sorted(labels)
    return [f"count_{lab}" for lab in labels_sorted]


def _export_sweep(
    *,
    session,
    sweep_id: int,
    output_dir: Path,
    group_size_answers: int,
) -> Dict[str, Any]:
    sweep = (
        session.query(Group)
        .filter(
            Group.id == int(sweep_id),
            Group.group_type == "mcq_sweep",
        )
        .first()
    )
    if sweep is None:
        raise ValueError(f"sweep_id={sweep_id} not found (group_type=mcq_sweep)")

    sweep_name = str(sweep.name or f"mcq_sweep_{sweep_id}")
    run_pairs = _sweep_runs(session, int(sweep_id))
    if not run_pairs:
        raise ValueError(f"sweep_id={sweep_id} has no linked mcq_run groups")

    rows: List[Dict[str, Any]] = []
    for run_group, run_pos in run_pairs:
        rows.extend(
            _run_rows(
                sweep_id=int(sweep_id),
                sweep_name=sweep_name,
                run_group=run_group,
                run_position=run_pos,
                group_size_answers=group_size_answers,
            )
        )

    count_cols = _ordered_count_columns(rows)
    base_cols = [
        "sweep_id",
        "sweep_name",
        "run_group_id",
        "run_position",
        "run_key",
        "deployment",
        "level",
        "subject",
        "options_per_question",
        "questions_per_test",
        "samples_per_combo",
        "group_index",
        "group_target_answers",
        "answers_in_group",
        "group_complete",
        "source",
    ]
    csv_cols = base_cols + count_cols

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"sweep_id={sweep_id}: no rows produced")
    for col in csv_cols:
        if col not in df.columns:
            df[col] = None
    df = df[csv_cols].sort_values(
        ["run_position", "run_group_id", "group_index"],
        kind="mergesort",
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"{int(sweep_id)}_{_safe_name(sweep_name)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep": {
            "id": int(sweep.id),
            "name": sweep_name,
            "created_at": (
                sweep.created_at.isoformat() if getattr(sweep, "created_at", None) else None
            ),
        },
        "group_size_answers": int(group_size_answers),
        "row_count": int(len(df)),
        "columns": list(csv_cols),
        "rows": df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return {
        "sweep_id": int(sweep.id),
        "sweep_name": sweep_name,
        "rows": int(len(df)),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "run_groups_linked": int(len(run_pairs)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sweep-id",
        type=int,
        action="append",
        default=[],
        help="mcq_sweep group id to export (can be repeated)",
    )
    p.add_argument(
        "--latest",
        type=int,
        default=0,
        help="Additionally export N most recent mcq_sweep groups",
    )
    p.add_argument(
        "--group-size-answers",
        type=int,
        default=1000,
        help="Answers per output row (default 1000 = 20 questions x 50 samples)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV/JSON files",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.group_size_answers < 1:
        print("--group-size-answers must be >= 1", file=sys.stderr)
        return 1

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL environment variable is required", file=sys.stderr)
        return 1

    db = DatabaseConnectionV2(db_url, enable_pgvector=False)
    db.init_db()

    with db.session_scope() as session:
        sweep_ids: List[int] = [int(x) for x in args.sweep_id]
        if int(args.latest) > 0:
            sweep_ids.extend(_latest_mcq_sweep_ids(session, int(args.latest)))
        # Preserve order, remove duplicates.
        seen = set()
        ordered: List[int] = []
        for sid in sweep_ids:
            if sid in seen:
                continue
            seen.add(sid)
            ordered.append(sid)
        if not ordered:
            print("No sweeps selected. Use --sweep-id and/or --latest.", file=sys.stderr)
            return 1

        reports: List[Dict[str, Any]] = []
        for sweep_id in ordered:
            report = _export_sweep(
                session=session,
                sweep_id=int(sweep_id),
                output_dir=args.output_dir.resolve(),
                group_size_answers=int(args.group_size_answers),
            )
            reports.append(report)
            print(
                f"[OK] sweep_id={report['sweep_id']} rows={report['rows']} "
                f"csv={report['csv_path']} json={report['json_path']}"
            )

    summary_path = args.output_dir.resolve() / "last_export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"exports": reports}, f, indent=2, ensure_ascii=False)
    print(f"[OK] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


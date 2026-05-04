#!/usr/bin/env python3
"""Fill missing embedding batches for snapshots using a baseline snapshot's model roster.

Collects distinct (provider, embedding_engine) from ``embedding_batch`` groups whose
``metadata_json`` matches the baseline snapshot's lineage key
``(source_dataframe_group_id, row_count)``, then for each target snapshot computes the
same key and calls :func:`study_query_llm.pipeline.embed.embed` for each missing pair.

When the target snapshot's row_count is strictly less than its source dataframe's
row_count, ``entry_max`` is set to the snapshot row_count (prefix truncation contract
used elsewhere in this repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set, Tuple

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.embed import embed

Pair = Tuple[str, str]
LineageKey = Tuple[int, int]


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if v:
            return v
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")


def _snapshot_lineage(session, snapshot_group_id: int) -> dict[str, Any]:
    g = (
        session.query(Group)
        .filter(
            Group.id == int(snapshot_group_id),
            Group.group_type == "dataset_snapshot",
        )
        .first()
    )
    if g is None:
        raise ValueError(f"dataset_snapshot id={snapshot_group_id} not found")
    md = dict(g.metadata_json or {})
    sdf = int(md.get("source_dataframe_group_id") or 0)
    rows = int(md.get("row_count") or 0)
    if sdf <= 0 or rows <= 0:
        raise ValueError(f"snapshot {snapshot_group_id} missing source_dataframe_group_id/row_count")
    dfg = (
        session.query(Group)
        .filter(Group.id == sdf, Group.group_type == "dataset_dataframe")
        .first()
    )
    df_md = dict((dfg.metadata_json if dfg else {}) or {})
    df_rows = int(df_md.get("row_count") or 0)
    return {
        "snapshot_group_id": int(snapshot_group_id),
        "source_dataframe_group_id": sdf,
        "snapshot_row_count": rows,
        "source_dataframe_row_count": df_rows,
        "lineage_key": (sdf, rows),
    }


def _pairs_for_lineage_key(session, key: LineageKey) -> Set[Pair]:
    sdf_id, entry_max = key
    out: Set[Pair] = set()
    for g in session.query(Group).filter(Group.group_type == "embedding_batch").all():
        md = dict(g.metadata_json or {})
        try:
            ksdf = int(md.get("source_dataframe_group_id") or 0)
            kem = int(md.get("entry_max") or 0)
        except (TypeError, ValueError):
            continue
        if ksdf != sdf_id or kem != entry_max:
            continue
        prov = str(md.get("provider") or "").strip()
        eng = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        if prov and eng:
            out.add((prov, eng))
    return out


def _entry_max_for_embed(lineage: dict[str, Any]) -> int | None:
    snap = int(lineage["snapshot_row_count"])
    df_rows = int(lineage["source_dataframe_row_count"])
    if df_rows > 0 and snap < df_rows:
        return snap
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-snapshot-id", type=int, default=9)
    p.add_argument(
        "--target-snapshot-ids",
        type=int,
        nargs="+",
        required=True,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--embed-timeout-seconds",
        type=float,
        default=7200.0,
        help="Per-embed() timeout (default: 7200 for large frames / slow providers)",
    )
    p.add_argument("--database-url", type=str, default=None)
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional report path (default: experimental_results/embedding_sweeps/fill_from_baseline_report.json)",
    )
    p.add_argument(
        "--providers",
        type=str,
        nargs="*",
        default=None,
        metavar="NAME",
        help=(
            "If set, only consider baseline (and fill) pairs whose provider matches "
            "one of these names (case-insensitive), e.g. `--providers openrouter`."
        ),
    )
    args = p.parse_args(argv)

    database_url = _resolve_database_url(args.database_url)
    conn = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    conn.init_db()

    with conn.session_scope() as session:
        base_lineage = _snapshot_lineage(session, int(args.baseline_snapshot_id))
        baseline_key: LineageKey = base_lineage["lineage_key"]
        want = _pairs_for_lineage_key(session, baseline_key)
        if not want:
            raise SystemExit(
                f"No embedding_batch rows found for baseline key={baseline_key!r}"
            )
        if args.providers:
            allow = {str(p).strip().lower() for p in args.providers if str(p).strip()}
            want = {(a, b) for a, b in want if str(a).strip().lower() in allow}
            if not want:
                raise SystemExit(
                    f"No baseline pairs left after --providers filter={sorted(allow)!r}"
                )

    report: list[dict[str, Any]] = []

    for tid in args.target_snapshot_ids:
        with conn.session_scope() as session:
            lin = _snapshot_lineage(session, int(tid))
            tkey: LineageKey = lin["lineage_key"]
            have = _pairs_for_lineage_key(session, tkey)
        missing = sorted(want - have)
        entry_max = _entry_max_for_embed(lin)
        row: dict[str, Any] = {
            "target_snapshot_group_id": int(tid),
            "lineage_key": [tkey[0], tkey[1]],
            "baseline_pair_count": len(want),
            "existing_pair_count": len(have),
            "missing_pair_count": len(missing),
            "entry_max": entry_max,
            "missing": [{"provider": a, "engine": b} for a, b in missing],
        }
        report.append(row)
        print(
            f"[PLAN] snapshot={tid} key={tkey} have={len(have)}/{len(want)} "
            f"missing={len(missing)} entry_max={entry_max}"
        )
        if args.dry_run or not missing:
            continue
        sdf = int(lin["source_dataframe_group_id"])
        created: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        for prov, engine in missing:
            print(f"[EMBED] snapshot={tid} provider={prov!r} engine={engine!r}")
            try:
                result = embed(
                    int(sdf),
                    deployment=str(engine),
                    provider=str(prov),
                    representation="full",
                    force=bool(args.force),
                    entry_max=entry_max,
                    db=conn,
                    write_intent=WriteIntent.CANONICAL,
                    timeout=float(args.embed_timeout_seconds),
                )
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                print(f"[ERROR] snapshot={tid} provider={prov!r} engine={engine!r}: {err}")
                errors.append({"provider": prov, "engine": engine, "error": err})
                continue
            created.append(
                {
                    "provider": prov,
                    "engine": engine,
                    "group_id": int(result.group_id),
                    "reused": bool((result.metadata or {}).get("reused")),
                }
            )
        row["created"] = created
        row["errors"] = errors

    out_path = Path(
        args.output_json
        if args.output_json
        else str(
            PROJECT_ROOT
            / "experimental_results"
            / "embedding_sweeps"
            / "fill_from_baseline_report.json"
        )
    )
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    if not args.dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[INFO] wrote {out_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))

    had_errors = any(bool(r.get("errors")) for r in report if isinstance(r, dict))
    return 1 if had_errors else 0

if __name__ == "__main__":
    raise SystemExit(main())

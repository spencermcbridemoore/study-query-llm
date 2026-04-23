#!/usr/bin/env python3
"""
Compute size, label cardinality, and text-length percentiles for the
"ready-to-run" snapshots used by the canonical pipeline:

    - bank77                            (default parser, label_mode='labeled')
    - twenty_newsgroups_6cat            (twenty_newsgroups_6cat_subquery_spec())
    - estela                            (default parser, label_mode='all')
    - sources_uncertainty_qc_pm         (pm-only parser, label_mode='labeled')

By default this also includes two research comparators that re-impose the
literature-convention 10 < len(text) <= 1000 window at snapshot time:

    - twenty_newsgroups_6cat_research   (6cat + length window)
    - estela_research                   (length window only)

The script runs each snapshot through acquire -> parse -> snapshot using a
temporary local SQLite + artifact directory, then derives statistics from the
materialized canonical dataframe filtered through the snapshot's
resolved_index. No data is written outside the chosen artifact dir / DB
(temporary by default).

Usage:
    python scripts/snapshot_inventory.py
    python scripts/snapshot_inventory.py --artifact-dir ./_inv_artifacts \
        --db ./_inv.sqlite3 --output-json ./snapshot_inventory.json
    python scripts/snapshot_inventory.py --skip-comparators
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs import (
    BANKING77_DATASET_SLUG,
    ESTELA_DATASET_SLUG,
    SOURCES_UNCERTAINTY_QC_SLUG,
    TWENTY_NEWSGROUPS_6CAT,
    TWENTY_NEWSGROUPS_DATASET_SLUG,
    estela_research_subquery_spec,
    twenty_newsgroups_6cat_subquery_spec,
    twenty_newsgroups_research_subquery_spec,
)
from study_query_llm.datasets.source_specs.parser_protocol import ParserCallable
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.datasets.source_specs.sources_uncertainty_zenodo import (
    parse_sources_uncertainty_pm_snapshot,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri, parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec
from study_query_llm.services.artifact_service import ArtifactService


@dataclass
class SnapshotPlan:
    label: str
    dataset_slug: str
    subquery_spec: SubquerySpec
    parser: Optional[ParserCallable] = None
    parser_id: Optional[str] = None
    parser_version: Optional[str] = None


def build_plans(include_comparators: bool) -> list[SnapshotPlan]:
    plans: list[SnapshotPlan] = [
        SnapshotPlan(
            label="bank77",
            dataset_slug=BANKING77_DATASET_SLUG,
            subquery_spec=SubquerySpec(label_mode="labeled"),
        ),
        SnapshotPlan(
            label="twenty_newsgroups_6cat",
            dataset_slug=TWENTY_NEWSGROUPS_DATASET_SLUG,
            subquery_spec=twenty_newsgroups_6cat_subquery_spec(),
        ),
        SnapshotPlan(
            label="estela",
            dataset_slug=ESTELA_DATASET_SLUG,
            subquery_spec=SubquerySpec(label_mode="all"),
        ),
        SnapshotPlan(
            label="sources_uncertainty_qc_pm",
            dataset_slug=SOURCES_UNCERTAINTY_QC_SLUG,
            subquery_spec=SubquerySpec(label_mode="labeled"),
            parser=parse_sources_uncertainty_pm_snapshot,
            parser_id="sources_uncertainty_qc.pm_only",
            parser_version="v1",
        ),
    ]
    if include_comparators:
        plans.extend(
            [
                SnapshotPlan(
                    label="twenty_newsgroups_6cat_research",
                    dataset_slug=TWENTY_NEWSGROUPS_DATASET_SLUG,
                    subquery_spec=twenty_newsgroups_research_subquery_spec(
                        newsgroups=TWENTY_NEWSGROUPS_6CAT
                    ),
                ),
                SnapshotPlan(
                    label="estela_research",
                    dataset_slug=ESTELA_DATASET_SLUG,
                    subquery_spec=estela_research_subquery_spec(),
                ),
            ]
        )
    return plans


def _load_dataframe(
    db: DatabaseConnectionV2, dataframe_group_id: int, artifact_dir: str
):
    import pandas as pd

    with db.session_scope() as session:
        uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(uri)
    table = pq.read_table(
        source=io.BytesIO(payload),
        columns=["position", "source_id", "text", "label", "label_name", "extra_json"],
    )
    frame = table.to_pandas()
    frame["position"] = frame["position"].astype(int)
    return frame


def _load_resolved_positions(
    snap_artifact_uris: dict[str, str], artifact_dir: str
) -> list[int]:
    uri = snap_artifact_uris.get("subquery_spec.json")
    if not uri:
        raise ValueError("snapshot missing subquery_spec.json artifact uri")
    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    payload = artifact_service.storage.read_from_uri(uri)
    spec_doc = json.loads(payload.decode("utf-8"))
    return [int(item[0]) for item in spec_doc.get("resolved_index", [])]


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=int)
    return {
        "min": float(arr.min()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _harvest(
    plan: SnapshotPlan, db: DatabaseConnectionV2, artifact_dir: str
) -> dict[str, Any]:
    spec = ACQUIRE_REGISTRY[plan.dataset_slug]

    print(f"\n[{plan.label}] acquire ...", flush=True)
    acquired = acquire(spec, db=db, artifact_dir=artifact_dir)
    print(
        f"[{plan.label}] acquire group_id={acquired.group_id} "
        f"reused={acquired.metadata.get('reused')}",
        flush=True,
    )

    print(f"[{plan.label}] parse ...", flush=True)
    parsed = parse(
        acquired.group_id,
        parser=plan.parser,
        parser_id=plan.parser_id,
        parser_version=plan.parser_version,
        db=db,
        artifact_dir=artifact_dir,
    )
    print(
        f"[{plan.label}] parse group_id={parsed.group_id} "
        f"reused={parsed.metadata.get('reused')}",
        flush=True,
    )

    print(f"[{plan.label}] snapshot ...", flush=True)
    snapped = snapshot(
        parsed.group_id,
        subquery_spec=plan.subquery_spec,
        db=db,
        artifact_dir=artifact_dir,
    )
    print(
        f"[{plan.label}] snapshot group_id={snapped.group_id} "
        f"reused={snapped.metadata.get('reused')} "
        f"row_count={snapped.metadata.get('row_count')}",
        flush=True,
    )

    positions = _load_resolved_positions(snapped.artifact_uris, artifact_dir)
    df = _load_dataframe(db, parsed.group_id, artifact_dir)
    sub = df[df["position"].isin(positions)]
    text_lens = [len(t) for t in sub["text"].astype(str).tolist()]
    labels = [int(x) for x in sub["label"].dropna().tolist()]
    n_categories = len(set(labels))
    label_names = sorted(
        str(name)
        for name in sub["label_name"].dropna().astype(str).unique().tolist()
    )

    return {
        "label": plan.label,
        "dataset_slug": plan.dataset_slug,
        "row_count": int(len(positions)),
        "n_categories": int(n_categories),
        "label_names": label_names,
        "spec_hash": snapped.metadata.get("spec_hash"),
        "lengths": _percentiles(text_lens),
    }


def _render_markdown_table(results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(
        "| Snapshot | Rows | Cats | min | p1 | p5 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |"
    )
    lines.append("|" + "---|" * 14)
    for r in results:
        L = r.get("lengths") or {}
        if not L:
            lines.append(
                f"| {r['label']} | {r['row_count']} | {r['n_categories']} | "
                f"(empty) |  |  |  |  |  |  |  |  |  |  |"
            )
            continue
        lines.append(
            f"| {r['label']} | {r['row_count']} | {r['n_categories']} | "
            f"{int(L['min'])} | {int(L['p1'])} | {int(L['p5'])} | "
            f"{int(L['p25'])} | {int(L['p50'])} | {int(L['p75'])} | "
            f"{int(L['p90'])} | {int(L['p95'])} | {int(L['p99'])} | "
            f"{int(L['max'])} | {L['mean']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory canonical snapshots: size, labels, char-length percentiles."
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Artifact directory (default: temporary).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite path for provenance DB (default: temporary file in tmp dir).",
    )
    parser.add_argument(
        "--skip-comparators",
        action="store_true",
        help="Skip the twenty_newsgroups_6cat_research and estela_research comparator snapshots.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the full results as JSON.",
    )
    args = parser.parse_args()

    plans = build_plans(include_comparators=not args.skip_comparators)
    os.environ["ARTIFACT_STORAGE_BACKEND"] = "local"
    os.environ["ARTIFACT_STORAGE_STRICT_MODE"] = "0"
    os.environ["ARTIFACT_RUNTIME_ENV"] = "dev"

    with tempfile.TemporaryDirectory(prefix="snapshot_inventory_") as tmpdir:
        artifact_dir = (
            args.artifact_dir
            if args.artifact_dir
            else os.path.join(tmpdir, "artifacts")
        )
        os.makedirs(artifact_dir, exist_ok=True)
        db_path = (
            args.db if args.db else os.path.join(tmpdir, "snapshot_inventory.sqlite3")
        )
        db_url = f"sqlite:///{Path(db_path).resolve().as_posix()}"
        print(f"[setup] artifact_dir = {artifact_dir}", flush=True)
        print(f"[setup] db_url       = {db_url}", flush=True)

        db = DatabaseConnectionV2(db_url, enable_pgvector=False)
        db.init_db()

        results: list[dict[str, Any]] = []
        failures: list[tuple[str, str]] = []
        for plan in plans:
            try:
                results.append(_harvest(plan, db, artifact_dir))
            except Exception as exc:
                print(f"\n[{plan.label}] FAILED: {exc}", flush=True)
                traceback.print_exc()
                failures.append((plan.label, str(exc)))

        print("\n" + "=" * 100)
        print("SNAPSHOT INVENTORY (character-length percentiles)")
        print("=" * 100 + "\n")
        print(_render_markdown_table(results))
        print()

        if failures:
            print("FAILURES:")
            for label, msg in failures:
                print(f"  - {label}: {msg}")
            print()

        if args.output_json:
            Path(args.output_json).write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[ok] wrote JSON to {args.output_json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

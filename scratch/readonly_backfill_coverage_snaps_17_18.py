"""Read-only: backfill provenance coverage for snapshots 17/18 vs manifest batch selection.

Manifest logic (clustering_analysis_backfill.build_manifest):
- Exactly one embedding_batch per (snapshot, engine, provider) -> use that batch id.
- Zero batches -> no_embedding_batch.
- Multiple batches -> collision_multiple_batches (no targets; script aborts on --execute).

Uses provenanced_runs.run_key LIKE backfill_exec__snap{sid}__emb{bid}__%
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
env = dotenv_values(ROOT / ".env")
url = (
    os.environ.get("CANONICAL_DATABASE_URL")
    or os.environ.get("DATABASE_URL")
    or env.get("CANONICAL_DATABASE_URL")
    or env.get("DATABASE_URL")
    or ""
).strip()
if not url:
    print("No database URL", file=sys.stderr)
    sys.exit(1)

SQL = """
WITH snaps AS (
  SELECT id AS snapshot_id,
         (metadata_json->>'source_dataframe_group_id')::int AS sdf_id,
         (metadata_json->>'row_count')::int AS entry_max
  FROM groups
  WHERE group_type = 'dataset_snapshot'
    AND id IN (17, 18)
),
candidates AS (
  SELECT s.snapshot_id,
         g.id AS embedding_batch_id,
         g.metadata_json->>'embedding_engine' AS embedding_engine
  FROM snaps s
  JOIN groups g
    ON g.group_type = 'embedding_batch'
   AND (g.metadata_json->>'source_dataframe_group_id')::int = s.sdf_id
   AND (g.metadata_json->>'entry_max')::int = s.entry_max
   AND lower(coalesce(g.metadata_json->>'provider', '')) = 'openrouter'
),
agg AS (
  SELECT snapshot_id,
         embedding_engine,
         COUNT(*)::int AS n_batches,
         MIN(embedding_batch_id) AS min_batch_id,
         MAX(embedding_batch_id) AS max_batch_id,
         ARRAY_AGG(embedding_batch_id ORDER BY embedding_batch_id) AS batch_ids
  FROM candidates
  GROUP BY snapshot_id, embedding_engine
),
resolved AS (
  SELECT snapshot_id,
         embedding_engine,
         CASE WHEN n_batches = 1 THEN min_batch_id ELSE NULL END AS selected_batch_id,
         n_batches,
         batch_ids
  FROM agg
),
touch AS (
  SELECT r.snapshot_id,
         r.embedding_engine,
         r.selected_batch_id,
         r.n_batches,
         r.batch_ids,
         COALESCE(
           (
             SELECT COUNT(*)::int
             FROM provenanced_runs pr
             WHERE pr.run_kind = 'execution'
               AND r.selected_batch_id IS NOT NULL
               AND pr.run_key LIKE 'backfill_exec__snap' || r.snapshot_id::text
                     || '__emb' || r.selected_batch_id::text || '__%'
           ),
           0
         ) AS backfill_pr_rows
  FROM resolved r
)
SELECT * FROM touch
ORDER BY embedding_engine, snapshot_id;
"""

OUT_CSV = ROOT / "scratch" / "backfill_coverage_snaps_17_18_readonly.csv"


def main() -> int:
    eng = create_engine(url, pool_pre_ping=True)
    with eng.connect() as conn:
        rows = conn.execute(text(SQL)).mappings().all()

    collisions = [r for r in rows if int(r["n_batches"]) > 1]
    singles = [r for r in rows if int(r["n_batches"]) == 1]
    untouched = [r for r in singles if int(r["backfill_pr_rows"]) == 0]

    print("total_snapshot_engine_rows", len(rows))
    print("collision_rows_manifest_would_skip", len(collisions))
    print("single_batch_rows", len(singles))
    print("single_batch_zero_provenance_rows", len(untouched))
    print("--- collisions (first 30) ---")
    for r in collisions[:30]:
        print(
            dict(
                snapshot_id=r["snapshot_id"],
                embedding_engine=r["embedding_engine"],
                n_batches=r["n_batches"],
                batch_ids=r["batch_ids"],
            )
        )
    if len(collisions) > 30:
        print("... truncated ...")
    print("--- engines with single batch AND zero backfill provenance (any snap) ---")
    engines_u = sorted({r["embedding_engine"] for r in untouched if r.get("embedding_engine")})
    for e in engines_u:
        print(e)
    print("distinct_engine_count_untouched_single_batch", len(engines_u))
    print("--- full untouched single-batch rows ---")
    for r in untouched:
        print(dict(r))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot_id",
                "embedding_engine",
                "selected_batch_id",
                "n_batches",
                "backfill_pr_rows",
                "batch_ids",
            ],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "snapshot_id": r["snapshot_id"],
                    "embedding_engine": r["embedding_engine"],
                    "selected_batch_id": r["selected_batch_id"],
                    "n_batches": r["n_batches"],
                    "backfill_pr_rows": r["backfill_pr_rows"],
                    "batch_ids": str(r["batch_ids"]),
                }
            )
    print("wrote", OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

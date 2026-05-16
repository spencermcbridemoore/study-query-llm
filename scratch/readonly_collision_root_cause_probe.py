"""Read-only probe: investigate collision_multiple_batches root causes.

Outputs a JSON report under scratch/exports with:
- sampled collision pair evidence across snapshots 6/9/10
- one non-collision comparison case from snapshot 17
- full 63-pair collision bucket tally

This script performs SELECT-only queries.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "experimental_results" / "backfill_per_pair" / "20260514T085641Z"
PAIR_CLASSIFICATION_PATH = ARTIFACT_DIR / "pair_classification.json"
INFEASIBLE_PATH = ARTIFACT_DIR / "infeasible_pairs.json"

# Requested sampling: 2-3 pairs per research snapshot; use 3 deterministic engines.
SAMPLED_ENGINES = (
    "baai/bge-base-en-v1.5",
    "openai/text-embedding-3-large",
    "perplexity/pplx-embed-v1-0.6b",
)
RESEARCH_SNAPSHOTS = (6, 9, 10)
COMPARISON_SNAPSHOT = 17
COMPARISON_ENGINE = "openai/text-embedding-3-large"

LEGACY_METHOD_NAMES = (
    "hdbscan",
    "kmeans+silhouette+kneedle",
    "gmm+bic+argmin",
)


def _load_database_url() -> str:
    env = dotenv_values(ROOT / ".env")
    url = (
        os.environ.get("DATABASE_URL")
        or env.get("DATABASE_URL")
        or os.environ.get("CANONICAL_DATABASE_URL")
        or env.get("CANONICAL_DATABASE_URL")
        or ""
    ).strip()
    if not url:
        raise RuntimeError("DATABASE_URL / CANONICAL_DATABASE_URL not found")
    return url


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return parsed


def _snapshot_lineage(conn, snapshot_id: int) -> dict[str, Any]:
    sql = text(
        """
        SELECT
            s.id AS snapshot_id,
            s.name AS snapshot_name,
            (s.metadata_json->>'source_dataframe_group_id')::int AS source_dataframe_group_id,
            (s.metadata_json->>'row_count')::int AS snapshot_row_count,
            (df.metadata_json->>'row_count')::int AS source_dataframe_row_count
        FROM groups s
        LEFT JOIN groups df
          ON df.id = (s.metadata_json->>'source_dataframe_group_id')::int
         AND df.group_type = 'dataset_dataframe'
        WHERE s.group_type = 'dataset_snapshot'
          AND s.id = :snapshot_id
        LIMIT 1
        """
    )
    row = conn.execute(sql, {"snapshot_id": int(snapshot_id)}).mappings().first()
    if row is None:
        raise RuntimeError(f"dataset_snapshot {snapshot_id} not found")
    return {
        "snapshot_id": int(row["snapshot_id"]),
        "snapshot_name": str(row.get("snapshot_name") or ""),
        "source_dataframe_group_id": int(row["source_dataframe_group_id"]),
        "snapshot_row_count": int(row["snapshot_row_count"]),
        "source_dataframe_row_count": int(row["source_dataframe_row_count"]),
    }


def _matching_batches_for_pair(
    conn,
    *,
    source_dataframe_group_id: int,
    source_dataframe_row_count: int,
    embedding_engine: str,
    provider: str | None,
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT
            g.id,
            g.created_at,
            g.name,
            (g.metadata_json->>'entry_max')::int AS entry_max,
            COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', '') AS embedding_engine,
            COALESCE(g.metadata_json->>'provider', '') AS provider,
            COALESCE(g.metadata_json->>'representation', '') AS representation,
            COALESCE(g.metadata_json->>'dataset_key', '') AS dataset_key,
            CASE
              WHEN COALESCE(g.metadata_json->>'dimension', '') ~ '^-?[0-9]+$'
              THEN (g.metadata_json->>'dimension')::int
              ELSE NULL
            END AS dimension,
            COALESCE(g.metadata_json->>'key_version', '') AS key_version
        FROM groups g
        WHERE g.group_type = 'embedding_batch'
          AND (g.metadata_json->>'source_dataframe_group_id')::int = :sdf_id
          AND (g.metadata_json->>'entry_max')::int = :entry_max
          AND COALESCE(g.metadata_json->>'embedding_engine', g.metadata_json->>'deployment', '') = :embedding_engine
          AND (
            :provider IS NULL
            OR lower(COALESCE(g.metadata_json->>'provider', '')) = lower(:provider)
          )
        ORDER BY g.id ASC
        """
    )
    rows = conn.execute(
        sql,
        {
            "sdf_id": int(source_dataframe_group_id),
            "entry_max": int(source_dataframe_row_count),
            "embedding_engine": str(embedding_engine),
            "provider": provider,
        },
    ).mappings().all()
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "id": int(row["id"]),
                "created_at": (
                    row["created_at"].isoformat()
                    if hasattr(row.get("created_at"), "isoformat")
                    else str(row.get("created_at") or "")
                ),
                "name": str(row.get("name") or ""),
                "entry_max": int(row["entry_max"]),
                "embedding_engine": str(row.get("embedding_engine") or ""),
                "provider": str(row.get("provider") or ""),
                "representation": str(row.get("representation") or ""),
                "dataset_key": str(row.get("dataset_key") or ""),
                "dimension": (int(row["dimension"]) if row.get("dimension") is not None else None),
                "key_version": str(row.get("key_version") or ""),
            }
        )
    return out


def _legacy_prior_run_usage(
    conn,
    *,
    snapshot_id: int,
    embedding_engine: str,
    candidate_batch_ids: list[int],
) -> dict[str, Any]:
    if not candidate_batch_ids:
        return {"rows": [], "distinct_used_batch_ids": []}

    sql = text(
        """
        WITH run_rows AS (
            SELECT
                pr.id AS provenanced_run_id,
                pr.created_at AS created_at,
                COALESCE(md.name, pr.metadata_json->>'analysis_key', '') AS method_name,
                COALESCE(
                  (pr.metadata_json->>'embedding_batch_group_id')::int,
                  (pr.config_json->>'embedding_batch_group_id')::int,
                  pr.source_group_id
                ) AS used_batch_id
            FROM provenanced_runs pr
            LEFT JOIN method_definitions md
              ON md.id = pr.method_definition_id
            WHERE pr.input_snapshot_group_id = :snapshot_id
              AND pr.run_status = 'completed'
              AND (
                pr.run_kind = 'analysis_execution'
                OR (
                  pr.run_kind = 'execution'
                  AND COALESCE(pr.metadata_json->>'execution_role', '') = 'analysis_execution'
                )
              )
              AND pr.run_key NOT LIKE 'backfill_exec__%%'
              AND COALESCE(md.name, pr.metadata_json->>'analysis_key', '') = ANY(:legacy_methods)
        )
        SELECT
            rr.provenanced_run_id,
            rr.created_at,
            rr.method_name,
            rr.used_batch_id
        FROM run_rows rr
        JOIN groups eb
          ON eb.id = rr.used_batch_id
         AND eb.group_type = 'embedding_batch'
        WHERE rr.used_batch_id = ANY(:candidate_batch_ids)
          AND COALESCE(eb.metadata_json->>'embedding_engine', eb.metadata_json->>'deployment', '') = :embedding_engine
        ORDER BY rr.created_at ASC, rr.provenanced_run_id ASC
        """
    )
    rows = conn.execute(
        sql,
        {
            "snapshot_id": int(snapshot_id),
            "legacy_methods": list(LEGACY_METHOD_NAMES),
            "candidate_batch_ids": [int(x) for x in candidate_batch_ids],
            "embedding_engine": str(embedding_engine),
        },
    ).mappings().all()

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        out_rows.append(
            {
                "provenanced_run_id": int(row["provenanced_run_id"]),
                "created_at": (
                    row["created_at"].isoformat()
                    if hasattr(row.get("created_at"), "isoformat")
                    else str(row.get("created_at") or "")
                ),
                "method_name": str(row.get("method_name") or ""),
                "used_batch_id": int(row["used_batch_id"]),
            }
        )
    distinct_batches = sorted({int(r["used_batch_id"]) for r in out_rows})
    return {
        "rows": out_rows,
        "distinct_used_batch_ids": distinct_batches,
    }


def _classify_collision_bucket(batches: list[dict[str, Any]]) -> str:
    if len(batches) <= 1:
        return "not_collision"
    providers = {str(b.get("provider") or "") for b in batches}
    representations = {str(b.get("representation") or "") for b in batches}
    dimensions = {b.get("dimension") for b in batches}
    dataset_keys = {str(b.get("dataset_key") or "") for b in batches}
    key_versions = {str(b.get("key_version") or "") for b in batches}

    # Bucket precedence: stale lineage (different data identity) before config drift.
    # Filter already locks source_dataframe + entry_max, so stale-lineage should be absent.
    if len(dataset_keys) > 1 and all(k for k in dataset_keys):
        return "f_other_dataset_key_drift"
    if len(providers) > 1:
        return "b_provider_drift"
    if len(representations) > 1:
        return "c_representation_drift"
    if len(dimensions) > 1:
        return "d_dimension_drift"
    if len(key_versions) > 1:
        return "f_other_key_version_drift"
    return "a_reembed_same_key"


def _pair_index(pair_rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    idx: dict[tuple[int, str], dict[str, Any]] = {}
    for row in pair_rows:
        snapshot_id = int(row["snapshot_id"])
        engine = str(row["embedding_engine"])
        idx[(snapshot_id, engine)] = row
    return idx


def main() -> int:
    classification = _load_json(PAIR_CLASSIFICATION_PATH)
    infeasible = _load_json(INFEASIBLE_PATH)
    pair_rows = list(classification.get("pairs") or [])
    infeasible_rows = list(infeasible.get("pairs") or [])
    index = _pair_index(pair_rows)

    sampled_pairs: list[dict[str, Any]] = []
    for snapshot_id in RESEARCH_SNAPSHOTS:
        for engine in SAMPLED_ENGINES:
            row = index.get((int(snapshot_id), str(engine)))
            if row is None:
                raise RuntimeError(f"Missing pair row for snapshot={snapshot_id}, engine={engine}")
            sampled_pairs.append(
                {
                    "snapshot_id": int(snapshot_id),
                    "embedding_engine": str(engine),
                    "provider": str(row.get("provider") or "") or None,
                    "pair_id": str(row.get("pair_id") or ""),
                    "reason_code": str(row.get("reason_code") or ""),
                }
            )

    comparison_pair_row = index.get((COMPARISON_SNAPSHOT, COMPARISON_ENGINE))
    if comparison_pair_row is None:
        raise RuntimeError(
            f"Missing comparison pair for snapshot={COMPARISON_SNAPSHOT}, engine={COMPARISON_ENGINE}"
        )
    comparison_pair = {
        "snapshot_id": int(COMPARISON_SNAPSHOT),
        "embedding_engine": str(COMPARISON_ENGINE),
        "provider": str(comparison_pair_row.get("provider") or "") or None,
        "pair_id": str(comparison_pair_row.get("pair_id") or ""),
        "reason_code": str(comparison_pair_row.get("reason_code") or ""),
        "classification": str(comparison_pair_row.get("classification") or ""),
        "manifest_status": str(comparison_pair_row.get("manifest_status") or ""),
    }

    db_url = _load_database_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    sampled_evidence: list[dict[str, Any]] = []
    comparison_evidence: dict[str, Any] = {}

    # Cache batch lookups so tally and evidence share the same underlying metadata.
    collision_batch_cache: dict[tuple[int, int, str, str | None], list[dict[str, Any]]] = {}
    legacy_usage_cache: dict[tuple[int, str, tuple[int, ...]], dict[str, Any]] = {}

    with engine.connect() as conn:
        for pair in sampled_pairs:
            lineage = _snapshot_lineage(conn, int(pair["snapshot_id"]))
            cache_key = (
                int(lineage["source_dataframe_group_id"]),
                int(lineage["source_dataframe_row_count"]),
                str(pair["embedding_engine"]),
                pair.get("provider"),
            )
            if cache_key not in collision_batch_cache:
                collision_batch_cache[cache_key] = _matching_batches_for_pair(
                    conn,
                    source_dataframe_group_id=int(lineage["source_dataframe_group_id"]),
                    source_dataframe_row_count=int(lineage["source_dataframe_row_count"]),
                    embedding_engine=str(pair["embedding_engine"]),
                    provider=pair.get("provider"),
                )
            batches = collision_batch_cache[cache_key]
            usage = _legacy_prior_run_usage(
                conn,
                snapshot_id=int(pair["snapshot_id"]),
                embedding_engine=str(pair["embedding_engine"]),
                candidate_batch_ids=[int(b["id"]) for b in batches],
            )
            sampled_evidence.append(
                {
                    **pair,
                    "snapshot_lineage": lineage,
                    "matching_batches": batches,
                    "legacy_prior_run_usage": usage,
                    "collision_bucket": _classify_collision_bucket(batches),
                }
            )

        comp_lineage = _snapshot_lineage(conn, int(comparison_pair["snapshot_id"]))
        comp_cache_key = (
            int(comp_lineage["source_dataframe_group_id"]),
            int(comp_lineage["source_dataframe_row_count"]),
            str(comparison_pair["embedding_engine"]),
            comparison_pair.get("provider"),
        )
        if comp_cache_key not in collision_batch_cache:
            collision_batch_cache[comp_cache_key] = _matching_batches_for_pair(
                conn,
                source_dataframe_group_id=int(comp_lineage["source_dataframe_group_id"]),
                source_dataframe_row_count=int(comp_lineage["source_dataframe_row_count"]),
                embedding_engine=str(comparison_pair["embedding_engine"]),
                provider=comparison_pair.get("provider"),
            )
        comp_batches = collision_batch_cache[comp_cache_key]
        comp_usage = _legacy_prior_run_usage(
            conn,
            snapshot_id=int(comparison_pair["snapshot_id"]),
            embedding_engine=str(comparison_pair["embedding_engine"]),
            candidate_batch_ids=[int(b["id"]) for b in comp_batches],
        )
        comparison_evidence = {
            **comparison_pair,
            "snapshot_lineage": comp_lineage,
            "matching_batches": comp_batches,
            "legacy_prior_run_usage": comp_usage,
            "collision_bucket": _classify_collision_bucket(comp_batches),
        }

        # Full infeasible tally by bucket across all 63 collision pairs.
        collision_pairs = [
            p
            for p in infeasible_rows
            if str(p.get("reason_code") or "") == "collision_multiple_batches"
        ]
        bucket_counts: Counter[str] = Counter()
        bucket_examples: dict[str, list[str]] = {}
        usage_profile_counter: Counter[str] = Counter()
        latest_alignment_counter: Counter[str] = Counter()
        usage_profile_examples: dict[str, list[str]] = {}
        latest_alignment_examples: dict[str, list[str]] = {}

        for row in collision_pairs:
            sdf_id = int(row["source_dataframe_group_id"])
            sdf_rows = int(row["source_dataframe_row_count"])
            emb_engine = str(row["embedding_engine"])
            provider = str(row.get("provider") or "") or None
            cache_key = (sdf_id, sdf_rows, emb_engine, provider)
            if cache_key not in collision_batch_cache:
                collision_batch_cache[cache_key] = _matching_batches_for_pair(
                    conn,
                    source_dataframe_group_id=sdf_id,
                    source_dataframe_row_count=sdf_rows,
                    embedding_engine=emb_engine,
                    provider=provider,
                )
            batches = collision_batch_cache[cache_key]
            bucket = _classify_collision_bucket(batches)
            bucket_counts[bucket] += 1
            if bucket not in bucket_examples:
                bucket_examples[bucket] = []
            if len(bucket_examples[bucket]) < 5:
                bucket_examples[bucket].append(str(row.get("pair_id") or ""))

            candidate_ids = tuple(int(b["id"]) for b in batches)
            usage_cache_key = (int(row["snapshot_id"]), emb_engine, candidate_ids)
            if usage_cache_key not in legacy_usage_cache:
                legacy_usage_cache[usage_cache_key] = _legacy_prior_run_usage(
                    conn,
                    snapshot_id=int(row["snapshot_id"]),
                    embedding_engine=emb_engine,
                    candidate_batch_ids=list(candidate_ids),
                )
            usage = legacy_usage_cache[usage_cache_key]
            used_ids = list(usage.get("distinct_used_batch_ids") or [])
            if len(used_ids) == 0:
                profile_key = "no_prior_legacy_runs"
            elif len(used_ids) == 1:
                profile_key = "one_prior_used_batch"
            else:
                profile_key = "multiple_prior_used_batches"
            usage_profile_counter[profile_key] += 1
            usage_profile_examples.setdefault(profile_key, [])
            if len(usage_profile_examples[profile_key]) < 5:
                usage_profile_examples[profile_key].append(str(row.get("pair_id") or ""))

            if len(used_ids) == 1 and candidate_ids:
                alignment_key = (
                    "prior_used_batch_is_latest_candidate"
                    if int(used_ids[0]) == int(max(candidate_ids))
                    else "prior_used_batch_not_latest_candidate"
                )
                latest_alignment_counter[alignment_key] += 1
                latest_alignment_examples.setdefault(alignment_key, [])
                if len(latest_alignment_examples[alignment_key]) < 5:
                    latest_alignment_examples[alignment_key].append(
                        str(row.get("pair_id") or "")
                    )

    total_collision_pairs = int(sum(bucket_counts.values()))
    bucket_tally = []
    for bucket_name, count in sorted(bucket_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        pct = (100.0 * float(count) / float(total_collision_pairs)) if total_collision_pairs else 0.0
        bucket_tally.append(
            {
                "bucket": bucket_name,
                "pair_count": int(count),
                "fraction_of_63": round(pct / 100.0, 4),
                "percent_of_63": round(pct, 2),
                "example_pair_ids": bucket_examples.get(bucket_name, []),
            }
        )

    by_snapshot = Counter(
        int(row["snapshot_id"])
        for row in infeasible_rows
        if str(row.get("reason_code") or "") == "collision_multiple_batches"
    )
    collisions_by_snapshot = {str(k): int(v) for k, v in sorted(by_snapshot.items())}

    out = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_dir": str(ARTIFACT_DIR),
        "sampled_pairs": sampled_pairs,
        "sampled_evidence": sampled_evidence,
        "comparison_evidence": comparison_evidence,
        "collision_tally": {
            "total_collision_pairs": total_collision_pairs,
            "collisions_by_snapshot": collisions_by_snapshot,
            "bucket_tally": bucket_tally,
            "legacy_usage_profile": {
                "counts": {k: int(v) for k, v in sorted(usage_profile_counter.items())},
                "examples": usage_profile_examples,
            },
            "latest_candidate_alignment": {
                "counts": {k: int(v) for k, v in sorted(latest_alignment_counter.items())},
                "examples": latest_alignment_examples,
            },
        },
        "notes": {
            "legacy_methods_for_prior_usage": list(LEGACY_METHOD_NAMES),
            "comparison_pair": comparison_pair,
            "selection_policy": "deterministic fixed engines across snapshots 6/9/10",
        },
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "scratch" / "exports" / f"collision_probe_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "collision_probe.json"
    out_path.write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT_JSON={out_path}")
    print(f"TOTAL_COLLISION_PAIRS={total_collision_pairs}")
    for row in bucket_tally:
        print(f"BUCKET {row['bucket']} {row['pair_count']} {row['percent_of_63']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

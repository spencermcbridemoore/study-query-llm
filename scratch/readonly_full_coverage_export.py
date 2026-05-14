"""Read-only export of clustering analysis coverage + per-cell metrics.

Outputs:
- scratch/exports/full_coverage_<UTC-yyyymmddTHHMMSSZ>/coverage_summary.md
- scratch/exports/full_coverage_<UTC-yyyymmddTHHMMSSZ>/analysis_metrics_per_cell.csv
- scratch/exports/full_coverage_<UTC-yyyymmddTHHMMSSZ>/analysis_labels_long.csv (or split files)
- scratch/exports/full_coverage_<UTC-yyyymmddTHHMMSSZ>/README.md
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from sqlalchemy import create_engine, text

from study_query_llm.pipeline.clustering.registry import iter_algorithm_specs

SNAPSHOT_SCOPE: dict[int, str] = {
    6: "sources_uncertainty_qc",
    9: "estela",
    10: "estela research",
    17: "banking77 contrast min l6",
    18: "banking77 contrast max l6",
}
SNAPSHOT_IDS = sorted(SNAPSHOT_SCOPE.keys())
LEGACY_METHOD_CANDIDATES = (
    "hdbscan",
    "kmeans+silhouette+kneedle",
    "gmm+bic+argmin",
)
MAX_LABEL_FILE_BYTES = 500 * 1024 * 1024


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isfinite(out):
            return out
        return None
    except (TypeError, ValueError):
        return None


def _normalize_col(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "metric"


def _flatten_numeric(
    value: Any,
    *,
    prefix: str,
    out: dict[str, float],
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        fv = _safe_float(value)
        if fv is not None:
            out[prefix] = fv
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}__{_normalize_col(str(key))}"
            _flatten_numeric(child, prefix=child_key, out=out)
        return
    # Lists are intentionally skipped to avoid exploding wide columns
    # (e.g., cluster label arrays, selection curve points).


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or "host"
        port = f":{parts.port}" if parts.port else ""
        userinfo = "***:***@"
        netloc = f"{userinfo}{hostname}{port}"
        path = parts.path or ""
        return urlunsplit((parts.scheme, netloc, path, "", ""))
    except Exception:
        return "<redacted>"


def _load_database_url(root: Path) -> str:
    env = dotenv_values(root / ".env")
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


def _fetch_snapshot_rows(conn) -> dict[int, dict[str, Any]]:
    sql = text(
        """
        SELECT
            g.id AS snapshot_id,
            g.name AS snapshot_name,
            g.metadata_json AS metadata_json
        FROM groups g
        WHERE g.group_type = 'dataset_snapshot'
          AND g.id = ANY(:snapshot_ids)
        ORDER BY g.id ASC
        """
    )
    rows = conn.execute(sql, {"snapshot_ids": SNAPSHOT_IDS}).mappings().all()
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        snap_id = int(row["snapshot_id"])
        md = dict(row["metadata_json"] or {})
        out[snap_id] = {
            "snapshot_id": snap_id,
            "snapshot_name": str(row.get("snapshot_name") or f"snapshot_{snap_id}"),
            "dataset_slug": str(md.get("dataset_slug") or ""),
            "row_count": _safe_int(md.get("row_count")) or 0,
            "source_dataframe_group_id": _safe_int(md.get("source_dataframe_group_id")) or 0,
            "scope_label": SNAPSHOT_SCOPE.get(snap_id, ""),
        }
    missing = sorted(set(SNAPSHOT_IDS) - set(out.keys()))
    if missing:
        raise RuntimeError(f"Missing dataset_snapshot ids in DB: {missing}")
    return out


def _fetch_embedding_batches(conn, source_dataframe_ids: set[int]) -> dict[int, dict[str, Any]]:
    # Pull all embedding_batch metadata once; filter in Python for robust parsing.
    sql = text(
        """
        SELECT
            g.id AS group_id,
            g.metadata_json AS metadata_json
        FROM groups g
        WHERE g.group_type = 'embedding_batch'
        ORDER BY g.id ASC
        """
    )
    rows = conn.execute(sql).mappings().all()
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        gid = int(row["group_id"])
        md = dict(row["metadata_json"] or {})
        sdf_id = _safe_int(md.get("source_dataframe_group_id"))
        if sdf_id is None or sdf_id not in source_dataframe_ids:
            continue
        engine = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        provider = str(md.get("provider") or md.get("embedding_provider") or "").strip()
        dim = _safe_int(md.get("dimension"))
        out[gid] = {
            "embedding_batch_group_id": gid,
            "source_dataframe_group_id": int(sdf_id),
            "embedding_engine": engine,
            "embedding_provider": provider,
            "embedding_dim": dim,
        }
    return out


def _fetch_completed_analysis_runs(conn) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT
            pr.id AS provenanced_run_id,
            pr.run_kind AS run_kind,
            pr.run_status AS run_status,
            pr.run_key AS analysis_run_key,
            pr.input_snapshot_group_id AS snapshot_id,
            pr.source_group_id AS source_group_id,
            pr.method_definition_id AS method_definition_id,
            pr.config_json AS config_json,
            pr.metadata_json AS metadata_json,
            pr.updated_at AS updated_at,
            md.name AS method_name,
            md.version AS method_version
        FROM provenanced_runs pr
        LEFT JOIN method_definitions md
          ON md.id = pr.method_definition_id
        WHERE pr.input_snapshot_group_id = ANY(:snapshot_ids)
          AND pr.run_status = 'completed'
          AND (
            pr.run_kind = 'analysis_execution'
            OR (
              pr.run_kind = 'execution'
              AND COALESCE(pr.metadata_json->>'execution_role', '') = 'analysis_execution'
            )
          )
        ORDER BY pr.id ASC
        """
    )
    rows = conn.execute(sql, {"snapshot_ids": SNAPSHOT_IDS}).mappings().all()
    return [dict(r) for r in rows]


def _fetch_analysis_results_for_groups(
    conn,
    analysis_group_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not analysis_group_ids:
        return out
    ids = sorted(analysis_group_ids)
    chunk_size = 1000
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        sql = text(
            """
            SELECT
                ar.id AS analysis_result_id,
                ar.analysis_group_id AS analysis_group_id,
                ar.result_key AS result_key,
                ar.result_value AS result_value,
                ar.result_json AS result_json
            FROM analysis_results ar
            WHERE ar.analysis_group_id = ANY(:group_ids)
            ORDER BY ar.id ASC
            """
        )
        rows = conn.execute(sql, {"group_ids": chunk}).mappings().all()
        for row in rows:
            gid = _safe_int(row.get("analysis_group_id"))
            if gid is None:
                continue
            out[int(gid)].append(dict(row))
    return out


def _extract_method_name(run: dict[str, Any]) -> str:
    md = dict(run.get("metadata_json") or {})
    method_name = str(run.get("method_name") or "").strip()
    if method_name:
        return method_name
    return str(md.get("analysis_key") or "").strip()


def _resolve_embedding_batch_id(run: dict[str, Any], embedding_batch_ids: set[int]) -> int | None:
    md = dict(run.get("metadata_json") or {})
    cfg = dict(run.get("config_json") or {})
    candidates = [
        _safe_int(md.get("embedding_batch_group_id")),
        _safe_int(cfg.get("embedding_batch_group_id")),
        _safe_int(run.get("source_group_id")),
    ]
    for cand in candidates:
        if cand is not None and cand in embedding_batch_ids:
            return int(cand)
    return None


def _collect_run_parameters(
    run: dict[str, Any],
    analysis_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = dict(run.get("config_json") or {})
    if isinstance(cfg.get("parameters"), dict):
        return dict(cfg["parameters"])
    for row in analysis_rows:
        payload = row.get("result_json")
        if isinstance(payload, dict) and isinstance(payload.get("parameters"), dict):
            return dict(payload["parameters"])
    return {}


def _collect_numeric_metrics(analysis_rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in analysis_rows:
        result_key = _normalize_col(str(row.get("result_key") or "metric"))
        scalar = _safe_float(row.get("result_value"))
        if scalar is not None:
            metrics[result_key] = scalar
        payload = row.get("result_json")
        if not isinstance(payload, dict):
            continue
        value = payload.get("value", payload)
        _flatten_numeric(value, prefix=f"metric_{result_key}", out=metrics)
    return metrics


def _extract_cluster_labels(analysis_rows: list[dict[str, Any]]) -> list[int | float | str] | None:
    # Prefer structured labels rows; fallback artifact parsing is intentionally omitted
    # to keep this export read-only without blob reads.
    preferred_keys = (
        "clustering_labels",
        "hdbscan_cluster_labels",
    )
    first_pass: list[dict[str, Any]] = []
    second_pass: list[dict[str, Any]] = []
    for row in analysis_rows:
        rk = str(row.get("result_key") or "").strip()
        if rk in preferred_keys:
            first_pass.append(row)
        else:
            second_pass.append(row)
    for row in first_pass + second_pass:
        payload = row.get("result_json")
        if not isinstance(payload, dict):
            continue
        value = payload.get("value", payload)
        if not isinstance(value, dict):
            continue
        labels = value.get("cluster_labels")
        if isinstance(labels, list):
            return labels
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=",",
            quotechar='"',
            doublequote=True,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    utc_now = datetime.now(timezone.utc)
    stamp = utc_now.strftime("%Y%m%dT%H%M%SZ")
    export_dir = root / "scratch" / "exports" / f"full_coverage_{stamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    db_url = _load_database_url(root)
    redacted_db_url = _redact_url(db_url)

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.connect() as conn:
        snapshot_rows = _fetch_snapshot_rows(conn)
        source_dataframe_ids = {
            int(row["source_dataframe_group_id"])
            for row in snapshot_rows.values()
            if int(row["source_dataframe_group_id"]) > 0
        }
        embedding_batches = _fetch_embedding_batches(conn, source_dataframe_ids)
        embedding_batch_ids = set(embedding_batches.keys())

        snapshot_to_engines: dict[int, set[str]] = defaultdict(set)
        for snap_id, srow in snapshot_rows.items():
            sdf_id = int(srow["source_dataframe_group_id"])
            for emb in embedding_batches.values():
                if int(emb["source_dataframe_group_id"]) != sdf_id:
                    continue
                if emb["embedding_engine"]:
                    snapshot_to_engines[snap_id].add(str(emb["embedding_engine"]))
        all_engines = sorted(
            {
                engine_name
                for engine_set in snapshot_to_engines.values()
                for engine_name in engine_set
                if engine_name
            }
        )

        completed_runs_raw = _fetch_completed_analysis_runs(conn)
        total_provenanced_runs_queried = len(completed_runs_raw)

        canonical_methods = sorted({spec.method_name for spec in iter_algorithm_specs()})
        legacy_present = sorted(
            {
                _extract_method_name(run)
                for run in completed_runs_raw
                if _extract_method_name(run) in LEGACY_METHOD_CANDIDATES
            }
        )
        method_names = sorted(set(canonical_methods + legacy_present))

        # Scope-filter runs to rows with resolvable embedding batch metadata
        # from the snapshot source-dataframe embedding inventory.
        filtered_runs: list[dict[str, Any]] = []
        for run in completed_runs_raw:
            snapshot_id = _safe_int(run.get("snapshot_id"))
            if snapshot_id is None or snapshot_id not in SNAPSHOT_SCOPE:
                continue
            method_name = _extract_method_name(run)
            if method_name not in method_names:
                continue
            embedding_batch_group_id = _resolve_embedding_batch_id(run, embedding_batch_ids)
            if embedding_batch_group_id is None:
                continue
            emb_meta = embedding_batches.get(embedding_batch_group_id)
            if not emb_meta:
                continue
            embedding_engine_name = str(emb_meta.get("embedding_engine") or "").strip()
            if not embedding_engine_name:
                continue
            row = dict(run)
            row["resolved_method_name"] = method_name
            row["embedding_batch_group_id"] = int(embedding_batch_group_id)
            row["embedding_engine"] = embedding_engine_name
            row["embedding_provider"] = str(emb_meta.get("embedding_provider") or "")
            row["embedding_dim"] = emb_meta.get("embedding_dim")
            filtered_runs.append(row)

        # Coverage grid
        coverage_counts: dict[tuple[int, str, str], int] = defaultdict(int)
        for row in filtered_runs:
            key = (
                int(row["snapshot_id"]),
                str(row["embedding_engine"]),
                str(row["resolved_method_name"]),
            )
            coverage_counts[key] += 1

        completed_cells_by_snapshot: dict[int, int] = {}
        expected_cells_by_snapshot: dict[int, int] = {}
        zero_coverage_pairs: list[tuple[int, str]] = []
        for snap_id in SNAPSHOT_IDS:
            expected = len(all_engines) * len(method_names)
            completed_cells = 0
            for engine_name in all_engines:
                row_total = 0
                for method_name in method_names:
                    cell_count = coverage_counts.get((snap_id, engine_name, method_name), 0)
                    if cell_count > 0:
                        completed_cells += 1
                    row_total += cell_count
                if row_total == 0:
                    zero_coverage_pairs.append((snap_id, engine_name))
            completed_cells_by_snapshot[snap_id] = completed_cells
            expected_cells_by_snapshot[snap_id] = expected

        grand_completed_cells = sum(completed_cells_by_snapshot.values())
        grand_expected_cells = sum(expected_cells_by_snapshot.values())
        grand_coverage_pct = (
            (100.0 * grand_completed_cells / grand_expected_cells)
            if grand_expected_cells
            else 0.0
        )

        # Pull analysis_results for selected runs.
        analysis_group_ids: set[int] = set()
        for row in filtered_runs:
            md = dict(row.get("metadata_json") or {})
            agid = _safe_int(md.get("analysis_group_id"))
            if agid is not None:
                analysis_group_ids.add(int(agid))
        analysis_rows_by_group = _fetch_analysis_results_for_groups(conn, analysis_group_ids)

    # Build metrics rows.
    metrics_rows: list[dict[str, Any]] = []
    labels_payloads: list[dict[str, Any]] = []
    for run in filtered_runs:
        snapshot_id = int(run["snapshot_id"])
        snapshot_meta = snapshot_rows[snapshot_id]
        method_name = str(run["resolved_method_name"])
        analysis_run_key = str(run.get("analysis_run_key") or "")
        run_md = dict(run.get("metadata_json") or {})
        analysis_key = str(run_md.get("analysis_key") or method_name)
        analysis_group_id = _safe_int(run_md.get("analysis_group_id"))
        group_rows = (
            analysis_rows_by_group.get(int(analysis_group_id), [])
            if analysis_group_id is not None
            else []
        )
        parameters = _collect_run_parameters(run, group_rows)
        numeric_metrics = _collect_numeric_metrics(group_rows)
        metric_row: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "snapshot_name": str(snapshot_meta.get("snapshot_name") or ""),
            "dataset_slug": str(snapshot_meta.get("dataset_slug") or ""),
            "embedding_batch_group_id": int(run["embedding_batch_group_id"]),
            "embedding_engine": str(run["embedding_engine"]),
            "embedding_provider": str(run["embedding_provider"]),
            "embedding_dim": run.get("embedding_dim"),
            "method_name": method_name,
            "analysis_key": analysis_key,
            "analysis_run_key": analysis_run_key,
            "provenanced_run_id": int(run["provenanced_run_id"]),
            "parameters_json": json.dumps(
                parameters,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "completed_at": (
                run.get("updated_at").isoformat()
                if hasattr(run.get("updated_at"), "isoformat")
                else str(run.get("updated_at") or "")
            ),
        }
        metric_row.update(numeric_metrics)
        metrics_rows.append(metric_row)

        labels = _extract_cluster_labels(group_rows)
        if labels is not None:
            labels_payloads.append(
                {
                    "snapshot_id": snapshot_id,
                    "analysis_run_key": analysis_run_key,
                    "labels": labels,
                }
            )

    metrics_csv_path = export_dir / "analysis_metrics_per_cell.csv"
    _write_csv(metrics_csv_path, metrics_rows)

    # Size-gated labels export.
    estimated_bytes_total = 0
    estimated_bytes_by_snapshot: dict[int, int] = defaultdict(int)
    for item in labels_payloads:
        key_len = len(str(item["analysis_run_key"]))
        row_count = len(item["labels"])
        estimate = row_count * (key_len + 24)
        estimated_bytes_total += estimate
        estimated_bytes_by_snapshot[int(item["snapshot_id"])] += estimate

    labels_written_files: list[Path] = []
    labels_skipped_snapshots: list[int] = []
    if estimated_bytes_total <= MAX_LABEL_FILE_BYTES:
        labels_csv_path = export_dir / "analysis_labels_long.csv"
        with labels_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["analysis_run_key", "snapshot_position", "cluster_label"],
                extrasaction="ignore",
                delimiter=",",
                quotechar='"',
                doublequote=True,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writeheader()
            for item in labels_payloads:
                analysis_run_key = str(item["analysis_run_key"])
                for idx, label in enumerate(item["labels"]):
                    writer.writerow(
                        {
                            "analysis_run_key": analysis_run_key,
                            "snapshot_position": int(idx),
                            "cluster_label": label,
                        }
                    )
        labels_written_files.append(labels_csv_path)
    else:
        for snap_id in SNAPSHOT_IDS:
            est = estimated_bytes_by_snapshot.get(snap_id, 0)
            if est == 0:
                continue
            if est > MAX_LABEL_FILE_BYTES:
                labels_skipped_snapshots.append(int(snap_id))
                continue
            per_snap_path = export_dir / f"analysis_labels_long__snap{snap_id}.csv"
            with per_snap_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["analysis_run_key", "snapshot_position", "cluster_label"],
                    extrasaction="ignore",
                    delimiter=",",
                    quotechar='"',
                    doublequote=True,
                    quoting=csv.QUOTE_ALL,
                    lineterminator="\n",
                )
                writer.writeheader()
                for item in labels_payloads:
                    if int(item["snapshot_id"]) != snap_id:
                        continue
                    analysis_run_key = str(item["analysis_run_key"])
                    for idx, label in enumerate(item["labels"]):
                        writer.writerow(
                            {
                                "analysis_run_key": analysis_run_key,
                                "snapshot_position": int(idx),
                                "cluster_label": label,
                            }
                        )
            labels_written_files.append(per_snap_path)

    # coverage_summary.md
    coverage_md_lines: list[str] = []
    coverage_md_lines.append("# Coverage Summary")
    coverage_md_lines.append("")
    coverage_md_lines.append("## Grand Totals")
    coverage_md_lines.append("")
    coverage_md_lines.append(
        f"- Snapshots in scope: **{len(SNAPSHOT_IDS)}** ({', '.join(str(s) for s in SNAPSHOT_IDS)})"
    )
    coverage_md_lines.append(f"- Embedding engines (union): **{len(all_engines)}**")
    coverage_md_lines.append(f"- Analysis methods tracked: **{len(method_names)}**")
    coverage_md_lines.append(f"- Completed cells: **{grand_completed_cells}**")
    coverage_md_lines.append(f"- Expected cells: **{grand_expected_cells}**")
    coverage_md_lines.append(f"- Coverage: **{grand_coverage_pct:.2f}%**")
    coverage_md_lines.append(
        f"- Completed provenanced run rows included in grid counts: **{len(filtered_runs)}**"
    )
    coverage_md_lines.append("")
    coverage_md_lines.append("### Zero-Coverage (Snapshot, Engine) Pairs")
    coverage_md_lines.append("")
    if zero_coverage_pairs:
        for snap_id, engine_name in sorted(zero_coverage_pairs):
            coverage_md_lines.append(f"- snapshot `{snap_id}` + engine `{engine_name}`")
    else:
        coverage_md_lines.append("- None")
    coverage_md_lines.append("")

    for snap_id in SNAPSHOT_IDS:
        snap = snapshot_rows[snap_id]
        dataset_slug = str(snap.get("dataset_slug") or "")
        row_count = int(snap.get("row_count") or 0)
        scope_label = str(snap.get("scope_label") or "")
        coverage_md_lines.append(f"## Snapshot {snap_id}")
        coverage_md_lines.append("")
        coverage_md_lines.append(
            f"`dataset_slug={dataset_slug}`, `row_count={row_count}`, represents: {scope_label}."
        )
        coverage_md_lines.append("")
        headers = ["embedding_engine"] + method_names + ["row_total"]
        coverage_md_lines.append("| " + " | ".join(headers) + " |")
        coverage_md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for engine_name in all_engines:
            values: list[str] = [engine_name]
            row_total = 0
            for method_name in method_names:
                count = coverage_counts.get((snap_id, engine_name, method_name), 0)
                values.append(str(count))
                row_total += count
            values.append(str(row_total))
            coverage_md_lines.append("| " + " | ".join(values) + " |")
        completed_cells = completed_cells_by_snapshot[snap_id]
        expected_cells = expected_cells_by_snapshot[snap_id]
        pct = (100.0 * completed_cells / expected_cells) if expected_cells else 0.0
        coverage_md_lines.append("")
        coverage_md_lines.append(
            f"- Completed cells: **{completed_cells}** / **{expected_cells}** "
            f"(**{pct:.2f}%**)"
        )
        coverage_md_lines.append("")

    coverage_md_lines.append("## Provenance")
    coverage_md_lines.append("")
    coverage_md_lines.append(f"- Redacted DB URL: `{redacted_db_url}`")
    coverage_md_lines.append(
        f"- UTC export timestamp: `{utc_now.strftime('%Y-%m-%dT%H:%M:%SZ')}`"
    )
    coverage_md_lines.append(
        f"- Total provenanced_runs rows queried: **{total_provenanced_runs_queried}**"
    )
    if labels_skipped_snapshots:
        coverage_md_lines.append(
            "- Label export skipped for snapshots exceeding 500MB estimate: "
            + ", ".join(str(s) for s in sorted(labels_skipped_snapshots))
        )
    (export_dir / "coverage_summary.md").write_text(
        "\n".join(coverage_md_lines) + "\n",
        encoding="utf-8",
    )

    # README.md
    readme_lines: list[str] = []
    readme_lines.append("# Full Coverage Export")
    readme_lines.append("")
    readme_lines.append("## Files")
    readme_lines.append("")
    readme_lines.append(
        "- `coverage_summary.md`: per-snapshot coverage matrix over "
        "(snapshot x embedding_engine x analysis_method), plus grand totals."
    )
    readme_lines.append(
        "- `analysis_metrics_per_cell.csv`: one row per completed analysis execution "
        "cell (provenanced run), including IDs, parameters, and flattened numeric metrics."
    )
    if labels_written_files:
        if len(labels_written_files) == 1 and labels_written_files[0].name == "analysis_labels_long.csv":
            readme_lines.append(
                "- `analysis_labels_long.csv`: long-format cluster assignments with "
                "`analysis_run_key`, `snapshot_position`, `cluster_label`."
            )
        else:
            names = ", ".join(f"`{path.name}`" for path in labels_written_files)
            readme_lines.append(
                "- Split label files due size estimate > 500MB total: " + names + "."
            )
    else:
        readme_lines.append(
            "- Label long-format CSV was not written for snapshots whose estimated "
            "file size exceeded 500MB."
        )
    readme_lines.append("")
    readme_lines.append("## Relationship Between Files")
    readme_lines.append("")
    readme_lines.append(
        "- `analysis_metrics_per_cell.csv` provides one row per `analysis_run_key`."
    )
    readme_lines.append(
        "- Label files provide many rows per `analysis_run_key` (one per snapshot position)."
    )
    readme_lines.append("")
    readme_lines.append("## SQL Join View")
    readme_lines.append("")
    readme_lines.append(
        "```sql\n"
        "CREATE VIEW analysis_cell_with_labels AS\n"
        "SELECT\n"
        "  m.*,\n"
        "  l.snapshot_position,\n"
        "  l.cluster_label\n"
        "FROM analysis_metrics_per_cell m\n"
        "LEFT JOIN analysis_labels_long l\n"
        "  ON m.analysis_run_key = l.analysis_run_key;\n"
        "```"
    )
    readme_lines.append("")
    readme_lines.append(
        "- If labels are split by snapshot, replace `analysis_labels_long` with a "
        "`UNION ALL` view over the split label tables."
    )
    readme_lines.append("")
    readme_lines.append(
        "- `snapshot_position` is the zero-based position within the analyzed snapshot-order "
        "label vector emitted for each `analysis_run_key`."
    )
    (export_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )

    csv_sizes: dict[str, int] = {}
    for path in [metrics_csv_path, *labels_written_files]:
        if path.exists():
            csv_sizes[path.name] = path.stat().st_size

    summary = {
        "export_dir": str(export_dir),
        "snapshot_completed_cells": completed_cells_by_snapshot,
        "snapshot_expected_cells": expected_cells_by_snapshot,
        "grand_completed_cells": grand_completed_cells,
        "grand_expected_cells": grand_expected_cells,
        "zero_coverage_pairs": [
            {"snapshot_id": sid, "embedding_engine": eng}
            for sid, eng in sorted(zero_coverage_pairs)
        ],
        "zero_coverage_top5": [
            {"snapshot_id": sid, "embedding_engine": eng}
            for sid, eng in sorted(zero_coverage_pairs)[:5]
        ],
        "csv_sizes_bytes": csv_sizes,
        "labels_skipped_snapshots": sorted(labels_skipped_snapshots),
        "method_count": len(method_names),
        "engine_count": len(all_engines),
        "provenanced_runs_rows_queried": total_provenanced_runs_queried,
    }
    (export_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"EXPORT_DIR={export_dir}")
    print(f"GRAND_COMPLETED_CELLS={grand_completed_cells}")
    for sid in SNAPSHOT_IDS:
        print(
            f"SNAPSHOT_{sid}_COMPLETED={completed_cells_by_snapshot[sid]} "
            f"/ {expected_cells_by_snapshot[sid]}"
        )
    top5 = ", ".join(
        f"({pair['snapshot_id']},{pair['embedding_engine']})"
        for pair in summary["zero_coverage_top5"]
    )
    print(f"ZERO_COVERAGE_TOP5={top5}")
    for name, size in sorted(csv_sizes.items()):
        print(f"CSV_SIZE {name} {size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

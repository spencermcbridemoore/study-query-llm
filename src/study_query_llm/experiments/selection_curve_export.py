"""Export per-k metrics from sweep-and-select clustering runs (kmeans / GMM) to CSV rows.

Sweep runners persist full ``k`` curves as JSON artifacts ending with
``_selection_curve.json``. This module loads those artifacts via storage URIs
recorded on ``analysis_results`` rows (``result_key="artifacts"``).
"""

from __future__ import annotations

import csv
import json
import logging
from typing import Any, Iterable, Iterator, Mapping, Sequence

from study_query_llm.db.models_v2 import AnalysisResult, MethodDefinition, ProvenancedRun
from study_query_llm.pipeline.clustering.registry import iter_algorithm_specs
from study_query_llm.services.artifact_service import ArtifactService

logger = logging.getLogger(__name__)


def default_sweep_method_names() -> tuple[str, ...]:
    """Registry methods with ``fit_mode == sweep_select`` (non-fixed-k sweep paths)."""
    return tuple(
        spec.method_name
        for spec in iter_algorithm_specs()
        if spec.fit_mode == "sweep_select"
    )


def _selection_curve_files(uris: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return ``(logical_filename, uri)`` pairs for sweep curve artifacts."""
    out: list[tuple[str, str]] = []
    for logical, uri in uris.items():
        if not isinstance(logical, str) or not isinstance(uri, str):
            continue
        if logical.lower().endswith("_selection_curve.json"):
            out.append((logical, uri))
    return sorted(out, key=lambda x: x[0])


def _provenance_by_analysis_group(
    session,
    group_ids: set[int],
) -> dict[int, ProvenancedRun]:
    """Map ``analysis_group_id`` -> latest matching provenanced_run by row id."""
    if not group_ids:
        return {}
    best: dict[int, ProvenancedRun] = {}
    candidates = (
        session.query(ProvenancedRun)
        .filter(ProvenancedRun.run_kind == "execution")
        .order_by(ProvenancedRun.id.asc())
        .all()
    )
    for pr in candidates:
        meta = pr.metadata_json or {}
        raw = meta.get("analysis_group_id")
        if raw is None:
            continue
        try:
            agid = int(raw)
        except (TypeError, ValueError):
            continue
        if agid not in group_ids:
            continue
        prev = best.get(agid)
        if prev is None or int(pr.id) > int(prev.id):
            best[agid] = pr
    return best


def _load_curve_doc(artifact_service: ArtifactService, uri: str) -> dict[str, Any]:
    raw = artifact_service.storage.read_from_uri(uri)
    doc = json.loads(raw.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"curve artifact is not a JSON object: {uri!r}")
    return doc


def iter_selection_curve_metric_rows(
    session,
    artifact_service: ArtifactService,
    *,
    method_names: Sequence[str] | None = None,
    analysis_group_ids: set[int] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one dict per (analysis run × curve file × candidate k)."""
    names: tuple[str, ...]
    if method_names is None:
        names = default_sweep_method_names()
    else:
        names = tuple(str(m) for m in method_names)

    q = (
        session.query(AnalysisResult, MethodDefinition)
        .join(MethodDefinition, AnalysisResult.method_definition_id == MethodDefinition.id)
        .filter(AnalysisResult.result_key == "artifacts")
        .filter(MethodDefinition.is_active.is_(True))
        .filter(MethodDefinition.name.in_(names))
        .order_by(AnalysisResult.id.asc())
    )
    pairs = q.all()
    gids = {int(ar.analysis_group_id) for ar, _md in pairs}
    prov_map = _provenance_by_analysis_group(session, gids)

    for ar, md in pairs:
        agid = int(ar.analysis_group_id)
        if analysis_group_ids is not None and agid not in analysis_group_ids:
            continue
        payload = ar.result_json or {}
        uris = payload.get("uris") if isinstance(payload.get("uris"), dict) else {}
        curves = _selection_curve_files(uris)
        if not curves:
            logger.warning(
                "No *_selection_curve.json in artifacts row: analysis_result_id=%s "
                "analysis_group_id=%s method=%s",
                ar.id,
                agid,
                md.name,
            )
            continue
        pr = prov_map.get(agid)
        pr_meta = dict(pr.metadata_json or {}) if pr is not None else {}
        base_common: dict[str, Any] = {
            "analysis_result_id": int(ar.id),
            "method_name": str(md.name),
            "method_version": str(md.version) if md.version else "",
            "analysis_group_id": agid,
            "provenanced_run_id": int(pr.id) if pr is not None else "",
            "analysis_run_key": str(pr.run_key) if pr is not None else "",
            "input_snapshot_group_id": pr_meta.get("input_snapshot_group_id", ""),
            "embedding_batch_group_id": pr_meta.get("embedding_batch_group_id", ""),
        }
        for logical, uri in curves:
            try:
                doc = _load_curve_doc(artifact_service, uri)
            except Exception as exc:
                logger.warning(
                    "Skip curve %s (analysis_group_id=%s): %s",
                    logical,
                    agid,
                    exc,
                )
                continue
            points = doc.get("points")
            if not isinstance(points, list) or not points:
                logger.warning(
                    "Curve %s has no points (analysis_group_id=%s)",
                    logical,
                    agid,
                )
                continue
            base = dict(base_common)
            base.update(
                {
                    "curve_logical_filename": logical,
                    "curve_uri": uri,
                    "selection_metric": doc.get("metric", ""),
                    "selection_rule": doc.get("selection_rule", ""),
                    "chosen_k": doc.get("chosen_k", ""),
                }
            )
            for pt in points:
                if not isinstance(pt, dict):
                    continue
                row = dict(base)
                for pk, pv in pt.items():
                    key = "candidate_k" if pk == "k" else str(pk)
                    row[key] = pv
                yield row


def write_metrics_csv(rows: Iterable[dict[str, Any]], path: str) -> int:
    """Write rows to UTF-8 CSV; returns row count written."""
    materialized = list(rows)
    if not materialized:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return 0
    fieldnames = sorted({k for r in materialized for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)

"""Derive MCQ analysis metrics from persisted probe_details (mcq_run metadata)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def compliance_rates_from_probe(probe_details: Dict[str, Any]) -> Dict[str, float]:
    """Map catalog result_keys for mcq_compliance from one run's probe_details."""
    summary = probe_details.get("summary") or {}
    attempted = float(summary.get("samples_attempted") or 0)
    successful_calls = float(summary.get("samples_with_successful_call") or 0)
    valid_keys = float(summary.get("samples_with_valid_answer_key") or 0)
    heading_n = float(summary.get("heading_present_count") or 0)

    parse_rate = (valid_keys / attempted) if attempted > 0 else 0.0
    q_compliance = (valid_keys / successful_calls) if successful_calls > 0 else 0.0
    format_rate = (heading_n / successful_calls) if successful_calls > 0 else 0.0

    return {
        "format_compliance_rate": format_rate,
        "question_count_compliance_rate": q_compliance,
        "answer_key_parse_rate": parse_rate,
    }


def sweep_pooled_distribution(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge pooled_distribution counts across run summaries."""
    total_by_label: Counter = Counter()
    labels: List[str] = []
    for s in summaries:
        dist = s.get("pooled_distribution") or {}
        if not labels and dist:
            labels = list(dist.keys())
        for lab, cell in dist.items():
            total_by_label[lab] += int((cell or {}).get("count", 0))
    total = sum(total_by_label.values())
    merged = {}
    for lab in labels or list(total_by_label.keys()):
        c = int(total_by_label.get(lab, 0))
        merged[lab] = {
            "count": c,
            "pct": (c / total) if total > 0 else math.nan,
        }
    return {"labels": labels or list(total_by_label.keys()), "pooled_distribution": merged, "total_answers": total}


def sweep_chi_square_vs_uniform(summaries: List[Dict[str, Any]]) -> Tuple[float, Optional[float]]:
    """Chi-square on aggregated pooled counts vs uniform; p_value None without scipy."""
    agg = sweep_pooled_distribution(summaries)
    labels = agg["labels"]
    total = agg["total_answers"]
    if total <= 0 or not labels:
        return math.nan, None
    k = len(labels)
    expected = float(total) / float(k)
    chi = 0.0
    for lab in labels:
        obs = int(agg["pooled_distribution"].get(lab, {}).get("count", 0))
        chi += ((obs - expected) ** 2) / expected if expected > 0 else 0.0
    return float(chi), None


def per_label_mean_stdev(summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Mean pooled pct per label across runs (simple average of run-level pcts)."""
    by_label: Dict[str, List[float]] = {}
    for s in summaries:
        dist = s.get("pooled_distribution") or {}
        for lab, cell in dist.items():
            p = (cell or {}).get("pct")
            if isinstance(p, (int, float)) and not math.isnan(float(p)):
                by_label.setdefault(lab, []).append(float(p))
    out: Dict[str, Dict[str, float]] = {}
    for lab, vals in by_label.items():
        if not vals:
            out[lab] = {"position_mean": math.nan, "position_stdev": math.nan}
            continue
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            out[lab] = {"position_mean": m, "position_stdev": 0.0}
        else:
            var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
            out[lab] = {"position_mean": m, "position_stdev": math.sqrt(var)}
    return out

"""CLI / driver: run MCQ analysis catalog for a fulfilled (or partially fulfilled) request."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Set

from study_query_llm.analysis.mcq_from_run import (
    compliance_rates_from_probe,
    per_label_mean_stdev,
    sweep_chi_square_vs_uniform,
    sweep_pooled_distribution,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.sweep_request_types import SWEEP_TYPE_MCQ
from study_query_llm.services.provenance_service import GROUP_TYPE_MCQ_RUN
from study_query_llm.services.sweep_request_service import SweepRequestService


def _probe_details_from_group(meta: Dict[str, Any]) -> Dict[str, Any]:
    if "probe_details" in meta and isinstance(meta["probe_details"], dict):
        return meta["probe_details"]
    summary = meta.get("result_summary")
    if isinstance(summary, dict):
        return {"summary": summary, "call_errors": [], "parse_failures": []}
    return {}


def _load_mcq_runs_for_keys(session, run_keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """run_key -> metadata_json for mcq_run groups."""
    out: Dict[str, Dict[str, Any]] = {}
    want = set(run_keys)
    all_mcq = session.query(Group).filter(Group.group_type == GROUP_TYPE_MCQ_RUN).all()
    for g in all_mcq:
        meta = dict(g.metadata_json or {})
        rk = meta.get("run_key")
        if rk and str(rk) in want:
            meta.setdefault("_group_id", int(g.id))
            out[str(rk)] = meta
    return out


def run_mcq_analyses_for_request(
    db: DatabaseConnectionV2,
    request_id: int,
    *,
    dry_run: bool = False,
    analysis_keys: Optional[List[str]] = None,
    orchestration_job_id: Optional[int] = None,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    """
    Execute analysis catalog for an MCQ sweep request.

    Records MethodService results and marks analysis keys complete.
    Returns a small report dict.
    """
    selected_keys = {str(k).strip() for k in (analysis_keys or []) if str(k).strip()}
    report: Dict[str, Any] = {
        "request_id": request_id,
        "recorded": [],
        "skipped": [],
        "dry_run": dry_run,
        "analysis_keys": sorted(selected_keys) if selected_keys else None,
    }
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req = svc.get_request(request_id)
        if not req:
            raise ValueError(f"request_id={request_id} not found")
        if (req.get("sweep_type") or "").lower() != SWEEP_TYPE_MCQ:
            raise ValueError(f"Request {request_id} is not sweep_type=mcq")

        progress = svc.compute_progress(request_id)
        completed_keys: List[str] = list(progress.get("completed_run_keys") or [])
        if not completed_keys:
            report["message"] = "No completed mcq runs to analyze"
            return report

        metas = _load_mcq_runs_for_keys(session, completed_keys)
        catalog = list(req.get("analysis_catalog") or [])
        if selected_keys:
            catalog = [
                entry
                for entry in catalog
                if isinstance(entry, dict)
                and str(entry.get("analysis_key") or "").strip() in selected_keys
            ]
        required_keys: Set[str] = {
            str(e.get("analysis_key"))
            for e in catalog
            if isinstance(e, dict) and e.get("required")
        }
        completed_already = {str(x) for x in (req.get("completed_analyses") or [])}

        summaries: List[Dict[str, Any]] = []
        run_ids_by_key: Dict[str, int] = {}
        for rk in completed_keys:
            meta = metas.get(rk) or {}
            details = _probe_details_from_group(meta)
            summ = details.get("summary") or meta.get("result_summary") or {}
            if isinstance(summ, dict):
                summaries.append(summ)
            gid = meta.get("_group_id")
            if gid is not None:
                run_ids_by_key[rk] = int(gid)

        sweep_source_id = run_ids_by_key.get(completed_keys[0])
        if sweep_source_id is None:
            raise RuntimeError("Could not resolve mcq_run group id for analysis provenance")

        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            akey = str(entry.get("analysis_key") or "")
            if skip_completed and akey in completed_already:
                report["skipped"].append({"analysis": akey, "reason": "already_completed"})
                continue
            scope = str(entry.get("scope") or "run")
            if akey == "mcq_compliance" and scope == "run":
                for rk in completed_keys:
                    meta = metas.get(rk) or {}
                    details = _probe_details_from_group(meta)
                    rates = compliance_rates_from_probe(details)
                    rid = run_ids_by_key.get(rk)
                    if rid is None:
                        continue
                    if dry_run:
                        report["recorded"].append({"analysis": akey, "run_key": rk, "rates": rates})
                        continue
                    for rk_metric, val in rates.items():
                        svc.record_analysis_result(
                            request_id=request_id,
                            source_group_id=rid,
                            analysis_key=akey,
                            result_key=rk_metric,
                            result_value=float(val),
                            orchestration_job_id=orchestration_job_id,
                            mark_complete=False,
                        )
                if not dry_run:
                    svc.mark_analysis_completed(request_id, akey)
                report["recorded"].append({"analysis": akey, "runs": len(completed_keys)})

            elif akey == "mcq_answer_position_distribution" and scope == "sweep":
                agg = sweep_pooled_distribution(summaries)
                means = per_label_mean_stdev(summaries)
                payload = {
                    "position_distribution": agg["pooled_distribution"],
                    "position_mean": means,
                    "position_stdev": {k: v["position_stdev"] for k, v in means.items()},
                }
                if dry_run:
                    report["recorded"].append({"analysis": akey, "payload_keys": list(payload.keys())})
                    continue
                svc.record_analysis_result(
                    request_id=request_id,
                    source_group_id=sweep_source_id,
                    analysis_key=akey,
                    result_key="position_distribution",
                    result_json=payload,
                    orchestration_job_id=orchestration_job_id,
                    mark_complete=False,
                )
                svc.mark_analysis_completed(request_id, akey)
                report["recorded"].append({"analysis": akey, "scope": "sweep"})

            elif akey == "mcq_answer_position_chi_square" and scope == "sweep":
                chi, p_val = sweep_chi_square_vs_uniform(summaries)
                if dry_run:
                    report["recorded"].append({"analysis": akey, "chi_square": chi, "p_value": p_val})
                    continue
                chi_payload = {
                    "chi_square": chi,
                    "p_value": p_val,
                    "note": None if p_val is not None else "p_value not computed (no scipy)",
                }
                svc.record_analysis_result(
                    request_id=request_id,
                    source_group_id=sweep_source_id,
                    analysis_key=akey,
                    result_key="chi_square",
                    result_value=float(chi) if chi == chi else None,
                    result_json=chi_payload,
                    orchestration_job_id=orchestration_job_id,
                    mark_complete=False,
                )
                svc.mark_analysis_completed(request_id, akey)
                report["recorded"].append({"analysis": akey, "scope": "sweep"})

        # Touch analysis_status if all required complete
        req2 = svc.get_request(request_id)
        done = set(req2.get("completed_analyses") or [])
        if required_keys <= done:
            report["all_required_complete"] = True

    return report


def run_enqueued_analysis_jobs_for_request(
    db: DatabaseConnectionV2,
    request_id: int,
    *,
    dry_run: bool = False,
    worker_id: str = "mcq-analyze-wrapper",
    lease_seconds: int = 300,
) -> Dict[str, Any]:
    """
    Compatibility wrapper: process orchestration ``analysis_run`` jobs for request.
    """
    report: Dict[str, Any] = {
        "request_id": int(request_id),
        "dry_run": bool(dry_run),
        "planned_jobs": 0,
        "processed_jobs": 0,
        "processed": [],
    }
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        req = svc.get_request(request_id)
        if not req:
            raise ValueError(f"request_id={request_id} not found")
        if (req.get("sweep_type") or "").lower() != SWEEP_TYPE_MCQ:
            raise ValueError(f"Request {request_id} is not sweep_type=mcq")
        svc.ensure_orchestration_jobs(request_id)
        jobs = repo.list_orchestration_jobs(
            request_group_id=int(request_id),
            job_type="analysis_run",
        )
        report["planned_jobs"] = len(jobs)
        report["job_statuses"] = [j.status for j in jobs]
        if dry_run:
            report["planned_analysis_keys"] = [
                str((j.payload_json or {}).get("analysis_key") or "")
                for j in jobs
            ]
            return report

    while True:
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            job = repo.claim_next_orchestration_job(
                worker_id=worker_id,
                lease_seconds=max(30, int(lease_seconds)),
                request_group_id=int(request_id),
                job_types=["analysis_run"],
            )
            if not job:
                break
            payload = dict(job.payload_json or {})
            analysis_key = str(payload.get("analysis_key") or "")
            try:
                job_report = run_mcq_analyses_for_request(
                    db,
                    int(request_id),
                    dry_run=False,
                    analysis_keys=[analysis_key] if analysis_key else None,
                    orchestration_job_id=int(job.id),
                    skip_completed=True,
                )
                repo.complete_orchestration_job(
                    int(job.id),
                    result_ref=f"analysis:{analysis_key}",
                )
                report["processed"].append(
                    {
                        "job_id": int(job.id),
                        "analysis_key": analysis_key,
                        "status": "completed",
                        "recorded": int(len(job_report.get("recorded") or [])),
                    }
                )
                report["processed_jobs"] = int(report["processed_jobs"]) + 1
            except Exception as exc:
                repo.fail_orchestration_job(
                    int(job.id),
                    error_json={"error": str(exc)[:1000]},
                )
                report["processed"].append(
                    {
                        "job_id": int(job.id),
                        "analysis_key": analysis_key,
                        "status": "failed",
                        "error": str(exc)[:1000],
                    }
                )
                raise
    return report


def build_analyze_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MCQ analysis driver for a sweep request.")
    p.add_argument("--request-id", type=int, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned metrics without writing to DB",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_analyze_arg_parser().parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL is required", file=sys.stderr)
        sys.exit(1)
    db = DatabaseConnectionV2(url, enable_pgvector=True)
    db.init_db()
    report = run_enqueued_analysis_jobs_for_request(
        db,
        args.request_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

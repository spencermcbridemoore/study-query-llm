"""Deterministic P0 control-plane baseline generation utilities."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import OrchestrationJobDependency
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.experiments.sweep_request_types import get_sweep_type_adapter
from study_query_llm.services.provenanced_run_service import canonical_run_fingerprint
from study_query_llm.services.sweep_request_service import SweepRequestService

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_TMP_SEGMENT_RE = re.compile(r"^tmp[0-9A-Za-z._-]*$")


def normalize_result_ref(result_ref: str | None) -> str | None:
    """Normalize result refs for deterministic parity comparisons.

    Rules:
    - Strip Windows drive roots.
    - Normalize path separators.
    - Replace timestamp tokens.
    - Collapse temp path segments to stable placeholders.
    - Keep semantic tails for path-like refs.
    """
    if result_ref is None:
        return None
    text = str(result_ref).strip()
    if not text:
        return text
    if text.startswith("analysis:"):
        return text
    if text.isdigit():
        return "<INT>"

    normalized = text.replace("\\", "/")
    normalized = _DRIVE_PREFIX_RE.sub("", normalized)
    normalized = _TIMESTAMP_RE.sub("<TIMESTAMP>", normalized)

    parts = [p for p in normalized.split("/") if p]
    folded: List[str] = []
    for part in parts:
        low = part.lower()
        if low in {"tmp", "temp"} or _TMP_SEGMENT_RE.match(low):
            folded.append("<TMP>")
            continue
        folded.append(part)

    if not folded:
        return normalized

    # Keep only semantic tail for file-like refs to avoid machine-local roots.
    if "." in folded[-1]:
        if len(folded) >= 2:
            return "/".join(folded[-2:])
        return folded[-1]
    return "/".join(folded)


def normalize_result_refs(result_refs: Iterable[str | None]) -> List[str]:
    out: List[str] = []
    for value in result_refs:
        normalized = normalize_result_ref(value)
        if normalized:
            out.append(normalized)
    return sorted(dict.fromkeys(out))


def _build_job_dependency_key_map(
    session: Any,
    jobs_by_id: Dict[int, Any],
) -> Dict[int, List[str]]:
    if not jobs_by_id:
        return {}
    rows = (
        session.query(OrchestrationJobDependency)
        .filter(OrchestrationJobDependency.job_id.in_(list(jobs_by_id.keys())))
        .all()
    )
    dep_keys_by_job_id: Dict[int, List[str]] = defaultdict(list)
    for row in rows:
        dep_job = jobs_by_id.get(int(row.depends_on_job_id))
        if dep_job is None:
            continue
        dep_keys_by_job_id[int(row.job_id)].append(str(dep_job.job_key))
    for job_id in list(dep_keys_by_job_id.keys()):
        dep_keys_by_job_id[job_id] = sorted(dep_keys_by_job_id[job_id])
    return dep_keys_by_job_id


def _summarize_orchestration_jobs(
    session: Any,
    repo: RawCallRepository,
    request_id: int,
) -> Dict[str, Any]:
    jobs = repo.list_orchestration_jobs(request_group_id=request_id)
    jobs_by_id = {int(job.id): job for job in jobs}
    dep_keys = _build_job_dependency_key_map(session, jobs_by_id)

    payload_key_sets: Dict[str, set[str]] = defaultdict(set)
    graph_rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    for job in jobs:
        payload = dict(job.payload_json or {})
        payload_key_sets[str(job.job_type)].update(payload.keys())
        status_counts[str(job.status)] += 1
        type_counts[str(job.job_type)] += 1
        graph_rows.append(
            {
                "job_type": str(job.job_type),
                "job_key": str(job.job_key),
                "base_run_key": job.base_run_key,
                "status": str(job.status),
                "depends_on_job_keys": dep_keys.get(int(job.id), []),
            }
        )

    return {
        "total_jobs": len(jobs),
        "job_type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "payload_key_sets": {
            key: sorted(value)
            for key, value in sorted(payload_key_sets.items(), key=lambda item: item[0])
        },
        "graph": sorted(
            graph_rows,
            key=lambda row: (row["job_type"], row["job_key"]),
        ),
    }


def _run_claim_complete_smoke(repo: RawCallRepository, request_id: int) -> Dict[str, Any]:
    claimed_jobs: List[Dict[str, Any]] = []
    while True:
        job = repo.claim_next_orchestration_job(
            worker_id="p0-baseline-worker",
            lease_seconds=60,
            request_group_id=request_id,
        )
        if job is None:
            break
        raw_ref = (
            "C:/Users/test/AppData/Local/Temp/session-abc123/"
            "2026-04-28T15:00:00Z/"
            f"{job.job_type}/{job.job_key}.json"
        )
        repo.complete_orchestration_job(int(job.id), result_ref=raw_ref)
        claimed_jobs.append(
            {
                "job_type": str(job.job_type),
                "job_key": str(job.job_key),
                "result_ref_raw": raw_ref,
                "result_ref_normalized": normalize_result_ref(raw_ref),
            }
        )
    return {
        "claimed_count": len(claimed_jobs),
        "claimed_jobs": claimed_jobs,
    }


def _materialize_runs_and_finalize(
    repo: RawCallRepository,
    svc: SweepRequestService,
    request_id: int,
) -> Dict[str, Any]:
    req = svc.get_request(request_id)
    if req is None:
        raise ValueError(f"Missing request_id={request_id}")
    adapter = get_sweep_type_adapter(req.get("sweep_type"))
    expected_keys = [str(x) for x in (req.get("expected_run_keys") or [])]
    for idx, run_key in enumerate(expected_keys):
        repo.create_group(
            group_type=adapter.run_group_type,
            name=f"baseline_{adapter.sweep_type}_run_{idx + 1}",
            metadata_json={"run_key": run_key},
        )
    sweep_id = svc.finalize_if_fulfilled(request_id)
    finalized_req = svc.get_request(request_id) or {}
    progress = svc.compute_progress(request_id)
    return {
        "request_status": str(finalized_req.get("request_status") or ""),
        "linked_sweep_id_present": bool(sweep_id),
        "expected_count": int(progress.get("expected_count") or 0),
        "completed_count": int(progress.get("completed_count") or 0),
        "missing_count": int(progress.get("missing_count") or 0),
    }


def _build_normalization_examples() -> Dict[str, str]:
    samples = {
        "windows_temp_path": (
            r"C:\Users\spenc\AppData\Local\Temp\sq-run-123\2026-04-28T15:00:00Z\job_shards\leaf.json"
        ),
        "unix_temp_path": "/tmp/sq-run-123/2026-04-28T15:00:00Z/job_shards/reduce.json",
        "integer_run_id": "12345",
        "analysis_result_ref": "analysis:mcq_compliance:2",
    }
    return {key: str(normalize_result_ref(value)) for key, value in samples.items()}


def build_p0_baseline_snapshot() -> Dict[str, Any]:
    """Build deterministic baseline snapshot for PR0 parity checks."""
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    with db.session_scope() as session:
        repo = RawCallRepository(session)
        svc = SweepRequestService(repo)
        # Force deterministic behavior regardless of developer shell flags.
        svc.use_derived_analysis_status = True
        svc.enable_analysis_jobs = True
        svc.record_analysis_parity = False

        clustering_request_id = svc.create_request(
            request_name="p0_baseline_clustering",
            algorithm="cosine_kllmeans_no_pca",
            fixed_config={"k_min": 2, "k_max": 3, "n_restarts": 2, "base_seed": 11},
            parameter_axes={
                "datasets": ["dbpedia"],
                "embedding_engines": ["engine/a"],
                "summarizers": ["None"],
            },
            entry_max=50,
            execution_mode="sharded",
            shard_config={"k_ranges": [[2, 3]], "tries_per_k": 2},
            sweep_type="clustering",
        )
        clustering_planned = _summarize_orchestration_jobs(session, repo, clustering_request_id)
        clustering_claimed = _run_claim_complete_smoke(repo, clustering_request_id)
        clustering_completed = _summarize_orchestration_jobs(session, repo, clustering_request_id)
        clustering_finalized = _materialize_runs_and_finalize(repo, svc, clustering_request_id)

        mcq_request_id = svc.create_request(
            request_name="p0_baseline_mcq",
            algorithm="mcq_answer_position_probe",
            fixed_config={"samples_per_combo": 3, "template_version": "v1"},
            parameter_axes={
                "levels": ["high school"],
                "subjects": ["physics"],
                "deployments": ["gpt-4o-mini"],
                "options_per_question": [4, 5],
                "questions_per_test": [10],
                "label_styles": ["upper"],
                "spread_correct_answer_uniformly": [False],
            },
            entry_max=None,
            sweep_type="mcq",
        )
        mcq_planned = _summarize_orchestration_jobs(session, repo, mcq_request_id)
        mcq_claimed = _run_claim_complete_smoke(repo, mcq_request_id)
        mcq_completed = _summarize_orchestration_jobs(session, repo, mcq_request_id)
        mcq_finalized = _materialize_runs_and_finalize(repo, svc, mcq_request_id)

    _, scheduling_hash = canonical_run_fingerprint(
        method_name="cosine_kllmeans_no_pca",
        method_version="1.0",
        config_json={
            "k_min": 2,
            "k_max": 3,
            "max_attempts": 3,
            "worker_id": "w1",
            "lease_seconds": 60,
        },
        determinism_class="pseudo_deterministic",
    )
    _, canonical_hash = canonical_run_fingerprint(
        method_name="cosine_kllmeans_no_pca",
        method_version="1.0",
        config_json={"k_min": 2, "k_max": 3},
        determinism_class="pseudo_deterministic",
    )

    return {
        "schema_version": 1,
        "normalization_examples": _build_normalization_examples(),
        "fingerprint_canary": {
            "scheduling_hash": scheduling_hash,
            "canonical_hash": canonical_hash,
            "matches": scheduling_hash == canonical_hash,
        },
        "clustering": {
            "planned": clustering_planned,
            "claimed_completion": {
                "claimed_count": clustering_claimed["claimed_count"],
                "normalized_result_refs": normalize_result_refs(
                    row["result_ref_raw"] for row in clustering_claimed["claimed_jobs"]
                ),
            },
            "completed_jobs": clustering_completed,
            "finalized_request": clustering_finalized,
        },
        "mcq": {
            "planned": mcq_planned,
            "claimed_completion": {
                "claimed_count": mcq_claimed["claimed_count"],
                "normalized_result_refs": normalize_result_refs(
                    row["result_ref_raw"] for row in mcq_claimed["claimed_jobs"]
                ),
            },
            "completed_jobs": mcq_completed,
            "finalized_request": mcq_finalized,
        },
    }


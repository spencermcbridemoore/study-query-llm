#!/usr/bin/env python3
"""Run one per-pair backfill worker for a single (snapshot, engine) pair.

This worker supports:
- phase0: embedding precompute (embed-then-cluster remediation)
- phase1: clustering backfill execution for one pair
- all: phase0 followed by phase1

Per-pair artifacts are written under ``<work_dir>/<phase>/`` and include:
- ``pair_spec.json``
- ``preflight_manifest.json`` (phase1)
- ``run_state.json`` (phase1 execution state)
- ``completed_keys_cache.json`` (phase1 debug cache)
- ``execute_stats.json`` (phase1)
- ``status.json`` (phase-level + top-level mirrors)
- ``worker.log`` (captured by orchestrator subprocess logging)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2  # noqa: E402
from study_query_llm.db.write_intent import WriteIntent  # noqa: E402
from study_query_llm.experiments.clustering_analysis_backfill import (  # noqa: E402
    build_manifest,
    fetch_completed_backfill_run_keys,
    preflight_manifest_blocking_issues,
    refresh_manifest_completion,
    run_backfill_execution,
    snapshot_lineage,
    verify_manifest_coverage,
)
from study_query_llm.pipeline.embed import embed  # noqa: E402


TRANSIENT_MARKERS = (
    "operationalerror",
    "connection",
    "timeout",
    "temporarily",
    "too many requests",
    "429",
    "503",
    "502",
    "network",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        value = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if value:
            return value
    raise SystemExit("No database URL found (CANONICAL_DATABASE_URL / DATABASE_URL)")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _engine_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(value).strip())


def _pair_id(snapshot_id: int, embedding_engine: str) -> str:
    return f"snap{int(snapshot_id)}__{_engine_slug(embedding_engine)}"


def _classify_exception(exc: Exception) -> str:
    rendered = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in rendered for marker in TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


def _single_pair_manifest(
    *,
    session: Any,
    snapshot_id: int,
    embedding_engine: str,
    provider: str,
) -> dict[str, Any]:
    manifest = build_manifest(
        session,
        embedding_engine=str(embedding_engine),
        provider=str(provider),
        snapshot_ids=[int(snapshot_id)],
    )
    return manifest


def _single_pair_manifest_status(manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pairs = list(manifest.get("pairs") or [])
    if not pairs:
        return "error", {"error": "manifest_missing_pairs"}
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        return str(pair.get("status") or "error"), pair
    return "error", {"error": "manifest_pair_not_object"}


def _run_phase0(
    *,
    db: DatabaseConnectionV2,
    snapshot_id: int,
    embedding_engine: str,
    provider: str,
    pair_spec: dict[str, Any],
    phase_dir: Path,
) -> dict[str, Any]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    result_path = phase_dir / "phase0_result.json"

    source_dataframe_group_id = int(pair_spec.get("source_dataframe_group_id") or 0)
    if source_dataframe_group_id <= 0:
        payload = {
            "phase": "phase0",
            "status": "failed",
            "failure_kind": "permanent",
            "reason_code": "lineage_missing",
            "error": "source_dataframe_group_id_missing",
            "updated_at": _now_iso(),
        }
        _write_json(result_path, payload)
        return payload

    try:
        embed_result = embed(
            int(source_dataframe_group_id),
            deployment=str(embedding_engine),
            provider=str(provider),
            representation="full",
            force=False,
            entry_max=None,
            db=db,
            write_intent=WriteIntent.CANONICAL,
        )
    except Exception as exc:  # noqa: BLE001
        failure_kind = _classify_exception(exc)
        payload = {
            "phase": "phase0",
            "status": "failed",
            "failure_kind": failure_kind,
            "reason_code": "embed_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "updated_at": _now_iso(),
        }
        _write_json(result_path, payload)
        return payload
    embed_payload = {
        "stage_name": embed_result.stage_name,
        "group_id": embed_result.group_id,
        "artifact_uris": dict(embed_result.artifact_uris or {}),
        "metadata": dict(embed_result.metadata or {}),
    }

    with db.session_scope() as session:
        manifest = _single_pair_manifest(
            session=session,
            snapshot_id=int(snapshot_id),
            embedding_engine=str(embedding_engine),
            provider=str(provider),
        )
    _write_json(phase_dir / "post_embed_manifest.json", manifest)
    pair_status, pair_entry = _single_pair_manifest_status(manifest)
    blockers = preflight_manifest_blocking_issues(manifest)

    if pair_status != "ok":
        reason_code = (
            "still_no_embedding_after_phase0"
            if pair_status == "no_embedding_batch"
            else f"manifest_status_{pair_status}"
        )
        payload = {
            "phase": "phase0",
            "status": "failed",
            "failure_kind": "permanent",
            "reason_code": reason_code,
            "manifest_status": pair_status,
            "manifest_pair": pair_entry,
            "preflight_blockers": blockers,
            "embed_result": embed_payload,
            "updated_at": _now_iso(),
        }
        _write_json(result_path, payload)
        return payload

    payload = {
        "phase": "phase0",
        "status": "completed",
        "failure_kind": None,
        "reason_code": None,
        "manifest_status": pair_status,
        "manifest_pair": pair_entry,
        "preflight_blockers": blockers,
        "embed_result": embed_payload,
        "updated_at": _now_iso(),
    }
    _write_json(result_path, payload)
    return payload


def _run_phase1(
    *,
    db: DatabaseConnectionV2,
    snapshot_id: int,
    embedding_engine: str,
    provider: str,
    pair_spec: dict[str, Any],
    phase_dir: Path,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = phase_dir / "preflight_manifest.json"
    run_state_path = phase_dir / "run_state.json"
    completed_cache_path = phase_dir / "completed_keys_cache.json"
    execute_stats_path = phase_dir / "execute_stats.json"
    final_manifest_path = phase_dir / "final_manifest.json"

    with db.session_scope() as session:
        manifest = _single_pair_manifest(
            session=session,
            snapshot_id=int(snapshot_id),
            embedding_engine=str(embedding_engine),
            provider=str(provider),
        )
    _write_json(preflight_path, manifest)

    blockers = preflight_manifest_blocking_issues(manifest)
    manifest_status, manifest_pair = _single_pair_manifest_status(manifest)
    if manifest_status != "ok":
        reason_code = (
            "still_no_embedding_after_phase0"
            if manifest_status == "no_embedding_batch"
            else f"manifest_status_{manifest_status}"
        )
        payload = {
            "phase": "phase1",
            "status": "failed",
            "failure_kind": "permanent",
            "reason_code": reason_code,
            "manifest_status": manifest_status,
            "manifest_pair": manifest_pair,
            "preflight_blockers": blockers,
            "updated_at": _now_iso(),
        }
        _write_json(phase_dir / "status.json", payload)
        return payload
    if blockers:
        payload = {
            "phase": "phase1",
            "status": "failed",
            "failure_kind": "permanent",
            "reason_code": "preflight_manifest_blocking_issues",
            "manifest_status": manifest_status,
            "manifest_pair": manifest_pair,
            "preflight_blockers": blockers,
            "updated_at": _now_iso(),
        }
        _write_json(phase_dir / "status.json", payload)
        return payload

    with db.session_scope() as session:
        completed_keys = sorted(fetch_completed_backfill_run_keys(session))
    _write_json(
        completed_cache_path,
        {
            "count": len(completed_keys),
            "sample_first_200": completed_keys[:200],
            "updated_at": _now_iso(),
        },
    )

    try:
        execute_stats = run_backfill_execution(
            db=db,
            manifest=manifest,
            run_state_path=str(run_state_path),
            dry_run=False,
            resume=bool(resume),
            force=bool(force),
        )
    except Exception as exc:  # noqa: BLE001
        failure_kind = _classify_exception(exc)
        payload = {
            "phase": "phase1",
            "status": "failed",
            "failure_kind": failure_kind,
            "reason_code": "execute_failed",
            "manifest_status": manifest_status,
            "manifest_pair": manifest_pair,
            "preflight_blockers": blockers,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "updated_at": _now_iso(),
        }
        _write_json(phase_dir / "status.json", payload)
        return payload
    _write_json(execute_stats_path, execute_stats)

    with db.session_scope() as session:
        refreshed = refresh_manifest_completion(session, dict(manifest))
    _write_json(final_manifest_path, refreshed)
    coverage = verify_manifest_coverage(refreshed)

    if not bool(coverage.get("coverage_complete")):
        payload = {
            "phase": "phase1",
            "status": "failed",
            "failure_kind": "permanent",
            "reason_code": "coverage_incomplete",
            "manifest_status": manifest_status,
            "manifest_pair": manifest_pair,
            "preflight_blockers": blockers,
            "execute_stats": execute_stats,
            "coverage": coverage,
            "updated_at": _now_iso(),
        }
        _write_json(phase_dir / "status.json", payload)
        return payload

    payload = {
        "phase": "phase1",
        "status": "completed",
        "failure_kind": None,
        "reason_code": None,
        "manifest_status": manifest_status,
        "manifest_pair": manifest_pair,
        "preflight_blockers": blockers,
        "execute_stats": execute_stats,
        "coverage": coverage,
        "updated_at": _now_iso(),
    }
    _write_json(phase_dir / "status.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", type=int, required=True)
    parser.add_argument("--embedding-engine", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("phase0", "phase1", "all"),
        default="all",
        help="phase0 (embed), phase1 (cluster), or all",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--resume", action="store_true", help="resume run_state.json in phase1")
    parser.add_argument("--force", action="store_true", help="pass force=True to analyze")
    args = parser.parse_args(argv)

    snapshot_id = int(args.snapshot_id)
    embedding_engine = str(args.embedding_engine).strip()
    provider = str(args.provider).strip()
    if not embedding_engine:
        raise SystemExit("--embedding-engine cannot be empty")
    if not provider:
        raise SystemExit("--provider cannot be empty")

    pair_id = _pair_id(snapshot_id, embedding_engine)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    db_url = _resolve_database_url(args.database_url)
    db = DatabaseConnectionV2(
        db_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    try:
        with db.session_scope() as session:
            lineage = snapshot_lineage(session, int(snapshot_id))
        pair_spec = {
            "pair_id": pair_id,
            "snapshot_id": int(snapshot_id),
            "embedding_engine": embedding_engine,
            "provider": provider,
            "source_dataframe_group_id": int(lineage.source_dataframe_group_id),
            "snapshot_row_count": int(lineage.snapshot_row_count),
            "source_dataframe_row_count": int(lineage.source_dataframe_row_count),
            "lineage_key": [
                int(lineage.source_dataframe_group_id),
                int(lineage.source_dataframe_row_count),
            ],
            "updated_at": _now_iso(),
        }
    except Exception as exc:  # noqa: BLE001
        failure_kind = _classify_exception(exc)
        payload = {
            "pair_id": pair_id,
            "phase": str(args.phase),
            "status": "failed",
            "failure_kind": failure_kind,
            "reason_code": "lineage_missing",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "updated_at": _now_iso(),
        }
        _write_json(work_dir / "status.json", payload)
        return 11 if failure_kind == "transient" else 10

    _write_json(work_dir / "pair_spec.json", pair_spec)

    overall_status: dict[str, Any] = {
        "pair_id": pair_id,
        "phase": str(args.phase),
        "status": "completed",
        "failure_kind": None,
        "reason_code": None,
        "phase_results": {},
        "updated_at": _now_iso(),
    }

    try:
        if args.phase in ("phase0", "all"):
            phase0_result = _run_phase0(
                db=db,
                snapshot_id=snapshot_id,
                embedding_engine=embedding_engine,
                provider=provider,
                pair_spec=pair_spec,
                phase_dir=work_dir / "phase0",
            )
            overall_status["phase_results"]["phase0"] = phase0_result
            if str(phase0_result.get("status")) != "completed":
                overall_status["status"] = "failed"
                overall_status["failure_kind"] = phase0_result.get("failure_kind")
                overall_status["reason_code"] = phase0_result.get("reason_code")

        if args.phase in ("phase1", "all") and overall_status["status"] == "completed":
            phase1_result = _run_phase1(
                db=db,
                snapshot_id=snapshot_id,
                embedding_engine=embedding_engine,
                provider=provider,
                pair_spec=pair_spec,
                phase_dir=work_dir / "phase1",
                resume=bool(args.resume),
                force=bool(args.force),
            )
            overall_status["phase_results"]["phase1"] = phase1_result
            if str(phase1_result.get("status")) != "completed":
                overall_status["status"] = "failed"
                overall_status["failure_kind"] = phase1_result.get("failure_kind")
                overall_status["reason_code"] = phase1_result.get("reason_code")

        overall_status["updated_at"] = _now_iso()
        _write_json(work_dir / "status.json", overall_status)
        if overall_status["status"] == "completed":
            return 0
        return 11 if overall_status.get("failure_kind") == "transient" else 10
    except Exception as exc:  # noqa: BLE001
        failure_kind = _classify_exception(exc)
        overall_status.update(
            {
                "status": "failed",
                "failure_kind": failure_kind,
                "reason_code": "worker_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "updated_at": _now_iso(),
            }
        )
        _write_json(work_dir / "status.json", overall_status)
        return 11 if failure_kind == "transient" else 10


if __name__ == "__main__":
    raise SystemExit(main())

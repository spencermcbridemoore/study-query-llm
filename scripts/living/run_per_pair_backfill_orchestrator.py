#!/usr/bin/env python3
"""Per-pair clustering backfill orchestrator.

Implements strict one-worker-per-(snapshot, engine) execution with:
- Pair-universe enumeration (UNION semantics)
- Pair classification (`clustering-ready`, `embed-then-cluster`, `infeasible`)
- Phase 0 embed precompute workers (for missing embeddings)
- Phase 1 clustering workers (one process per pair)
- Bounded concurrency + retry policy
- Final verification using ``scratch/readonly_full_coverage_export.py``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
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

from study_query_llm.config import config  # noqa: E402
from study_query_llm.db.connection_v2 import DatabaseConnectionV2  # noqa: E402
from study_query_llm.db.models_v2 import Group  # noqa: E402
from study_query_llm.db.write_intent import WriteIntent  # noqa: E402
from study_query_llm.experiments.clustering_analysis_backfill import (  # noqa: E402
    build_manifest,
    fetch_completed_backfill_run_keys,
    snapshot_lineage,
)
from study_query_llm.services.model_registry import ModelRegistry  # noqa: E402


SNAPSHOT_SCOPE: tuple[int, ...] = (6, 9, 10, 17, 18)
FORCED_ENGINES: tuple[str, ...] = ("qwen/qwen3-embedding-4b", "embed-v-4-0")
DEFAULT_MAX_WORKERS = 8
DEFAULT_PAIR_MAX_RETRIES = 2
PHASE0_TIMEOUT_SECONDS = 3 * 60 * 60
PHASE1_TIMEOUT_SECONDS = 6 * 60 * 60
TRANSIENT_EXIT_CODES = {11}
EMBEDDING_LEVEL_INFEASIBLE_METHOD_COUNT = 26


@dataclass(frozen=True)
class PairKey:
    snapshot_id: int
    embedding_engine: str

    @property
    def pair_id(self) -> str:
        return f"snap{int(self.snapshot_id)}__{engine_slug(self.embedding_engine)}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def engine_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(value).strip())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return raw


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        value = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if value:
            return value
    raise SystemExit("No database URL found (CANONICAL_DATABASE_URL / DATABASE_URL)")


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, float]:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"started_at={utc_now()}\n")
        handle.write("cmd=" + " ".join(command) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = 124
            handle.write(f"\nterminated_due_to_timeout_seconds={timeout_seconds}\n")
    return int(returncode), float(time.time() - start)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_openrouter_price_per_1k_tokens(pricing_payload: dict[str, Any]) -> float | None:
    candidates = []
    for key in ("prompt", "input", "prompt_tokens", "input_tokens", "per_token"):
        raw = pricing_payload.get(key)
        if raw is None:
            continue
        try:
            candidates.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return None
    per_token = min(candidates)
    if per_token <= 0:
        return None
    return per_token * 1000.0


def _estimate_tokens_for_rows(row_count: int) -> int:
    # Coarse preflight estimate only; exact token count would require materializing texts.
    # Uses a conservative fixed estimate to surface order-of-magnitude budget.
    return int(max(0, row_count) * 256)


def _parse_worker_status(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {
            "status": "failed",
            "failure_kind": "transient",
            "reason_code": "missing_status_file",
        }
    try:
        payload = _read_json(status_path)
    except Exception:
        return {
            "status": "failed",
            "failure_kind": "transient",
            "reason_code": "invalid_status_file",
        }
    if not isinstance(payload, dict):
        return {
            "status": "failed",
            "failure_kind": "transient",
            "reason_code": "invalid_status_file",
        }
    return payload


def _classify_pair_from_manifest(
    *,
    pair_key: PairKey,
    provider: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    status = "error"
    pair_entry: dict[str, Any] = {}
    pairs = list(manifest.get("pairs") or [])
    if pairs and isinstance(pairs[0], dict):
        pair_entry = dict(pairs[0])
        status = str(pair_entry.get("status") or "error")

    classification = "infeasible"
    reason_code = "lineage_missing"
    if status == "ok":
        classification = "clustering-ready"
        reason_code = ""
    elif status == "no_embedding_batch":
        classification = "embed-then-cluster"
        reason_code = ""
    elif status == "collision_multiple_batches":
        classification = "infeasible"
        reason_code = "collision_multiple_batches"
    elif status == "error":
        classification = "infeasible"
        reason_code = "lineage_missing"
    else:
        classification = "infeasible"
        reason_code = f"manifest_status_{status}"

    return {
        "pair_id": pair_key.pair_id,
        "snapshot_id": int(pair_key.snapshot_id),
        "embedding_engine": str(pair_key.embedding_engine),
        "provider": str(provider),
        "classification": classification,
        "reason_code": reason_code,
        "manifest_status": status,
        "manifest_pair": pair_entry,
    }


def _resolve_provider_from_registry(embedding_engine: str) -> str | None:
    # Prefer a deterministic provider based on configured embedding backends.
    # OpenRouter engines commonly include "/" slug paths.
    eng = str(embedding_engine).strip()
    if not eng:
        return None
    if "/" in eng:
        return "openrouter"
    try:
        available = set(config.get_available_providers())
    except Exception:
        available = set()
    if "openrouter" in available:
        return "openrouter"
    if "azure" in available:
        return "azure"
    if "openai" in available:
        return "openai"
    if "local" in available:
        return "local"
    if "huggingface" in available:
        return "huggingface"
    if "ollama" in available:
        return "ollama"
    return None


async def _openrouter_pricing_table() -> dict[str, float]:
    registry = ModelRegistry()
    data = await registry.refresh_provider("openrouter")
    details = data.get("model_details")
    if not isinstance(details, dict):
        return {}
    out: dict[str, float] = {}
    for model_id, row in details.items():
        if not isinstance(model_id, str) or not isinstance(row, dict):
            continue
        pricing = row.get("pricing")
        if not isinstance(pricing, dict):
            continue
        per_1k = _extract_openrouter_price_per_1k_tokens(pricing)
        if per_1k is not None:
            out[model_id] = per_1k
    return out


def _deterministic_engine_universe(
    *,
    session: Any,
    snapshot_ids: list[int],
) -> tuple[
    list[str],
    dict[int, dict[str, Any]],
    dict[tuple[int, str], str],
    dict[tuple[int, str], list[str]],
]:
    source_df_by_snapshot: dict[int, int] = {}
    row_count_by_snapshot: dict[int, int] = {}
    source_row_count_by_snapshot: dict[int, int] = {}
    for sid in snapshot_ids:
        lineage = snapshot_lineage(session, int(sid))
        source_df_by_snapshot[int(sid)] = int(lineage.source_dataframe_group_id)
        row_count_by_snapshot[int(sid)] = int(lineage.snapshot_row_count)
        source_row_count_by_snapshot[int(sid)] = int(lineage.source_dataframe_row_count)

    engine_set: set[str] = set()
    provider_values: dict[tuple[int, str], set[str]] = defaultdict(set)
    for group in session.query(Group).filter(Group.group_type == "embedding_batch").all():
        md = dict(group.metadata_json or {})
        sdf_id = _safe_int(md.get("source_dataframe_group_id"))
        if sdf_id is None:
            continue
        if sdf_id not in set(source_df_by_snapshot.values()):
            continue
        engine = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        if not engine:
            continue
        provider = str(md.get("provider") or "").strip()
        engine_set.add(engine)
        for sid, sid_sdf in source_df_by_snapshot.items():
            if sid_sdf == sdf_id and provider:
                provider_values[(int(sid), engine)].add(provider)

    engine_set.update(FORCED_ENGINES)

    snapshot_meta: dict[int, dict[str, Any]] = {}
    for sid in snapshot_ids:
        snapshot_meta[int(sid)] = {
            "snapshot_id": int(sid),
            "source_dataframe_group_id": int(source_df_by_snapshot[int(sid)]),
            "snapshot_row_count": int(row_count_by_snapshot[int(sid)]),
            "source_dataframe_row_count": int(source_row_count_by_snapshot[int(sid)]),
        }
    provider_observed: dict[tuple[int, str], str] = {}
    provider_candidates: dict[tuple[int, str], list[str]] = {}
    for key, values in provider_values.items():
        ordered = sorted(v for v in values if v)
        provider_candidates[key] = ordered
        if len(ordered) == 1:
            provider_observed[key] = ordered[0]
    return sorted(engine_set), snapshot_meta, provider_observed, provider_candidates


def _run_coverage_export() -> dict[str, Any]:
    script_path = PROJECT_ROOT / "scratch" / "readonly_full_coverage_export.py"
    if not script_path.is_file():
        raise RuntimeError(f"Coverage export script not found: {script_path}")
    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Coverage export failed (exit={proc.returncode}):\n{combined}")
    export_dir: Path | None = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("EXPORT_DIR="):
            export_dir = Path(line.split("=", 1)[1].strip())
            break
    if export_dir is None:
        # Fallback: detect latest export dir
        exports_root = PROJECT_ROOT / "scratch" / "exports"
        candidates = sorted(
            [p for p in exports_root.glob("full_coverage_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            export_dir = candidates[0]
    if export_dir is None:
        raise RuntimeError("Could not resolve coverage export directory from script output.")
    summary_path = export_dir / "export_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing export summary: {summary_path}")
    summary = _read_json(summary_path)
    return {
        "export_dir": str(export_dir),
        "summary_path": str(summary_path),
        "summary": summary,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }


def _update_pair_status(
    *,
    work_root: Path,
    pair_id: str,
    payload: dict[str, Any],
) -> None:
    _write_json(work_root / "pairs" / pair_id / "status.json", payload)


def _retry_backoff_seconds(attempt_index: int) -> float:
    # attempt_index starts at 0
    return float(min(300.0, 2.0 ** attempt_index))


def _run_worker_phase(
    *,
    phase: str,
    pair: dict[str, Any],
    work_root: Path,
    database_url: str,
    max_retries: int,
) -> dict[str, Any]:
    snapshot_id = int(pair["snapshot_id"])
    embedding_engine = str(pair["embedding_engine"])
    provider = str(pair["provider"])
    pair_id = str(pair["pair_id"])

    pair_root = work_root / "pairs" / pair_id
    pair_root.mkdir(parents=True, exist_ok=True)
    logs_dir = pair_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    max_attempts = max(1, int(max_retries) + 1)
    for attempt in range(max_attempts):
        log_path = logs_dir / f"{phase}.attempt{attempt + 1}.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "living" / "run_per_pair_backfill_worker.py"),
            "--snapshot-id",
            str(snapshot_id),
            "--embedding-engine",
            embedding_engine,
            "--provider",
            provider,
            "--work-dir",
            str(pair_root),
            "--phase",
            phase,
            "--database-url",
            database_url,
            "--resume",
        ]
        timeout_seconds = PHASE0_TIMEOUT_SECONDS if phase == "phase0" else PHASE1_TIMEOUT_SECONDS
        exit_code, elapsed_seconds = _run_subprocess(
            command,
            cwd=PROJECT_ROOT,
            log_path=log_path,
            timeout_seconds=timeout_seconds,
        )
        status_payload = _parse_worker_status(pair_root / "status.json")
        failure_kind = str(status_payload.get("failure_kind") or "")
        reason_code = str(status_payload.get("reason_code") or "")
        terminal_ok = int(exit_code) == 0 and str(status_payload.get("status")) == "completed"
        attempt_payload = {
            "attempt": attempt + 1,
            "exit_code": int(exit_code),
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "log_path": str(log_path),
            "failure_kind": failure_kind,
            "reason_code": reason_code,
            "terminal_ok": terminal_ok,
            "status_payload": status_payload,
            "updated_at": utc_now(),
        }
        attempts.append(attempt_payload)
        if terminal_ok:
            return {
                "status": "completed",
                "attempts": attempts,
                "failure_kind": None,
                "reason_code": None,
            }
        transient = int(exit_code) in TRANSIENT_EXIT_CODES or failure_kind == "transient"
        if transient and attempt + 1 < max_attempts:
            time.sleep(_retry_backoff_seconds(attempt))
            continue
        return {
            "status": "failed",
            "attempts": attempts,
            "failure_kind": "transient" if transient else "permanent",
            "reason_code": reason_code or ("worker_nonzero_exit" if exit_code != 0 else "worker_failed"),
        }
    return {
        "status": "failed",
        "attempts": attempts,
        "failure_kind": "transient",
        "reason_code": "retry_budget_exhausted",
    }


def _collect_infeasible_pairs(classification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in classification_rows:
        if str(row.get("classification")) == "infeasible":
            out.append(dict(row))
    return out


def _run_pairs_with_bounded_concurrency(
    *,
    pairs: list[dict[str, Any]],
    phase: str,
    work_root: Path,
    database_url: str,
    max_workers: int,
    pair_max_retries: int,
    halt_on_first_permanent: bool,
    halt_on_failure_rate: float | None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not pairs:
        return {}, False

    ordered_pairs = sorted(
        [dict(p) for p in pairs],
        key=lambda row: (int(row.get("snapshot_id") or 0), str(row.get("embedding_engine") or "")),
    )
    results: dict[str, dict[str, Any]] = {}
    total_failures = 0
    total_finished = 0
    halted = False
    queue = list(ordered_pairs)

    def _submit(executor: ThreadPoolExecutor, pair_row: dict[str, Any]):
        future = executor.submit(
            _run_worker_phase,
            phase=phase,
            pair=pair_row,
            work_root=work_root,
            database_url=database_url,
            max_retries=pair_max_retries,
        )
        return future, pair_row

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        in_flight: dict[Any, dict[str, Any]] = {}
        while queue and len(in_flight) < max(1, int(max_workers)):
            pair_row = queue.pop(0)
            future, pair_payload = _submit(executor, pair_row)
            in_flight[future] = pair_payload

        while in_flight:
            done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                pair_row = in_flight.pop(future)
                pair_id = str(pair_row["pair_id"])
                phase_result = future.result()
                results[pair_id] = phase_result

                status_payload = {
                    "pair_id": pair_id,
                    "phase": phase,
                    "status": phase_result["status"],
                    "failure_kind": phase_result.get("failure_kind"),
                    "reason_code": phase_result.get("reason_code"),
                    "attempts": phase_result.get("attempts", []),
                    "updated_at": utc_now(),
                }
                _update_pair_status(work_root=work_root, pair_id=pair_id, payload=status_payload)

                total_finished += 1
                if phase_result["status"] != "completed":
                    total_failures += 1

                is_permanent = str(phase_result.get("failure_kind")) == "permanent"
                if halt_on_first_permanent and is_permanent:
                    halted = True
                if halt_on_failure_rate is not None and total_finished > 0:
                    failure_rate = float(total_failures) / float(total_finished)
                    if failure_rate >= float(halt_on_failure_rate):
                        halted = True

                if halted:
                    queue = []

            if halted:
                continue

            while queue and len(in_flight) < max(1, int(max_workers)):
                pair_row = queue.pop(0)
                future, pair_payload = _submit(executor, pair_row)
                in_flight[future] = pair_payload

    return results, halted


def _verify_acceptance(
    *,
    work_root: Path,
    infeasible_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = _run_coverage_export()
    summary = dict(coverage.get("summary") or {})

    infeasible_map: dict[tuple[int, str], dict[str, Any]] = {}
    total_infeasible_missing_methods = 0
    for row in infeasible_pairs:
        sid = int(row["snapshot_id"])
        eng = str(row["embedding_engine"])
        missing_methods = _safe_int(row.get("missing_method_count")) or 0
        total_infeasible_missing_methods += int(missing_methods)
        infeasible_map[(sid, eng)] = dict(row)

    zero_coverage_pairs = list(summary.get("zero_coverage_pairs") or [])
    unexpected_zero_coverage: list[dict[str, Any]] = []
    for pair in zero_coverage_pairs:
        if not isinstance(pair, dict):
            continue
        sid = _safe_int(pair.get("snapshot_id"))
        eng = str(pair.get("embedding_engine") or "")
        if sid is None:
            continue
        if (int(sid), eng) not in infeasible_map:
            unexpected_zero_coverage.append({"snapshot_id": int(sid), "embedding_engine": eng})

    grand_completed_cells = _safe_int(summary.get("grand_completed_cells")) or 0
    grand_expected_cells = _safe_int(summary.get("grand_expected_cells")) or 0
    expected_completed_modulo_infeasible = grand_expected_cells - total_infeasible_missing_methods
    completed_cells_ok = int(grand_completed_cells) == int(expected_completed_modulo_infeasible)

    payload = {
        "generated_at": utc_now(),
        "coverage_export_dir": coverage.get("export_dir"),
        "coverage_summary_path": coverage.get("summary_path"),
        "grand_completed_cells": grand_completed_cells,
        "grand_expected_cells": grand_expected_cells,
        "expected_completed_modulo_infeasible": expected_completed_modulo_infeasible,
        "infeasible_missing_method_total": total_infeasible_missing_methods,
        "completed_cells_ok": bool(completed_cells_ok),
        "unexpected_zero_coverage_pairs": unexpected_zero_coverage,
        "unexpected_zero_coverage_count": len(unexpected_zero_coverage),
        "zero_coverage_pairs_total": len(zero_coverage_pairs),
        "coverage_summary": summary,
    }
    _write_json(work_root / "verification_summary.json", payload)
    return payload


def _acceptance_markdown(
    *,
    work_root: Path,
    orchestrator_summary: dict[str, Any],
    verification_summary: dict[str, Any],
) -> None:
    lines = [
        "# Per-Pair Backfill Acceptance Summary",
        "",
        f"- generated_at: `{utc_now()}`",
        f"- work_dir: `{work_root}`",
        "",
        "## Pair Outcomes",
        "",
        f"- total_pairs: **{orchestrator_summary.get('total_pairs', 0)}**",
        f"- clustering_ready_initial: **{orchestrator_summary.get('initial_class_counts', {}).get('clustering-ready', 0)}**",
        f"- embed_then_cluster_initial: **{orchestrator_summary.get('initial_class_counts', {}).get('embed-then-cluster', 0)}**",
        f"- infeasible_final: **{orchestrator_summary.get('final_class_counts', {}).get('infeasible', 0)}**",
        "",
        "## Coverage Verification",
        "",
        f"- completed_cells_ok: **{verification_summary.get('completed_cells_ok')}**",
        f"- grand_completed_cells: **{verification_summary.get('grand_completed_cells')}**",
        f"- expected_completed_modulo_infeasible: **{verification_summary.get('expected_completed_modulo_infeasible')}**",
        f"- unexpected_zero_coverage_count: **{verification_summary.get('unexpected_zero_coverage_count')}**",
        "",
    ]
    (work_root / "ACCEPTANCE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_has_embedding_level_reason(row: dict[str, Any]) -> bool:
    return str(row.get("reason_code") or "") in {
        "provider_unresolved",
        "provider_ambiguous",
        "lineage_missing",
        "collision_multiple_batches",
        "embed_failed",
        "still_no_embedding_after_phase0",
    }


def _pair_manifest_refresh(
    *,
    db: DatabaseConnectionV2,
    pair: dict[str, Any],
) -> dict[str, Any]:
    with db.session_scope() as session:
        return build_manifest(
            session,
            embedding_engine=str(pair["embedding_engine"]),
            provider=str(pair["provider"]),
            snapshot_ids=[int(pair["snapshot_id"])],
        )


def _compute_missing_method_count_for_pair(
    *,
    db: DatabaseConnectionV2,
    pair: dict[str, Any],
) -> int:
    reason_code = str(pair.get("reason_code") or "")
    if reason_code in {
        "provider_unresolved",
        "provider_ambiguous",
        "lineage_missing",
        "collision_multiple_batches",
        "embed_failed",
        "still_no_embedding_after_phase0",
    }:
        return int(EMBEDDING_LEVEL_INFEASIBLE_METHOD_COUNT)

    manifest = _pair_manifest_refresh(db=db, pair=pair)
    targets = list(manifest.get("targets") or [])
    if not targets:
        return int(EMBEDDING_LEVEL_INFEASIBLE_METHOD_COUNT)
    with db.session_scope() as session:
        completed = fetch_completed_backfill_run_keys(session)
    missing = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        run_key = str(target.get("analysis_run_key") or "")
        if run_key and run_key not in completed:
            missing += 1
    return int(missing)


def _load_price_overrides(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = _read_json(path)
    out: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            out[key] = numeric
    return out


def _determine_budget_estimate(
    *,
    pairs: list[dict[str, Any]],
    max_workers: int,
    provider_price_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    class_counts = Counter(str(row.get("classification")) for row in pairs)
    embed_pairs = [row for row in pairs if str(row.get("classification")) == "embed-then-cluster"]
    phase0_row_volume = sum(_safe_int(row.get("snapshot_row_count")) or 0 for row in embed_pairs)
    runnable = [row for row in pairs if str(row.get("classification")) in ("clustering-ready", "embed-then-cluster")]

    pricing_table: dict[str, float] = {}
    try:
        pricing_table = asyncio.run(_openrouter_pricing_table())
    except Exception:
        pricing_table = {}
    overrides = dict(provider_price_overrides or {})
    pricing_table.update(overrides)

    token_volume_by_engine: dict[str, int] = defaultdict(int)
    for row in embed_pairs:
        engine = str(row.get("embedding_engine") or "")
        tokens = _estimate_tokens_for_rows(_safe_int(row.get("snapshot_row_count")) or 0)
        token_volume_by_engine[engine] += int(tokens)

    cost_estimates_by_engine: dict[str, float | None] = {}
    for engine, tokens in sorted(token_volume_by_engine.items()):
        per_1k = pricing_table.get(engine)
        if per_1k is None:
            cost_estimates_by_engine[engine] = None
        else:
            cost_estimates_by_engine[engine] = float(tokens) / 1000.0 * float(per_1k)

    total_runnable_pairs = len(runnable)
    waves = (total_runnable_pairs + max(1, int(max_workers)) - 1) // max(1, int(max_workers))

    return {
        "generated_at": utc_now(),
        "pair_counts_by_class": dict(class_counts),
        "phase0_missing_embedding_pairs": len(embed_pairs),
        "phase0_row_volume": int(phase0_row_volume),
        "phase0_token_volume_by_engine": dict(sorted(token_volume_by_engine.items())),
        "phase0_cost_estimates_usd_by_engine": cost_estimates_by_engine,
        "max_workers": int(max_workers),
        "total_runnable_pairs": int(total_runnable_pairs),
        "worst_case_worker_waves": int(waves),
        "pricing_source": (
            "model_registry_openrouter_best_effort+operator_overrides"
            if overrides
            else "model_registry_openrouter_best_effort"
        ),
        "provider_price_overrides": overrides,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dry-run", "all"),
        default="all",
        help="dry-run builds inventories and budgets only; all runs phase0+phase1+verify",
    )
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--pair-max-retries", type=int, default=DEFAULT_PAIR_MAX_RETRIES)
    parser.add_argument("--halt-on-first-permanent", action="store_true")
    parser.add_argument("--halt-on-failure-rate", type=float, default=None)
    parser.add_argument("--accept-budget", action="store_true")
    parser.add_argument("--provider-price-json", type=Path, default=None)
    parser.add_argument(
        "--snapshot-ids",
        type=int,
        nargs="+",
        default=list(SNAPSHOT_SCOPE),
        help=f"snapshot ids in scope (default: {list(SNAPSHOT_SCOPE)})",
    )
    parser.add_argument(
        "--run-stamp",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        help="timestamp suffix for work directory",
    )
    args = parser.parse_args(argv)

    database_url = _resolve_database_url(args.database_url)
    db = DatabaseConnectionV2(
        database_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )

    run_stamp = str(args.run_stamp).strip()
    work_root = PROJECT_ROOT / "experimental_results" / "backfill_per_pair" / run_stamp
    work_root.mkdir(parents=True, exist_ok=True)

    snapshot_ids = sorted({int(s) for s in args.snapshot_ids})
    with db.session_scope() as session:
        (
            engine_universe,
            snapshot_meta,
            provider_observed,
            provider_candidates,
        ) = _deterministic_engine_universe(
            session=session,
            snapshot_ids=snapshot_ids,
        )

    pair_rows: list[dict[str, Any]] = []
    provider_resolution: dict[str, str] = {}
    unresolved_pairs: list[dict[str, Any]] = []
    for sid in snapshot_ids:
        for engine in engine_universe:
            key = PairKey(snapshot_id=int(sid), embedding_engine=str(engine))
            observed_provider = provider_observed.get((int(sid), str(engine)))
            candidates = list(provider_candidates.get((int(sid), str(engine)), []))
            provider = observed_provider or provider_resolution.get(engine) or _resolve_provider_from_registry(engine)
            if provider:
                provider_resolution[engine] = provider
            row = {
                "pair_id": key.pair_id,
                "snapshot_id": int(sid),
                "embedding_engine": str(engine),
                "provider": str(provider or ""),
                "source_dataframe_group_id": int(snapshot_meta[int(sid)]["source_dataframe_group_id"]),
                "snapshot_row_count": int(snapshot_meta[int(sid)]["snapshot_row_count"]),
                "source_dataframe_row_count": int(snapshot_meta[int(sid)]["source_dataframe_row_count"]),
                "lineage_key": [
                    int(snapshot_meta[int(sid)]["source_dataframe_group_id"]),
                    int(snapshot_meta[int(sid)]["source_dataframe_row_count"]),
                ],
                "classification": "unknown",
                "reason_code": "",
                "manifest_status": "",
                "manifest_pair": {},
                "provider_candidates": candidates,
            }
            if len(candidates) > 1:
                row["classification"] = "infeasible"
                row["reason_code"] = "provider_ambiguous"
                unresolved_pairs.append(row)
            elif not provider:
                row["classification"] = "infeasible"
                row["reason_code"] = "provider_unresolved"
                unresolved_pairs.append(row)
            pair_rows.append(row)

    pair_universe_payload = {
        "generated_at": utc_now(),
        "snapshot_ids": snapshot_ids,
        "engine_universe": engine_universe,
        "forced_engines": list(FORCED_ENGINES),
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
    }
    _write_json(work_root / "pair_universe.json", pair_universe_payload)

    classification_rows: list[dict[str, Any]] = []
    with db.session_scope() as session:
        for row in pair_rows:
            pair_key = PairKey(
                snapshot_id=int(row["snapshot_id"]),
                embedding_engine=str(row["embedding_engine"]),
            )
            provider = str(row.get("provider") or "")
            if not provider:
                classification_rows.append(dict(row))
                continue
            manifest = build_manifest(
                session,
                embedding_engine=str(pair_key.embedding_engine),
                provider=str(provider),
                snapshot_ids=[int(pair_key.snapshot_id)],
            )
            pair_dir = work_root / "pairs" / pair_key.pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            _write_json(pair_dir / "pair_spec.json", row)
            _write_json(pair_dir / "preflight_manifest.json", manifest)
            classified = _classify_pair_from_manifest(
                pair_key=pair_key,
                provider=str(provider),
                manifest=manifest,
            )
            classified.update(
                {
                    "source_dataframe_group_id": int(row["source_dataframe_group_id"]),
                    "snapshot_row_count": int(row["snapshot_row_count"]),
                    "source_dataframe_row_count": int(row["source_dataframe_row_count"]),
                    "lineage_key": list(row["lineage_key"]),
                }
            )
            classification_rows.append(classified)
            _update_pair_status(
                work_root=work_root,
                pair_id=pair_key.pair_id,
                payload={
                    "pair_id": pair_key.pair_id,
                    "status": "classified",
                    "classification": classified["classification"],
                    "reason_code": classified.get("reason_code") or "",
                    "updated_at": utc_now(),
                },
            )

    for row in unresolved_pairs:
        _update_pair_status(
            work_root=work_root,
            pair_id=str(row["pair_id"]),
            payload={
                "pair_id": str(row["pair_id"]),
                "status": "failed",
                "classification": "infeasible",
                "reason_code": "provider_unresolved",
                "updated_at": utc_now(),
            },
        )

    _write_json(
        work_root / "pair_classification.json",
        {"generated_at": utc_now(), "pairs": classification_rows},
    )
    classification_summary = {
        "generated_at": utc_now(),
        "snapshot_ids": snapshot_ids,
        "engine_count": len(engine_universe),
        "pair_count": len(classification_rows),
        "class_counts": dict(Counter(str(r.get("classification")) for r in classification_rows)),
        "provider_resolution": dict(sorted(provider_resolution.items())),
    }
    _write_json(work_root / "classification_summary.json", classification_summary)
    infeasible_pairs = _collect_infeasible_pairs(classification_rows)
    _write_json(work_root / "infeasible_pairs.json", {"generated_at": utc_now(), "pairs": infeasible_pairs})

    provider_price_overrides = _load_price_overrides(args.provider_price_json)
    budget = _determine_budget_estimate(
        pairs=classification_rows,
        max_workers=max(1, int(args.max_workers)),
        provider_price_overrides=provider_price_overrides,
    )
    _write_json(work_root / "budget_estimate.json", budget)
    budget_md = [
        "# Budget Estimate",
        "",
        f"- generated_at: `{budget['generated_at']}`",
        f"- pair_counts_by_class: `{budget['pair_counts_by_class']}`",
        f"- phase0_missing_embedding_pairs: **{budget['phase0_missing_embedding_pairs']}**",
        f"- phase0_row_volume: **{budget['phase0_row_volume']}**",
        f"- worst_case_worker_waves: **{budget['worst_case_worker_waves']}**",
    ]
    (work_root / "budget_estimate.md").write_text("\n".join(budget_md) + "\n", encoding="utf-8")

    if args.phase == "dry-run":
        summary = {
            "generated_at": utc_now(),
            "phase": "dry-run",
            "work_dir": str(work_root),
            "initial_class_counts": dict(Counter(str(r.get("classification")) for r in classification_rows)),
        }
        _write_json(work_root / "orchestrator_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0

    if not bool(args.accept_budget):
        raise SystemExit("Budget estimate generated. Re-run with --accept-budget to execute.")

    max_workers = max(1, int(args.max_workers))
    pair_max_retries = max(0, int(args.pair_max_retries))

    initial_class_counts = Counter(str(r.get("classification")) for r in classification_rows)
    phase0_pairs = [
        r for r in classification_rows if str(r.get("classification")) == "embed-then-cluster"
    ]
    phase1_pairs = [
        r for r in classification_rows if str(r.get("classification")) == "clustering-ready"
    ]

    # Phase 0: embed workers
    phase0_results, phase0_halted = _run_pairs_with_bounded_concurrency(
        pairs=phase0_pairs,
        phase="phase0",
        work_root=work_root,
        database_url=database_url,
        max_workers=max_workers,
        pair_max_retries=pair_max_retries,
        halt_on_first_permanent=bool(args.halt_on_first_permanent),
        halt_on_failure_rate=float(args.halt_on_failure_rate) if args.halt_on_failure_rate is not None else None,
    )
    if phase0_halted:
        for row in phase0_pairs:
            pair_id = str(row["pair_id"])
            if pair_id in phase0_results:
                continue
            phase0_results[pair_id] = {
                "status": "failed",
                "attempts": [],
                "failure_kind": "permanent",
                "reason_code": "halted_by_operator_policy",
            }
            _update_pair_status(
                work_root=work_root,
                pair_id=pair_id,
                payload={
                    "pair_id": pair_id,
                    "phase": "phase0",
                    "status": "failed",
                    "failure_kind": "permanent",
                    "reason_code": "halted_by_operator_policy",
                    "attempts": [],
                    "updated_at": utc_now(),
                },
            )

    # Reclassify phase0 pairs
    reclassified_pairs: list[dict[str, Any]] = []
    with db.session_scope() as session:
        for row in classification_rows:
            if str(row.get("classification")) != "embed-then-cluster":
                reclassified_pairs.append(dict(row))
                continue
            pair_key = PairKey(
                snapshot_id=int(row["snapshot_id"]),
                embedding_engine=str(row["embedding_engine"]),
            )
            provider = str(row.get("provider") or "")
            manifest = build_manifest(
                session,
                embedding_engine=str(pair_key.embedding_engine),
                provider=provider,
                snapshot_ids=[int(pair_key.snapshot_id)],
            )
            classified = _classify_pair_from_manifest(
                pair_key=pair_key,
                provider=provider,
                manifest=manifest,
            )
            classified.update(
                {
                    "source_dataframe_group_id": int(row["source_dataframe_group_id"]),
                    "snapshot_row_count": int(row["snapshot_row_count"]),
                    "source_dataframe_row_count": int(row["source_dataframe_row_count"]),
                    "lineage_key": list(row["lineage_key"]),
                }
            )
            if classified["classification"] == "embed-then-cluster":
                classified["classification"] = "infeasible"
                classified["reason_code"] = "still_no_embedding_after_phase0"
            reclassified_pairs.append(classified)
            _write_json(work_root / "pairs" / pair_key.pair_id / "post_phase0_manifest.json", manifest)

    classification_rows = reclassified_pairs
    infeasible_pairs = _collect_infeasible_pairs(classification_rows)
    _write_json(
        work_root / "pair_classification_post_phase0.json",
        {"generated_at": utc_now(), "pairs": classification_rows},
    )
    post_phase0_summary = {
        "generated_at": utc_now(),
        "class_counts": dict(Counter(str(r.get("classification")) for r in classification_rows)),
        "phase0_completed_pairs": sum(
            1 for result in phase0_results.values() if str(result.get("status")) == "completed"
        ),
        "phase0_failed_pairs": sum(
            1 for result in phase0_results.values() if str(result.get("status")) != "completed"
        ),
        "phase0_halted": bool(phase0_halted),
    }
    _write_json(work_root / "phase0_summary.json", post_phase0_summary)
    _write_json(work_root / "infeasible_pairs.json", {"generated_at": utc_now(), "pairs": infeasible_pairs})

    phase1_pairs = [
        r for r in classification_rows if str(r.get("classification")) == "clustering-ready"
    ]

    # Phase 1: clustering workers
    phase1_results, phase1_halted = _run_pairs_with_bounded_concurrency(
        pairs=phase1_pairs,
        phase="phase1",
        work_root=work_root,
        database_url=database_url,
        max_workers=max_workers,
        pair_max_retries=pair_max_retries,
        halt_on_first_permanent=bool(args.halt_on_first_permanent),
        halt_on_failure_rate=float(args.halt_on_failure_rate) if args.halt_on_failure_rate is not None else None,
    )
    if phase1_halted:
        for row in phase1_pairs:
            pair_id = str(row["pair_id"])
            if pair_id in phase1_results:
                continue
            phase1_results[pair_id] = {
                "status": "failed",
                "attempts": [],
                "failure_kind": "permanent",
                "reason_code": "halted_by_operator_policy",
            }
            _update_pair_status(
                work_root=work_root,
                pair_id=pair_id,
                payload={
                    "pair_id": pair_id,
                    "phase": "phase1",
                    "status": "failed",
                    "failure_kind": "permanent",
                    "reason_code": "halted_by_operator_policy",
                    "attempts": [],
                    "updated_at": utc_now(),
                },
            )

    # Final class reconciliation after phase1
    final_rows: list[dict[str, Any]] = []
    for row in classification_rows:
        if str(row.get("classification")) != "clustering-ready":
            final_rows.append(dict(row))
            continue
        pair_id = str(row["pair_id"])
        result = phase1_results.get(pair_id)
        if result and str(result.get("status")) == "completed":
            final_rows.append(dict(row))
            continue
        failed_row = dict(row)
        failed_row["classification"] = "infeasible"
        failed_row["reason_code"] = str(
            (result or {}).get("reason_code") or "phase1_failed"
        )
        final_rows.append(failed_row)

    # Compute infeasible missing-method counts
    final_infeasible_pairs = _collect_infeasible_pairs(final_rows)
    for row in final_infeasible_pairs:
        row["missing_method_count"] = _compute_missing_method_count_for_pair(db=db, pair=row)
    _write_json(work_root / "infeasible_pairs.json", {"generated_at": utc_now(), "pairs": final_infeasible_pairs})

    verification_summary = _verify_acceptance(
        work_root=work_root,
        infeasible_pairs=final_infeasible_pairs,
    )
    verification_summary["embedding_level_infeasible_method_count"] = int(
        EMBEDDING_LEVEL_INFEASIBLE_METHOD_COUNT
    )
    verification_summary["embedding_level_infeasible_pair_count"] = int(
        sum(1 for row in final_infeasible_pairs if _pair_has_embedding_level_reason(row))
    )
    _write_json(work_root / "verification_summary.json", verification_summary)

    final_class_counts = Counter(str(r.get("classification")) for r in final_rows)
    orchestrator_summary = {
        "generated_at": utc_now(),
        "phase": "all",
        "work_dir": str(work_root),
        "snapshot_ids": snapshot_ids,
        "engine_count": len(engine_universe),
        "total_pairs": len(final_rows),
        "initial_class_counts": dict(initial_class_counts),
        "final_class_counts": dict(final_class_counts),
        "phase0_completed": sum(1 for r in phase0_results.values() if str(r.get("status")) == "completed"),
        "phase0_failed": sum(1 for r in phase0_results.values() if str(r.get("status")) != "completed"),
        "phase1_completed": sum(1 for r in phase1_results.values() if str(r.get("status")) == "completed"),
        "phase1_failed": sum(1 for r in phase1_results.values() if str(r.get("status")) != "completed"),
        "verification_summary_path": str(work_root / "verification_summary.json"),
        "infeasible_pairs_path": str(work_root / "infeasible_pairs.json"),
    }
    _write_json(work_root / "orchestrator_summary.json", orchestrator_summary)
    _acceptance_markdown(
        work_root=work_root,
        orchestrator_summary=orchestrator_summary,
        verification_summary=verification_summary,
    )

    print(json.dumps(orchestrator_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

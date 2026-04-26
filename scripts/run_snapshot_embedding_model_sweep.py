#!/usr/bin/env python3
"""Run embedding-model sweeps for a snapshot's source dataframe.

This script executes `pipeline.embed()` across many embedding models with:
- bounded model-level concurrency (`--engine-concurrency`)
- provider-budget-aware effective concurrency math
- optional one-pass serial retry for failures (`--retry-failed-serial`)
- optional model pre-validation + exclusion controls
- optional chunk circuit-breaker fallback profile
- snapshot-aware target resolution (snapshot -> source_dataframe_group_id)

Example:
  python scripts/run_snapshot_embedding_model_sweep.py \
    --snapshot-group-id 9 \
    --provider openrouter \
    --max-models 24 \
    --engine-concurrency 4 \
    --retry-failed-serial \
    --force
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.embed import embed
from study_query_llm.providers.factory import ProviderFactory


@dataclass
class SweepRunConfig:
    snapshot_group_id: int
    source_dataframe_group_id: int
    provider: str
    models: List[str]
    database_url: str
    requested_engine_concurrency: int
    engine_concurrency: int
    provider_concurrency_budget: int | None
    requested_chunk_worker_concurrency: int
    chunk_worker_concurrency: int
    chunk_circuit_breaker_enabled: bool
    chunk_failure_fallback_threshold: int
    disable_parallel_chunks_env: bool
    retry_failed_serial: bool
    force: bool
    entry_max: int | None
    chunk_size: int
    max_retries: int
    initial_wait_seconds: float
    max_wait_seconds: float
    singleflight_lease_seconds: int
    singleflight_wait_timeout_seconds: float
    singleflight_poll_seconds: float
    embed_timeout_seconds: float
    pair_timeout_seconds: float
    pre_validate_models: bool
    validation_concurrency: int
    validation_cache_ttl_seconds: float
    validation_timeout_seconds: float
    excluded_models: List[str]
    excluded_models_file: str | None


UNAVAILABLE_MODEL_PATTERNS = (
    "does not exist",
    "not found",
    "invalid model",
    "unknown model",
    "no endpoints found",
    "unsupported model",
    "not available",
)

AUTO_QUARANTINE_ERROR_PATTERNS = (
    "nonetype",
    "malformedembeddingresponseerror",
    "embedding response is missing 'data'",
    "embedding response returned empty 'data'",
    "does not exist",
    "not found",
    "invalid model",
    "unknown model",
)

MODEL_VALIDATION_CACHE_PATH = (
    PROJECT_ROOT / ".cache" / "embedding_model_validation_cache.json"
)
DISABLE_PARALLEL_CHUNKS_ENV = "SQLLM_DISABLE_PARALLEL_CHUNKS"


def _coalesce_env(key: str, env_file: Dict[str, Any]) -> str:
    env_val = os.environ.get(key)
    if env_val is not None and str(env_val).strip() != "":
        return str(env_val).strip()
    file_val = env_file.get(key)
    if file_val is None:
        return ""
    return str(file_val).strip()


def _normalize_model_name(value: str) -> str:
    return str(value or "").strip().lower()


def _env_flag_enabled(name: str) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_effective_engine_concurrency(
    *,
    engine_concurrency: int,
    provider_budget: int | None,
    chunk_worker_concurrency: int,
) -> int:
    requested = max(1, int(engine_concurrency))
    if provider_budget is None or int(provider_budget) <= 0:
        return requested
    chunk_workers = max(1, int(chunk_worker_concurrency))
    # Budget math is intentionally enforced at the outer model-dispatch boundary.
    # Each model run executes inside its own asyncio.run()/thread context.
    budget_limited = max(1, math.floor(int(provider_budget) / chunk_workers))
    return max(1, min(requested, budget_limited))


def _read_excluded_models_file(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise ValueError(f"exclude-models file not found: {file_path}")
    raw = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".json":
        payload = json.loads(raw)
        if isinstance(payload, dict):
            models = payload.get("models", [])
        else:
            models = payload
        if not isinstance(models, list):
            raise ValueError(
                "exclude-models JSON must be a list or an object with `models` list"
            )
        return [str(model).strip() for model in models if str(model).strip()]
    return [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _resolve_excluded_models(
    *,
    explicit: List[str],
    file_path: str | None,
) -> List[str]:
    from_file: List[str] = []
    if file_path:
        from_file = _read_excluded_models_file(file_path)
    merged: List[str] = []
    seen: set[str] = set()
    for model in [*explicit, *from_file]:
        normalized = _normalize_model_name(model)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        merged.append(normalized)
    return merged


def _is_model_unavailable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(pattern in msg for pattern in UNAVAILABLE_MODEL_PATTERNS)


def _load_validation_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, dict)
    }


def _save_validation_cache(path: Path, payload: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


async def _probe_model_best_effort(
    *,
    provider: str,
    model: str,
    timeout_seconds: float,
) -> Tuple[bool, Optional[str]]:
    embedding_provider = ProviderFactory().create_embedding_provider(provider)
    async with embedding_provider:
        try:
            is_valid = await asyncio.wait_for(
                embedding_provider.validate_model(model),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            if _is_model_unavailable_error(exc):
                return False, f"{type(exc).__name__}: {exc}"
            # Non-deterministic probe failures (rate limits/network) are not hard-fail.
            return True, f"inconclusive validate_model: {type(exc).__name__}: {exc}"
        if not is_valid:
            return False, "validate_model returned False"

        if provider != "openrouter":
            return True, None

        try:
            await asyncio.wait_for(
                embedding_provider.create_embeddings(
                    ["model availability probe"], model
                ),
                timeout=timeout_seconds,
            )
            return True, None
        except Exception as exc:
            if _is_model_unavailable_error(exc):
                return False, f"{type(exc).__name__}: {exc}"
            return True, f"inconclusive probe call: {type(exc).__name__}: {exc}"


async def _best_effort_filter_models(
    *,
    provider: str,
    models: List[str],
    validation_concurrency: int,
    validation_cache_ttl_seconds: float,
    validation_timeout_seconds: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not models:
        return [], []

    now = time.time()
    cache = _load_validation_cache(MODEL_VALIDATION_CACHE_PATH)
    cache_changed = False
    unavailable: List[Dict[str, Any]] = []
    keep_models: List[str] = []
    needs_probe: List[str] = []

    for model in models:
        cache_key = f"{provider}:{_normalize_model_name(model)}"
        entry = cache.get(cache_key)
        checked_at = float(entry.get("checked_at_unix") or 0.0) if entry else 0.0
        is_fresh = (now - checked_at) <= float(validation_cache_ttl_seconds)
        if entry and is_fresh:
            if bool(entry.get("ok")):
                keep_models.append(model)
            else:
                unavailable.append(
                    {
                        "model": model,
                        "source": "validation_cache",
                        "reason": str(entry.get("reason") or "cached unavailable"),
                    }
                )
            continue
        needs_probe.append(model)

    if needs_probe:
        semaphore = asyncio.Semaphore(max(1, int(validation_concurrency)))

        async def _probe_one(model: str) -> Dict[str, Any]:
            async with semaphore:
                ok, reason = await _probe_model_best_effort(
                    provider=provider,
                    model=model,
                    timeout_seconds=float(validation_timeout_seconds),
                )
                return {"model": model, "ok": bool(ok), "reason": reason}

        probe_results = await asyncio.gather(*[_probe_one(model) for model in needs_probe])
        for row in probe_results:
            model = str(row["model"])
            ok = bool(row["ok"])
            reason = row.get("reason")
            cache_key = f"{provider}:{_normalize_model_name(model)}"
            cache[cache_key] = {
                "ok": ok,
                "reason": str(reason or ""),
                "checked_at_unix": now,
            }
            cache_changed = True
            if ok:
                keep_models.append(model)
            else:
                unavailable.append(
                    {
                        "model": model,
                        "source": "probe",
                        "reason": str(reason or "probe unavailable"),
                    }
                )

    if cache_changed:
        _save_validation_cache(MODEL_VALIDATION_CACHE_PATH, cache)

    keep_order = {model: idx for idx, model in enumerate(models)}
    keep_models.sort(key=lambda model: keep_order.get(model, 0))
    return keep_models, unavailable


def _compute_auto_quarantine_candidates(
    *,
    final_errors: List[Dict[str, Any]],
    prefiltered_unavailable: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()

    for row in prefiltered_unavailable:
        model = str(row.get("model") or "").strip()
        if not model:
            continue
        norm = _normalize_model_name(model)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(
            {
                "model": model,
                "reason": str(row.get("reason") or "filtered unavailable model"),
                "source": str(row.get("source") or "prevalidation"),
            }
        )

    for row in final_errors:
        model = str(row.get("model") or "").strip()
        err = str(row.get("error") or "").strip()
        if not model or not err:
            continue
        err_lower = err.lower()
        if not any(pattern in err_lower for pattern in AUTO_QUARANTINE_ERROR_PATTERNS):
            continue
        norm = _normalize_model_name(model)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(
            {
                "model": model,
                "reason": err,
                "source": "error_pattern",
            }
        )

    return out


def _resolve_database_url(explicit_database_url: Optional[str]) -> str:
    if explicit_database_url and explicit_database_url.strip():
        return explicit_database_url.strip()
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    resolved = _coalesce_env("CANONICAL_DATABASE_URL", env_file) or _coalesce_env(
        "DATABASE_URL", env_file
    )
    if not resolved:
        raise ValueError("Missing CANONICAL_DATABASE_URL / DATABASE_URL")
    return resolved


def _resolve_snapshot_target(database_url: str, snapshot_group_id: int) -> dict[str, Any]:
    conn = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    conn.init_db()

    with conn.session_scope() as session:
        snapshot_group = (
            session.query(Group)
            .filter(
                Group.id == int(snapshot_group_id),
                Group.group_type == "dataset_snapshot",
            )
            .first()
        )
        if snapshot_group is None:
            raise ValueError(
                f"dataset_snapshot group id={snapshot_group_id} not found"
            )
        snapshot_md = dict(snapshot_group.metadata_json or {})
        source_dataframe_group_id = int(
            snapshot_md.get("source_dataframe_group_id") or 0
        )
        if source_dataframe_group_id <= 0:
            raise ValueError(
                "dataset_snapshot is missing metadata_json['source_dataframe_group_id']"
            )

        dataframe_group = (
            session.query(Group)
            .filter(
                Group.id == int(source_dataframe_group_id),
                Group.group_type == "dataset_dataframe",
            )
            .first()
        )
        dataframe_md = dict((dataframe_group.metadata_json if dataframe_group else {}) or {})

    return {
        "snapshot_group_id": int(snapshot_group_id),
        "snapshot_row_count": int(snapshot_md.get("row_count") or 0),
        "dataset_slug": str(snapshot_md.get("dataset_slug") or ""),
        "source_dataframe_group_id": int(source_dataframe_group_id),
        "source_dataframe_row_count": int(dataframe_md.get("row_count") or 0),
    }


async def _list_embedding_models(provider: str, max_models: int) -> List[str]:
    deployments = await ProviderFactory().list_provider_deployments(
        provider, modality="embedding"
    )
    unique: List[str] = []
    seen: set[str] = set()
    for dep in deployments:
        model = str(dep.id).strip()
        if not model:
            continue
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(model)
    if max_models > 0:
        return unique[:max_models]
    return unique


def _run_embed_for_model_sync(
    *,
    database_url: str,
    dataframe_group_id: int,
    provider: str,
    model: str,
    force: bool,
    entry_max: int | None,
    chunk_size: int,
    chunk_worker_concurrency: int,
    chunk_circuit_breaker_enabled: bool,
    chunk_failure_fallback_threshold: int,
    max_retries: int,
    initial_wait_seconds: float,
    max_wait_seconds: float,
    singleflight_lease_seconds: int,
    singleflight_wait_timeout_seconds: float,
    singleflight_poll_seconds: float,
    embed_timeout_seconds: float,
) -> dict[str, Any]:
    conn = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    conn.init_db()
    result = embed(
        dataframe_group_id=int(dataframe_group_id),
        deployment=str(model),
        provider=str(provider),
        representation="full",
        force=bool(force),
        entry_max=int(entry_max) if entry_max is not None else None,
        db=conn,
        write_intent=WriteIntent.CANONICAL,
        chunk_size=int(chunk_size),
        chunk_worker_concurrency=int(chunk_worker_concurrency),
        chunk_circuit_breaker_enabled=bool(chunk_circuit_breaker_enabled),
        chunk_failure_fallback_threshold=int(chunk_failure_fallback_threshold),
        max_retries=int(max_retries),
        initial_wait=float(initial_wait_seconds),
        max_wait=float(max_wait_seconds),
        singleflight_lease_seconds=int(singleflight_lease_seconds),
        singleflight_wait_timeout_seconds=float(singleflight_wait_timeout_seconds),
        singleflight_poll_seconds=float(singleflight_poll_seconds),
        timeout=float(embed_timeout_seconds),
    )
    return {
        "model": str(model),
        "embedding_batch_group_id": int(result.group_id),
        "reused": bool((result.metadata or {}).get("reused")),
    }


async def _run_models(
    *,
    models: List[str],
    config: SweepRunConfig,
    concurrency: int,
    attempt_label: str,
) -> List[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def _run_one(model: str) -> dict[str, Any]:
        async with semaphore:
            started = time.perf_counter()
            try:
                payload = await asyncio.wait_for(
                    asyncio.to_thread(
                        _run_embed_for_model_sync,
                        database_url=config.database_url,
                        dataframe_group_id=config.source_dataframe_group_id,
                        provider=config.provider,
                        model=model,
                        force=config.force,
                        entry_max=config.entry_max,
                        chunk_size=config.chunk_size,
                        chunk_worker_concurrency=config.chunk_worker_concurrency,
                        chunk_circuit_breaker_enabled=config.chunk_circuit_breaker_enabled,
                        chunk_failure_fallback_threshold=config.chunk_failure_fallback_threshold,
                        max_retries=config.max_retries,
                        initial_wait_seconds=config.initial_wait_seconds,
                        max_wait_seconds=config.max_wait_seconds,
                        singleflight_lease_seconds=config.singleflight_lease_seconds,
                        singleflight_wait_timeout_seconds=config.singleflight_wait_timeout_seconds,
                        singleflight_poll_seconds=config.singleflight_poll_seconds,
                        embed_timeout_seconds=config.embed_timeout_seconds,
                    ),
                    timeout=float(config.pair_timeout_seconds),
                )
                elapsed = round(time.perf_counter() - started, 2)
                return {
                    "model": str(model),
                    "ok": True,
                    "attempt": attempt_label,
                    "elapsed_seconds": elapsed,
                    "embedding_batch_group_id": int(payload["embedding_batch_group_id"]),
                    "reused": bool(payload["reused"]),
                }
            except Exception as exc:
                elapsed = round(time.perf_counter() - started, 2)
                return {
                    "model": str(model),
                    "ok": False,
                    "attempt": attempt_label,
                    "elapsed_seconds": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    return await asyncio.gather(*[_run_one(model) for model in models])


def _build_summary(
    *,
    config: SweepRunConfig,
    target: dict[str, Any],
    models: List[str],
    discovered_models: List[str],
    excluded_models: List[str],
    unavailable_models: List[dict[str, Any]],
    parallel_wall_seconds: float,
    parallel_results: List[dict[str, Any]],
    retry_wall_seconds: float,
    retry_results: List[dict[str, Any]],
) -> dict[str, Any]:
    final_by_model: dict[str, dict[str, Any]] = {
        row["model"]: row for row in parallel_results
    }
    for row in retry_results:
        final_by_model[row["model"]] = row

    final_results = [final_by_model[m] for m in models if m in final_by_model]
    final_success = [r for r in final_results if bool(r.get("ok"))]
    final_errors = [r for r in final_results if not bool(r.get("ok"))]
    created_count = sum(1 for r in final_success if not bool(r.get("reused")))
    reused_count = sum(1 for r in final_success if bool(r.get("reused")))
    auto_quarantine_candidates = _compute_auto_quarantine_candidates(
        final_errors=final_errors,
        prefiltered_unavailable=unavailable_models,
    )

    failed_first_pass_models = {
        r["model"] for r in parallel_results if not bool(r.get("ok"))
    }
    recovered_by_retry = sum(
        1
        for r in retry_results
        if bool(r.get("ok")) and str(r["model"]) in failed_first_pass_models
    )

    total_wall_seconds = round(parallel_wall_seconds + retry_wall_seconds, 2)
    sequential_estimate_seconds = round(
        sum(float(r.get("elapsed_seconds") or 0.0) for r in final_results),
        2,
    )
    estimated_speedup_vs_serial = (
        round(sequential_estimate_seconds / total_wall_seconds, 2)
        if total_wall_seconds > 0
        else None
    )

    return {
        "provider": config.provider,
        "target": target,
        "discovered_model_count": len(discovered_models),
        "model_count": len(models),
        "models_discovered": discovered_models,
        "models_excluded": excluded_models,
        "models_unavailable_prefiltered": unavailable_models,
        "models_attempted": models,
        "requested_engine_concurrency": int(config.requested_engine_concurrency),
        "engine_concurrency": int(config.engine_concurrency),
        "provider_concurrency_budget": (
            int(config.provider_concurrency_budget)
            if config.provider_concurrency_budget is not None
            else None
        ),
        "requested_chunk_worker_concurrency": int(
            config.requested_chunk_worker_concurrency
        ),
        "chunk_worker_concurrency": int(config.chunk_worker_concurrency),
        "chunk_circuit_breaker_enabled": bool(config.chunk_circuit_breaker_enabled),
        "chunk_failure_fallback_threshold": int(
            config.chunk_failure_fallback_threshold
        ),
        "disable_parallel_chunks_env": bool(config.disable_parallel_chunks_env),
        "retry_failed_serial": bool(config.retry_failed_serial),
        "force": bool(config.force),
        "entry_max": int(config.entry_max) if config.entry_max is not None else None,
        "chunk_size": int(config.chunk_size),
        "max_retries": int(config.max_retries),
        "initial_wait_seconds": float(config.initial_wait_seconds),
        "max_wait_seconds": float(config.max_wait_seconds),
        "singleflight_lease_seconds": int(config.singleflight_lease_seconds),
        "singleflight_wait_timeout_seconds": float(
            config.singleflight_wait_timeout_seconds
        ),
        "singleflight_poll_seconds": float(config.singleflight_poll_seconds),
        "embed_timeout_seconds": float(config.embed_timeout_seconds),
        "pair_timeout_seconds": float(config.pair_timeout_seconds),
        "pre_validate_models": bool(config.pre_validate_models),
        "validation_concurrency": int(config.validation_concurrency),
        "validation_cache_ttl_seconds": float(config.validation_cache_ttl_seconds),
        "validation_timeout_seconds": float(config.validation_timeout_seconds),
        "excluded_models_file": config.excluded_models_file,
        "parallel_pass": {
            "wall_seconds": round(parallel_wall_seconds, 2),
            "success_count": sum(1 for r in parallel_results if bool(r.get("ok"))),
            "error_count": sum(1 for r in parallel_results if not bool(r.get("ok"))),
            "results": parallel_results,
        },
        "serial_retry_pass": {
            "wall_seconds": round(retry_wall_seconds, 2),
            "success_count": sum(1 for r in retry_results if bool(r.get("ok"))),
            "error_count": sum(1 for r in retry_results if not bool(r.get("ok"))),
            "recovered_count": recovered_by_retry,
            "results": retry_results,
        },
        "final": {
            "wall_seconds": total_wall_seconds,
            "success_count": len(final_success),
            "error_count": len(final_errors),
            "created_count": created_count,
            "reused_count": reused_count,
            "sequential_estimate_seconds": sequential_estimate_seconds,
            "estimated_speedup_vs_serial": estimated_speedup_vs_serial,
            "auto_quarantine_candidates": auto_quarantine_candidates,
            "results": final_results,
        },
    }


async def _async_main(args: argparse.Namespace) -> int:
    provider = str(args.provider).strip().lower()
    if not provider:
        raise ValueError("provider must be non-empty")

    database_url = _resolve_database_url(args.database_url)
    target = _resolve_snapshot_target(database_url, int(args.snapshot_group_id))
    snapshot_rows = int(target.get("snapshot_row_count") or 0)
    source_df_rows = int(target.get("source_dataframe_row_count") or 0)
    if args.entry_max is not None:
        effective_entry_max: int | None = int(args.entry_max)
    elif (
        snapshot_rows > 0
        and source_df_rows > 0
        and snapshot_rows < source_df_rows
    ):
        # embed() truncates the canonical dataframe to texts[:entry_max]; when the
        # snapshot is a strict subset of the source dataframe, default entry_max to
        # the snapshot row_count so embedding_batch keys align with snapshot lineage
        # (matches how partial-view sweeps must be invoked explicitly).
        effective_entry_max = snapshot_rows
        print(
            "[INFO] auto entry_max from snapshot row_count "
            f"(partial dataframe view): entry_max={effective_entry_max} "
            f"snapshot_row_count={snapshot_rows} "
            f"source_dataframe_row_count={source_df_rows}"
        )
    else:
        effective_entry_max = None

    discovered_models = await _list_embedding_models(provider, int(args.max_models))
    if not discovered_models:
        raise ValueError(
            f"No embedding models discovered for provider={provider!r}"
        )
    excluded_models = _resolve_excluded_models(
        explicit=list(args.exclude_model or []),
        file_path=(
            str(args.exclude_models_file).strip()
            if args.exclude_models_file is not None
            else None
        ),
    )
    excluded_set = {_normalize_model_name(model) for model in excluded_models}
    models = [
        model
        for model in discovered_models
        if _normalize_model_name(model) not in excluded_set
    ]
    if args.pre_validate_models and models:
        models, unavailable_models = await _best_effort_filter_models(
            provider=provider,
            models=models,
            validation_concurrency=max(1, int(args.validation_concurrency)),
            validation_cache_ttl_seconds=float(args.validation_cache_ttl_seconds),
            validation_timeout_seconds=float(args.validation_timeout_seconds),
        )
    else:
        unavailable_models = []
    if not models:
        raise ValueError(
            "No models remain after applying exclusions/pre-validation filters. "
            "Adjust --exclude-model/--exclude-models-file or disable "
            "--pre-validate-models."
        )

    requested_engine_concurrency = max(1, int(args.engine_concurrency))
    requested_chunk_worker_concurrency = max(1, int(args.chunk_worker_concurrency))
    disable_parallel_chunks_env = _env_flag_enabled(DISABLE_PARALLEL_CHUNKS_ENV)
    effective_chunk_worker_concurrency = (
        1 if disable_parallel_chunks_env else requested_chunk_worker_concurrency
    )
    provider_concurrency_budget = (
        int(args.provider_concurrency_budget)
        if int(args.provider_concurrency_budget) > 0
        else None
    )
    effective_engine_concurrency = _resolve_effective_engine_concurrency(
        engine_concurrency=requested_engine_concurrency,
        provider_budget=provider_concurrency_budget,
        chunk_worker_concurrency=effective_chunk_worker_concurrency,
    )

    config = SweepRunConfig(
        snapshot_group_id=int(target["snapshot_group_id"]),
        source_dataframe_group_id=int(target["source_dataframe_group_id"]),
        provider=provider,
        models=models,
        database_url=database_url,
        requested_engine_concurrency=requested_engine_concurrency,
        engine_concurrency=effective_engine_concurrency,
        provider_concurrency_budget=provider_concurrency_budget,
        requested_chunk_worker_concurrency=requested_chunk_worker_concurrency,
        chunk_worker_concurrency=effective_chunk_worker_concurrency,
        chunk_circuit_breaker_enabled=bool(args.chunk_circuit_breaker_fallback),
        chunk_failure_fallback_threshold=max(
            1, int(args.chunk_failure_fallback_threshold)
        ),
        disable_parallel_chunks_env=disable_parallel_chunks_env,
        retry_failed_serial=bool(args.retry_failed_serial),
        force=bool(args.force),
        entry_max=effective_entry_max,
        chunk_size=max(1, int(args.chunk_size)),
        max_retries=max(1, int(args.max_retries)),
        initial_wait_seconds=max(0.01, float(args.initial_wait_seconds)),
        max_wait_seconds=max(0.01, float(args.max_wait_seconds)),
        singleflight_lease_seconds=max(1, int(args.singleflight_lease_seconds)),
        singleflight_wait_timeout_seconds=max(
            1.0, float(args.singleflight_wait_timeout_seconds)
        ),
        singleflight_poll_seconds=max(0.01, float(args.singleflight_poll_seconds)),
        embed_timeout_seconds=float(args.embed_timeout_seconds),
        pair_timeout_seconds=float(args.pair_timeout_seconds),
        pre_validate_models=bool(args.pre_validate_models),
        validation_concurrency=max(1, int(args.validation_concurrency)),
        validation_cache_ttl_seconds=float(args.validation_cache_ttl_seconds),
        validation_timeout_seconds=float(args.validation_timeout_seconds),
        excluded_models=excluded_models,
        excluded_models_file=(
            str(args.exclude_models_file).strip()
            if args.exclude_models_file is not None
            else None
        ),
    )

    print(
        "[INFO] target "
        f"snapshot_group_id={target['snapshot_group_id']} "
        f"snapshot_row_count={target['snapshot_row_count']} "
        f"source_dataframe_group_id={target['source_dataframe_group_id']} "
        f"source_dataframe_row_count={target['source_dataframe_row_count']} "
        f"dataset_slug={target['dataset_slug']!r}"
    )
    print(
        "[INFO] run config "
        f"provider={provider} models={len(models)} "
        f"discovered_models={len(discovered_models)} "
        f"engine_concurrency={config.engine_concurrency} "
        f"requested_engine_concurrency={config.requested_engine_concurrency} "
        f"provider_concurrency_budget={config.provider_concurrency_budget} "
        f"requested_chunk_worker_concurrency={config.requested_chunk_worker_concurrency} "
        f"chunk_worker_concurrency={config.chunk_worker_concurrency} "
        f"chunk_circuit_breaker_enabled={config.chunk_circuit_breaker_enabled} "
        f"chunk_failure_fallback_threshold={config.chunk_failure_fallback_threshold} "
        f"disable_parallel_chunks_env={config.disable_parallel_chunks_env} "
        f"retry_failed_serial={config.retry_failed_serial} "
        f"force={config.force} "
        f"entry_max={config.entry_max} "
        f"chunk_size={config.chunk_size} "
        f"max_retries={config.max_retries} "
        f"initial_wait_seconds={config.initial_wait_seconds} "
        f"max_wait_seconds={config.max_wait_seconds} "
        f"singleflight_lease_seconds={config.singleflight_lease_seconds} "
        f"singleflight_wait_timeout_seconds={config.singleflight_wait_timeout_seconds} "
        f"singleflight_poll_seconds={config.singleflight_poll_seconds} "
        f"embed_timeout_seconds={config.embed_timeout_seconds} "
        f"pair_timeout_seconds={config.pair_timeout_seconds} "
        f"pre_validate_models={config.pre_validate_models} "
        f"validation_concurrency={config.validation_concurrency} "
        f"validation_cache_ttl_seconds={config.validation_cache_ttl_seconds} "
        f"validation_timeout_seconds={config.validation_timeout_seconds}"
    )
    print(
        "[INFO] filtering "
        f"excluded_models={len(excluded_models)} "
        f"unavailable_prefiltered={len(unavailable_models)}"
    )
    if (
        config.entry_max is None
        and int(target["source_dataframe_row_count"]) > int(target["snapshot_row_count"])
    ):
        print(
            "[INFO] source dataframe is larger than snapshot row_count. "
            "Use --entry-max to bound run size when you want a smaller test pass."
        )
    print(f"[INFO] models={models}")
    if excluded_models:
        print(f"[INFO] excluded_models={excluded_models}")
    if unavailable_models:
        print(f"[INFO] unavailable_prefiltered={unavailable_models}")

    parallel_start = time.perf_counter()
    parallel_results = await _run_models(
        models=models,
        config=config,
        concurrency=config.engine_concurrency,
        attempt_label="parallel",
    )
    parallel_wall_seconds = round(time.perf_counter() - parallel_start, 2)

    failed_models = [r["model"] for r in parallel_results if not bool(r.get("ok"))]
    retry_results: List[dict[str, Any]] = []
    retry_wall_seconds = 0.0
    if failed_models and config.retry_failed_serial:
        print(
            f"[INFO] serial retry requested for failed_models={len(failed_models)}"
        )
        retry_start = time.perf_counter()
        retry_results = await _run_models(
            models=failed_models,
            config=config,
            concurrency=1,
            attempt_label="serial_retry",
        )
        retry_wall_seconds = round(time.perf_counter() - retry_start, 2)

    summary = _build_summary(
        config=config,
        target=target,
        models=models,
        discovered_models=discovered_models,
        excluded_models=excluded_models,
        unavailable_models=unavailable_models,
        parallel_wall_seconds=parallel_wall_seconds,
        parallel_results=parallel_results,
        retry_wall_seconds=retry_wall_seconds,
        retry_results=retry_results,
    )

    final = summary["final"]
    print(
        "[DONE] "
        f"wall={final['wall_seconds']}s "
        f"success={final['success_count']}/{len(models)} "
        f"errors={final['error_count']} "
        f"created={final['created_count']} "
        f"reused={final['reused_count']} "
        f"speedup_vs_serial_est={final['estimated_speedup_vs_serial']}x"
    )
    print("SUMMARY_JSON=" + json.dumps(summary, ensure_ascii=True, sort_keys=True))

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True),
            encoding="utf-8",
        )
        print(f"[INFO] wrote summary to {output_path}")

    return 0 if int(final["error_count"]) == 0 else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run embedding sweep for a snapshot's source dataframe with "
            "bounded model-level concurrency."
        )
    )
    parser.add_argument(
        "--snapshot-group-id",
        type=int,
        required=True,
        help="dataset_snapshot group id to target",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        help="Embedding provider name (default: openrouter)",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=30,
        help="Maximum number of embedding models to try (<=0 means all discovered)",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        help=(
            "Model id to skip. Repeat flag to exclude multiple models. "
            "Applied after discovery."
        ),
    )
    parser.add_argument(
        "--exclude-models-file",
        type=str,
        default=None,
        help=(
            "Optional file of model ids to exclude (json list or newline-delimited txt)."
        ),
    )
    parser.add_argument(
        "--pre-validate-models",
        dest="pre_validate_models",
        action="store_true",
        default=True,
        help="Best-effort pre-validate model availability before parallel sweep (default: enabled)",
    )
    parser.add_argument(
        "--no-pre-validate-models",
        dest="pre_validate_models",
        action="store_false",
        help="Disable pre-validation filter pass",
    )
    parser.add_argument(
        "--validation-concurrency",
        type=int,
        default=4,
        help="Concurrent pre-validation probes (default: 4)",
    )
    parser.add_argument(
        "--validation-cache-ttl-seconds",
        type=float,
        default=900.0,
        help="TTL for model pre-validation cache entries (default: 900s)",
    )
    parser.add_argument(
        "--validation-timeout-seconds",
        type=float,
        default=15.0,
        help="Per-model timeout for pre-validation probes (default: 15s)",
    )
    parser.add_argument(
        "--engine-concurrency",
        type=int,
        default=4,
        help="Max concurrent model runs (default: 4)",
    )
    parser.add_argument(
        "--provider-concurrency-budget",
        type=int,
        default=0,
        help=(
            "Optional provider-wide concurrency budget. "
            "When >0, effective_engine_concurrency is bounded by "
            "floor(provider_budget / max(1, chunk_worker_concurrency))."
        ),
    )
    parser.add_argument(
        "--chunk-worker-concurrency",
        type=int,
        default=1,
        help=(
            "Requested chunk-worker concurrency for budget calculations and "
            "embed() runtime knobs. Effective value is forced to 1 when "
            f"{DISABLE_PARALLEL_CHUNKS_ENV}=1."
        ),
    )
    parser.add_argument(
        "--chunk-circuit-breaker-fallback",
        action="store_true",
        help=(
            "Enable chunk circuit-breaker fallback. When enabled and chunk failures "
            "reach threshold, remaining undispatched chunks downgrade to serial."
        ),
    )
    parser.add_argument(
        "--chunk-failure-fallback-threshold",
        type=int,
        default=2,
        help=(
            "Chunk failure threshold for circuit-breaker fallback (default: 2). "
            "Only used when --chunk-circuit-breaker-fallback is set."
        ),
    )
    parser.add_argument(
        "--retry-failed-serial",
        dest="retry_failed_serial",
        action="store_true",
        default=True,
        help="Retry failed models once sequentially (default: enabled)",
    )
    parser.add_argument(
        "--no-retry-failed-serial",
        dest="retry_failed_serial",
        action="store_false",
        help="Disable serial retry for failed models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass force=True to embed() (recompute even when cache exists)",
    )
    parser.add_argument(
        "--entry-max",
        type=int,
        default=None,
        help=(
            "Optional max rows passed to embed() per model (truncates canonical "
            "dataframe to texts[:entry_max]). When omitted and snapshot_row_count is "
            "strictly less than source_dataframe_row_count, the sweep defaults "
            "entry_max to snapshot_row_count so batches align with partial snapshots."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Chunk size passed to embed()",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="EmbeddingService max retries per request/batch (default: 6)",
    )
    parser.add_argument(
        "--initial-wait-seconds",
        type=float,
        default=1.0,
        help="Initial retry wait seconds for EmbeddingService (default: 1.0)",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=30.0,
        help="Maximum retry wait seconds for EmbeddingService (default: 30.0)",
    )
    parser.add_argument(
        "--singleflight-lease-seconds",
        type=int,
        default=45,
        help="Singleflight lease seconds for deduped cache keys (default: 45)",
    )
    parser.add_argument(
        "--singleflight-wait-timeout-seconds",
        type=float,
        default=90.0,
        help="Singleflight wait timeout seconds (default: 90.0)",
    )
    parser.add_argument(
        "--singleflight-poll-seconds",
        type=float,
        default=0.1,
        help="Singleflight poll interval seconds (default: 0.1)",
    )
    parser.add_argument(
        "--embed-timeout-seconds",
        type=float,
        default=300.0,
        help="embed() internal timeout (seconds)",
    )
    parser.add_argument(
        "--pair-timeout-seconds",
        type=float,
        default=420.0,
        help="Per-model outer timeout (seconds)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Explicit DB URL (defaults to CANONICAL_DATABASE_URL / DATABASE_URL)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional output path for summary JSON",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Parallel fixed-k clustering sweep across snapshot x embedding-batch pairs."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import dotenv_values
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from study_query_llm.db.connection_v2 import DatabaseConnectionV2  # noqa: E402
from study_query_llm.db.models_v2 import Group  # noqa: E402
from study_query_llm.db.write_intent import default_write_intent_for_connection  # noqa: E402
from study_query_llm.experiments.clustering_analysis_backfill import (  # noqa: E402
    analysis_run_key_for_target,
    embedding_batches_for_lineage_and_engine,
    expand_parameter_variants,
    fetch_completed_backfill_run_keys,
    line_run_key,
    snapshot_lineage,
)
from study_query_llm.pipeline.analyze import analyze  # noqa: E402
from study_query_llm.pipeline.clustering.registry import iter_algorithm_specs  # noqa: E402
DEFAULT_SNAPSHOTS = [6, 9, 10, 17, 18]
DEFAULT_PROVIDER = "openrouter"
DEFAULT_K_MIN, DEFAULT_K_MAX, DEFAULT_MAX_WORKERS = 2, 10, 8
OPENROUTER_ENGINE_CACHE = PROJECT_ROOT / ".cache" / "model_validation" / "openrouter_embedding_models.json"
@dataclass
class Target:
    snapshot_group_id: int
    embedding_batch_group_id: int
    embedding_engine: str
    method_name: str
    parameters: dict[str, Any]
    analysis_run_key: str
    run_key: str
    method_version: str = "1.0"

def is_fixed_k_spec(spec) -> bool:
    if getattr(spec, "fit_mode", None) != "single_fit":
        return False
    required = list((spec.parameters_schema or {}).get("required") or [])
    return "k" in required

def utc_now_iso() -> str: return datetime.now(timezone.utc).isoformat()

def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        value = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if value:
            return value
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")

def default_state_path() -> Path:
    return PROJECT_ROOT / "experimental_results" / "backfill_manifests" / f"parallel_fixed_k_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"

def dedupe_preserve(values: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

def load_openrouter_engines(cache_path: Path = OPENROUTER_ENGINE_CACHE) -> list[str]:
    if not cache_path.exists():
        raise SystemExit(f"OpenRouter engine cache not found: {cache_path}")
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    models = raw.get("models")
    if not isinstance(models, dict):
        raise SystemExit(f"Invalid engine cache shape in {cache_path}: expected 'models' dict")
    names: list[str] = []
    for model_name, info in models.items():
        if not isinstance(info, dict):
            continue
        if bool(info.get("valid")) is not True:
            continue
        if str(info.get("provider") or "").strip().lower() != "openrouter":
            continue
        names.append(str(model_name))
    return sorted(set(names))

def fixed_k_method_names() -> list[str]: return sorted(str(spec.method_name) for spec in iter_algorithm_specs() if is_fixed_k_spec(spec))

def get_embedding_dim_for_batch(session, batch_id: int) -> int:
    row = (
        session.query(Group)
        .filter(Group.id == int(batch_id), Group.group_type == "embedding_batch")
        .first()
    )
    metadata = dict((row.metadata_json if row else {}) or {})
    for key in ("embedding_dim", "embedding_dimension", "dimension", "dim"):
        raw = metadata.get(key)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0

def build_targets(
    db: DatabaseConnectionV2,
    snapshot_ids: list[int],
    engines: list[str],
    provider: str | None,
    k_min: int,
    k_max: int,
) -> tuple[list[Target], dict[str, Any]]:
    specs = [spec for spec in iter_algorithm_specs() if is_fixed_k_spec(spec)]
    fixed_k_range = range(int(k_min), int(k_max) + 1)
    targets: list[Target] = []
    seen_keys: set[str] = set()
    duplicate_keys: list[str] = []
    diag: dict[str, Any] = {"fixed_k_method_names": sorted(str(spec.method_name) for spec in specs), "checked_pairs": 0, "matched_pairs": 0, "missing_pairs": [], "multi_batch_pairs": [], "snapshot_errors": [], "expansion_errors": [], "duplicate_analysis_run_keys": []}
    with db.session_scope() as session:
        for snapshot_group_id in snapshot_ids:
            try:
                lineage = snapshot_lineage(session, int(snapshot_group_id))
            except Exception as exc:  # noqa: BLE001
                diag["snapshot_errors"].append({"snapshot_group_id": int(snapshot_group_id), "error": f"{type(exc).__name__}: {exc}"})
                continue
            for engine in engines:
                diag["checked_pairs"] += 1
                batches = embedding_batches_for_lineage_and_engine(
                    session, lineage.lineage_key, str(engine), provider=provider
                )
                if not batches:
                    diag["missing_pairs"].append({"snapshot_group_id": int(snapshot_group_id), "embedding_engine": str(engine), "provider": provider, "lineage_key": [int(lineage.lineage_key[0]), int(lineage.lineage_key[1])]})
                    continue
                diag["matched_pairs"] += 1
                batch = batches[-1]
                if len(batches) > 1:
                    diag["multi_batch_pairs"].append({"snapshot_group_id": int(snapshot_group_id), "embedding_engine": str(engine), "provider": provider, "embedding_batch_group_ids": [int(item.group_id) for item in batches], "selected_embedding_batch_group_id": int(batch.group_id)})
                    print(
                        "[warn] multiple embedding batches matched pair "
                        f"snap={snapshot_group_id} engine={engine}; selecting latest group_id={batch.group_id}",
                        file=sys.stderr,
                    )
                embedding_dim = get_embedding_dim_for_batch(session, int(batch.group_id))
                n_samples = int(lineage.snapshot_row_count)
                for spec in specs:
                    try:
                        variants = expand_parameter_variants(
                            spec, n_samples=n_samples, embedding_dim=embedding_dim, fixed_k_range=fixed_k_range
                        )
                    except Exception as exc:  # noqa: BLE001
                        diag["expansion_errors"].append({"snapshot_group_id": int(snapshot_group_id), "embedding_batch_group_id": int(batch.group_id), "embedding_engine": str(batch.embedding_engine), "method_name": str(spec.method_name), "error": f"{type(exc).__name__}: {exc}"})
                        print(
                            "[warn] skipped parameter expansion "
                            f"snap={snapshot_group_id} emb={batch.group_id} "
                            f"engine={batch.embedding_engine} method={spec.method_name}: {exc}",
                            file=sys.stderr,
                        )
                        continue
                    for params in variants:
                        params_dict = dict(params)
                        key = analysis_run_key_for_target(
                            snapshot_group_id=int(snapshot_group_id),
                            embedding_batch_group_id=int(batch.group_id),
                            method_name=str(spec.method_name),
                            parameters=params_dict,
                        )
                        if key in seen_keys:
                            duplicate_keys.append(key)
                            continue
                        seen_keys.add(key)
                        targets.append(Target(snapshot_group_id=int(snapshot_group_id), embedding_batch_group_id=int(batch.group_id), embedding_engine=str(batch.embedding_engine), method_name=str(spec.method_name), parameters=params_dict, analysis_run_key=key, run_key=line_run_key(snapshot_group_id=int(snapshot_group_id), embedding_batch_group_id=int(batch.group_id))))
    diag["duplicate_analysis_run_keys"] = sorted(set(duplicate_keys))
    diag["planned_targets"] = len(targets)
    return targets, diag

def run_one_target(target: Target, db: DatabaseConnectionV2, force: bool) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        analyze(
            target.snapshot_group_id,
            target.embedding_batch_group_id,
            method_name=target.method_name,
            run_key=target.run_key,
            analysis_run_key=target.analysis_run_key,
            request_group_id=None,
            method_version=target.method_version,
            parameters=target.parameters,
            force=force,
            db=db,
        )
        status, error = "completed", None
    except Exception as exc:  # noqa: BLE001
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    result = {
        "analysis_run_key": target.analysis_run_key,
        "status": status,
        "elapsed_s": time.perf_counter() - started,
        "snapshot_group_id": target.snapshot_group_id,
        "embedding_batch_group_id": target.embedding_batch_group_id,
        "embedding_engine": target.embedding_engine,
        "method_name": target.method_name,
        "parameters": target.parameters,
    }
    if error is not None:
        result["error"] = error
    return result

def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=int, nargs="*", default=list(DEFAULT_SNAPSHOTS)); parser.add_argument("--engines", nargs="*", default=None); parser.add_argument("--provider", default=DEFAULT_PROVIDER); parser.add_argument("--k-min", type=int, default=DEFAULT_K_MIN); parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX); parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS); parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--force", action="store_true"); parser.add_argument("--state-file", type=Path, default=None); parser.add_argument("--list-methods", action="store_true"); parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    if args.list_methods:
        for name in fixed_k_method_names():
            print(name)
        return 0
    snapshots = dedupe_preserve([int(x) for x in list(args.snapshots or [])])
    if not snapshots:
        raise SystemExit("No snapshot ids provided")
    engines = load_openrouter_engines() if args.engines is None else dedupe_preserve([str(x).strip() for x in list(args.engines) if str(x).strip()])
    if not engines:
        raise SystemExit("No embedding engines provided")
    k_min, k_max = int(args.k_min), int(args.k_max)
    if k_min < DEFAULT_K_MIN:
        print(f"[warn] --k-min {k_min} < {DEFAULT_K_MIN}; clamping to {DEFAULT_K_MIN}", file=sys.stderr)
        k_min = DEFAULT_K_MIN
    if k_max > DEFAULT_K_MAX:
        print(f"[warn] --k-max {k_max} > {DEFAULT_K_MAX}; expand_parameter_variants() applies internal cap {DEFAULT_K_MAX}", file=sys.stderr)
    if k_max < k_min:
        raise SystemExit(f"Invalid k-range: k_min={k_min}, k_max={k_max}")
    max_workers = max(1, int(args.max_workers))
    db_url = _resolve_database_url(args.database_url)
    if not (os.environ.get("SQLLM_WRITE_INTENT") or "").strip():
        os.environ["SQLLM_WRITE_INTENT"] = str(default_write_intent_for_connection(db_url))
    db = DatabaseConnectionV2(db_url, quiet=True)
    targets, diag = build_targets(
        db=db,
        snapshot_ids=snapshots,
        engines=engines,
        provider=str(args.provider),
        k_min=k_min,
        k_max=k_max,
    )
    print(f"plan snapshots={len(snapshots)} engines={len(engines)} methods={len(diag['fixed_k_method_names'])} k=[{k_min},{k_max}]")
    print(f"pairs checked={diag['checked_pairs']} matched={diag['matched_pairs']} missing={len(diag['missing_pairs'])}")
    if diag["multi_batch_pairs"]:
        print(f"multi_batch_pairs={len(diag['multi_batch_pairs'])}")
    if diag["snapshot_errors"]:
        print(f"snapshot_errors={len(diag['snapshot_errors'])}")
    if diag["expansion_errors"]:
        print(f"expansion_errors={len(diag['expansion_errors'])}")
    if diag["duplicate_analysis_run_keys"]:
        print(f"duplicate_analysis_run_keys={len(diag['duplicate_analysis_run_keys'])}")
    print(f"planned_targets={len(targets)}")

    with db.session_scope() as session:
        completed = fetch_completed_backfill_run_keys(session)
    already_completed = sum(1 for target in targets if target.analysis_run_key in completed)
    pending = [target for target in targets if args.force or target.analysis_run_key not in completed]
    print(f"already_completed={already_completed} to_run={len(pending)} force={bool(args.force)}")
    if args.dry_run:
        return 0
    state_file = Path(args.state_file) if args.state_file else default_state_path()
    state: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "wall_clock_s": None,
        "max_workers": max_workers,
        "k_range": {"k_min": k_min, "k_max": k_max},
        "snapshot_ids": snapshots,
        "engines": engines,
        "provider": str(args.provider),
        "force": bool(args.force),
        "diag": diag,
        "totals": {
            "planned_targets": len(targets),
            "already_completed": already_completed,
            "to_run": len(pending),
            "completed": 0,
            "failed": 0,
        },
        "results": {},
    }
    write_state(state_file, state)
    print(f"state_file={state_file}")
    run_started = time.perf_counter()
    total = len(pending)
    if total == 0:
        state["finished_at"], state["wall_clock_s"] = utc_now_iso(), 0.0
        write_state(state_file, state)
        print("nothing to run")
        return 0
    done_count = 0
    completed_count = 0
    failed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_one_target, target, db, bool(args.force)): target for target in pending}
        for future in as_completed(future_map):
            result = future.result()
            done_count += 1
            status = str(result.get("status") or "failed")
            if status == "completed":
                completed_count += 1
            else:
                failed_count += 1
            state["results"][str(result["analysis_run_key"])] = result
            state["totals"]["completed"] = completed_count
            state["totals"]["failed"] = failed_count
            should_log = (done_count % 25 == 0) or (status == "failed") or (done_count == total)
            if not should_log:
                continue
            elapsed_total = time.perf_counter() - run_started
            rate = done_count / elapsed_total if elapsed_total > 0 else 0.0
            remaining = total - done_count
            eta_s = remaining / rate if rate > 0 else 0.0
            k_value = (result.get("parameters") or {}).get("k")
            status_token = "OK" if status == "completed" else "FAIL"
            print(
                f"[{done_count}/{total} {status_token}] snap={result['snapshot_group_id']} "
                f"emb={result['embedding_batch_group_id']} engine={result['embedding_engine']} "
                f"method={result['method_name']} k={k_value} elapsed_s={float(result.get('elapsed_s') or 0.0):.2f} "
                f"rate={rate:.2f}/s eta_s={eta_s:.1f}"
            )
            if status == "failed":
                print(f"  error={result.get('error')}", file=sys.stderr)
            write_state(state_file, state)
    state["finished_at"] = utc_now_iso()
    state["wall_clock_s"] = time.perf_counter() - run_started
    write_state(state_file, state)
    print(f"finished completed={completed_count} failed={failed_count} wall_clock_s={state['wall_clock_s']:.2f}")
    return 0 if failed_count == 0 else 2

if __name__ == "__main__": raise SystemExit(main())

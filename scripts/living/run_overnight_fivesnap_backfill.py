#!/usr/bin/env python3
"""Overnight operator workflow: bundled clustering backfill across fixed snapshots.

Locks snapshots **6, 9, 10, 21, 22**, builds the intersection roster of
``(provider, embedding_engine)`` pairs present on **all** snapshots for matching
lineage, runs **preflight** manifests with :func:`preflight_manifest_blocking_issues`,
then executes **runnable** pairs sequentially via
``scripts/living/run_two_engine_backfill_sharded.py`` (8 shards / 8 parallel)
with an isolated ``work/<pair_id>/`` directory per pair.

Phases (subcommands):

1. ``init`` — write ``snapshots.json`` + ``pairs.candidate.json``
2. ``preflight`` — per-pair dry manifests under ``preflight/`` + ``queue.json``
3. ``execute`` — timeboxed sequential ``sharded.py all`` runs + ``progress.json``
4. ``verify-package`` — per-attempted-pair verify + acceptance + consolidated outputs

Artifacts live under ``experimental_results/overnight_5snap/<UTC>/`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.db.connection_v2 import DatabaseConnectionV2  # noqa: E402
from study_query_llm.db.models_v2 import Group  # noqa: E402
from study_query_llm.experiments.clustering_analysis_backfill import (  # noqa: E402
    build_manifest,
    preflight_manifest_blocking_issues,
    snapshot_lineage,
    validate_registry_expansion_or_raise,
)

SHARDED_SCRIPT = PROJECT_ROOT / "scripts" / "living" / "run_two_engine_backfill_sharded.py"

DEFAULT_SNAPSHOT_IDS: tuple[int, ...] = (6, 9, 10, 21, 22)


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if v:
            return v
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")


def pair_slug(provider: str, embedding_engine: str) -> str:
    def _slug(s: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s).strip())

    return f"{_slug(provider)}__{_slug(embedding_engine)}"


def provider_engine_map_for_snapshot(session: Any, snapshot_group_id: int) -> dict[tuple[str, str], str]:
    """Map (provider_lower, embedding_engine) -> provider string as stored on batch."""
    lin = snapshot_lineage(session, int(snapshot_group_id))
    sdf_id, entry_max = lin.lineage_key
    out: dict[tuple[str, str], str] = {}
    for g in session.query(Group).filter(Group.group_type == "embedding_batch").all():
        md = dict(g.metadata_json or {})
        try:
            ksdf = int(md.get("source_dataframe_group_id") or 0)
            kem = int(md.get("entry_max") or 0)
        except (TypeError, ValueError):
            continue
        if ksdf != sdf_id or kem != entry_max:
            continue
        prov = str(md.get("provider") or "").strip()
        eng = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        if not eng:
            continue
        out[(prov.lower(), eng)] = prov
    return out


def intersection_candidate_pairs(session: Any, snapshot_ids: Sequence[int]) -> list[dict[str, str]]:
    if not snapshot_ids:
        return []
    maps = [provider_engine_map_for_snapshot(session, sid) for sid in snapshot_ids]
    keys = set.intersection(*(set(m.keys()) for m in maps))
    pairs: list[dict[str, str]] = []
    for prov_lower, eng in sorted(keys):
        prov_stored = maps[0][(prov_lower, eng)]
        pid = pair_slug(prov_stored, eng)
        pairs.append(
            {
                "pair_id": pid,
                "provider": prov_stored,
                "embedding_engine": eng,
            }
        )
    return pairs


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return raw


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


def cmd_init(*, run_dir: Path, snapshot_ids: list[int]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = run_dir / "snapshots.json"
    _write_json(snapshots_path, {"snapshot_ids": list(snapshot_ids)})
    db = DatabaseConnectionV2(_resolve_database_url(None), enable_pgvector=False)
    with db.session_scope() as session:
        pairs = intersection_candidate_pairs(session, snapshot_ids)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_ids": list(snapshot_ids),
        "pairs": pairs,
    }
    _write_json(run_dir / "pairs.candidate.json", payload)
    print(json.dumps({"run_dir": str(run_dir.resolve()), "pair_count": len(pairs)}, indent=2))


def _load_snapshots_file(run_dir: Path) -> list[int]:
    p = run_dir / "snapshots.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}; run init first")
    raw = _read_json(p)
    ids = raw.get("snapshot_ids")
    if not isinstance(ids, list) or not ids:
        raise SystemExit("snapshots.json requires non-empty snapshot_ids list")
    out: list[int] = []
    for x in ids:
        try:
            out.append(int(x))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid snapshot id: {x!r}") from exc
    return out


def cmd_preflight(*, run_dir: Path, database_url: str | None) -> None:
    snapshot_ids = _load_snapshots_file(run_dir)
    preflight_dir = run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    cand_path = run_dir / "pairs.candidate.json"
    if not cand_path.is_file():
        raise SystemExit(f"missing {cand_path}; run init first")
    cand = _read_json(cand_path)
    pairs = cand.get("pairs") or []
    if not isinstance(pairs, list):
        raise SystemExit("pairs.candidate.json pairs must be a list")

    validate_registry_expansion_or_raise()
    db = DatabaseConnectionV2(_resolve_database_url(database_url), enable_pgvector=False)

    runnable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    with db.session_scope() as session:
        for entry in pairs:
            if not isinstance(entry, dict):
                skipped.append({"entry": entry, "reason": "not_an_object"})
                continue
            prov = entry.get("provider")
            eng = entry.get("embedding_engine")
            pid = entry.get("pair_id")
            if not isinstance(prov, str) or not isinstance(eng, str) or not isinstance(pid, str):
                skipped.append({"entry": entry, "reason": "missing_or_invalid_fields"})
                continue
            manifest = build_manifest(
                session,
                embedding_engine=eng,
                provider=prov,
                snapshot_ids=snapshot_ids,
            )
            pre_path = preflight_dir / f"{pid}.json"
            _write_json(pre_path, manifest)
            reasons = preflight_manifest_blocking_issues(manifest)
            row = {
                "pair_id": pid,
                "provider": prov,
                "embedding_engine": eng,
                "preflight_manifest": str(pre_path),
            }
            if reasons:
                row["reasons"] = list(reasons)
                blocked.append(row)
            else:
                runnable.append(row)

    queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_ids": snapshot_ids,
        "runnable": sorted(runnable, key=lambda r: str(r.get("pair_id") or "")),
        "blocked": sorted(blocked, key=lambda r: str(r.get("pair_id") or "")),
        "skipped": skipped,
    }
    _write_json(run_dir / "queue.json", queue)
    print(
        json.dumps(
            {
                "runnable": len(runnable),
                "blocked": len(blocked),
                "skipped": len(skipped),
            },
            indent=2,
        )
    )


def _load_queue(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "queue.json"
    if not p.is_file():
        raise SystemExit(f"missing {p}; run preflight first")
    return _read_json(p)


def _load_progress(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "progress.json"
    if not p.is_file():
        return {
            "version": 1,
            "pairs": {},
            "deferred_due_to_timebox": [],
        }
    return _read_json(p)


def _save_progress(run_dir: Path, progress: dict[str, Any]) -> None:
    _write_json(run_dir / "progress.json", progress)


def cmd_execute(
    *,
    run_dir: Path,
    database_url: str | None,
    max_wall_hours: float,
    next_pair_floor_minutes: float,
    shard_count: int,
    max_parallel: int,
) -> None:
    queue = _load_queue(run_dir)
    snapshots_path = run_dir / "snapshots.json"
    if not snapshots_path.is_file():
        raise SystemExit(f"missing {snapshots_path}")

    runnable = list(queue.get("runnable") or [])
    ids_sorted = sorted(
        [r for r in runnable if isinstance(r, dict)],
        key=lambda r: str(r.get("pair_id") or ""),
    )

    progress = _load_progress(run_dir)
    pairs_state: dict[str, Any] = dict(progress.get("pairs") or {})
    deferred: list[str] = list(progress.get("deferred_due_to_timebox") or [])

    max_wall = float(max_wall_hours) * 3600.0
    floor_s = float(next_pair_floor_minutes) * 60.0
    wall_start = float(progress.get("wall_start_epoch") or 0.0) or None

    work_root = run_dir / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if wall_start is None:
        wall_start = time.time()
        progress["wall_start_epoch"] = wall_start
        progress["wall_budget_seconds"] = max_wall
        progress["next_pair_floor_seconds"] = floor_s

    def remaining_s() -> float:
        return max_wall - (time.time() - float(wall_start))

    if remaining_s() < floor_s:
        print(
            json.dumps(
                {
                    "halt": "insufficient_remaining_budget_for_floor",
                    "remaining_seconds": remaining_s(),
                    "floor_seconds": floor_s,
                },
                indent=2,
            ),
            flush=True,
        )
        _save_progress(run_dir, progress)
        return

    for row in ids_sorted:
        pid = str(row.get("pair_id") or "")
        if not pid:
            continue
        st = pairs_state.get(pid) or {}
        if st.get("status") == "completed" and int(st.get("exit_code") or -1) == 0:
            continue

        rem = remaining_s()
        if rem < floor_s:
            try:
                start_idx = ids_sorted.index(row)
            except ValueError:
                start_idx = 0
            for r2 in ids_sorted[start_idx:]:
                p2 = str(r2.get("pair_id") or "")
                if not p2:
                    continue
                st2 = pairs_state.get(p2) or {}
                done = st2.get("status") == "completed" and int(st2.get("exit_code") or -1) == 0
                if done:
                    continue
                if p2 not in deferred:
                    deferred.append(p2)
            progress["deferred_due_to_timebox"] = sorted(set(deferred))
            print(json.dumps({"deferred_due_to_timebox": progress["deferred_due_to_timebox"]}, indent=2))
            break

        prov = str(row.get("provider") or "")
        eng = str(row.get("embedding_engine") or "")
        wdir = work_root / pid
        wdir.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            sys.executable,
            str(SHARDED_SCRIPT),
            "all",
            "--work-dir",
            str(wdir),
            "--engines",
            eng,
            "--shard-count",
            str(int(shard_count)),
            "--max-parallel",
            str(int(max_parallel)),
            "--snapshot-ids-file",
            str(snapshots_path),
        ]
        if prov:
            cmd.extend(["--provider", prov])
        if database_url:
            cmd.extend(["--database-url", database_url])

        log_path = logs_dir / f"{pid}.log"
        t0 = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"cmd: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        rc = int(proc.returncode)
        elapsed = time.time() - t0
        status = "completed" if rc == 0 else "partial"
        pairs_state[pid] = {
            "provider": prov,
            "embedding_engine": eng,
            "work_dir": str(wdir.resolve()),
            "exit_code": rc,
            "status": status,
            "elapsed_seconds": round(elapsed, 3),
            "log_path": str(log_path.resolve()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        progress["pairs"] = pairs_state
        progress["deferred_due_to_timebox"] = deferred
        _save_progress(run_dir, progress)
        print(json.dumps({"pair_id": pid, "exit_code": rc, "status": status}, indent=2), flush=True)


def _run_sharded_phase(
    *,
    phase: str,
    work_dir: Path,
    engine: str,
    provider: str,
    snapshots_path: Path,
    database_url: str | None,
    shard_count: int,
    max_parallel: int,
) -> int:
    cmd: list[str] = [
        sys.executable,
        str(SHARDED_SCRIPT),
        phase,
        "--work-dir",
        str(work_dir),
        "--engines",
        engine,
        "--shard-count",
        str(shard_count),
        "--max-parallel",
        str(max_parallel),
        "--snapshot-ids-file",
        str(snapshots_path),
    ]
    if provider:
        cmd.extend(["--provider", provider])
    if database_url:
        cmd.extend(["--database-url", database_url])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(proc.returncode)


def cmd_verify_package(*, run_dir: Path, database_url: str | None, shard_count: int, max_parallel: int) -> None:
    snapshots_path = run_dir / "snapshots.json"
    if not snapshots_path.is_file():
        raise SystemExit(f"missing {snapshots_path}")

    progress = _load_progress(run_dir)
    pairs_state: dict[str, Any] = dict(progress.get("pairs") or {})
    queue = _load_queue(run_dir)

    summaries: list[dict[str, Any]] = []

    for pid, st in sorted(pairs_state.items(), key=lambda kv: kv[0]):
        if not isinstance(st, dict):
            continue
        wdir = Path(str(st.get("work_dir") or ""))
        if not wdir.is_dir():
            summaries.append({"pair_id": pid, "status": "missing_work_dir"})
            continue
        prov = str(st.get("provider") or "")
        eng = str(st.get("embedding_engine") or "")

        vrc = _run_sharded_phase(
            phase="verify",
            work_dir=wdir,
            engine=eng,
            provider=prov,
            snapshots_path=snapshots_path,
            database_url=database_url,
            shard_count=shard_count,
            max_parallel=max_parallel,
        )
        verify_report_path = wdir / "verify_report.json"
        if vrc == 0 and verify_report_path.is_file():
            _run_sharded_phase(
                phase="acceptance",
                work_dir=wdir,
                engine=eng,
                provider=prov,
                snapshots_path=snapshots_path,
                database_url=database_url,
                shard_count=shard_count,
                max_parallel=max_parallel,
            )

        cov: dict[str, Any] | None = None
        if verify_report_path.is_file():
            try:
                rep = json.loads(verify_report_path.read_text(encoding="utf-8"))
                eng_rep = rep.get(eng) if isinstance(rep, dict) else None
                if isinstance(eng_rep, dict):
                    cov = eng_rep.get("coverage")  # type: ignore[assignment]
            except json.JSONDecodeError:
                cov = None

        summaries.append(
            {
                "pair_id": pid,
                "execute_status": st.get("status"),
                "execute_exit_code": st.get("exit_code"),
                "verify_exit_code": vrc,
                "coverage": cov,
                "work_dir": str(wdir.resolve()),
            }
        )

    blocked_ids = {str(b.get("pair_id")) for b in (queue.get("blocked") or []) if isinstance(b, dict)}
    runnable_ids = {str(b.get("pair_id")) for b in (queue.get("runnable") or []) if isinstance(b, dict)}
    attempted = set(pairs_state.keys())
    not_started = sorted(runnable_ids - attempted)
    deferred = list(progress.get("deferred_due_to_timebox") or [])

    consolidated = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir.resolve()),
        "per_pair": summaries,
        "blocked_pair_ids": sorted(blocked_ids),
        "runnable_not_attempted": not_started,
        "deferred_due_to_timebox": sorted(set(deferred)),
    }
    _write_json(run_dir / "CONSOLIDATED_SUMMARY.json", consolidated)

    # RESUME_COMMANDS.md
    lines = [
        "# Overnight five-snapshot backfill — resume commands",
        "",
        f"- run_dir: `{run_dir.resolve()}`",
        f"- snapshots: `{snapshots_path.resolve()}`",
        "",
        "## Continue orchestrator execute (remaining runnable / deferred)",
        "",
        "```powershell",
        f"python scripts/living/run_overnight_fivesnap_backfill.py execute --run-dir \"{run_dir.resolve()}\"",
        "```",
        "",
        "## Per-pair sharded resume (same work dirs — preserves shard state)",
        "",
    ]
    for pid in sorted(set(not_started) | set(deferred) | {p for p, s in pairs_state.items() if (s.get("exit_code") or 0) != 0}):
        row = next(
            (r for r in (queue.get("runnable") or []) if isinstance(r, dict) and str(r.get("pair_id")) == pid),
            None,
        )
        if row is None:
            continue
        prov = str(row.get("provider") or "")
        eng = str(row.get("embedding_engine") or "")
        wdir = run_dir / "work" / pid
        cmd = (
            f"python scripts/living/run_two_engine_backfill_sharded.py all "
            f'--work-dir "{wdir.resolve()}" --engines {eng!r} '
            f"--shard-count {shard_count} --max-parallel {max_parallel} "
            f'--snapshot-ids-file "{snapshots_path.resolve()}"'
        )
        if prov:
            cmd += f" --provider {prov!r}"
        lines.extend([f"### {pid}", "", "```powershell", cmd, "```", ""])
    (run_dir / "RESUME_COMMANDS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote": str((run_dir / "CONSOLIDATED_SUMMARY.json").resolve())}, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Write snapshots.json + pairs.candidate.json")
    p_init.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run root (default: experimental_results/overnight_5snap/<UTC>)",
    )
    p_init.add_argument(
        "--snapshot-ids",
        type=int,
        nargs="*",
        default=list(DEFAULT_SNAPSHOT_IDS),
        help=f"snapshot group ids (default: {list(DEFAULT_SNAPSHOT_IDS)})",
    )

    p_pre = sub.add_parser("preflight", help="Dry manifests + queue.json")
    p_pre.add_argument("--run-dir", type=Path, required=True)
    p_pre.add_argument("--database-url", default=None)

    p_ex = sub.add_parser("execute", help="Sequential timeboxed sharded runs")
    p_ex.add_argument("--run-dir", type=Path, required=True)
    p_ex.add_argument("--database-url", default=None)
    p_ex.add_argument("--max-wall-hours", type=float, default=24.0)
    p_ex.add_argument("--next-pair-floor-minutes", type=float, default=90.0)
    p_ex.add_argument("--shard-count", type=int, default=8)
    p_ex.add_argument("--max-parallel", type=int, default=8)

    p_vp = sub.add_parser("verify-package", help="Verify + acceptance + consolidated outputs")
    p_vp.add_argument("--run-dir", type=Path, required=True)
    p_vp.add_argument("--database-url", default=None)
    p_vp.add_argument("--shard-count", type=int, default=8)
    p_vp.add_argument("--max-parallel", type=int, default=8)

    args = p.parse_args(argv)

    if args.command == "init":
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else PROJECT_ROOT / "experimental_results" / "overnight_5snap" / ts
        )
        cmd_init(run_dir=run_dir, snapshot_ids=list(args.snapshot_ids))
        return 0
    if args.command == "preflight":
        cmd_preflight(run_dir=Path(args.run_dir), database_url=args.database_url)
        return 0
    if args.command == "execute":
        cmd_execute(
            run_dir=Path(args.run_dir),
            database_url=args.database_url,
            max_wall_hours=float(args.max_wall_hours),
            next_pair_floor_minutes=float(args.next_pair_floor_minutes),
            shard_count=int(args.shard_count),
            max_parallel=int(args.max_parallel),
        )
        return 0
    if args.command == "verify-package":
        cmd_verify_package(
            run_dir=Path(args.run_dir),
            database_url=args.database_url,
            shard_count=int(args.shard_count),
            max_parallel=int(args.max_parallel),
        )
        return 0
    raise SystemExit("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())

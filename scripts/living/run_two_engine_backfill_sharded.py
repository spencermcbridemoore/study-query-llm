#!/usr/bin/env python3
"""Orchestrate two-engine clustering backfill with snapshot sharding and parallel workers.

Runs the workflow described in the operator plan: preflight manifests, deterministic
round-robin shards (default 8), parallel ``backfill_all_variant_clustering_analysis``
processes per engine, then global verify + acceptance summary.

Default engines complement ``text-embedding-3-small``:

- ``text-embedding-3-large``
- ``embed-v-4-0``

Override with ``--engines``. Uses ``--snapshot-ids-file`` on child processes to avoid
long argv lists.

Living library: :mod:`study_query_llm.experiments.clustering_analysis_backfill`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from study_query_llm.experiments.clustering_analysis_backfill import (  # noqa: E402
    build_manifest,
    ok_snapshot_ids_from_manifest,
    preflight_manifest_blocking_issues,
    refresh_manifest_completion,
    round_robin_shard_snapshot_ids,
    validate_registry_expansion_or_raise,
    validate_shards_partition_exact,
    verify_manifest_coverage,
)

_BACKFILL_SCRIPT = PROJECT_ROOT / "scripts" / "living" / "backfill_all_variant_clustering_analysis.py"

DEFAULT_ENGINES: tuple[str, ...] = ("text-embedding-3-large", "embed-v-4-0")


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if v:
            return v
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")


def engine_slug(embedding_engine: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in embedding_engine.strip())


def _run_backfill_subprocess(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(_BACKFILL_SCRIPT), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return int(proc.returncode), out


def cmd_preflight(
    *,
    engines: list[str],
    work_dir: Path,
    provider: str | None,
    database_url: str | None,
) -> dict[str, Any]:
    validate_registry_expansion_or_raise()
    work_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"engines": list(engines), "manifests": {}}
    for eng in engines:
        slug_e = engine_slug(eng)
        manifest_path = work_dir / f"{slug_e}.preflight.json"
        cmd = [
            "--embedding-engine",
            eng,
            "--dry-run",
            "--manifest-out",
            str(manifest_path),
        ]
        if provider:
            cmd.extend(["--provider", provider])
        if database_url:
            cmd.extend(["--database-url", database_url])
        rc, text = _run_backfill_subprocess(cmd)
        print(text)
        if rc != 0:
            raise SystemExit(f"preflight failed for {eng!r} exit={rc}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        blockers = preflight_manifest_blocking_issues(manifest)
        out["manifests"][eng] = str(manifest_path)
        if blockers:
            print(f"blockers for {eng}:")
            for b in blockers:
                print(f"  - {b}")
            raise SystemExit(2)
    return out


def cmd_shard(
    *,
    engines: list[str],
    work_dir: Path,
    shard_count: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"shard_count": shard_count, "engines": {}}
    for eng in engines:
        slug_e = engine_slug(eng)
        preflight_path = work_dir / f"{slug_e}.preflight.json"
        if not preflight_path.exists():
            raise SystemExit(f"missing preflight manifest: {preflight_path}")
        manifest = json.loads(preflight_path.read_text(encoding="utf-8"))
        ids = ok_snapshot_ids_from_manifest(manifest)
        shards = round_robin_shard_snapshot_ids(ids, shard_count)
        validate_shards_partition_exact(ids, shards)
        shard_paths: list[str] = []
        for i, shard_ids in enumerate(shards):
            spath = work_dir / f"{slug_e}.shard{i}.snapshot_ids.json"
            spath.write_text(
                json.dumps(
                    {
                        "embedding_engine": eng,
                        "shard_index": i,
                        "shard_count": shard_count,
                        "snapshot_ids": shard_ids,
                    },
                    indent=2,
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            shard_paths.append(str(spath))
        summary["engines"][eng] = {
            "ok_snapshots": len(ids),
            "shard_files": shard_paths,
        }
        print(f"{eng}: {len(ids)} ok snapshots -> {shard_count} shard files")
    return summary


def _execute_one_shard(
    *,
    eng: str,
    shard_index: int,
    work_dir: Path,
    provider: str | None,
    database_url: str | None,
    dry_run: bool,
    force: bool,
) -> tuple[int, str]:
    slug_e = engine_slug(eng)
    shard_file = work_dir / f"{slug_e}.shard{shard_index}.snapshot_ids.json"
    if not shard_file.exists():
        return 1, f"missing shard file {shard_file}"
    raw_ids = json.loads(shard_file.read_text(encoding="utf-8")).get("snapshot_ids") or []
    if not raw_ids:
        return 0, f"skip empty shard {eng} shard{shard_index}"

    cmd = [
        "--embedding-engine",
        eng,
        "--execute",
        "--resume",
        "--snapshot-ids-file",
        str(shard_file),
        "--run-state",
        str(work_dir / f"{slug_e}.shard{shard_index}.state.json"),
        "--manifest-out",
        str(work_dir / f"{slug_e}.shard{shard_index}.exec.json"),
    ]
    if dry_run:
        cmd.append("--dry-run")
    if force:
        cmd.append("--force")
    if provider:
        cmd.extend(["--provider", provider])
    if database_url:
        cmd.extend(["--database-url", database_url])
    rc, text = _run_backfill_subprocess(cmd)
    return rc, text


def cmd_execute(
    *,
    engines: list[str],
    work_dir: Path,
    shard_count: int,
    provider: str | None,
    database_url: str | None,
    dry_run: bool,
    force: bool,
    max_parallel: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for eng in engines:
        print(f"=== execute engine {eng!r} ({max_parallel} workers) ===")
        futs = []
        codes: list[int] = []
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            for i in range(shard_count):
                futs.append(
                    ex.submit(
                        _execute_one_shard,
                        eng=eng,
                        shard_index=i,
                        work_dir=work_dir,
                        provider=provider,
                        database_url=database_url,
                        dry_run=dry_run,
                        force=force,
                    )
                )
            for fut in as_completed(futs):
                rc, text = fut.result()
                print(text)
                codes.append(rc)
        if any(c != 0 for c in codes):
            raise SystemExit(f"execute had non-zero exits for {eng}: {codes}")
        results[eng] = {"exit_codes": codes}
    return results


def cmd_verify(
    *,
    engines: list[str],
    work_dir: Path,
    provider: str | None,
    database_url: str | None,
) -> dict[str, Any]:
    db = DatabaseConnectionV2(_resolve_database_url(database_url), enable_pgvector=False)
    report: dict[str, Any] = {}
    for eng in engines:
        slug_e = engine_slug(eng)
        final_path = work_dir / f"{slug_e}.final.json"
        cmd = [
            "--embedding-engine",
            eng,
            "--dry-run",
            "--manifest-out",
            str(final_path),
        ]
        if provider:
            cmd.extend(["--provider", provider])
        if database_url:
            cmd.extend(["--database-url", database_url])
        rc, text = _run_backfill_subprocess(cmd)
        print(text)
        if rc != 0:
            raise SystemExit(f"final manifest failed for {eng} rc={rc}")

        manifest = json.loads(final_path.read_text(encoding="utf-8"))
        blockers = preflight_manifest_blocking_issues(manifest)
        if blockers:
            print(f"final manifest blockers for {eng}: {blockers}")

        with db.session_scope() as session:
            refreshed = refresh_manifest_completion(session, dict(manifest))
        cov = verify_manifest_coverage(refreshed)
        report[eng] = {
            "final_manifest": str(final_path),
            "coverage": cov,
            "refreshed_missing_count": refreshed.get("missing_count"),
            "total_targets": refreshed.get("total_targets"),
        }
        print(json.dumps({eng: report[eng]}, indent=2))
        if not cov.get("coverage_complete"):
            raise SystemExit(f"coverage incomplete for {eng}")
    return report


def cmd_acceptance(*, engines: list[str], work_dir: Path, verify_report: dict[str, Any]) -> Path:
    lines = [
        "# Two-engine sharded backfill acceptance",
        "",
        f"- generated_at: {datetime.now(timezone.utc).isoformat()}",
        f"- work_dir: `{work_dir}`",
        "",
    ]
    for eng in engines:
        r = verify_report.get(eng) or {}
        cov = r.get("coverage") or {}
        lines.extend(
            [
                f"## {eng}",
                "",
                f"- coverage_complete: {cov.get('coverage_complete')}",
                f"- missing_keys_remaining: {cov.get('missing_keys_remaining')}",
                f"- expected_target_keys: {cov.get('expected_target_keys')}",
                f"- total_targets (manifest): {r.get('total_targets')}",
                f"- refreshed_missing_count: {r.get('refreshed_missing_count')}",
                f"- final_manifest: `{r.get('final_manifest')}`",
                "",
            ]
        )
    out_path = work_dir / "ACCEPTANCE_SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "phase",
        choices=("preflight", "shard", "execute", "verify", "acceptance", "all"),
        help="workflow phase",
    )
    p.add_argument(
        "--engines",
        nargs="+",
        default=list(DEFAULT_ENGINES),
        help=f"embedding_engine strings (default: {DEFAULT_ENGINES})",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="artifact directory (default: experimental_results/backfill_manifests/two_engine_<utc>)",
    )
    p.add_argument("--shard-count", type=int, default=8, help="parallel shards per engine")
    p.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="concurrent child processes per engine (default: shard-count)",
    )
    p.add_argument("--provider", default=None)
    p.add_argument("--database-url", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="for execute phase: pass --dry-run to children (no analyze writes)",
    )
    p.add_argument("--force", action="store_true", help="pass --force to execute children")
    args = p.parse_args(argv)

    engines = [str(e).strip() for e in args.engines if str(e).strip()]
    if len(engines) < 1:
        raise SystemExit("need at least one --engines value")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work_dir = (
        Path(args.work_dir)
        if args.work_dir is not None
        else PROJECT_ROOT / "experimental_results" / "backfill_manifests" / f"two_engine_{ts}"
    )
    shard_count = max(1, int(args.shard_count))
    max_parallel = max(1, min(int(args.max_parallel) if args.max_parallel is not None else shard_count, shard_count))

    verify_report_path = work_dir / "verify_report.json"

    print(f"work_dir: {work_dir.resolve()}", flush=True)

    if args.phase == "preflight":
        cmd_preflight(
            engines=engines,
            work_dir=work_dir,
            provider=args.provider,
            database_url=args.database_url,
        )
        return 0

    if args.phase == "shard":
        cmd_shard(engines=engines, work_dir=work_dir, shard_count=shard_count)
        return 0

    if args.phase == "execute":
        cmd_execute(
            engines=engines,
            work_dir=work_dir,
            shard_count=shard_count,
            provider=args.provider,
            database_url=args.database_url,
            dry_run=bool(args.dry_run),
            force=bool(args.force),
            max_parallel=max_parallel,
        )
        return 0

    if args.phase == "verify":
        rep = cmd_verify(
            engines=engines,
            work_dir=work_dir,
            provider=args.provider,
            database_url=args.database_url,
        )
        verify_report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return 0

    if args.phase == "acceptance":
        if not verify_report_path.exists():
            raise SystemExit(f"missing {verify_report_path}; run verify first")
        verify_report = json.loads(verify_report_path.read_text(encoding="utf-8"))
        cmd_acceptance(engines=engines, work_dir=work_dir, verify_report=verify_report)
        return 0

    # all
    cmd_preflight(
        engines=engines,
        work_dir=work_dir,
        provider=args.provider,
        database_url=args.database_url,
    )
    cmd_shard(engines=engines, work_dir=work_dir, shard_count=shard_count)
    cmd_execute(
        engines=engines,
        work_dir=work_dir,
        shard_count=shard_count,
        provider=args.provider,
        database_url=args.database_url,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        max_parallel=max_parallel,
    )
    rep = cmd_verify(
        engines=engines,
        work_dir=work_dir,
        provider=args.provider,
        database_url=args.database_url,
    )
    verify_report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    cmd_acceptance(engines=engines, work_dir=work_dir, verify_report=rep)
    print(json.dumps({"work_dir": str(work_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

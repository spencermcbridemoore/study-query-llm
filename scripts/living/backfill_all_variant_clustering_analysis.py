#!/usr/bin/env python3
"""Plan and execute bundled clustering analysis across snapshots for one embedding engine.

Matches ``dataset_snapshot`` groups to ``embedding_batch`` rows on lineage key
``(source_dataframe_group_id, source_dataframe_row_count)`` against the batch
``entry_max`` (the full-dataframe embed row count): ``analyze`` consumes the full
embedding matrix sliced by the snapshot resolved-index, so the matching batch is
the full-dataframe embed, not a per-snapshot prefix.

Writes manifests under ``experimental_results/backfill_manifests/``. Uses deterministic
``analysis_run_key`` values prefixed with ``backfill_exec__`` (see
:mod:`study_query_llm.experiments.clustering_analysis_backfill`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    preflight_manifest_blocking_issues,
    refresh_manifest_completion,
    run_backfill_execution,
    validate_registry_expansion_or_raise,
    verify_manifest_coverage,
)


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if v:
            return v
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")


def _default_manifest_path(embedding_engine: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in embedding_engine)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = PROJECT_ROOT / "experimental_results" / "backfill_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"clustering_backfill__{safe}__{ts}.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embedding-engine",
        default="text-embedding-3-small",
        help="embedding_engine metadata value to match on embedding_batch groups",
    )
    p.add_argument("--provider", default=None, help="optional provider filter")
    p.add_argument(
        "--snapshot-ids",
        type=int,
        nargs="*",
        default=None,
        help="optional subset of dataset_snapshot group ids (default: all snapshots)",
    )
    p.add_argument(
        "--snapshot-ids-file",
        type=Path,
        default=None,
        help=(
            "JSON file {\"snapshot_ids\": [int, ...]} for large shard lists "
            "(alternative to --snapshot-ids)"
        ),
    )
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="write manifest JSON here (default: experimental_results/backfill_manifests/...)",
    )
    p.add_argument(
        "--validate-registry-only",
        action="store_true",
        help="expand all registry methods with synthetic sizes and exit (no DB)",
    )
    p.add_argument("--dry-run", action="store_true", help="build manifest / execution dry-run only")
    p.add_argument(
        "--execute",
        action="store_true",
        help="run analyze() for targets not completed in DB",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="keep existing run-state JSON entries when executing",
    )
    p.add_argument(
        "--run-state",
        type=Path,
        default=None,
        help="path to run-state JSON for resume (recommended with --execute)",
    )
    p.add_argument("--force", action="store_true", help="pass force=True into analyze()")
    p.add_argument("--database-url", default=None)
    p.add_argument(
        "--verify-manifest",
        type=Path,
        default=None,
        help="load manifest JSON, refresh completion from DB, print coverage report",
    )
    args = p.parse_args(argv)

    snapshot_ids_arg: list[int] | None = None
    if args.snapshot_ids_file is not None:
        if args.snapshot_ids is not None:
            raise SystemExit("Use only one of --snapshot-ids or --snapshot-ids-file")
        raw = json.loads(Path(args.snapshot_ids_file).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise SystemExit("--snapshot-ids-file must contain a JSON object")
        ids = raw.get("snapshot_ids")
        if not isinstance(ids, list):
            raise SystemExit("--snapshot-ids-file requires key snapshot_ids (list)")
        snapshot_ids_arg = []
        for x in ids:
            try:
                snapshot_ids_arg.append(int(x))
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"invalid snapshot id in file: {x!r}") from exc
    elif args.snapshot_ids is not None:
        snapshot_ids_arg = list(args.snapshot_ids)

    if args.validate_registry_only:
        validate_registry_expansion_or_raise()
        print("registry expansion validation OK")
        return 0

    if args.verify_manifest:
        path = Path(args.verify_manifest)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        db = DatabaseConnectionV2(_resolve_database_url(args.database_url), enable_pgvector=False, write_intent=WriteIntent.CANONICAL)
        with db.session_scope() as session:
            refreshed = refresh_manifest_completion(session, manifest)
        report = verify_manifest_coverage(refreshed)
        print(json.dumps({"coverage": report, "manifest_path": str(path)}, indent=2))
        return 0 if report.get("coverage_complete") else 2

    manifest_path = Path(args.manifest_out) if args.manifest_out else _default_manifest_path(
        str(args.embedding_engine)
    )

    db = DatabaseConnectionV2(_resolve_database_url(args.database_url), enable_pgvector=False, write_intent=WriteIntent.CANONICAL)

    validate_registry_expansion_or_raise()

    with db.session_scope() as session:
        manifest = build_manifest(
            session,
            embedding_engine=str(args.embedding_engine),
            provider=args.provider,
            snapshot_ids=snapshot_ids_arg,
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote manifest: {manifest_path}")

    blockers = preflight_manifest_blocking_issues(manifest)
    if blockers:
        print("preflight issues:")
        for b in blockers:
            print(f"  - {b}")
        if args.execute:
            print("abort: resolve preflight issues before --execute")
            return 2

    stats: dict[str, Any] | None = None
    if args.execute:
        stats = run_backfill_execution(
            db=db,
            manifest=manifest,
            run_state_path=str(args.run_state) if args.run_state else None,
            dry_run=bool(args.dry_run),
            resume=bool(args.resume),
            force=bool(args.force),
        )
        print(json.dumps(stats, indent=2))
        if stats.get("halted_reason"):
            return 3

    cov = verify_manifest_coverage(manifest)
    print(json.dumps({"preflight_coverage": cov}, indent=2))
    return 2 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())

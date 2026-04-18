#!/usr/bin/env python3
"""Run BANKING77 through acquire -> snapshot -> embed -> analyze."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from study_query_llm.datasets.source_specs.banking77 import BANKING77_DATASET_SLUG
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.analyze import analyze
from study_query_llm.pipeline.embed import embed
from study_query_llm.pipeline.snapshot import snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BANKING77 pipeline stages end-to-end with provenance persistence.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL override (defaults to env DATABASE_URL).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Artifact base directory (default: artifacts).",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=BANKING77_DATASET_SLUG,
        help="Dataset slug from source spec registry (default: banking77).",
    )
    parser.add_argument(
        "--embedding-deployment",
        type=str,
        default="text-embedding-3-large",
        help="Embedding deployment/model name for stage 3.",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="azure",
        help="Embedding provider for stage 3 (default: azure).",
    )
    parser.add_argument(
        "--embedding-representation",
        type=str,
        choices=["full", "intent_mean", "sparse"],
        default="full",
        help="Embedding representation persisted by stage 3.",
    )
    parser.add_argument(
        "--embedding-chunk-size",
        type=int,
        default=None,
        help="Optional chunk size for provider embedding calls.",
    )
    parser.add_argument(
        "--embedding-timeout-seconds",
        type=float,
        default=1800.0,
        help="Embed-stage timeout passed to embedding helper.",
    )
    parser.add_argument(
        "--analysis-method",
        type=str,
        default="bank77_structural_summary",
        help="Method name persisted by stage 4 analysis.",
    )
    parser.add_argument(
        "--analysis-run-key",
        type=str,
        default=None,
        help="Deterministic run key; defaults to UTC timestamp key.",
    )
    parser.add_argument(
        "--analysis-method-version",
        type=str,
        default=None,
        help="Optional method version for analysis result rows.",
    )
    parser.add_argument(
        "--force-acquire",
        action="store_true",
        help="Bypass stage-1 idempotent reuse.",
    )
    parser.add_argument(
        "--force-snapshot",
        action="store_true",
        help="Bypass stage-2 idempotent reuse.",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Bypass stage-3 idempotent reuse.",
    )
    parser.add_argument(
        "--force-analyze",
        action="store_true",
        help="Bypass stage-4 completed-run reuse.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Stop after embedding stage (skip stage 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    database_url = str(args.database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required (env or --database-url)")

    acquire_spec = ACQUIRE_REGISTRY.get(str(args.dataset_slug))
    if acquire_spec is None:
        known = ", ".join(sorted(ACQUIRE_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset slug {args.dataset_slug!r}. Known: {known}")

    run_key = (
        str(args.analysis_run_key).strip()
        if args.analysis_run_key
        else datetime.now(timezone.utc).strftime("bank77_%Y%m%d_%H%M%S")
    )

    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()

    acquired = acquire(
        acquire_spec,
        force=bool(args.force_acquire),
        db=db,
        artifact_dir=str(args.artifact_dir),
    )
    snapped = snapshot(
        acquired.group_id,
        force=bool(args.force_snapshot),
        db=db,
        artifact_dir=str(args.artifact_dir),
    )
    embedded = embed(
        snapped.group_id,
        deployment=str(args.embedding_deployment),
        provider=str(args.embedding_provider),
        representation=str(args.embedding_representation),
        force=bool(args.force_embed),
        db=db,
        artifact_dir=str(args.artifact_dir),
        chunk_size=args.embedding_chunk_size,
        timeout=float(args.embedding_timeout_seconds),
    )

    analyzed = None
    if not args.skip_analysis:
        analyzed = analyze(
            embedded.group_id,
            method_name=str(args.analysis_method),
            method_version=args.analysis_method_version,
            run_key=run_key,
            force=bool(args.force_analyze),
            db=db,
            artifact_dir=str(args.artifact_dir),
            parameters={
                "dataset_slug": str(args.dataset_slug),
                "embedding_representation": str(args.embedding_representation),
                "embedding_deployment": str(args.embedding_deployment),
                "embedding_provider": str(args.embedding_provider),
            },
        )

    summary = {
        "dataset_slug": str(args.dataset_slug),
        "acquire": {
            "group_id": int(acquired.group_id),
            "metadata": acquired.metadata,
            "artifact_uris": acquired.artifact_uris,
        },
        "snapshot": {
            "group_id": int(snapped.group_id),
            "metadata": snapped.metadata,
            "artifact_uris": snapped.artifact_uris,
        },
        "embed": {
            "group_id": int(embedded.group_id),
            "metadata": embedded.metadata,
            "artifact_uris": embedded.artifact_uris,
        },
        "analyze": (
            {
                "group_id": int(analyzed.group_id),
                "run_id": int(analyzed.run_id or 0),
                "metadata": analyzed.metadata,
                "artifact_uris": analyzed.artifact_uris,
            }
            if analyzed is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

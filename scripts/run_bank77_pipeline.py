#!/usr/bin/env python3
"""Run BANKING77 through acquire -> parse -> snapshot -> embed -> analyze."""

from __future__ import annotations

import argparse
import hashlib
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
from study_query_llm.pipeline.hdbscan_runner import run_hdbscan_analysis
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot

EMBEDDING_REPRESENTATION_ALIAS: dict[str, str] = {
    "intent_mean": "label_centroid",
}


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
        choices=["full", "label_centroid", "intent_mean"],
        default="full",
        help=(
            "Representation used by stage 5 analysis (embed stage is always full). "
            "Legacy alias: intent_mean -> label_centroid."
        ),
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
        help="Method name persisted by stage 5 analysis.",
    )
    parser.add_argument(
        "--analysis-strategy",
        type=str,
        choices=["default", "hdbscan"],
        default="default",
        help="Analysis implementation strategy (default or hdbscan).",
    )
    parser.add_argument(
        "--analysis-run-key",
        type=str,
        default=None,
        help="Deterministic run key; hdbscan defaults to params-hash key.",
    )
    parser.add_argument(
        "--analysis-method-version",
        type=str,
        default=None,
        help="Optional method version for analysis result rows.",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=5,
        help="HDBSCAN min_cluster_size (used only with --analysis-strategy hdbscan).",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="Optional HDBSCAN min_samples (used only with --analysis-strategy hdbscan).",
    )
    parser.add_argument(
        "--hdbscan-metric",
        type=str,
        default="euclidean",
        help="HDBSCAN distance metric (used only with --analysis-strategy hdbscan).",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default="eom",
        help="HDBSCAN cluster selection method.",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon.",
    )
    parser.add_argument(
        "--hdbscan-alpha",
        type=float,
        default=1.0,
        help="HDBSCAN alpha parameter.",
    )
    parser.add_argument(
        "--hdbscan-allow-single-cluster",
        action="store_true",
        help="Enable HDBSCAN allow_single_cluster.",
    )
    parser.add_argument(
        "--hdbscan-normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before HDBSCAN fit.",
    )
    parser.add_argument(
        "--force-acquire",
        action="store_true",
        help="Bypass stage-1 idempotent reuse.",
    )
    parser.add_argument(
        "--force-snapshot",
        action="store_true",
        help="Bypass stage-3 idempotent reuse.",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Bypass stage-2 idempotent reuse.",
    )
    parser.add_argument(
        "--force-embed",
        action="store_true",
        help="Bypass stage-4 idempotent reuse.",
    )
    parser.add_argument(
        "--force-analyze",
        action="store_true",
        help="Bypass stage-5 completed-run reuse.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Stop after embedding stage (skip stage 4).",
    )
    return parser.parse_args()


def _safe_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    return cleaned.strip("_") or "value"


def _canonical_embedding_representation(value: str) -> str:
    normalized = str(value).strip().lower()
    return EMBEDDING_REPRESENTATION_ALIAS.get(normalized, normalized)


def _validate_embedding_representation_for_analysis(args: argparse.Namespace) -> None:
    strategy = str(args.analysis_strategy).strip().lower()
    representation = _canonical_embedding_representation(args.embedding_representation)
    if strategy == "hdbscan" and representation != "full":
        raise ValueError(
            "--analysis-strategy hdbscan requires --embedding-representation full; "
            f"got {representation!r}"
        )


def _resolve_analysis_method_name(args: argparse.Namespace) -> str:
    method_name = str(args.analysis_method).strip()
    if (
        str(args.analysis_strategy) == "hdbscan"
        and method_name == "bank77_structural_summary"
    ):
        return "bank77_hdbscan_analysis"
    return method_name


def _build_analysis_parameters(args: argparse.Namespace) -> dict[str, object]:
    canonical_representation = _canonical_embedding_representation(args.embedding_representation)
    params: dict[str, object] = {
        "dataset_slug": str(args.dataset_slug),
        "representation_type": canonical_representation,
        "embedding_representation": canonical_representation,
        "embedding_deployment": str(args.embedding_deployment),
        "embedding_provider": str(args.embedding_provider),
        "analysis_strategy": str(args.analysis_strategy),
    }
    if str(args.analysis_strategy) == "hdbscan":
        params.update(
            {
                "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
                "hdbscan_min_samples": (
                    int(args.hdbscan_min_samples)
                    if args.hdbscan_min_samples is not None
                    else None
                ),
                "hdbscan_metric": str(args.hdbscan_metric),
                "hdbscan_cluster_selection_method": str(
                    args.hdbscan_cluster_selection_method
                ),
                "hdbscan_cluster_selection_epsilon": float(
                    args.hdbscan_cluster_selection_epsilon
                ),
                "hdbscan_alpha": float(args.hdbscan_alpha),
                "hdbscan_allow_single_cluster": bool(args.hdbscan_allow_single_cluster),
                "hdbscan_normalize_embeddings": bool(args.hdbscan_normalize_embeddings),
            }
        )
    return params


def _resolve_run_key(
    args: argparse.Namespace,
    *,
    method_name: str,
    parameters: dict[str, object],
) -> str:
    if args.analysis_run_key:
        return str(args.analysis_run_key).strip()
    strategy = str(args.analysis_strategy)
    if strategy != "hdbscan":
        return datetime.now(timezone.utc).strftime("bank77_%Y%m%d_%H%M%S")
    canonical = json.dumps(
        {
            "method_name": method_name,
            "parameters": parameters,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    dataset_token = _safe_token(str(parameters.get("dataset_slug") or "dataset"))
    deployment_token = _safe_token(str(parameters.get("embedding_deployment") or "embed"))
    return f"{dataset_token}_{deployment_token}_hdbscan_{digest}"


def _resolve_method_runner(args: argparse.Namespace):
    if str(args.analysis_strategy) == "hdbscan":
        return run_hdbscan_analysis
    return None


def main() -> None:
    args = _parse_args()
    database_url = str(args.database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required (env or --database-url)")

    acquire_spec = ACQUIRE_REGISTRY.get(str(args.dataset_slug))
    if acquire_spec is None:
        known = ", ".join(sorted(ACQUIRE_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset slug {args.dataset_slug!r}. Known: {known}")
    args.embedding_representation = _canonical_embedding_representation(
        str(args.embedding_representation)
    )
    _validate_embedding_representation_for_analysis(args)

    db = DatabaseConnectionV2(database_url, enable_pgvector=False)
    db.init_db()

    acquired = acquire(
        acquire_spec,
        force=bool(args.force_acquire),
        db=db,
        artifact_dir=str(args.artifact_dir),
    )
    parsed = parse(
        acquired.group_id,
        force=bool(args.force_parse),
        db=db,
        artifact_dir=str(args.artifact_dir),
    )
    snapped = snapshot(
        parsed.group_id,
        force=bool(args.force_snapshot),
        db=db,
        artifact_dir=str(args.artifact_dir),
    )
    embedded = embed(
        parsed.group_id,
        deployment=str(args.embedding_deployment),
        provider=str(args.embedding_provider),
        representation="full",
        force=bool(args.force_embed),
        db=db,
        artifact_dir=str(args.artifact_dir),
        chunk_size=args.embedding_chunk_size,
        timeout=float(args.embedding_timeout_seconds),
    )

    analyzed = None
    if not args.skip_analysis:
        analysis_method_name = _resolve_analysis_method_name(args)
        analysis_parameters = _build_analysis_parameters(args)
        run_key = _resolve_run_key(
            args,
            method_name=analysis_method_name,
            parameters=analysis_parameters,
        )
        analyzed = analyze(
            snapped.group_id,
            embedded.group_id,
            method_name=analysis_method_name,
            method_version=args.analysis_method_version,
            run_key=run_key,
            force=bool(args.force_analyze),
            db=db,
            artifact_dir=str(args.artifact_dir),
            parameters=analysis_parameters,
            method_runner=_resolve_method_runner(args),
        )

    summary = {
        "dataset_slug": str(args.dataset_slug),
        "acquire": {
            "group_id": int(acquired.group_id),
            "metadata": acquired.metadata,
            "artifact_uris": acquired.artifact_uris,
        },
        "parse": {
            "group_id": int(parsed.group_id),
            "metadata": parsed.metadata,
            "artifact_uris": parsed.artifact_uris,
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

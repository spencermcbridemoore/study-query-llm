"""
Ingest pkl sweep results into the DB as run/step groups with raw metric values.

Reads no_pca_50runs_*.pkl and experimental_sweep_*.pkl files, computes metrics
(ARI, silhouette, etc.), and stores them as Group(type=run) + Group(type=step)
entries with metric_specs provenance and SHA-256 checksums.

Usage:
  python scripts/ingest_sweep_to_db.py --data-dir experimental_results
  python scripts/ingest_sweep_to_db.py --data-dir experimental_results --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from sqlalchemy import text as sa_text

from study_query_llm.config import config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.models_v2 import Group
from study_query_llm.services.provenance_service import ProvenanceService
from study_query_llm.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

from study_query_llm.experiments.result_metrics import (
    rows_from_50runs as _rows_from_50runs,
    rows_from_sweep as _rows_from_sweep,
    dist_from_result as _dist_from_z,
    METRICS,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metric_specs() -> dict:
    """Build the metric_specs dict with live library versions."""
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "not installed"

    return {
        "objective": {
            "source": "pkl_stored",
            "description": "KLLMeans cosine objective (sum of min distances to centroids)",
        },
        "dispersion": {
            "source": "derived",
            "formula": "objective / n_samples",
        },
        "cosine_sim": {
            "source": "derived",
            "formula": "1.0 - dispersion",
        },
        "cosine_sim_norm": {
            "source": "derived",
            "formula": "(cosine_sim + 1.0) / 2.0",
        },
        "ari": {
            "compute_fn": "sklearn.metrics.adjusted_rand_score",
            "library": "scikit-learn",
            "library_version": sklearn_version,
            "params": {},
            "inputs": ["ground_truth_labels", "predicted_labels"],
        },
        "silhouette": {
            "compute_fn": "sklearn.metrics.silhouette_score",
            "library": "scikit-learn",
            "library_version": sklearn_version,
            "params": {"metric": "precomputed"},
            "inputs": ["distance_matrix", "predicted_labels"],
        },
    }


def build_ingestion_env() -> dict:
    """Capture the compute environment at ingestion time."""
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "not installed"

    return {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn_version,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def load_and_group_pkl(pkl_path: Path, data_type: str) -> dict | None:
    """
    Load a pkl file and extract rows grouped by k.

    Returns a dict with metadata and per-k metric arrays, or None on failure.
    """
    try:
        if pkl_path.stat().st_size == 0:
            return None
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        logger.warning("Skip %s: %s", pkl_path.name, e)
        return None

    if not isinstance(data, dict):
        return None

    if data_type == "50runs":
        if "result" not in data:
            return None
        rows = _rows_from_50runs({"data": data})
    else:
        meta = data.get("metadata") or {}
        if "embedding_engine" not in meta:
            return None
        rows = _rows_from_sweep({"data": data})

    if not rows:
        return None

    first = rows[0]
    dataset = first["dataset"]
    engine = first["engine"]
    summarizer = first["summarizer"]

    pkl_meta = data.get("metadata") or {}
    n_restarts = pkl_meta.get("n_restarts", 1)
    if data_type == "50runs" and n_restarts == 1:
        n_restarts = 50

    # Group rows by k -> lists of metric values
    by_k: dict[int, dict[str, list]] = defaultdict(lambda: {m: [] for m in METRICS})
    n_samples = 0
    for row in rows:
        k = row["k"]
        if row["n_samples"]:
            n_samples = row["n_samples"]
        for m in METRICS:
            val = row.get(m)
            if val is not None:
                by_k[k][m].append(val)

    k_values = sorted(by_k.keys())

    return {
        "dataset": dataset,
        "engine": engine,
        "summarizer": summarizer,
        "data_type": data_type,
        "n_restarts": n_restarts,
        "n_samples": n_samples,
        "k_range": [min(k_values), max(k_values)] if k_values else [0, 0],
        "by_k": dict(by_k),
        "pkl_path": pkl_path,
    }


def ingest_one_pkl(
    info: dict,
    repository: RawCallRepository,
    provenance: ProvenanceService,
    metric_specs: dict,
    ingestion_env: dict,
    dry_run: bool = False,
) -> int | None:
    """Ingest one parsed pkl's data into DB. Returns run group ID or None."""
    pkl_path: Path = info["pkl_path"]
    source_file = pkl_path.name

    sha = file_sha256(pkl_path)
    byte_size = pkl_path.stat().st_size

    if dry_run:
        print(f"  [DRY RUN] Would create run for {source_file}")
        print(f"    dataset={info['dataset']}, engine={info['engine']}, "
              f"summarizer={info['summarizer']}, data_type={info['data_type']}")
        print(f"    k_range={info['k_range']}, n_restarts={info['n_restarts']}")
        print(f"    sha256={sha[:16]}..., size={byte_size}")
        n_steps = len(info["by_k"])
        print(f"    Would create {n_steps} step groups (one per k)")
        return None

    # Idempotency: check if already ingested
    existing = repository.session.query(Group).filter(
        Group.group_type == "clustering_run",
        sa_text("metadata_json->>'source_file' = :sf"),
    ).params(sf=source_file).first()
    if existing:
        logger.info("  [SKIP] Already ingested: %s (run group %d)", source_file, existing.id)
        return None

    run_metadata = {
        "algorithm": "cosine_kllmeans_no_pca",
        "dataset": info["dataset"],
        "embedding_engine": info["engine"],
        "summarizer": info["summarizer"],
        "n_restarts": info["n_restarts"],
        "k_range": info["k_range"],
        "data_type": info["data_type"],
        "source_file": source_file,
        "source_file_sha256": sha,
        "source_file_byte_size": byte_size,
        "metric_specs": metric_specs,
        "ingestion_env": ingestion_env,
    }

    run_id = provenance.create_run_group(
        algorithm="cosine_kllmeans_no_pca",
        config=run_metadata,
        name=f"sweep_{info['dataset']}_{info['engine']}_{info['data_type']}",
        description=(
            f"Ingested from {source_file}: "
            f"{info['dataset']}/{info['engine']}/{info['summarizer']}"
        ),
    )

    # Overwrite metadata_json to include everything (create_run_group nests config)
    run_group = repository.get_group_by_id(run_id)
    run_group.metadata_json = run_metadata
    repository.session.flush()

    # Create step groups per k
    for k in sorted(info["by_k"].keys()):
        metrics_for_k = info["by_k"][k]
        step_metadata = {
            "k": k,
            "n_samples": info["n_samples"],
        }
        for m in METRICS:
            vals = metrics_for_k.get(m, [])
            step_metadata[f"{m}s"] = vals

        step_id = provenance.create_step_group(
            parent_run_id=run_id,
            step_name=f"k={k}",
            step_type="clustering",
            metadata=step_metadata,
        )
        repository.create_group_link(
            parent_group_id=run_id,
            child_group_id=step_id,
            link_type="clustering_step",
            position=k,
        )

    logger.info(
        "  Ingested %s -> run group %d (%d steps)",
        source_file, run_id, len(info["by_k"]),
    )
    return run_id


def main():
    parser = argparse.ArgumentParser(description="Ingest pkl sweep results into the database")
    parser.add_argument(
        "--data-dir", type=Path,
        default=REPO_ROOT / "experimental_results",
        help="Directory containing pkl files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be ingested without writing to DB",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print(f"Data dir: {data_dir}")
    print(f"Dry run: {args.dry_run}")

    metric_specs = build_metric_specs()
    ingestion_env = build_ingestion_env()
    print(f"sklearn version: {ingestion_env['sklearn_version']}")
    print(f"numpy version: {ingestion_env['numpy_version']}")

    # Scan pkl files
    pkl_infos: list[dict] = []

    for p in sorted(data_dir.glob("no_pca_50runs_*.pkl")):
        info = load_and_group_pkl(p, "50runs")
        if info:
            pkl_infos.append(info)

    for p in sorted(data_dir.glob("experimental_sweep_*.pkl")):
        info = load_and_group_pkl(p, "sweep")
        if info:
            pkl_infos.append(info)

    if not pkl_infos:
        print("[ERROR] No valid pkl files found.")
        sys.exit(1)

    print(f"Found {len(pkl_infos)} pkl file(s) to ingest.")

    if args.dry_run:
        for info in pkl_infos:
            ingest_one_pkl(info, None, None, metric_specs, ingestion_env, dry_run=True)
        print("Dry run complete.")
        return

    db = DatabaseConnectionV2(config.database.connection_string)
    db.init_db()

    created = 0
    skipped = 0
    with db.session_scope() as session:
        repository = RawCallRepository(session)
        provenance = ProvenanceService(repository)

        for info in pkl_infos:
            run_id = ingest_one_pkl(
                info, repository, provenance,
                metric_specs, ingestion_env, dry_run=False,
            )
            if run_id is not None:
                created += 1
            else:
                skipped += 1

    print(f"\nDone. Created {created} run group(s), skipped {skipped}.")


if __name__ == "__main__":
    main()

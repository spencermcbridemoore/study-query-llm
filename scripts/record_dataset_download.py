#!/usr/bin/env python3
"""
Layer 0: download public dataset files, record SHA-256 + acquisition.json.

Examples:
  python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem
  python scripts/record_dataset_download.py --dataset ausem --dry-run
  python scripts/record_dataset_download.py --dataset ausem --output-dir ./data/acquisitions/ausem \\
    --persist-db --dataset-group-name acquire_ausem

--persist-db requires DATABASE_URL and ARTIFACT_STORAGE_BACKEND=azure_blob.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from study_query_llm.datasets.acquisition import (
    acquisition_manifest_sha256,
    build_acquisition_manifest,
    download_acquisition_files,
    fetch_url,
    write_acquisition_bundle,
)
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.provenance_service import ProvenanceService


def main() -> None:
    parser = argparse.ArgumentParser(description="Record dataset download provenance (layer 0)")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(ACQUIRE_REGISTRY.keys()),
        help="Dataset slug with pinned source URLs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write acquisition.json and files/ subtree here (required unless --dry-run or --persist-db only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch files and print manifest JSON; do not write disk or database",
    )
    parser.add_argument(
        "--persist-db",
        action="store_true",
        help="Create dataset group and store blobs via ArtifactService (Azure required)",
    )
    parser.add_argument(
        "--dataset-group-name",
        default=None,
        help="Group name for ProvenanceService.create_dataset_group (default: acquire_<slug>)",
    )
    args = parser.parse_args()

    cfg = ACQUIRE_REGISTRY[args.dataset]
    specs = cfg.file_specs()
    files = download_acquisition_files(specs, fetch=fetch_url)

    manifest = build_acquisition_manifest(
        dataset_slug=cfg.slug,
        source=cfg.source_metadata(),
        files=files,
        runner_script="scripts/record_dataset_download.py",
    )
    mhash = acquisition_manifest_sha256(manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"\nmanifest_sha256={mhash}", file=sys.stderr)
        return

    if args.output_dir is None and not args.persist_db:
        parser.error("Provide --output-dir, or use --dry-run, or add --persist-db")

    if args.output_dir is not None:
        out = write_acquisition_bundle(args.output_dir, manifest, files)
        print(f"[OK] Wrote bundle: {out.parent}")
        print(f"manifest_sha256={mhash}")

    if args.persist_db:
        backend = (os.environ.get("ARTIFACT_STORAGE_BACKEND") or "local").strip().lower()
        if backend != "azure_blob":
            raise SystemExit(
                "ERROR: --persist-db requires ARTIFACT_STORAGE_BACKEND=azure_blob "
                f"(got {backend!r}). See .env.example and scripts/check_azure_blob_storage.py"
            )
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise SystemExit("ERROR: DATABASE_URL is not set")

        group_name = args.dataset_group_name or f"acquire_{cfg.slug}"
        manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )

        db = DatabaseConnectionV2(db_url, enable_pgvector=False)
        db.init_db()
        with db.session_scope() as session:
            repo = RawCallRepository(session)
            provenance = ProvenanceService(repo)
            artifacts = ArtifactService(repository=repo)

            group_id = provenance.create_dataset_group(
                name=group_name,
                description=f"Layer-0 acquisition: {cfg.slug}",
                metadata={
                    "acquisition_layer": "0",
                    "dataset_slug": cfg.slug,
                    "acquired_at": manifest["acquired_at"],
                    "manifest_sha256": mhash,
                    "file_count": len(files),
                },
            )

            for f in files:
                logical = f.relative_path.replace("/", "_").replace("\\", "_")
                artifacts.store_group_blob_artifact(
                    group_id=group_id,
                    step_name="acquisition",
                    logical_filename=logical,
                    data=f.data,
                    artifact_type="dataset_acquisition_file",
                    content_type="text/csv",
                    metadata={
                        "source_url": f.url,
                        "sha256": f.sha256,
                        "relative_path": f.relative_path,
                        "byte_size": f.byte_size,
                    },
                )

            artifacts.store_group_blob_artifact(
                group_id=group_id,
                step_name="acquisition",
                logical_filename="acquisition.json",
                data=manifest_bytes,
                artifact_type="dataset_acquisition_manifest",
                content_type="application/json",
                metadata={
                    "manifest_sha256": mhash,
                    "dataset_slug": cfg.slug,
                },
            )

        print(f"[OK] Persisted acquisition to dataset group id={group_id} name={group_name!r}")


if __name__ == "__main__":
    main()

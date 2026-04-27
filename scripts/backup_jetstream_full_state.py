#!/usr/bin/env python3
"""
Automate a full Jetstream backup: DB dump/upload/verify + artifact blob copy.

This script orchestrates existing DB backup scripts and then mirrors all blobs from
the active artifact container into a dated backup prefix.

Default flow:
  1. python scripts/dump_postgres_for_jetstream_migration.py --from-jetstream
  2. python scripts/upload_jetstream_pg_dump_to_blob.py --dump-path <newest-dump>
  3. python scripts/verify_db_backup_inventory.py --strict-azure
  4. Copy every blob from source artifact container to destination backup prefix
  5. Write a local JSON receipt under backup_pg_dumps/

Usage:
  python scripts/backup_jetstream_full_state.py
  python scripts/backup_jetstream_full_state.py --dry-run
  python scripts/backup_jetstream_full_state.py --dump-path pg_migration_dumps/jetstream_for_local_20260426_000000Z.dump
  python scripts/backup_jetstream_full_state.py --artifact-destination-container artifacts-dev-backups
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

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
DUMP_DIR = REPO / "pg_migration_dumps"
RECEIPT_DIR = REPO / "backup_pg_dumps"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _resolve_artifact_source_container_from_env() -> str:
    runtime_env = (os.environ.get("ARTIFACT_RUNTIME_ENV") or "dev").strip().lower()
    base = (os.environ.get("AZURE_STORAGE_CONTAINER") or "artifacts").strip()
    explicit = (os.environ.get(f"AZURE_STORAGE_CONTAINER_{runtime_env.upper()}") or "").strip()
    if explicit:
        return explicit
    if runtime_env in {"dev", "stage", "prod"}:
        return f"{base}-{runtime_env}"
    return base


def _resolve_destination_connection_string(
    *,
    source_connection_string: str,
    destination_env_var: str,
) -> str:
    override = (os.environ.get(destination_env_var) or "").strip()
    return override or source_connection_string


def _build_destination_blob_name(
    *,
    backup_prefix: str,
    source_container: str,
    source_blob_name: str,
) -> str:
    prefix = backup_prefix.strip().strip("/")
    normalized_name = source_blob_name.strip().lstrip("/")
    if prefix:
        return f"{prefix}/{source_container}/{normalized_name}"
    return f"{source_container}/{normalized_name}"


def _latest_jetstream_dump() -> Path | None:
    if not DUMP_DIR.is_dir():
        return None
    candidates = sorted(
        DUMP_DIR.glob("jetstream_for_local_*.dump"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_python_script(script_name: str, args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        check=False,
    )
    return int(proc.returncode)


def _resolve_dump_path(dump_path_arg: str | None) -> Path:
    if dump_path_arg:
        candidate = Path(dump_path_arg)
        if not candidate.is_absolute():
            candidate = REPO / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Dump file not found: {candidate}")
        return candidate
    latest = _latest_jetstream_dump()
    if latest is None:
        raise FileNotFoundError(
            "No jetstream_for_local_*.dump found under pg_migration_dumps/."
        )
    return latest


def _run_db_backup_flow(*, dump_path_arg: str | None, dry_run: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok",
        "dump_path": None,
        "steps": [],
    }

    if dry_run:
        print("[DRY RUN] DB backup flow", flush=True)
        if dump_path_arg:
            path = _resolve_dump_path(dump_path_arg)
            result["dump_path"] = str(path)
            print(f"[DRY RUN] would upload existing dump: {path}", flush=True)
        else:
            rc = _run_python_script(
                "dump_postgres_for_jetstream_migration.py",
                ["--from-jetstream", "--dry-run"],
            )
            result["steps"].append(
                {
                    "name": "dump_postgres_for_jetstream_migration.py",
                    "return_code": rc,
                }
            )
            if rc != 0:
                raise RuntimeError("Dry-run dump command failed.")
            print(
                "[DRY RUN] would run upload_jetstream_pg_dump_to_blob.py "
                "and verify_db_backup_inventory.py --strict-azure",
                flush=True,
            )
        return result

    dump_path: Path
    if dump_path_arg:
        dump_path = _resolve_dump_path(dump_path_arg)
    else:
        rc_dump = _run_python_script(
            "dump_postgres_for_jetstream_migration.py",
            ["--from-jetstream"],
        )
        result["steps"].append(
            {
                "name": "dump_postgres_for_jetstream_migration.py",
                "return_code": rc_dump,
            }
        )
        if rc_dump != 0:
            raise RuntimeError("Jetstream dump step failed.")
        dump_path = _resolve_dump_path(None)

    result["dump_path"] = str(dump_path)

    rc_upload = _run_python_script(
        "upload_jetstream_pg_dump_to_blob.py",
        ["--dump-path", str(dump_path)],
    )
    result["steps"].append(
        {
            "name": "upload_jetstream_pg_dump_to_blob.py",
            "return_code": rc_upload,
        }
    )
    if rc_upload != 0:
        raise RuntimeError("Dump upload step failed.")

    rc_verify = _run_python_script(
        "verify_db_backup_inventory.py",
        ["--strict-azure"],
    )
    result["steps"].append(
        {
            "name": "verify_db_backup_inventory.py",
            "return_code": rc_verify,
        }
    )
    if rc_verify != 0:
        raise RuntimeError("DB backup verification step failed.")

    return result


def _collect_blob_stats(container_client, *, prefix: str | None = None) -> tuple[int, int]:
    total_count = 0
    total_bytes = 0
    list_kwargs: dict[str, str] = {}
    if prefix:
        list_kwargs["name_starts_with"] = prefix
    for blob in container_client.list_blobs(**list_kwargs):
        total_count += 1
        total_bytes += int(getattr(blob, "size", 0) or 0)
    return total_count, total_bytes


def _copy_blob(
    *,
    source_container_client,
    destination_container_client,
    source_blob_name: str,
    destination_blob_name: str,
    overwrite: bool,
    transfer_concurrency: int,
) -> tuple[str, int]:
    source_blob_client = source_container_client.get_blob_client(source_blob_name)
    destination_blob_client = destination_container_client.get_blob_client(
        destination_blob_name
    )

    if not overwrite and destination_blob_client.exists():
        existing = destination_blob_client.get_blob_properties()
        return "skipped_existing", int(getattr(existing, "size", 0) or 0)

    source_properties = source_blob_client.get_blob_properties()
    downloader = source_blob_client.download_blob(max_concurrency=transfer_concurrency)
    destination_blob_client.upload_blob(
        downloader.chunks(),
        overwrite=overwrite,
        content_settings=source_properties.content_settings,
        metadata=source_properties.metadata or None,
        max_concurrency=transfer_concurrency,
    )
    return "copied", int(getattr(source_properties, "size", 0) or 0)


def _run_artifact_backup_flow(
    *,
    source_container: str,
    destination_container: str,
    backup_prefix: str,
    destination_connection_string_env: str,
    max_workers: int,
    transfer_concurrency: int,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_connection_string = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip()
    if not source_connection_string:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is required for artifact backup."
        )

    destination_connection_string = _resolve_destination_connection_string(
        source_connection_string=source_connection_string,
        destination_env_var=destination_connection_string_env,
    )

    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob import BlobServiceClient

    source_service = BlobServiceClient.from_connection_string(source_connection_string)
    destination_service = BlobServiceClient.from_connection_string(
        destination_connection_string
    )
    source_client = source_service.get_container_client(source_container)
    destination_client = destination_service.get_container_client(destination_container)

    if not dry_run:
        try:
            destination_client.create_container()
        except ResourceExistsError:
            pass

    source_blobs = [
        (blob.name, int(getattr(blob, "size", 0) or 0))
        for blob in source_client.list_blobs()
    ]
    source_count = len(source_blobs)
    source_bytes = sum(size for _, size in source_blobs)
    destination_prefix = f"{backup_prefix.strip('/')}/{source_container}".strip("/")

    result: dict[str, Any] = {
        "status": "ok",
        "dry_run": bool(dry_run),
        "source_container": source_container,
        "destination_container": destination_container,
        "destination_prefix": destination_prefix,
        "source_blob_count": source_count,
        "source_total_bytes": source_bytes,
        "copied_blob_count": 0,
        "copied_total_bytes": 0,
        "skipped_existing_blob_count": 0,
        "errors": [],
        "destination_blob_count": None,
        "destination_total_bytes": None,
        "parity_ok": None,
    }

    if dry_run:
        return result

    def _task(payload: tuple[str, int]) -> tuple[str, str, int]:
        source_blob_name, _size = payload
        destination_blob_name = _build_destination_blob_name(
            backup_prefix=backup_prefix,
            source_container=source_container,
            source_blob_name=source_blob_name,
        )
        action, transferred = _copy_blob(
            source_container_client=source_client,
            destination_container_client=destination_client,
            source_blob_name=source_blob_name,
            destination_blob_name=destination_blob_name,
            overwrite=overwrite,
            transfer_concurrency=transfer_concurrency,
        )
        return action, source_blob_name, transferred

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_task, item) for item in source_blobs]
        for fut in as_completed(futures):
            try:
                action, blob_name, transferred = fut.result()
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(str(exc))
                continue

            if action == "copied":
                result["copied_blob_count"] = int(result["copied_blob_count"]) + 1
                result["copied_total_bytes"] = int(result["copied_total_bytes"]) + int(
                    transferred
                )
            elif action == "skipped_existing":
                result["skipped_existing_blob_count"] = int(
                    result["skipped_existing_blob_count"]
                ) + 1
            else:
                result["errors"].append(
                    f"Unexpected copy action={action!r} for blob={blob_name!r}"
                )

    if result["errors"]:
        result["status"] = "failed"
        return result

    destination_count, destination_bytes = _collect_blob_stats(
        destination_client,
        prefix=f"{destination_prefix}/",
    )
    result["destination_blob_count"] = destination_count
    result["destination_total_bytes"] = destination_bytes
    parity_ok = destination_count == source_count and destination_bytes == source_bytes
    result["parity_ok"] = parity_ok
    if not parity_ok:
        result["status"] = "failed"
        result["errors"].append(
            "Destination prefix stats do not match source container stats."
        )
    return result


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automate full Jetstream backup (DB dump/upload/verify + artifact container copy)."
        )
    )
    parser.add_argument(
        "--dump-path",
        default=None,
        help=(
            "Existing .dump file path. If omitted, create a new Jetstream dump first "
            "(unless --skip-db-backup)."
        ),
    )
    parser.add_argument(
        "--skip-db-backup",
        action="store_true",
        help="Skip DB dump/upload/verify orchestration.",
    )
    parser.add_argument(
        "--skip-artifact-backup",
        action="store_true",
        help="Skip artifact container backup step.",
    )
    parser.add_argument(
        "--artifact-source-container",
        default=None,
        help=(
            "Source artifact container. Default resolves from env using ARTIFACT_RUNTIME_ENV "
            "and AZURE_STORAGE_CONTAINER."
        ),
    )
    parser.add_argument(
        "--artifact-destination-container",
        default=None,
        help="Destination backup container (default: <source-container>-backups).",
    )
    parser.add_argument(
        "--artifact-backup-prefix",
        default=None,
        help=(
            "Backup prefix in destination container (default: jetstream-full-state/<UTC-stamp>)."
        ),
    )
    parser.add_argument(
        "--destination-connection-string-env",
        default="AZURE_BACKUP_STORAGE_CONNECTION_STRING",
        help=(
            "Env var for destination storage connection string. "
            "Falls back to AZURE_STORAGE_CONNECTION_STRING when unset."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel blob copy workers (default: ARTIFACT_MIRROR_MAX_CONCURRENCY or 4).",
    )
    parser.add_argument(
        "--transfer-concurrency",
        type=int,
        default=1,
        help="Per-blob SDK transfer concurrency for download/upload streams (default: 1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination blobs if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without writing dumps/blobs/receipt.",
    )
    parser.add_argument(
        "--receipt-path",
        default=None,
        help=(
            "Local receipt path (default: backup_pg_dumps/jetstream_full_state_backup_<ts>.receipt.json)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(REPO / ".env", encoding="utf-8")
    args = _parse_args(argv)
    stamp = _utc_stamp()
    started_at = _utc_now_iso()
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = int((os.environ.get("ARTIFACT_MIRROR_MAX_CONCURRENCY") or "4").strip())
    max_workers = max(1, int(max_workers))
    transfer_concurrency = max(1, int(args.transfer_concurrency))

    source_container = (
        (args.artifact_source_container or "").strip()
        or _resolve_artifact_source_container_from_env()
    )
    destination_container = (
        (args.artifact_destination_container or "").strip()
        or f"{source_container}-backups"
    )
    backup_prefix = (
        (args.artifact_backup_prefix or "").strip().strip("/")
        or f"jetstream-full-state/{stamp}"
    )

    if args.receipt_path:
        receipt_path = Path(args.receipt_path)
        if not receipt_path.is_absolute():
            receipt_path = REPO / receipt_path
        receipt_path = receipt_path.resolve()
    else:
        receipt_path = (
            RECEIPT_DIR / f"jetstream_full_state_backup_{stamp}.receipt.json"
        ).resolve()

    receipt: dict[str, Any] = {
        "status": "ok",
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "dry_run": bool(args.dry_run),
        "config": {
            "skip_db_backup": bool(args.skip_db_backup),
            "skip_artifact_backup": bool(args.skip_artifact_backup),
            "source_container": source_container,
            "destination_container": destination_container,
            "backup_prefix": backup_prefix,
            "destination_connection_string_env": args.destination_connection_string_env,
            "max_workers": max_workers,
            "transfer_concurrency": transfer_concurrency,
            "overwrite": bool(args.overwrite),
            "dump_path_arg": args.dump_path,
        },
        "db_backup": None,
        "artifact_backup": None,
        "receipt_path": str(receipt_path),
        "error": None,
    }

    print(f"Backup stamp: {stamp}", flush=True)
    print(f"Dry run: {args.dry_run}", flush=True)

    try:
        if args.skip_db_backup:
            receipt["db_backup"] = {"status": "skipped"}
            print("DB backup step: skipped", flush=True)
        else:
            print("Running DB backup flow...", flush=True)
            receipt["db_backup"] = _run_db_backup_flow(
                dump_path_arg=args.dump_path,
                dry_run=bool(args.dry_run),
            )
            print("DB backup flow: complete", flush=True)

        if args.skip_artifact_backup:
            receipt["artifact_backup"] = {"status": "skipped"}
            print("Artifact backup step: skipped", flush=True)
        else:
            print("Running artifact backup flow...", flush=True)
            artifact_result = _run_artifact_backup_flow(
                source_container=source_container,
                destination_container=destination_container,
                backup_prefix=backup_prefix,
                destination_connection_string_env=args.destination_connection_string_env,
                max_workers=max_workers,
                transfer_concurrency=transfer_concurrency,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
            receipt["artifact_backup"] = artifact_result
            if artifact_result.get("status") != "ok":
                raise RuntimeError(
                    "Artifact backup failed: "
                    + "; ".join(artifact_result.get("errors") or ["unknown error"])
                )
            print("Artifact backup flow: complete", flush=True)

    except Exception as exc:  # noqa: BLE001
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {receipt['error']}", file=sys.stderr, flush=True)
        if not args.dry_run:
            receipt["finished_at_utc"] = _utc_now_iso()
            _write_receipt(receipt_path, receipt)
            print(f"Wrote failure receipt: {receipt_path}", file=sys.stderr, flush=True)
        return 1

    receipt["finished_at_utc"] = _utc_now_iso()
    if not args.dry_run:
        _write_receipt(receipt_path, receipt)
        print(f"Wrote receipt: {receipt_path}", flush=True)
    else:
        print("[DRY RUN] receipt not written.", flush=True)

    print("Backup completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Archive MCQ-linked artifact blobs to a frozen prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _normalize_archive_prefix(prefix: str) -> str:
    token = "<YYYYMMDD>"
    normalized = (prefix or "").strip().replace("\\", "/")
    if token in normalized:
        normalized = normalized.replace(token, datetime.now(timezone.utc).strftime("%Y%m%d"))
    return normalized.strip("/")


def _is_azure_blob_uri(uri: str) -> bool:
    parsed = urlparse((uri or "").strip())
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return ".blob.core.windows.net" in host


def _parse_azure_blob_uri(uri: str) -> tuple[str, str, str]:
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    if ".blob.core.windows.net" not in host.lower():
        raise ValueError(f"Not an Azure blob URI: {uri}")
    path = unquote(parsed.path or "").strip("/")
    container, _, blob_path = path.partition("/")
    if not container or not blob_path:
        raise ValueError(f"Azure blob URI missing container/path: {uri}")
    account_url = f"{parsed.scheme}://{host}"
    return account_url, container, blob_path


def _uri_to_local_path(uri: str) -> Path | None:
    parsed = urlparse((uri or "").strip())
    if parsed.scheme == "":
        return Path(uri)
    # Windows absolute paths can be parsed as scheme="c", path="\\foo\\bar".
    if len(parsed.scheme) == 1 and parsed.path.startswith(("\\", "/")):
        return Path(f"{parsed.scheme}:{parsed.path}")
    if parsed.scheme != "file":
        return None
    raw_path = unquote(parsed.path or "")
    if os.name == "nt" and raw_path.startswith("/") and len(raw_path) > 2 and raw_path[2] == ":":
        raw_path = raw_path.lstrip("/")
    return Path(raw_path)


def _local_destination_path(
    *,
    source_path: Path,
    artifact_root: Path,
    archive_prefix: str,
) -> Path:
    source_resolved = source_path.resolve()
    archive_root = (artifact_root / Path(archive_prefix)).resolve()
    try:
        if source_resolved.is_relative_to(archive_root):
            return source_resolved
    except ValueError:
        pass

    root_resolved = artifact_root.resolve()
    try:
        relative = source_resolved.relative_to(root_resolved)
        suffix = Path(relative.as_posix())
    except ValueError:
        digest = hashlib.sha256(str(source_resolved).encode("utf-8")).hexdigest()[:12]
        suffix = Path("external") / f"{digest}_{source_resolved.name}"
    return artifact_root / Path(archive_prefix) / suffix


def _archive_local_uri(
    *,
    source_uri: str,
    artifact_root: Path,
    archive_prefix: str,
    dry_run: bool,
) -> tuple[str, int]:
    source_path = _uri_to_local_path(source_uri)
    if source_path is None:
        raise ValueError(f"Unsupported local URI: {source_uri}")
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source artifact does not exist: {source_path}")
    destination = _local_destination_path(
        source_path=source_path,
        artifact_root=artifact_root,
        archive_prefix=archive_prefix,
    )
    size = int(source_path.stat().st_size)
    if not dry_run and source_path != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_path.read_bytes())
    return str(destination.resolve()), size


def _archive_azure_uri(
    *,
    source_uri: str,
    archive_prefix: str,
    destination_container: str | None,
    connection_string: str,
    dry_run: bool,
) -> tuple[str, int]:
    from azure.storage.blob import BlobServiceClient

    _, source_container, source_blob_path = _parse_azure_blob_uri(source_uri)
    destination_blob_path = (
        f"{archive_prefix}/{source_container}/{source_blob_path}".strip("/")
    )
    destination_container_name = (destination_container or source_container).strip()

    service = BlobServiceClient.from_connection_string(connection_string)
    source_blob = service.get_blob_client(container=source_container, blob=source_blob_path)
    destination_blob = service.get_blob_client(
        container=destination_container_name,
        blob=destination_blob_path,
    )

    source_props = source_blob.get_blob_properties()
    size = int(source_props.size or 0)
    if not dry_run:
        payload = source_blob.download_blob().readall()
        destination_blob.upload_blob(payload, overwrite=True)
        size = len(payload)
    return destination_blob.url, size


def _load_backup_call_artifact_uris(backup_json_path: Path) -> list[str]:
    doc = json.loads(backup_json_path.read_text(encoding="utf-8"))
    artifacts = doc.get("call_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Backup JSON must contain call_artifacts array.")
    uris = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("uri") or "").strip()
        if uri:
            uris.append(uri)
    return sorted(set(uris))


def archive_from_backup(
    *,
    backup_json_path: Path,
    archive_prefix: str,
    artifact_root: Path,
    destination_container: str | None,
    connection_string: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    source_uris = _load_backup_call_artifact_uris(backup_json_path)
    uri_remap: dict[str, str] = {}
    copied = 0
    total_bytes = 0
    errors: list[str] = []

    for source_uri in source_uris:
        try:
            if _is_azure_blob_uri(source_uri):
                if not connection_string:
                    raise ValueError(
                        "AZURE_STORAGE_CONNECTION_STRING is required for Azure blob URIs."
                    )
                destination_uri, size = _archive_azure_uri(
                    source_uri=source_uri,
                    archive_prefix=archive_prefix,
                    destination_container=destination_container,
                    connection_string=connection_string,
                    dry_run=dry_run,
                )
            else:
                destination_uri, size = _archive_local_uri(
                    source_uri=source_uri,
                    artifact_root=artifact_root,
                    archive_prefix=archive_prefix,
                    dry_run=dry_run,
                )
            uri_remap[source_uri] = destination_uri
            copied += 1
            total_bytes += int(size)
        except Exception as exc:
            errors.append(f"{source_uri}: {exc}")

    return {
        "uri_remap": uri_remap,
        "copied_count": copied,
        "source_uri_count": len(source_uris),
        "total_bytes": total_bytes,
        "errors": errors,
    }


def _write_uri_remap(
    *,
    uri_remap: dict[str, str],
    archive_prefix: str,
    artifact_root: Path,
    backup_json_path: Path,
    dry_run: bool,
) -> dict[str, str]:
    remap_json = json.dumps(uri_remap, indent=2, ensure_ascii=False) + "\n"
    local_receipt = backup_json_path.with_suffix(".uri_remap.json")
    archive_receipt = artifact_root / Path(archive_prefix) / "uri_remap.json"
    if not dry_run:
        local_receipt.write_text(remap_json, encoding="utf-8")
        archive_receipt.parent.mkdir(parents=True, exist_ok=True)
        archive_receipt.write_text(remap_json, encoding="utf-8")
    return {
        "local_receipt": str(local_receipt),
        "archive_receipt": str(archive_receipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy MCQ-linked call artifact blobs into frozen archive prefix.",
    )
    parser.add_argument(
        "--backup-json",
        type=Path,
        required=True,
        help="Path produced by scripts/backup_mcq_db_to_json.py (contains call_artifacts URIs).",
    )
    parser.add_argument(
        "--archive-prefix",
        default="mcq-archive/<YYYYMMDD>/",
        help="Archive prefix (default: mcq-archive/<YYYYMMDD>/).",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact root directory for local URIs (default: ARTIFACT_DIR or artifacts).",
    )
    parser.add_argument(
        "--destination-container",
        default=None,
        help="Optional Azure container override (default: same source container).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print remap details without copying bytes.",
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env", encoding="utf-8")
    backup_json_path = (
        args.backup_json if args.backup_json.is_absolute() else REPO / args.backup_json
    )
    if not backup_json_path.is_file():
        print(f"ERROR: backup JSON not found: {backup_json_path}", file=sys.stderr)
        return 1

    archive_prefix = _normalize_archive_prefix(args.archive_prefix)
    if not archive_prefix:
        print("ERROR: archive prefix resolves to empty.", file=sys.stderr)
        return 1
    artifact_root = args.artifact_root or Path(
        (os.environ.get("ARTIFACT_DIR") or "artifacts").strip()
    )
    if not artifact_root.is_absolute():
        artifact_root = REPO / artifact_root
    connection_string = (os.environ.get("AZURE_STORAGE_CONNECTION_STRING") or "").strip() or None

    result = archive_from_backup(
        backup_json_path=backup_json_path,
        archive_prefix=archive_prefix,
        artifact_root=artifact_root,
        destination_container=args.destination_container,
        connection_string=connection_string,
        dry_run=bool(args.dry_run),
    )
    if result["errors"]:
        print("ERROR: archive encountered failures:", file=sys.stderr)
        for line in result["errors"]:
            print(f"  - {line}", file=sys.stderr)
        return 1

    receipts = _write_uri_remap(
        uri_remap=result["uri_remap"],
        archive_prefix=archive_prefix,
        artifact_root=artifact_root,
        backup_json_path=backup_json_path,
        dry_run=bool(args.dry_run),
    )

    print(f"Archive prefix: {archive_prefix}")
    print(f"Source URIs: {result['source_uri_count']}")
    print(f"Copied: {result['copied_count']}")
    print(f"Total bytes: {result['total_bytes']}")
    print(f"URI remap receipt (local): {receipts['local_receipt']}")
    print(f"URI remap receipt (archive): {receipts['archive_receipt']}")
    if args.dry_run:
        print("DRY RUN: no bytes copied, no files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

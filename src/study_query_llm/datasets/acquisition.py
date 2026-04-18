"""
Layer 0: record provenance for downloaded dataset files (URLs, checksums, timestamps).

Uses stdlib HTTP only; no extra dependencies.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ACQUISITION_SCHEMA_VERSION = "1.0"
DEFAULT_USER_AGENT = "study-query-llm-dataset-acquisition/1.0 (+https://github.com/spencermcbridemoore/study-query-llm)"
DEFAULT_TIMEOUT_SEC = 120.0


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zenodo_file_download_url(record_id: int, filename: str) -> str:
    """
    Public Zenodo file URL suitable for GET (browser-style download parameter).

    Example: https://zenodo.org/records/16912394/files/sources_v2.xlsx?download=1
    """
    name = str(filename).lstrip("/")
    if ".." in name or "/" in name:
        raise ValueError(f"Unsafe Zenodo filename: {filename!r}")
    rid = int(record_id)
    return f"https://zenodo.org/records/{rid}/files/{name}?download=1"


def fetch_url(url: str, *, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> bytes:
    """GET a URL and return response body bytes."""
    req = Request(
        url,
        headers={"User-Agent": DEFAULT_USER_AGENT},
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return resp.read()
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} fetching {url!r}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Failed to fetch {url!r}: {e.reason}") from e


def _try_git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


@dataclass(frozen=True)
class FileFetchSpec:
    """One file to download for an acquisition bundle."""

    relative_path: str
    url: str


@dataclass(frozen=True)
class FetchedFile:
    relative_path: str
    url: str
    data: bytes
    sha256: str
    byte_size: int


def download_acquisition_files(
    specs: List[FileFetchSpec],
    *,
    fetch: Callable[[str], bytes] = fetch_url,
) -> List[FetchedFile]:
    """Download each spec; compute SHA-256 and size per file."""
    out: List[FetchedFile] = []
    for spec in specs:
        data = fetch(spec.url)
        out.append(
            FetchedFile(
                relative_path=spec.relative_path,
                url=spec.url,
                data=data,
                sha256=sha256_hex(data),
                byte_size=len(data),
            )
        )
    return out


def build_acquisition_manifest(
    *,
    dataset_slug: str,
    source: Dict[str, Any],
    files: List[FetchedFile],
    runner_script: str,
    extra_runner: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the acquisition.json payload (sorted file order for stability)."""
    acquired_at = datetime.now(timezone.utc).isoformat()
    runner: Dict[str, Any] = {
        "script": runner_script,
        "git_commit": _try_git_commit(),
    }
    if extra_runner:
        runner.update(extra_runner)

    sorted_files = sorted(files, key=lambda f: f.relative_path)
    return {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "dataset_slug": dataset_slug,
        "acquired_at": acquired_at,
        "source": source,
        "runner": runner,
        "files": [
            {
                "relative_path": f.relative_path,
                "url": f.url,
                "sha256": f.sha256,
                "byte_size": f.byte_size,
            }
            for f in sorted_files
        ],
    }


def write_acquisition_bundle(
    output_dir: Path,
    manifest: Dict[str, Any],
    files: List[FetchedFile],
    *,
    files_subdir: str = "files",
) -> Path:
    """
    Write manifest to output_dir/acquisition.json and bytes under output_dir/files_subdir/...

    Returns path to acquisition.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_files = output_dir / files_subdir
    for f in files:
        dest = base_files / f.relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.data)

    manifest_path = output_dir / "acquisition.json"
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    return manifest_path


def acquisition_manifest_sha256(manifest: Dict[str, Any]) -> str:
    """Stable hash of canonical manifest JSON (for idempotency hints)."""
    payload = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_fingerprint(*, dataset_slug: str, manifest: Dict[str, Any]) -> str:
    """
    Compute semantic acquisition fingerprint independent from timestamps.

    Fingerprint contract:
    - dataset slug
    - source.pinning_identity when available
    - sorted (relative_path, sha256) pairs
    """
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    pinning_identity = source.get("pinning_identity")
    files_raw = manifest.get("files")
    file_pairs: list[tuple[str, str]] = []
    if isinstance(files_raw, list):
        for item in files_raw:
            if not isinstance(item, dict):
                continue
            rel_path = str(item.get("relative_path") or "").strip()
            digest = str(item.get("sha256") or "").strip()
            if rel_path and digest:
                file_pairs.append((rel_path, digest))
    file_pairs.sort()

    payload = {
        "dataset_slug": str(dataset_slug),
        "pinning_identity": pinning_identity,
        "files": file_pairs,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

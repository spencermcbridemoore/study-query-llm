#!/usr/bin/env python
"""Context-hygiene backup helper (Phase B.1 spec).

Modes (mutually exclusive):
  default (create):     write zip + sibling manifest CSV; abort-if-exists optional.
  --dry-run:            walk include set; emit count + projected uncompressed bytes; no writes.
  --verify-zip ZIP:     random-sample N manifest entries, re-hash from zip, fail on mismatch.

Manifest CSV columns: relpath, bytes, sha256.

Include / exclude sets are taken verbatim from
.cursor/plans/context_hygiene_cleanup.md (B.1).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import os
import random
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

INCLUDE_DIR_PREFIXES = (
    ".claude/worktrees/",
    "experimental_results/",
    "scratch/",
    "backup_pg_dumps/",
    "pg_migration_dumps/",
    "data/embedding_cache/",
    "artifacts/",
    "logs/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".cache/",
)
INCLUDE_TOPLEVEL_DIR_PREFIX = "backup_"
INCLUDE_ROOT_FILES = (
    "custom_sweep_output.log",
    "no_pca_multi_embedding_sweep_output.log",
    "sweep_curves.csv",
    "gcm-diagnose.log",
)

# Files in scratch/ that stay in-repo and therefore should NOT enter the backup
SCRATCH_KEEP = (
    "scratch/claude/README.md",
    "scratch/local/README.md",
    "scratch/local/jetstream-remote-build-and-restart.ps1",
)

EXCLUDE_EXACT = (
    ".env",
    ".env.example",
    "deploy/jetstream/.env.jetstream",
    "deploy/jetstream/terraform/terraform.tfvars",
)
EXCLUDE_PREFIXES = (
    ".git/",
    "deploy/jetstream/terraform/.terraform/",
)
EXCLUDE_GLOBS = (
    "deploy/jetstream/terraform/*.tfstate*",
    "**/*.tfstate",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa*",
)


def is_excluded(rel: str) -> bool:
    if rel in EXCLUDE_EXACT:
        return True
    for prefix in EXCLUDE_PREFIXES:
        if rel.startswith(prefix):
            return True
    for pattern in EXCLUDE_GLOBS:
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def is_included(rel: str) -> bool:
    if rel in SCRATCH_KEEP:
        return False
    if rel in INCLUDE_ROOT_FILES:
        return True
    for prefix in INCLUDE_DIR_PREFIXES:
        if rel.startswith(prefix):
            return True
    head = rel.split("/", 1)[0]
    if head.startswith(INCLUDE_TOPLEVEL_DIR_PREFIX) and "/" in rel:
        return True
    # Any nested __pycache__
    if "/__pycache__/" in "/" + rel:
        return True
    return False


def walk_repo():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT, topdown=True):
        if ".git" in dirnames:
            dirnames.remove(".git")
        dirnames.sort()
        for fname in sorted(filenames):
            full = Path(dirpath) / fname
            try:
                rel = full.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            yield full, rel


def collect_backup_set():
    for full, rel in walk_repo():
        if is_excluded(rel):
            continue
        if is_included(rel):
            yield full, rel


def sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def cmd_dry_run(args):
    count = 0
    total_bytes = 0
    for full, rel in collect_backup_set():
        try:
            sz = full.stat().st_size
        except OSError:
            continue
        count += 1
        total_bytes += sz
    print("DRY RUN")
    print(f"  files: {count}")
    print(f"  projected_uncompressed_bytes: {total_bytes}")
    print(f"  projected_uncompressed_gb: {total_bytes / (1024 ** 3):.2f}")
    print(f"  zip_path:      {args.zip_path}")
    print(f"  manifest_path: {args.manifest_path}")
    print(f"  scratch_keep_list_excluded: {list(SCRATCH_KEEP)}")


def cmd_create(args):
    zip_path = Path(args.zip_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    if args.abort_if_exists:
        if zip_path.exists():
            sys.stderr.write(f"ERROR: zip exists, aborting: {zip_path}\n")
            sys.exit(2)
        if manifest_path.exists():
            sys.stderr.write(f"ERROR: manifest exists, aborting: {manifest_path}\n")
            sys.exit(2)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating backup zip: {zip_path}")
    written = 0
    bytes_written = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf, \
         manifest_path.open("w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["relpath", "bytes", "sha256"])
        for full, rel in collect_backup_set():
            try:
                sz = full.stat().st_size
                digest = sha256_of_file(full)
            except OSError as e:
                sys.stderr.write(f"WARN skip unreadable {rel}: {e}\n")
                continue
            zf.write(full, arcname=rel)
            w.writerow([rel, sz, digest])
            written += 1
            bytes_written += sz
            if written % 500 == 0:
                print(f"  progress: {written} files, {bytes_written / (1024 ** 3):.2f} GB raw")
    zip_size = zip_path.stat().st_size
    print(f"DONE files={written} raw_gb={bytes_written / (1024 ** 3):.2f} zip_gb={zip_size / (1024 ** 3):.2f}")
    print(f"Manifest: {manifest_path}")


def cmd_verify(args):
    zip_path = Path(args.verify_zip).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    if not zip_path.exists():
        sys.stderr.write(f"ERROR zip not found: {zip_path}\n")
        sys.exit(2)
    if not manifest_path.exists():
        sys.stderr.write(f"ERROR manifest not found: {manifest_path}\n")
        sys.exit(2)
    with manifest_path.open("r", encoding="utf-8") as mf:
        rows = list(csv.DictReader(mf))
    total = len(rows)
    if total == 0:
        sys.stderr.write("ERROR manifest is empty\n")
        sys.exit(2)
    sample_size = min(args.sample_size, total)
    rng = random.Random(20260531)  # deterministic seed for reproducibility
    sample = rng.sample(rows, sample_size)

    mismatches = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for i, row in enumerate(sample, 1):
            rel = row["relpath"]
            expected_sha = row["sha256"]
            expected_bytes = int(row["bytes"])
            try:
                with zf.open(rel) as zfh:
                    data = zfh.read()
            except KeyError:
                mismatches.append((rel, "missing-in-zip"))
                continue
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != expected_sha or len(data) != expected_bytes:
                mismatches.append((rel, f"sha-mismatch expected={expected_sha[:12]} got={actual_sha[:12]}"))
            if i % 200 == 0:
                print(f"  verified {i}/{sample_size}")
    print("VERIFY summary")
    print(f"  manifest_rows: {total}")
    print(f"  sample_size:   {sample_size}")
    print(f"  mismatches:    {len(mismatches)}")
    if mismatches:
        sys.stderr.write("MISMATCH DETAIL (first 20):\n")
        for rel, reason in mismatches[:20]:
            sys.stderr.write(f"  {rel}: {reason}\n")
        sys.exit(3)
    print("RESULT: PASS")


def main():
    ap = argparse.ArgumentParser(description="Context-hygiene backup helper (B.1)")
    ap.add_argument("--zip-path", help="Destination zip path")
    ap.add_argument("--manifest-path", help="Companion manifest CSV path")
    ap.add_argument("--abort-if-exists", action="store_true",
                    help="Fail if zip or manifest already exists (create mode)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="List include set; emit count + projected bytes; no writes")
    mode.add_argument("--verify-zip", metavar="ZIP_PATH",
                      help="Verify a previously created zip against its manifest")
    ap.add_argument("--sample-size", type=int, default=1000,
                    help="Random-sample size for --verify-zip (default 1000)")
    args = ap.parse_args()

    if args.dry_run:
        if not args.zip_path or not args.manifest_path:
            ap.error("--dry-run requires --zip-path and --manifest-path (for output labels)")
        cmd_dry_run(args)
    elif args.verify_zip:
        if not args.manifest_path:
            ap.error("--verify-zip requires --manifest-path")
        cmd_verify(args)
    else:
        if not args.zip_path or not args.manifest_path:
            ap.error("create mode requires --zip-path and --manifest-path")
        cmd_create(args)


if __name__ == "__main__":
    main()
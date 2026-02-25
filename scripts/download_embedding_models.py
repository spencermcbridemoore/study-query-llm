"""
Download all recommended embedding models to the local HuggingFace cache.

Models are stored in ~/.cache/huggingface (or HF_CACHE_DIR if set) using
HF's content-addressed snapshot format.  Once downloaded, each model loads
instantly into TEI without any network activity — even with no internet.

Run this once before your first sweep:

    python scripts/download_embedding_models.py

Already-cached models are skipped (HF hub checks file hashes, not just presence).
Pass --dry-run to see what would be downloaded without actually downloading.

Approximate total download: ~52 GB (varies by what is already cached).
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Suppress the HF hub symlinks warning on Windows — we handle degraded mode
# (file copies instead of symlinks) gracefully.  The cache still works; it
# just uses a bit more disk space since duplicate blobs are not deduped.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------
# (tier, model_id, approx_size_gb, notes)
MODELS = [
    # ── Tier 1: State-of-the-art (MTEB top 10) ─────────────────────────────
    ("1", "Qwen/Qwen3-Embedding-0.6B",              0.6,  "MTEB #4 — small, punches above its weight"),
    ("1", "Qwen/Qwen3-Embedding-4B",                8.0,  "MTEB #3 — sweet spot quality/VRAM"),
    ("1", "Qwen/Qwen3-Embedding-8B",               15.0,  "MTEB #2 — best available, ~15 GB VRAM"),
    ("1", "Alibaba-NLP/gte-Qwen2-7B-instruct",     15.0,  "MTEB #6 — strong multilingual"),
    ("1", "intfloat/multilingual-e5-large-instruct", 1.1, "MTEB #7 — best multilingual at small size"),
    # ── Tier 2: Proven workhorses ────────────────────────────────────────────
    ("2", "Alibaba-NLP/gte-Qwen2-1.5B-instruct",   3.5,  "MTEB #15 — mid-size sweet spot"),
    ("2", "Snowflake/snowflake-arctic-embed-l-v2.0", 1.1, "MTEB #35 — retrieval-optimised"),
    ("2", "WhereIsAI/UAE-Large-V1",                 0.7,  "MTEB #52 — strong BERT-family"),
    ("2", "BAAI/bge-m3",                            2.3,  "Multi-function: dense + sparse + colbert"),
    ("2", "BAAI/bge-large-en-v1.5",                 0.7,  "Solid English baseline"),
    ("2", "Alibaba-NLP/gte-large-en-v1.5",          0.9,  "Excellent English-only"),
    # ── Tier 3: Lightweight / baselines ─────────────────────────────────────
    ("3", "nomic-ai/nomic-embed-text-v1.5",         0.3,  "Fast 137M baseline"),
    ("3", "nomic-ai/nomic-embed-text-v2-moe",       1.9,  "MoE architecture, novel for research"),
    ("3", "sentence-transformers/all-mpnet-base-v2", 0.4, "Classic 109M baseline"),
]

TOTAL_GB = sum(m[2] for m in MODELS)


def _hf_cache_dir() -> str:
    return (
        os.environ.get("HF_CACHE_DIR", "")
        or os.environ.get("HF_HOME", "")
        or str(Path.home() / ".cache" / "huggingface")
    )


def _is_cached(model_id: str, cache_dir: str, token: Optional[str]) -> bool:
    """Return True if the model snapshot already exists in the cache."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def download_model(
    model_id: str,
    cache_dir: str,
    token: Optional[str],
    dry_run: bool,
) -> tuple[bool, float]:
    """
    Download *model_id* to *cache_dir*.

    On Windows without Developer Mode, the HF cache falls back from symlinks
    to plain file copies automatically (degraded mode).  If symlink creation
    still fails with WinError 1314, we retry using ``local_dir`` mode which
    downloads files directly without any symlink step.

    Returns:
        (already_cached, elapsed_seconds)
    """
    from huggingface_hub import snapshot_download

    if _is_cached(model_id, cache_dir, token):
        return True, 0.0

    if dry_run:
        return False, 0.0

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            local_files_only=False,
        )
    except OSError as exc:
        # WinError 1314: symlink privilege not held — retry with local_dir
        # which copies files directly into a flat folder (no symlinks needed).
        if getattr(exc, "winerror", None) == 1314:
            safe_name = model_id.replace("/", "--")
            local_dir = Path(cache_dir) / "local_downloads" / safe_name
            local_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"      [warn] Symlink creation blocked (WinError 1314).\n"
                f"             Retrying with direct copy to: {local_dir}\n"
                f"             To fix permanently, enable Windows Developer Mode:\n"
                f"             Settings > Privacy & Security > For Developers > ON"
            )
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                token=token,
            )
        else:
            raise

    return False, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check cache status without downloading anything.",
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3"],
        help="Only download models from a specific tier (default: all tiers).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="Download a single model by its HuggingFace ID (overrides --tier).",
    )
    args = parser.parse_args()

    # Resolve env
    from dotenv import load_dotenv
    load_dotenv()
    cache_dir = _hf_cache_dir()
    token = os.environ.get("HF_API_TOKEN") or None

    # Filter model list
    if args.model:
        targets = [m for m in MODELS if m[1] == args.model]
        if not targets:
            # Allow ad-hoc models not in the registry
            targets = [("?", args.model, 0.0, "ad-hoc")]
    elif args.tier:
        targets = [m for m in MODELS if m[0] == args.tier]
    else:
        targets = MODELS

    total_gb = sum(m[2] for m in targets)

    print("=" * 70)
    print("  HuggingFace Embedding Model Downloader")
    print("=" * 70)
    print(f"  Cache dir : {cache_dir}")
    print(f"  HF token  : {'set (' + token[:8] + '...)' if token else 'NOT SET (public models only)'}")
    print(f"  Models    : {len(targets)}  (~{total_gb:.0f} GB total if not cached)")
    if args.dry_run:
        print("  Mode      : DRY RUN — no downloads")
    print("=" * 70)
    print()

    results = {"cached": [], "downloaded": [], "failed": []}

    for tier, model_id, size_gb, notes in targets:
        label = f"[Tier {tier}] {model_id}"
        size_str = f"{size_gb:.1f} GB" if size_gb > 0 else "?"
        print(f"  >>  {label}")
        print(f"      {notes}  (~{size_str})")

        try:
            already_cached, elapsed = download_model(model_id, cache_dir, token, args.dry_run)
        except Exception as exc:
            print(f"      [FAILED]: {exc}")
            results["failed"].append(model_id)
            print()
            continue

        if already_cached:
            print("      [cached] Already in cache -- skipping")
            results["cached"].append(model_id)
        elif args.dry_run:
            print("      [pending] Would download")
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"      [done] Downloaded in {mins}m {secs}s")
            results["downloaded"].append(model_id)
        print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Already cached : {len(results['cached'])}")
    if not args.dry_run:
        print(f"  Downloaded     : {len(results['downloaded'])}")
    print(f"  Failed         : {len(results['failed'])}")
    if results["failed"]:
        for m in results["failed"]:
            print(f"    - {m}")
    print()
    if not results["failed"]:
        print("  All models ready.  Start a sweep with LocalDockerTEIManager.")
    else:
        print("  Some models failed.  Check your HF_API_TOKEN and network.")
        sys.exit(1)


if __name__ == "__main__":
    main()

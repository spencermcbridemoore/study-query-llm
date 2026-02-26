"""
Download all recommended summarizer/paraphraser LLMs via Ollama.

Models are stored in Ollama's local blob store and served via its
OpenAI-compatible ``/v1/chat/completions`` endpoint on localhost:11434.

Run this once before your first local sweep:

    python scripts/download_summarizer_models.py

Already-pulled models are skipped.
Pass --dry-run to see what would be downloaded without actually pulling.

Prerequisites:
  - Ollama installed and running (``ollama serve`` or background service)
  - Enough disk space (~50 GB for all tiers)
  - RTX 4090 24 GB recommended for Tier 1 models
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Ollama executable discovery
# ---------------------------------------------------------------------------
# Ollama may not be on PATH (Windows installs to AppData).  Try the common
# locations before falling back to the bare name.
_OLLAMA_PATHS = [
    Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
    Path("/usr/local/bin/ollama"),
    Path("/usr/bin/ollama"),
]


def _find_ollama() -> str:
    """Return the path to the ``ollama`` executable."""
    for p in _OLLAMA_PATHS:
        if p.exists():
            return str(p)
    return "ollama"


OLLAMA_BIN = _find_ollama()

# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------
# (tier, ollama_tag, approx_vram_gb, notes)
MODELS = [
    # -- Tier 1: Best quality (fits alongside small embedders on 24 GB) ----
    ("1", "qwen2.5:32b",       17.0, "Best instruction following, near GPT-4o for short summarisation"),
    ("1", "mistral-small:24b", 13.0, "Mistral's best open small model, excellent conciseness"),
    ("1", "gemma2:27b",        15.0, "Google's best open 27B, strong instruction following"),
    # -- Tier 2: More VRAM headroom (fits alongside 7B embedders) ----------
    ("2", "qwen2.5:14b",       8.0,  "Excellent balance, leaves ~16 GB for embedder"),
    ("2", "llama3.1:8b",       4.5,  "Solid baseline, maximum VRAM headroom"),
    ("2", "gemma2:9b",         5.0,  "Compact, fast, good instruction following"),
]

TOTAL_VRAM = sum(m[2] for m in MODELS)

# ---------------------------------------------------------------------------
# Ollama API helpers
# ---------------------------------------------------------------------------
OLLAMA_API = "http://localhost:11434"


def _ollama_api_available() -> bool:
    """Return True if the Ollama API responds."""
    try:
        urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def _list_pulled_models() -> set[str]:
    """Return the set of model tags already available locally."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_API}/api/tags", timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {m["name"] for m in data.get("models", [])}
    except Exception:
        return set()


def _is_pulled(tag: str, pulled: set[str]) -> bool:
    """Check if *tag* is already pulled (handles implicit :latest suffix)."""
    if tag in pulled:
        return True
    if ":" not in tag and f"{tag}:latest" in pulled:
        return True
    return False


def pull_model(tag: str, dry_run: bool) -> tuple[bool, float]:
    """Pull an Ollama model.  Returns (already_pulled, elapsed_seconds)."""
    pulled = _list_pulled_models()
    if _is_pulled(tag, pulled):
        return True, 0.0
    if dry_run:
        return False, 0.0

    t0 = time.time()
    result = subprocess.run(
        [OLLAMA_BIN, "pull", tag],
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"ollama pull {tag} failed (exit code {result.returncode})")
    return False, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check status without pulling anything.",
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2"],
        help="Only pull models from a specific tier (default: all).",
    )
    parser.add_argument(
        "--model",
        metavar="TAG",
        help="Pull a single model by its Ollama tag (overrides --tier).",
    )
    args = parser.parse_args()

    # Verify Ollama is running
    if not _ollama_api_available():
        print("ERROR: Ollama API is not reachable at http://localhost:11434")
        print("       Start Ollama with: ollama serve")
        sys.exit(1)

    # Filter model list
    if args.model:
        targets = [m for m in MODELS if m[1] == args.model]
        if not targets:
            targets = [("?", args.model, 0.0, "ad-hoc")]
    elif args.tier:
        targets = [m for m in MODELS if m[0] == args.tier]
    else:
        targets = MODELS

    total_vram = sum(m[2] for m in targets)

    print("=" * 70)
    print("  Ollama Summarizer Model Downloader")
    print("=" * 70)
    print(f"  Ollama bin : {OLLAMA_BIN}")
    print(f"  API        : {OLLAMA_API}")
    print(f"  Models     : {len(targets)}  (~{total_vram:.0f} GB VRAM total at Q4)")
    if args.dry_run:
        print("  Mode       : DRY RUN -- no downloads")
    print("=" * 70)
    print()

    results = {"cached": [], "downloaded": [], "failed": []}

    for tier, tag, vram_gb, notes in targets:
        label = f"[Tier {tier}] {tag}"
        print(f"  >>  {label}")
        print(f"      {notes}  (~{vram_gb:.1f} GB VRAM)")

        try:
            already_pulled, elapsed = pull_model(tag, args.dry_run)
        except Exception as exc:
            print(f"      [FAILED]: {exc}")
            results["failed"].append(tag)
            print()
            continue

        if already_pulled:
            print("      [cached] Already pulled -- skipping")
            results["cached"].append(tag)
        elif args.dry_run:
            print("      [pending] Would pull")
        else:
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"      [done] Pulled in {mins}m {secs}s")
            results["downloaded"].append(tag)
        print()

    # Summary
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Already pulled : {len(results['cached'])}")
    if not args.dry_run:
        print(f"  Downloaded     : {len(results['downloaded'])}")
    print(f"  Failed         : {len(results['failed'])}")
    if results["failed"]:
        for m in results["failed"]:
            print(f"    - {m}")
    print()
    if not results["failed"]:
        print("  All models ready.  Add (model, 'local_llm') tuples to LLM_SUMMARIZERS.")
    else:
        print("  Some models failed.  Check that Ollama is running and has disk space.")
        sys.exit(1)


if __name__ == "__main__":
    main()

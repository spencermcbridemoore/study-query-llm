"""
Quick benchmark: TEI model load time + embedding latency on local RTX 4090.

Usage:
    python scripts/benchmark_tei_models.py

Tests each model in sequence:
  - Container start + model load time
  - Single-text latency (p50, p95)
  - Batch-32 throughput (texts/sec)
  - GPU VRAM used by the model

Requires Docker to be running and TEI image 89-1.9 to be pulled.
"""

import asyncio
import time
import statistics
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import sys
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HF_CACHE = str(Path.home() / ".cache" / "huggingface")
TEI_PORT  = 9900
WARMUP_RUNS = 3
TIMED_RUNS  = 10

SAMPLE_TEXTS_1  = ["What is the Pythagorean theorem?"]
SAMPLE_TEXTS_32 = [
    f"Question {i}: Explain a key concept in physics or mathematics."
    for i in range(32)
]

MODELS = [
    # (model_id, display_name)
    ("nomic-ai/nomic-embed-text-v1.5",           "nomic-v1.5          137M "),
    ("WhereIsAI/UAE-Large-V1",                   "UAE-Large-V1        335M "),
    ("BAAI/bge-large-en-v1.5",                   "bge-large-en-v1.5   335M "),
    ("Snowflake/snowflake-arctic-embed-l-v2.0",  "arctic-embed-l-v2   568M "),
    ("Alibaba-NLP/gte-large-en-v1.5",            "gte-large-en-v1.5   434M "),
    ("intfloat/multilingual-e5-large-instruct",  "me5-large-instruct  560M "),
    ("BAAI/bge-m3",                              "bge-m3              570M "),
    ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",      "gte-Qwen2-1.5B      1.5B "),
    ("Qwen/Qwen3-Embedding-0.6B",                "Qwen3-Embed-0.6B    0.6B "),
    ("Qwen/Qwen3-Embedding-4B",                  "Qwen3-Embed-4B      4B   "),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct",        "gte-Qwen2-7B        7.6B "),
    ("Qwen/Qwen3-Embedding-8B",                  "Qwen3-Embed-8B      8B   "),
]


def _start_container(model_id: str) -> tuple:
    import docker, docker.types
    client = docker.from_env()
    name = "tei-bench-" + model_id.replace("/", "-").replace(".", "-").lower()
    try:
        old = client.containers.get(name)
        old.stop(); old.remove()
    except Exception:
        pass

    container = client.containers.run(
        image="ghcr.io/huggingface/text-embeddings-inference:1.5",
        command=["--model-id", model_id, "--port", "80"],
        name=name,
        detach=True,
        ports={"80/tcp": TEI_PORT},
        volumes={HF_CACHE: {"bind": "/data", "mode": "ro"}},
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        remove=False,
    )
    return client, container, name


def _wait_healthy(timeout: int = 300, interval: float = 2.0) -> float:
    url = f"http://localhost:{TEI_PORT}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _embed(texts: list[str], model_id: str) -> float:
    """POST to /v1/embeddings, return elapsed seconds."""
    import json
    body = json.dumps({"model": model_id, "input": texts}).encode()
    req  = urllib.request.Request(
        f"http://localhost:{TEI_PORT}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60):
        pass
    return time.perf_counter() - t0


def _gpu_vram_mb() -> Optional[int]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return None


def _stop_container(client, container, name: str):
    try:
        container.stop()
        container.remove()
    except Exception:
        pass


def run_benchmark():
    print("=" * 72)
    print("  TEI GPU Benchmark  —  RTX 4090  (89-1.9 image)")
    print(f"  HF cache: {HF_CACHE}")
    print("=" * 72)
    print()
    print(f"{'Model':<32}  {'Load':>7}  {'1-text p50':>10}  {'1-text p95':>10}  {'32-batch t/s':>12}  {'VRAM':>6}")
    print("-" * 80)

    results = []

    for model_id, display in MODELS:
        sys.stdout.write(f"  {display:<30}  starting... ")
        sys.stdout.flush()

        t_start = time.time()
        client, container, name = _start_container(model_id)
        healthy = _wait_healthy(timeout=300)
        load_secs = time.time() - t_start

        if not healthy:
            _stop_container(client, container, name)
            print("TIMEOUT")
            continue

        vram = _gpu_vram_mb()

        # Warmup
        for _ in range(WARMUP_RUNS):
            try:
                _embed(SAMPLE_TEXTS_1, model_id)
            except Exception:
                pass

        # Single-text timing
        single_times = []
        for _ in range(TIMED_RUNS):
            try:
                single_times.append(_embed(SAMPLE_TEXTS_1, model_id))
            except Exception as e:
                print(f"\n    ERROR: {e}")
                break

        # Batch-32 timing
        batch_times = []
        for _ in range(max(3, TIMED_RUNS // 2)):
            try:
                batch_times.append(_embed(SAMPLE_TEXTS_32, model_id))
            except Exception:
                break

        _stop_container(client, container, name)

        if not single_times:
            print("FAILED")
            continue

        p50   = statistics.median(single_times) * 1000
        p95   = sorted(single_times)[int(len(single_times) * 0.95)] * 1000
        tps   = (32 / statistics.median(batch_times)) if batch_times else 0
        vram_str = f"{vram}MB" if vram else "n/a"

        print(
            f"\r  {display:<30}  {load_secs:>6.0f}s  "
            f"{p50:>9.0f}ms  {p95:>9.0f}ms  "
            f"{tps:>11.0f}/s  {vram_str:>6}"
        )
        results.append((display, load_secs, p50, p95, tps, vram_str))

    print()
    print("=" * 80)
    print("  Done.")


if __name__ == "__main__":
    run_benchmark()

"""
Benchmark the 4 models that previously failed due to CUDA < 12.9.
Now using TEI 89-1.9 (Ada Lovelace optimised, requires CUDA 12.9+).

Usage:
    python scripts/benchmark_tei_failed_models.py
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

HF_CACHE   = str(Path.home() / ".cache" / "huggingface")
TEI_PORT   = 9900
WARMUP_RUNS = 3
TIMED_RUNS  = 10
TEI_IMAGE   = "ghcr.io/huggingface/text-embeddings-inference:89-1.9"

SAMPLE_TEXTS_1  = ["What is the Pythagorean theorem?"]
SAMPLE_TEXTS_32 = [
    f"Question {i}: Explain a key concept in physics or mathematics."
    for i in range(32)
]

# Only the 4 models that timed out with TEI 1.5 / CUDA 12.6
MODELS = [
    ("BAAI/bge-m3",               "bge-m3              570M "),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embed-0.6B    0.6B "),
    ("Qwen/Qwen3-Embedding-4B",   "Qwen3-Embed-4B      4B   "),
    ("Qwen/Qwen3-Embedding-8B",   "Qwen3-Embed-8B      8B   "),
]


def _start_container(model_id: str) -> tuple:
    import docker
    import docker.types
    client = docker.from_env()
    name = "tei-bench-" + model_id.replace("/", "-").replace(".", "-").lower()
    try:
        old = client.containers.get(name)
        old.stop()
        old.remove()
    except Exception:
        pass

    container = client.containers.run(
        image=TEI_IMAGE,
        command=["--model-id", model_id, "--port", "80"],
        name=name,
        detach=True,
        ports={"80/tcp": TEI_PORT},
        volumes={HF_CACHE: {"bind": "/data", "mode": "rw"}},
        device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],
        remove=False,
    )
    return client, container, name


def _wait_healthy(timeout: int = 400, interval: float = 2.0) -> bool:
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


def _embed(texts: list, model_id: str) -> float:
    """POST to /v1/embeddings, return elapsed seconds."""
    import json
    body = json.dumps({"model": model_id, "input": texts}).encode()
    req = urllib.request.Request(
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


def _tail_logs(container, lines: int = 15) -> str:
    try:
        return container.logs(tail=lines).decode("utf-8", errors="replace")
    except Exception:
        return ""


def run_benchmark():
    print("=" * 80)
    print(f"  TEI GPU Benchmark (previously failed models)  --  RTX 4090")
    print(f"  Image : {TEI_IMAGE}")
    print(f"  Cache : {HF_CACHE}")
    print("=" * 80)
    print()
    print(
        f"{'Model':<32}  {'Load':>7}  {'1-text p50':>10}  "
        f"{'1-text p95':>10}  {'32-batch t/s':>12}  {'VRAM':>7}"
    )
    print("-" * 82)

    for model_id, display in MODELS:
        sys.stdout.write(f"  {display:<30}  starting...")
        sys.stdout.flush()

        t_start = time.time()
        try:
            client, container, name = _start_container(model_id)
        except Exception as exc:
            print(f"  DOCKER ERROR: {exc}")
            continue

        healthy = _wait_healthy(timeout=400)
        load_secs = time.time() - t_start

        if not healthy:
            logs = _tail_logs(container)
            _stop_container(client, container, name)
            print(f"  TIMEOUT after {load_secs:.0f}s")
            if logs:
                print("  Last container logs:")
                for line in logs.strip().splitlines()[-8:]:
                    print(f"    {line}")
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
            print("  FAILED (no successful inference)")
            continue

        p50 = statistics.median(single_times) * 1000
        p95 = sorted(single_times)[int(len(single_times) * 0.95)] * 1000
        tps = (32 / statistics.median(batch_times)) if batch_times else 0
        vram_str = f"{vram}MB" if vram else "n/a"

        print(
            f"\r  {display:<30}  {load_secs:>6.0f}s  "
            f"{p50:>9.0f}ms  {p95:>9.0f}ms  "
            f"{tps:>11.0f}/s  {vram_str:>7}"
        )

    print()
    print("=" * 80)
    print("  Done.")


if __name__ == "__main__":
    run_benchmark()

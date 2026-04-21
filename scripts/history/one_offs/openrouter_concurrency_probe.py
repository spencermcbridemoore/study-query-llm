#!/usr/bin/env python3
"""Minimal OpenRouter load probe: tiny completions, no result storage, high fan-out.

Fires many concurrent chat completions (semaphore-limited) to measure how
OpenRouter behaves under parallel load. Does not use the MCQ probe, DB, or
disk artifacts beyond an optional JSON summary path.

Examples:
  python scripts/history/one_offs/openrouter_concurrency_probe.py --concurrency 30 --requests 60
  python scripts/history/one_offs/openrouter_concurrency_probe.py --matrix 15,30,60 --requests-per-phase 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import study_query_llm.config  # noqa: F401 - load .env

from study_query_llm.config import Config
from study_query_llm.providers.factory import ProviderFactory


@dataclass
class PhaseResult:
    concurrency: int
    requests_planned: int
    success_count: int
    error_count: int
    elapsed_seconds: float
    latency_ms_mean: Optional[float]
    latency_ms_p95: Optional[float]
    errors_sample: List[str]


def _percentile(sorted_vals: Sequence[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


async def _run_phase(
    *,
    provider,
    concurrency: int,
    num_requests: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
    max_tokens_max: Optional[int] = None,
) -> PhaseResult:
    sem = asyncio.Semaphore(max(1, concurrency))
    latencies_ok: List[float] = []
    errors: List[str] = []

    async def one(i: int) -> None:
        nonlocal latencies_ok, errors
        async with sem:
            mt = int(max_tokens)
            if max_tokens_max is not None and int(max_tokens_max) >= mt:
                mt = random.randint(mt, int(max_tokens_max))
            t0 = time.perf_counter()
            try:
                resp = await provider.complete(
                    prompt,
                    temperature=temperature,
                    max_tokens=mt,
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if getattr(resp, "latency_ms", None) is not None:
                    try:
                        dt_ms = float(resp.latency_ms)
                    except (TypeError, ValueError):
                        pass
                latencies_ok.append(dt_ms)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:200]}")

    started = time.perf_counter()
    await asyncio.gather(*[one(i) for i in range(num_requests)])
    elapsed = time.perf_counter() - started

    lat_sorted = sorted(latencies_ok)
    mean_lat = float(statistics.mean(latencies_ok)) if latencies_ok else None
    p95 = _percentile(lat_sorted, 0.95) if lat_sorted else None

    return PhaseResult(
        concurrency=concurrency,
        requests_planned=num_requests,
        success_count=len(latencies_ok),
        error_count=len(errors),
        elapsed_seconds=elapsed,
        latency_ms_mean=mean_lat,
        latency_ms_p95=p95,
        errors_sample=errors[:12],
    )


async def _amain(args: argparse.Namespace) -> int:
    fresh = Config()
    if "openrouter" in fresh._provider_configs:
        del fresh._provider_configs["openrouter"]
    or_cfg = fresh.get_provider_config("openrouter")
    model = (args.model or "").strip() or (or_cfg.model or "")
    if not model:
        print("[FATAL] Set --model or OPENROUTER_MODEL.", file=sys.stderr)
        return 1

    factory = ProviderFactory(fresh)
    provider = factory.create_chat_provider("openrouter", model)

    phases: List[Tuple[int, int]] = []
    if args.matrix:
        for c in args.matrix:
            phases.append((int(c), int(args.requests_per_phase)))
    else:
        phases.append((int(args.concurrency), int(args.requests)))

    results: List[PhaseResult] = []
    for conc, nreq in phases:
        if int(args.max_tokens_max) > int(args.max_tokens):
            tok_hdr = f"{int(args.max_tokens)}-{int(args.max_tokens_max)} (random per request)"
        else:
            tok_hdr = str(int(args.max_tokens))
        print(
            f"\n[PHASE] concurrency={conc} requests={nreq} "
            f"prompt={args.prompt[:40]!r} max_tokens={tok_hdr}",
            flush=True,
        )
        pr = await _run_phase(
            provider=provider,
            concurrency=conc,
            num_requests=nreq,
            prompt=args.prompt,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            max_tokens_max=int(args.max_tokens_max) if int(args.max_tokens_max) > 0 else None,
        )
        results.append(pr)
        mean_s = f"{pr.latency_ms_mean:.1f}" if pr.latency_ms_mean is not None else "n/a"
        p95_s = f"{pr.latency_ms_p95:.1f}" if pr.latency_ms_p95 is not None else "n/a"
        print(
            f"  ok={pr.success_count} err={pr.error_count} "
            f"elapsed={pr.elapsed_seconds:.2f}s mean_ms={mean_s} p95_ms={p95_s}",
            flush=True,
        )
        if pr.errors_sample:
            print("  sample errors:", flush=True)
            for e in pr.errors_sample[:5]:
                print(f"    {e}", flush=True)

    out_payload: Dict[str, Any] = {
        "model": model,
        "max_tokens_range": (
            [int(args.max_tokens), int(args.max_tokens_max)]
            if int(args.max_tokens_max) > int(args.max_tokens)
            else int(args.max_tokens)
        ),
        "phases": [asdict(r) for r in results],
    }
    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote {out_path}")

    if hasattr(provider, "close"):
        await provider.close()  # type: ignore[misc]

    # Probe tool: exit 0 after reporting (errors are in stdout / JSON).
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenRouter minimal concurrency stress probe.")
    p.add_argument(
        "--concurrency",
        type=int,
        default=30,
        help="Max in-flight requests (single phase if --matrix not set).",
    )
    p.add_argument(
        "--requests",
        type=int,
        default=60,
        help="Total requests for single-phase run.",
    )
    p.add_argument(
        "--matrix",
        type=str,
        default="",
        help="Comma-separated concurrency levels, e.g. 15,30,60 (runs --requests-per-phase each).",
    )
    p.add_argument(
        "--requests-per-phase",
        type=int,
        default=60,
        help="Requests per matrix phase (default 60).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="OpenRouter model id (default: OPENROUTER_MODEL).",
    )
    p.add_argument("--prompt", type=str, default="Reply with exactly: OK")
    p.add_argument("--max-tokens", type=int, default=4)
    p.add_argument(
        "--max-tokens-max",
        type=int,
        default=0,
        help="If > --max-tokens, each request picks a random cap in [max-tokens, max-tokens-max].",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path for JSON summary (under repo root if relative).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.matrix.strip():
        args.matrix = [int(x.strip()) for x in args.matrix.split(",") if x.strip()]
    else:
        args.matrix = []
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())



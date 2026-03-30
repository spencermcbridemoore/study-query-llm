#!/usr/bin/env python3
"""Ping OpenRouter chat models with one minimal completion each (no sweep, no DB).

Use before large MCQ runs to see which model IDs respond. Requires
OPENROUTER_API_KEY and uses ProviderFactory(openrouter, model_id).

Default model list matches models intended for the [3,4,5,6] options sweep.

Examples:
  python scripts/verify_openrouter_chat_models.py
  python scripts/verify_openrouter_chat_models.py --models openai/gpt-oss-20b,allenai/olmo-3.1-32b-instruct
  python scripts/verify_openrouter_chat_models.py --parallel 2 --json-out experimental_results/openrouter_model_ping.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import study_query_llm.config  # noqa: F401 — load .env

from study_query_llm.config import Config
from study_query_llm.providers.factory import ProviderFactory

# Models to verify before expanding mcq_sweep_highschool_college_20q_*_openrouter configs.
DEFAULT_MODEL_IDS: List[str] = [
    "allenai/olmo-3.1-32b-instruct",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-0.6b-04-28",
    "qwen/qwen3-1.7b",
    "qwen/qwen3-8b",
    "qwen/qwen3-14b",
    "qwen/qwen3-4b",
    "qwen/qwen3-32b",
]


@dataclass
class ModelPingResult:
    model: str
    ok: bool
    latency_ms: Optional[float]
    error: str
    preview: str


async def _ping_one(
    factory: ProviderFactory,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> ModelPingResult:
    provider = None
    try:
        provider = factory.create_chat_provider("openrouter", model)
        t0 = time.perf_counter()
        resp = await provider.complete(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if getattr(resp, "latency_ms", None) is not None:
            try:
                elapsed_ms = float(resp.latency_ms)
            except (TypeError, ValueError):
                pass
        text = (getattr(resp, "text", None) or "").strip().replace("\n", " ")[:120]
        return ModelPingResult(
            model=model,
            ok=True,
            latency_ms=elapsed_ms,
            error="",
            preview=text,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc)[:500]}"
        return ModelPingResult(
            model=model,
            ok=False,
            latency_ms=None,
            error=err,
            preview="",
        )
    finally:
        if provider is not None and hasattr(provider, "close"):
            try:
                await provider.close()  # type: ignore[misc]
            except Exception:
                pass


async def _run(
    models: List[str],
    *,
    parallel: int,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> List[ModelPingResult]:
    fresh = Config()
    if "openrouter" in fresh._provider_configs:
        del fresh._provider_configs["openrouter"]
    factory = ProviderFactory(fresh)

    sem = asyncio.Semaphore(max(1, parallel))

    async def run_one(m: str) -> ModelPingResult:
        async with sem:
            print(f"[TRY] {m}", flush=True)
            return await _ping_one(
                factory,
                m,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

    return list(await asyncio.gather(*[run_one(m) for m in models]))


def _parse_models_arg(s: str) -> List[str]:
    parts = [p.strip() for p in s.replace("\n", ",").split(",")]
    return [p for p in parts if p]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify OpenRouter chat models respond to one minimal completion."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model ids (default: built-in list for MCQ multi-model sweep).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Max concurrent verification calls (default 4).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Reply with exactly the word: OK",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional JSON path (relative to repo root).",
    )
    args = parser.parse_args()

    models = _parse_models_arg(args.models) if args.models.strip() else list(DEFAULT_MODEL_IDS)
    if not models:
        print("[FATAL] No models to check.", file=sys.stderr)
        return 1

    try:
        fresh = Config()
        fresh.get_provider_config("openrouter")
    except Exception as exc:
        print(f"[FATAL] OpenRouter not configured: {exc}", file=sys.stderr)
        return 1

    results = asyncio.run(
        _run(
            models,
            parallel=int(args.parallel),
            prompt=str(args.prompt),
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
        )
    )

    ok_n = sum(1 for r in results if r.ok)
    print("\n" + "=" * 72)
    print(f"{'MODEL':<48} {'STATUS':<8} {'MS':>10}")
    print("=" * 72)
    for r in results:
        ms = f"{r.latency_ms:.0f}" if r.latency_ms is not None else "-"
        st = "OK" if r.ok else "FAIL"
        print(f"{r.model:<48} {st:<8} {ms:>10}")
        if not r.ok:
            print(f"  -> {r.error[:200]}")
        elif r.preview:
            print(f"  -> {r.preview[:100]}")
    print("=" * 72)
    print(f"Summary: {ok_n}/{len(results)} OK")

    payload: Dict[str, Any] = {
        "models_checked": [r.model for r in results],
        "ok_count": ok_n,
        "fail_count": len(results) - ok_n,
        "results": [asdict(r) for r in results],
    }
    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Wrote {out_path}")

    return 0 if ok_n == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Probe OpenRouter embedding models with the longest texts from a snapshot's embed slice.

Loads canonical dataframe texts (truncated with the same ``entry_max`` rule as
``fill_snapshot_embeddings_from_baseline``), takes the ``--top-n`` longest by
character length, then for each OpenRouter model in the baseline snapshot's
embedding roster:

1. Calls ``create_embeddings`` once with all ``top_n`` strings in a **single**
   request (mirrors a fat batch).
2. If that fails or times out, retries with **only the single longest** string
   to distinguish "batch too big" vs "one document over limit".

Uses direct :class:`OpenAICompatibleEmbeddingProvider` (no DB writes). Results
are printed as JSON and optionally written with ``--output-json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.config import Config
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.embed import _load_dataframe_texts
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri
from study_query_llm.providers.openai_compatible_embedding_provider import (
    OpenAICompatibleEmbeddingProvider,
)

Pair = Tuple[str, str]
LineageKey = Tuple[int, int]


def _resolve_database_url(explicit: str | None) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        v = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if v:
            return v
    raise SystemExit("No database URL (CANONICAL_DATABASE_URL / DATABASE_URL)")


def _snapshot_lineage(session, snapshot_group_id: int) -> dict[str, Any]:
    g = (
        session.query(Group)
        .filter(
            Group.id == int(snapshot_group_id),
            Group.group_type == "dataset_snapshot",
        )
        .first()
    )
    if g is None:
        raise ValueError(f"dataset_snapshot id={snapshot_group_id} not found")
    md = dict(g.metadata_json or {})
    sdf = int(md.get("source_dataframe_group_id") or 0)
    rows = int(md.get("row_count") or 0)
    if sdf <= 0 or rows <= 0:
        raise ValueError(f"snapshot {snapshot_group_id} missing lineage fields")
    dfg = (
        session.query(Group)
        .filter(Group.id == sdf, Group.group_type == "dataset_dataframe")
        .first()
    )
    df_md = dict((dfg.metadata_json if dfg else {}) or {})
    df_rows = int(df_md.get("row_count") or 0)
    return {
        "source_dataframe_group_id": sdf,
        "snapshot_row_count": rows,
        "source_dataframe_row_count": df_rows,
    }


def _entry_max(lineage: dict[str, Any]) -> int | None:
    snap = int(lineage["snapshot_row_count"])
    df_rows = int(lineage["source_dataframe_row_count"])
    if df_rows > 0 and snap < df_rows:
        return snap
    return None


def _pairs_for_lineage_key(session, key: LineageKey) -> Set[Pair]:
    sdf_id, entry_max = key
    out: Set[Pair] = set()
    for g in session.query(Group).filter(Group.group_type == "embedding_batch").all():
        md = dict(g.metadata_json or {})
        try:
            ksdf = int(md.get("source_dataframe_group_id") or 0)
            kem = int(md.get("entry_max") or 0)
        except (TypeError, ValueError):
            continue
        if ksdf != sdf_id or kem != entry_max:
            continue
        prov = str(md.get("provider") or "").strip()
        eng = str(md.get("embedding_engine") or md.get("deployment") or "").strip()
        if prov and eng:
            out.add((prov, eng))
    return out


def _openrouter_models_from_baseline(
    session, baseline_snapshot_id: int
) -> List[str]:
    lin = _snapshot_lineage(session, int(baseline_snapshot_id))
    key: LineageKey = (int(lin["source_dataframe_group_id"]), int(lin["snapshot_row_count"]))
    engines = sorted(
        {
            eng
            for prov, eng in _pairs_for_lineage_key(session, key)
            if str(prov).strip().lower() == "openrouter"
        }
    )
    if not engines:
        raise SystemExit("No openrouter engines found on baseline snapshot batches")
    return engines


async def _one_probe(
    *,
    provider: OpenAICompatibleEmbeddingProvider,
    model: str,
    texts: List[str],
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        results = await asyncio.wait_for(
            provider.create_embeddings(texts, model=model),
            timeout=timeout_s,
        )
        elapsed = time.perf_counter() - started
        dim = len(results[0].vector) if results else 0
        return {
            "ok": True,
            "model": model,
            "input_count": len(texts),
            "elapsed_seconds": round(elapsed, 2),
            "output_vectors": len(results),
            "dimension": dim,
        }
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "model": model,
            "input_count": len(texts),
            "elapsed_seconds": round(elapsed, 2),
            "error": f"TimeoutError after {timeout_s}s",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "ok": False,
            "model": model,
            "input_count": len(texts),
            "elapsed_seconds": round(elapsed, 2),
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_all(
    *,
    models: List[str],
    long_texts: List[str],
    longest_one: List[str],
    batch_timeout: float,
    single_timeout: float,
) -> List[dict[str, Any]]:
    cfg = Config().get_provider_config("openrouter")
    api_key = str(cfg.api_key or "").strip()
    endpoint = str(cfg.endpoint or "").strip().rstrip("/")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY missing for openrouter provider config")
    if not endpoint:
        raise SystemExit("OPENROUTER_ENDPOINT missing")

    prov = OpenAICompatibleEmbeddingProvider(
        base_url=endpoint if endpoint.endswith("/v1") else f"{endpoint}/v1",
        api_key=api_key,
        provider_label="openrouter",
    )
    rows: List[dict[str, Any]] = []
    try:
        for model in models:
            batch = await _one_probe(
                provider=prov,
                model=model,
                texts=long_texts,
                timeout_s=batch_timeout,
            )
            batch["phase"] = "batch_20"
            rows.append(batch)
            if not batch["ok"]:
                solo = await _one_probe(
                    provider=prov,
                    model=model,
                    texts=longest_one,
                    timeout_s=single_timeout,
                )
                solo["phase"] = "single_longest"
                rows.append(solo)
    finally:
        await prov.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-group-id", type=int, default=3)
    p.add_argument("--baseline-snapshot-id", type=int, default=9)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=240.0,
        help="Timeout for one API call with all top_n texts (default: 240)",
    )
    p.add_argument(
        "--single-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for single-longest fallback call (default: 120)",
    )
    p.add_argument("--artifact-dir", type=str, default="artifacts")
    p.add_argument("--database-url", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    args = p.parse_args(argv)

    database_url = _resolve_database_url(args.database_url)
    conn = DatabaseConnectionV2(
        database_url,
        enable_pgvector=True,
        write_intent=WriteIntent.CANONICAL,
    )
    conn.init_db()

    with conn.session_scope() as session:
        lin = _snapshot_lineage(session, int(args.snapshot_group_id))
        sdf = int(lin["source_dataframe_group_id"])
        em = _entry_max(lin)
        parquet_uri = find_dataframe_parquet_uri(session, sdf)
        models = _openrouter_models_from_baseline(session, int(args.baseline_snapshot_id))

    texts = _load_dataframe_texts(
        dataframe_parquet_uri=parquet_uri,
        artifact_dir=str(args.artifact_dir),
    )
    if em is not None:
        texts = texts[: int(em)]
    if not texts:
        raise SystemExit("No texts loaded for probe")

    indexed = sorted(enumerate(texts), key=lambda t: len(t[1]), reverse=True)
    top_n = max(1, int(args.top_n))
    chosen = indexed[:top_n]
    long_texts = [t for _, t in chosen]
    longest_one = [long_texts[0]] if long_texts else []

    lens = [len(t) for t in long_texts]
    report: dict[str, Any] = {
        "snapshot_group_id": int(args.snapshot_group_id),
        "baseline_snapshot_id": int(args.baseline_snapshot_id),
        "source_dataframe_group_id": sdf,
        "entry_max_used": em,
        "text_row_count": len(texts),
        "top_n": top_n,
        "longest_char_lengths": lens,
        "longest_row_indices": [i for i, _ in chosen],
        "openrouter_model_count": len(models),
        "models": models,
        "probes": asyncio.run(
            _run_all(
                models=models,
                long_texts=long_texts,
                longest_one=longest_one,
                batch_timeout=float(args.batch_timeout_seconds),
                single_timeout=float(args.single_timeout_seconds),
            )
        ),
    }

    text = json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True)
    print(text)
    out_path = args.output_json
    if out_path:
        path = Path(out_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[INFO] wrote {path}", file=sys.stderr)

    failed = [r for r in report["probes"] if not r.get("ok")]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

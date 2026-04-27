#!/usr/bin/env python3
"""Create a prompt snapshot from CSV and run OpenRouter chat sweeps.

Pipeline flow implemented by this script:
1) acquire local CSV as a canonical dataset group
2) parse CSV into canonical dataset_dataframe (text = prompt column)
3) snapshot all prompt rows (dataset_snapshot)
4) run OpenRouter chat inference using an available model roster
   (llama / qwen / closed_cheap)
5) persist per-model response artifacts + execution lineage

Design choice:
- Inference calls are intentionally *not* written as one RawCall row per prompt.
  Instead, each model run is stored as group-level artifacts (CSV + JSONL) and
  linked through `ProvenancedRun`. This keeps DB row growth controlled while
  preserving lineage from source data -> snapshot -> model response pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pyarrow.parquet as pq
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.datasets.source_specs.registry import DatasetAcquireConfig
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact, Group
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.db.write_intent import WriteIntent
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import find_dataframe_parquet_uri, parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.method_service import MethodService
from study_query_llm.services.provenanced_run_service import ProvenancedRunService


# Candidate rosters from the earlier OpenRouter availability audit.
MODEL_ROSTERS: Dict[str, List[str]] = {
    # Ordered by parameter scale (ascending) where known from slug.
    "llama": [
        "meta-llama/llama-3.2-1b-instruct",
        "meta-llama/llama-3.2-3b-instruct",
        "meta-llama/llama-3-8b-instruct",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3-70b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-4-scout",
        "meta-llama/llama-4-maverick",
    ],
    # Ordered by size where known.
    "qwen": [
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen3-8b",
        "qwen/qwen3-14b",
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen3-32b",
        "qwen/qwen-2.5-72b-instruct",
        "qwen/qwen-2-72b-instruct",
        "qwen/qwen3-235b-a22b",
    ],
    # "Closed-source cross-company anchors (cheap tiers only)".
    # Kept in source order from the user-provided list.
    # Note: x-ai/grok-3-mini intentionally excluded due long-tail latency.
    "closed_cheap": [
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-haiku-4.5",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-lite-001",
        "google/gemini-2.5-flash",
        "x-ai/grok-4-mini",
    ],
}

PROMPT_RESPONSE_METHOD_NAME = "openrouter_prompt_only_responses"
PROMPT_RESPONSE_METHOD_VERSION = "1.0"


@dataclass(frozen=True)
class PromptRecord:
    snapshot_position: int
    source_id: str
    prompt: str
    extra: Dict[str, Any]


def _sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "value"


def _candidate_models_for_roster(roster_name: str) -> List[str]:
    roster_key = str(roster_name or "").strip().lower()
    if roster_key not in MODEL_ROSTERS:
        raise ValueError(
            f"Unknown model roster {roster_name!r}. "
            f"Available rosters: {sorted(MODEL_ROSTERS.keys())}"
        )
    return list(MODEL_ROSTERS[roster_key])


def _parse_model_allowlist(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    values: List[str] = []
    seen: set[str] = set()
    for token in str(raw).split(","):
        model = str(token).strip()
        if not model:
            continue
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(model)
    return values


def _resolve_database_url(explicit: Optional[str]) -> str:
    env_file = dotenv_values(PROJECT_ROOT / ".env")
    for key in ("CANONICAL_DATABASE_URL", "DATABASE_URL"):
        candidate = (explicit or os.environ.get(key) or env_file.get(key) or "").strip()
        if candidate:
            return candidate
    raise SystemExit("No database URL found (CANONICAL_DATABASE_URL or DATABASE_URL).")


def _default_dataset_slug(csv_path: Path) -> str:
    stem = _sanitize_token(csv_path.stem)
    return f"mosart_{stem}"


def _build_local_csv_acquire_config(
    *,
    dataset_slug: str,
    csv_path: Path,
    acquisition_relative_path: str,
) -> DatasetAcquireConfig:
    csv_sha256 = _sha256_bytes(csv_path.read_bytes())
    csv_size = int(csv_path.stat().st_size)

    def _file_specs() -> List[FileFetchSpec]:
        return [
            FileFetchSpec(
                relative_path=str(acquisition_relative_path),
                url=f"local://{csv_path.as_posix()}",
            )
        ]

    def _source_metadata() -> Dict[str, Any]:
        return {
            "source_type": "local_csv",
            "path": str(csv_path),
            "pinning_identity": {
                "sha256": csv_sha256,
                "bytes": csv_size,
            },
        }

    return DatasetAcquireConfig(
        slug=str(dataset_slug),
        file_specs=_file_specs,
        source_metadata=_source_metadata,
    )


def _make_local_fetch() -> Callable[[str], bytes]:
    def _fetch(url: str) -> bytes:
        if not str(url).startswith("local://"):
            raise ValueError(f"Unsupported local fetch URL: {url!r}")
        raw = str(url)[len("local://") :]
        target = Path(raw)
        if not target.exists():
            raise FileNotFoundError(f"Local acquisition path not found: {target}")
        return target.read_bytes()

    return _fetch


def _make_prompt_parser(
    *,
    acquisition_relative_path: str,
    prompt_column: str,
    source_id_column: Optional[str],
) -> Callable[[ParserContext], Iterable[SnapshotRow]]:
    prompt_col = str(prompt_column)
    source_col = str(source_id_column).strip() if source_id_column else ""
    relative_path = str(acquisition_relative_path)

    def _parser(ctx: ParserContext) -> Iterable[SnapshotRow]:
        csv_path = ctx.artifact_dir_local / relative_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Parser could not locate acquired CSV: {csv_path}")

        rows: List[SnapshotRow] = []
        seen_source_ids: Dict[str, int] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV has no header row.")
            if prompt_col not in reader.fieldnames:
                raise ValueError(
                    f"Prompt column {prompt_col!r} not found. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row_index, row in enumerate(reader):
                normalized = {
                    str(key): ("" if value is None else str(value))
                    for key, value in row.items()
                }
                prompt = str(normalized.get(prompt_col, ""))
                if not prompt.strip():
                    raise ValueError(
                        f"Row {row_index} has empty prompt in column {prompt_col!r}."
                    )

                candidate_source_id = ""
                if source_col:
                    candidate_source_id = str(normalized.get(source_col, "")).strip()
                if not candidate_source_id:
                    candidate_source_id = str(normalized.get("prompt_id", "")).strip()
                if not candidate_source_id:
                    candidate_source_id = str(normalized.get("question_id", "")).strip()
                if not candidate_source_id:
                    candidate_source_id = f"row_{row_index:06d}"

                seen_count = seen_source_ids.get(candidate_source_id, 0)
                seen_source_ids[candidate_source_id] = seen_count + 1
                if seen_count > 0:
                    source_id = f"{candidate_source_id}__dup{seen_count:03d}"
                else:
                    source_id = candidate_source_id

                rows.append(
                    SnapshotRow(
                        position=len(rows),
                        source_id=source_id,
                        text=prompt,
                        label=None,
                        label_name=None,
                        extra=normalized,
                    )
                )

        if not rows:
            raise ValueError("CSV parser produced zero prompt rows.")
        return rows

    return _parser


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _latest_artifact_uri(
    repo: RawCallRepository,
    *,
    group_id: int,
    artifact_type: str,
) -> str:
    artifacts = repo.list_group_artifacts(group_id=group_id, artifact_types=[artifact_type])
    if not artifacts:
        raise ValueError(f"group id={group_id} has no artifact type={artifact_type!r}")
    latest = sorted(artifacts, key=lambda a: int(a.id))[-1]
    return str(latest.uri)


def _load_snapshot_prompts(
    *,
    db_conn: DatabaseConnectionV2,
    snapshot_group_id: int,
    artifact_dir: str,
) -> tuple[int, List[PromptRecord]]:
    with db_conn.session_scope() as session:
        repo = RawCallRepository(session)
        snapshot_group = (
            session.query(Group)
            .filter(
                Group.id == int(snapshot_group_id),
                Group.group_type == "dataset_snapshot",
            )
            .first()
        )
        if snapshot_group is None:
            raise ValueError(f"dataset_snapshot group id={snapshot_group_id} not found")
        snapshot_meta = dict(snapshot_group.metadata_json or {})
        dataframe_group_id = int(snapshot_meta.get("source_dataframe_group_id") or -1)
        if dataframe_group_id <= 0:
            raise ValueError(
                "dataset_snapshot metadata missing source_dataframe_group_id"
            )
        subquery_uri = _latest_artifact_uri(
            repo,
            group_id=int(snapshot_group_id),
            artifact_type="dataset_subquery_spec",
        )
        parquet_uri = find_dataframe_parquet_uri(session, int(dataframe_group_id))

    artifact_service = ArtifactService(artifact_dir=artifact_dir)
    subquery_payload = json.loads(
        artifact_service.storage.read_from_uri(subquery_uri).decode("utf-8")
    )
    resolved_index = list(subquery_payload.get("resolved_index") or [])
    if not resolved_index:
        raise ValueError("snapshot resolved_index is empty")

    table_payload = artifact_service.storage.read_from_uri(parquet_uri)
    table = pq.read_table(
        io.BytesIO(table_payload),
        columns=["position", "source_id", "text", "extra_json"],
    )
    frame = table.to_pandas().set_index("position", drop=False)

    prompts: List[PromptRecord] = []
    for item in resolved_index:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Invalid resolved_index entry: {item!r}")
        position = int(item[0])
        fallback_source_id = str(item[1])
        if position not in frame.index:
            raise ValueError(f"resolved position {position} missing from dataframe parquet")

        row = frame.loc[position]
        row_source_id = str(row["source_id"] or "").strip() or fallback_source_id
        prompt = str(row["text"])
        extra_raw = row.get("extra_json")
        extra: Dict[str, Any] = {}
        if isinstance(extra_raw, str) and extra_raw.strip():
            loaded = json.loads(extra_raw)
            if isinstance(loaded, dict):
                extra = {str(k): v for k, v in loaded.items()}

        prompts.append(
            PromptRecord(
                snapshot_position=int(position),
                source_id=row_source_id,
                prompt=prompt,
                extra=extra,
            )
        )

    return int(dataframe_group_id), prompts


async def _resolve_openrouter_chat_models() -> set[str]:
    factory = ProviderFactory()
    deployments = await factory.list_provider_deployments(
        "openrouter",
        modality="chat",
    )
    return {str(info.id).strip().lower() for info in deployments}


async def _close_provider_if_supported(provider: Any) -> None:
    close_fn = getattr(provider, "close", None)
    if close_fn is None:
        return
    result = close_fn()
    if asyncio.iscoroutine(result):
        await result


async def _run_model_inference(
    *,
    model: str,
    prompts: Sequence[PromptRecord],
    concurrency: int,
    temperature: float,
    max_tokens: Optional[int],
    max_retries: int,
    initial_wait: float,
    max_wait: float,
    progress_every: int,
) -> List[Dict[str, Any]]:
    factory = ProviderFactory()
    provider = factory.create_chat_provider("openrouter", model)
    service = InferenceService(
        provider=provider,
        repository=None,
        require_db_persistence=False,
        max_retries=int(max_retries),
        initial_wait=float(initial_wait),
        max_wait=float(max_wait),
        preprocess=False,
    )
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(prompts)
    done_counter = 0
    done_lock = asyncio.Lock()
    rows: List[Optional[Dict[str, Any]]] = [None] * total

    async def _one(index: int, rec: PromptRecord) -> None:
        nonlocal done_counter
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await service.run_inference(
                    rec.prompt,
                    temperature=float(temperature),
                    max_tokens=max_tokens,
                )
                metadata = dict(response.get("metadata") or {})
                row = {
                    "snapshot_position": int(rec.snapshot_position),
                    "source_id": str(rec.source_id),
                    "prompt": str(rec.prompt),
                    "response": str(response.get("response") or ""),
                    "status": "success",
                    "error": "",
                    "latency_ms": metadata.get("latency_ms"),
                    "tokens": metadata.get("tokens"),
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                }
            except Exception as exc:
                row = {
                    "snapshot_position": int(rec.snapshot_position),
                    "source_id": str(rec.source_id),
                    "prompt": str(rec.prompt),
                    "response": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": None,
                    "tokens": None,
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                }
            rows[index] = row

        async with done_lock:
            done_counter += 1
            if (
                done_counter == total
                or done_counter == 1
                or done_counter % max(1, int(progress_every)) == 0
            ):
                print(
                    f"[MODEL {model}] progress {done_counter}/{total}",
                    flush=True,
                )

    try:
        await asyncio.gather(*[_one(idx, rec) for idx, rec in enumerate(prompts)])
    finally:
        await _close_provider_if_supported(provider)

    final_rows = [row for row in rows if row is not None]
    if len(final_rows) != total:
        raise RuntimeError(
            f"Model {model} completed with row mismatch: {len(final_rows)} != {total}"
        )
    return final_rows


async def _run_models_with_workers(
    *,
    models: Sequence[str],
    prompts: Sequence[PromptRecord],
    model_concurrency: int,
    prompt_concurrency: int,
    temperature: float,
    max_tokens: Optional[int],
    max_retries: int,
    initial_wait: float,
    max_wait: float,
    progress_every: int,
    on_model_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    ordered_models = list(models)
    total_models = len(ordered_models)
    if total_models <= 0:
        return []

    model_slots = max(1, int(model_concurrency))
    model_sem = asyncio.Semaphore(model_slots)
    results: List[Optional[Dict[str, Any]]] = [None] * total_models

    async def _one(model_position: int, model: str) -> None:
        result_payload: Optional[Dict[str, Any]] = None
        async with model_sem:
            model_started = time.perf_counter()
            print(
                f"[RUN] model={model} ({model_position + 1}/{total_models}) "
                f"active_model_workers<= {model_slots}",
                flush=True,
            )
            result_rows = await _run_model_inference(
                model=model,
                prompts=prompts,
                concurrency=int(prompt_concurrency),
                temperature=float(temperature),
                max_tokens=max_tokens,
                max_retries=int(max_retries),
                initial_wait=float(initial_wait),
                max_wait=float(max_wait),
                progress_every=int(progress_every),
            )
            elapsed_seconds = round(time.perf_counter() - model_started, 3)
            failures = sum(1 for row in result_rows if row["status"] != "success")
            successes = len(result_rows) - failures
            print(
                f"[INFERENCE DONE] model={model} "
                f"success={successes} failed={failures} elapsed={elapsed_seconds}s",
                flush=True,
            )
            result_payload = {
                "model_position": int(model_position),
                "model": model,
                "rows": result_rows,
                "elapsed_seconds": elapsed_seconds,
                "success_count": int(successes),
                "failure_count": int(failures),
            }
            results[model_position] = result_payload
        if on_model_complete is not None:
            if result_payload is None:
                raise RuntimeError(f"Missing model result payload for {model}")
            on_model_complete(result_payload)

    await asyncio.gather(
        *[_one(model_position, model) for model_position, model in enumerate(ordered_models)]
    )
    final = [row for row in results if row is not None]
    if len(final) != total_models:
        raise RuntimeError(
            f"Model worker result mismatch: got {len(final)} expected {total_models}"
        )
    return final


def _serialize_csv(rows: Sequence[Dict[str, Any]], ordered_columns: Sequence[str]) -> str:
    sink = io.StringIO()
    writer = csv.DictWriter(
        sink,
        fieldnames=[str(col) for col in ordered_columns],
        extrasaction="ignore",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return sink.getvalue()


def _serialize_jsonl(rows: Sequence[Dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _artifact_uri_by_id(session: Any, artifact_id: int) -> str:
    artifact = session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _ensure_prompt_response_method(repo: RawCallRepository) -> int:
    method_service = MethodService(repo)
    existing = method_service.get_method(
        PROMPT_RESPONSE_METHOD_NAME,
        version=PROMPT_RESPONSE_METHOD_VERSION,
    )
    if existing is not None:
        return int(existing.id)
    return int(
        method_service.register_method(
            name=PROMPT_RESPONSE_METHOD_NAME,
            version=PROMPT_RESPONSE_METHOD_VERSION,
            description=(
                "Prompt-only chat completion sweep over dataset_snapshot prompts "
                "with per-model response artifacts."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "provider": {"type": "string"},
                    "model": {"type": "string"},
                    "temperature": {"type": "number"},
                    "max_tokens": {"type": ["integer", "null"]},
                    "concurrency": {"type": "integer"},
                    "prompt_column": {"type": "string"},
                },
            },
        )
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-roster",
        type=str,
        choices=sorted(MODEL_ROSTERS.keys()),
        default="llama",
        help="Model roster key to run (default: llama).",
    )
    parser.add_argument(
        "--model-allowlist",
        type=str,
        default=None,
        help=(
            "Optional comma-separated explicit model IDs to run. "
            "When provided, these IDs are used instead of the roster default order."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=(
            "c:/Users/spenc/OneDrive/Documents/Claude/Projects/MOSART/"
            "llm_prompts_astro03_astro07_no_preprompt.csv"
        ),
        help="Path to source CSV file containing prompts.",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="user_prompt",
        help="CSV column to send to model as the prompt.",
    )
    parser.add_argument(
        "--source-id-column",
        type=str,
        default="prompt_id",
        help="CSV column used for stable source_id (fallbacks used if missing/empty).",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="Dataset slug override (default derives from CSV filename).",
    )
    parser.add_argument(
        "--acquisition-relative-path",
        type=str,
        default="source.csv",
        help="Filename used inside acquisition artifact bundle.",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Optional DB URL override (defaults from env/.env).",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="artifacts",
        help="Artifact base directory/URI resolver.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for chat inference.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Completion max tokens per prompt (<=0 means provider default).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max in-flight prompt calls per model.",
    )
    parser.add_argument(
        "--model-concurrency",
        type=int,
        default=1,
        help="Number of models to run concurrently (worker slots).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="InferenceService retry attempts.",
    )
    parser.add_argument(
        "--initial-wait-seconds",
        type=float,
        default=1.0,
        help="Retry backoff initial wait seconds.",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=30.0,
        help="Retry backoff max wait seconds.",
    )
    parser.add_argument(
        "--target-model-count",
        type=int,
        default=9,
        help="Maximum number of available Llama models to run (ascending-size order).",
    )
    parser.add_argument(
        "--require-model-count",
        type=int,
        default=9,
        help="Fail fast if fewer than this many candidate models are available.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional cap for prompt rows (for smoke tests).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Progress log interval during per-model runs.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Combined output CSV path (defaults under experimental_results/prompt_sweeps).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Run summary JSON path (defaults next to output CSV).",
    )
    parser.add_argument("--force-acquire", action="store_true", help="Force acquire stage.")
    parser.add_argument("--force-parse", action="store_true", help="Force parse stage.")
    parser.add_argument("--force-snapshot", action="store_true", help="Force snapshot stage.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV path does not exist: {csv_path}")
    if not csv_path.is_file():
        raise SystemExit(f"CSV path is not a file: {csv_path}")

    dataset_slug = str(args.dataset_slug or _default_dataset_slug(csv_path))
    database_url = _resolve_database_url(args.database_url)
    max_tokens = int(args.max_tokens) if int(args.max_tokens) > 0 else None

    db_conn = DatabaseConnectionV2(
        database_url,
        enable_pgvector=False,
        write_intent=WriteIntent.CANONICAL,
    )
    db_conn.init_db()

    print(f"[SETUP] csv={csv_path}", flush=True)
    print(f"[SETUP] dataset_slug={dataset_slug}", flush=True)

    acquire_cfg = _build_local_csv_acquire_config(
        dataset_slug=dataset_slug,
        csv_path=csv_path,
        acquisition_relative_path=str(args.acquisition_relative_path),
    )
    acquire_result = acquire(
        acquire_cfg,
        force=bool(args.force_acquire),
        db=db_conn,
        write_intent=WriteIntent.CANONICAL,
        artifact_dir=str(args.artifact_dir),
        fetch=_make_local_fetch(),
    )
    print(
        f"[ACQUIRE] dataset_group_id={acquire_result.group_id} reused={acquire_result.metadata.get('reused')}",
        flush=True,
    )

    parser = _make_prompt_parser(
        acquisition_relative_path=str(args.acquisition_relative_path),
        prompt_column=str(args.prompt_column),
        source_id_column=str(args.source_id_column),
    )
    parse_result = parse(
        int(acquire_result.group_id),
        parser=parser,
        parser_id="local_csv_prompt_parser",
        parser_version="1.0",
        force=bool(args.force_parse),
        db=db_conn,
        write_intent=WriteIntent.CANONICAL,
        artifact_dir=str(args.artifact_dir),
    )
    print(
        f"[PARSE] dataframe_group_id={parse_result.group_id} reused={parse_result.metadata.get('reused')}",
        flush=True,
    )

    snapshot_result = snapshot(
        int(parse_result.group_id),
        subquery_spec=SubquerySpec(label_mode="all"),
        force=bool(args.force_snapshot),
        db=db_conn,
        write_intent=WriteIntent.CANONICAL,
        artifact_dir=str(args.artifact_dir),
    )
    print(
        f"[SNAPSHOT] snapshot_group_id={snapshot_result.group_id} reused={snapshot_result.metadata.get('reused')}",
        flush=True,
    )

    dataframe_group_id, prompts = _load_snapshot_prompts(
        db_conn=db_conn,
        snapshot_group_id=int(snapshot_result.group_id),
        artifact_dir=str(args.artifact_dir),
    )
    if args.max_prompts is not None:
        prompts = prompts[: max(0, int(args.max_prompts))]
    if not prompts:
        raise SystemExit("No prompts available after snapshot loading/max-prompts filter.")
    print(f"[PROMPTS] loaded={len(prompts)}", flush=True)

    candidate_models = _candidate_models_for_roster(str(args.model_roster))
    allowlist = _parse_model_allowlist(args.model_allowlist)
    if allowlist:
        candidate_models = list(allowlist)
        print(f"[MODELS] allowlist_requested={len(allowlist)}", flush=True)
        for index, model in enumerate(allowlist, start=1):
            print(f"  ALLOW {index:02d}. {model}", flush=True)
    available_models = asyncio.run(_resolve_openrouter_chat_models())
    selected_models = [
        model for model in candidate_models if model.lower() in available_models
    ]
    if args.target_model_count is not None:
        selected_models = selected_models[: max(1, int(args.target_model_count))]
    if len(selected_models) < int(args.require_model_count):
        raise SystemExit(
            f"Only {len(selected_models)} candidate models available; "
            f"required at least {args.require_model_count}."
        )
    print(f"[MODELS] selected={len(selected_models)}", flush=True)
    for index, model in enumerate(selected_models, start=1):
        print(f"  {index:02d}. {model}", flush=True)

    run_started = datetime.now(timezone.utc)
    run_stamp = run_started.strftime("%Y%m%d_%H%M%S")
    run_token = f"{_sanitize_token(dataset_slug)}_snap{int(snapshot_result.group_id)}_{run_stamp}"

    with db_conn.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = int(
            repo.create_group(
                group_type="custom",
                name=f"prompt_sweep_request:{run_token}",
                description="Prompt-only OpenRouter Llama sweep request",
                metadata_json={
                    "run_token": run_token,
                    "provider": "openrouter",
                    "model_roster": str(args.model_roster),
                    "dataset_group_id": int(acquire_result.group_id),
                    "dataframe_group_id": int(dataframe_group_id),
                    "snapshot_group_id": int(snapshot_result.group_id),
                    "prompt_column": str(args.prompt_column),
                    "source_id_column": str(args.source_id_column),
                    "candidate_models": list(candidate_models),
                    "selected_models": list(selected_models),
                    "target_model_count": int(args.target_model_count),
                    "model_concurrency": int(args.model_concurrency),
                    "prompt_concurrency": int(args.concurrency),
                    "prompt_count": len(prompts),
                    "created_at": run_started.isoformat(),
                },
            )
        )
        repo.create_group_link(
            parent_group_id=request_group_id,
            child_group_id=int(snapshot_result.group_id),
            link_type="depends_on",
        )
        method_definition_id = _ensure_prompt_response_method(repo)

    merged_rows_by_position: Dict[int, List[Dict[str, Any]]] = {}
    model_reports_by_position: Dict[int, Dict[str, Any]] = {}

    def _persist_completed_model(model_result: Dict[str, Any]) -> None:
        model_position = int(model_result["model_position"])
        model = str(model_result["model"])
        result_rows = list(model_result["rows"])
        elapsed_seconds = float(model_result["elapsed_seconds"])
        successes = int(model_result["success_count"])
        failures = int(model_result["failure_count"])
        if model_position in model_reports_by_position:
            raise RuntimeError(f"Duplicate model completion callback for position={model_position}")

        merged_rows: List[Dict[str, Any]] = []
        for row, prompt_record in zip(result_rows, prompts):
            merged = dict(prompt_record.extra)
            merged.update(
                {
                    "source_id": row["source_id"],
                    "snapshot_position": row["snapshot_position"],
                    "model": model,
                    "provider": "openrouter",
                    "response": row["response"],
                    "status": row["status"],
                    "error": row["error"],
                    "latency_ms": row["latency_ms"],
                    "tokens": row["tokens"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
            )
            merged_rows.append(merged)

        base_columns = sorted(
            {
                key
                for merged in merged_rows
                for key in merged.keys()
                if key
                not in {
                    "source_id",
                    "snapshot_position",
                    "model",
                    "provider",
                    "response",
                    "status",
                    "error",
                    "latency_ms",
                    "tokens",
                    "elapsed_seconds",
                }
            }
        )
        ordered_columns = [
            *base_columns,
            "source_id",
            "snapshot_position",
            "provider",
            "model",
            "response",
            "status",
            "error",
            "latency_ms",
            "tokens",
            "elapsed_seconds",
        ]

        csv_text = _serialize_csv(merged_rows, ordered_columns=ordered_columns)
        jsonl_text = _serialize_jsonl(merged_rows)

        with db_conn.session_scope() as session:
            repo = RawCallRepository(session)
            artifacts = ArtifactService(
                repository=repo,
                artifact_dir=str(args.artifact_dir),
                write_intent=WriteIntent.CANONICAL,
            )
            safe_model = _sanitize_token(model)
            result_group_id = int(
                repo.create_group(
                    group_type="summarization_batch",
                    name=f"prompt_responses:{safe_model}:{run_stamp}",
                    description=f"Prompt response artifact batch for {model}",
                    metadata_json={
                        "run_token": run_token,
                        "provider": "openrouter",
                        "model": model,
                        "model_position": int(model_position),
                        "request_group_id": int(request_group_id),
                        "snapshot_group_id": int(snapshot_result.group_id),
                        "source_dataframe_group_id": int(dataframe_group_id),
                        "prompt_count": len(prompts),
                        "success_count": successes,
                        "failure_count": failures,
                        "temperature": float(args.temperature),
                        "max_tokens": max_tokens,
                        "prompt_column": str(args.prompt_column),
                        "source_id_column": str(args.source_id_column),
                    },
                )
            )
            repo.create_group_link(
                parent_group_id=request_group_id,
                child_group_id=result_group_id,
                link_type="contains",
                position=int(model_position),
            )
            repo.create_group_link(
                parent_group_id=result_group_id,
                child_group_id=int(snapshot_result.group_id),
                link_type="depends_on",
            )

            csv_artifact_id = int(
                artifacts.store_group_blob_artifact(
                    group_id=result_group_id,
                    step_name="inference",
                    logical_filename=f"{safe_model}.responses.csv",
                    data=csv_text.encode("utf-8"),
                    artifact_type="model_response_csv",
                    content_type="text/csv",
                    metadata={
                        "provider": "openrouter",
                        "model": model,
                        "run_token": run_token,
                        "row_count": len(merged_rows),
                        "success_count": successes,
                        "failure_count": failures,
                    },
                )
            )
            jsonl_artifact_id = int(
                artifacts.store_group_blob_artifact(
                    group_id=result_group_id,
                    step_name="inference",
                    logical_filename=f"{safe_model}.responses.jsonl",
                    data=jsonl_text.encode("utf-8"),
                    artifact_type="model_response_jsonl",
                    content_type="application/jsonl",
                    metadata={
                        "provider": "openrouter",
                        "model": model,
                        "run_token": run_token,
                        "row_count": len(merged_rows),
                    },
                )
            )
            csv_uri = _artifact_uri_by_id(session, csv_artifact_id)
            jsonl_uri = _artifact_uri_by_id(session, jsonl_artifact_id)

            provenanced = ProvenancedRunService(repo)
            _ = provenanced.record_method_execution(
                request_group_id=int(request_group_id),
                run_key=f"openrouter:{model}",
                source_group_id=int(result_group_id),
                result_group_id=int(result_group_id),
                input_snapshot_group_id=int(snapshot_result.group_id),
                method_definition_id=int(method_definition_id),
                determinism_class="non_deterministic",
                config_json={
                    "provider": "openrouter",
                    "model": model,
                    "temperature": float(args.temperature),
                    "max_tokens": max_tokens,
                    "concurrency": int(args.concurrency),
                    "model_concurrency": int(args.model_concurrency),
                    "prompt_column": str(args.prompt_column),
                },
                result_ref=csv_uri,
                metadata_json={
                    "dataset_snapshot_ids": [int(snapshot_result.group_id)],
                    "csv_artifact_uri": csv_uri,
                    "jsonl_artifact_uri": jsonl_uri,
                    "success_count": successes,
                    "failure_count": failures,
                    "elapsed_seconds": elapsed_seconds,
                },
                run_status="completed",
            )

        merged_rows_by_position[int(model_position)] = list(merged_rows)
        model_reports_by_position[int(model_position)] = {
            "model": model,
            "provider": "openrouter",
            "model_position": int(model_position),
            "prompt_count": len(prompts),
            "success_count": successes,
            "failure_count": failures,
            "elapsed_seconds": elapsed_seconds,
            "result_group_id": int(result_group_id),
            "csv_artifact_id": int(csv_artifact_id),
            "jsonl_artifact_id": int(jsonl_artifact_id),
            "csv_artifact_uri": csv_uri,
            "jsonl_artifact_uri": jsonl_uri,
        }
        print(
            f"[DONE] model={model} success={successes} failed={failures} elapsed={elapsed_seconds}s",
            flush=True,
        )

    _ = asyncio.run(
        _run_models_with_workers(
            models=selected_models,
            prompts=prompts,
            model_concurrency=int(args.model_concurrency),
            prompt_concurrency=int(args.concurrency),
            temperature=float(args.temperature),
            max_tokens=max_tokens,
            max_retries=int(args.max_retries),
            initial_wait=float(args.initial_wait_seconds),
            max_wait=float(args.max_wait_seconds),
            progress_every=int(args.progress_every),
            on_model_complete=_persist_completed_model,
        )
    )

    if len(model_reports_by_position) != len(selected_models):
        raise RuntimeError(
            f"Persisted model count mismatch: {len(model_reports_by_position)} != {len(selected_models)}"
        )
    ordered_positions = sorted(model_reports_by_position.keys())
    model_reports: List[Dict[str, Any]] = [
        model_reports_by_position[pos] for pos in ordered_positions
    ]
    all_output_rows: List[Dict[str, Any]] = []
    for position in ordered_positions:
        all_output_rows.extend(merged_rows_by_position.get(position, []))

    out_dir = PROJECT_ROOT / "experimental_results" / "prompt_sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_csv = out_dir / f"{run_token}.responses.csv"
    output_csv = Path(args.output_csv).expanduser() if args.output_csv else default_csv
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    combined_base_columns = sorted(
        {
            key
            for row in all_output_rows
            for key in row.keys()
            if key
            not in {
                "source_id",
                "snapshot_position",
                "provider",
                "model",
                "response",
                "status",
                "error",
                "latency_ms",
                "tokens",
                "elapsed_seconds",
            }
        }
    )
    combined_columns = [
        *combined_base_columns,
        "source_id",
        "snapshot_position",
        "provider",
        "model",
        "response",
        "status",
        "error",
        "latency_ms",
        "tokens",
        "elapsed_seconds",
    ]
    combined_csv_text = _serialize_csv(all_output_rows, ordered_columns=combined_columns)
    output_csv.write_text(combined_csv_text, encoding="utf-8")

    default_json = output_csv.with_suffix(".summary.json")
    output_json = Path(args.output_json).expanduser() if args.output_json else default_json
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    run_finished = datetime.now(timezone.utc)
    summary = {
        "run_token": run_token,
        "provider": "openrouter",
        "model_roster": str(args.model_roster),
        "csv_path": str(csv_path),
        "dataset_slug": dataset_slug,
        "dataset_group_id": int(acquire_result.group_id),
        "dataframe_group_id": int(parse_result.group_id),
        "snapshot_group_id": int(snapshot_result.group_id),
        "request_group_id": int(request_group_id),
        "method_definition_id": int(method_definition_id),
        "prompt_column": str(args.prompt_column),
        "source_id_column": str(args.source_id_column),
        "prompt_count": len(prompts),
        "selected_models": selected_models,
        "target_model_count": int(args.target_model_count),
        "model_concurrency": int(args.model_concurrency),
        "prompt_concurrency": int(args.concurrency),
        "model_reports": model_reports,
        "combined_output_csv": str(output_csv),
        "started_at": run_started.isoformat(),
        "finished_at": run_finished.isoformat(),
        "duration_seconds": round((run_finished - run_started).total_seconds(), 3),
    }
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    total_failures = sum(int(row["failure_count"]) for row in model_reports)
    print(f"[WRITE] combined CSV -> {output_csv}", flush=True)
    print(f"[WRITE] summary JSON -> {output_json}", flush=True)
    print(
        f"[SUMMARY] models={len(model_reports)} prompts/model={len(prompts)} "
        f"total_rows={len(all_output_rows)} failures={total_failures}",
        flush=True,
    )
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

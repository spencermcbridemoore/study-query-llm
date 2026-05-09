"""Phase-1 foundations for MCQ first-token logprob execution."""

from __future__ import annotations

import asyncio
import io
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Literal, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.providers.factory import ProviderFactory
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.inference_service import InferenceService
from study_query_llm.services.method_runtime_registry import (
    MethodRunnerContext,
    MethodRunnerResult,
)

PermutationStrategy = Literal["full_120", "latin_squares_25", "single_latin_square_5"]

_PICTURE_REF_REGEX = re.compile(
    r"\b(figure|diagram|graph|plot|sketch|illustrat|depicted)\b|includegraphics",
    flags=re.IGNORECASE,
)
_IMG_TAG_REGEX = re.compile(r"<img\b", flags=re.IGNORECASE)
_MARKDOWN_IMAGE_REGEX = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_EXTERNAL_IMAGE_URL_REGEX = re.compile(
    r"https?://\S+\.(png|jpg|jpeg|gif|svg)",
    flags=re.IGNORECASE,
)
_SAFE_TOKEN_REGEX = re.compile(r"[^A-Za-z0-9._-]+")

# Keys MethodRunnerResult.metadata_json sets from runner state; caller metadata cannot override.
_MCQ_LOGPROB_RUNNER_METADATA_PROTECTED_KEYS: frozenset[str] = frozenset(
    {
        "dataset_name",
        "dataset_version",
        "imported_run_id",
        "provider",
        "model",
        "permutation_strategy",
        "prompt_template_version",
    }
)

_OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E")
_RATE_LIMIT_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
_FORMAT_SUFFIXES: tuple[str, ...] = ("Answer:", "Answer: ", "The correct answer is ")

_REQUIRED_DATASET_COLUMNS: tuple[str, ...] = (
    "ItemID",
    "ItemType",
    "ProblemBody",
    "OptionA",
    "OptionB",
    "OptionC",
    "OptionD",
    "OptionE",
    "CorrectLetter",
    "temp_bank_id",
)

_MIDTERM_FILTER_COLUMNS: tuple[str, ...] = (
    "ProblemBody",
    "ProblemBodyTemplate",
    "OptionA",
    "OptionB",
    "OptionC",
    "OptionD",
    "OptionE",
)

# Baseline relabel keeps Latin-square structure but avoids duplicating k=1 rows.
_BASELINE_REMAP: dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 5, 5: 4}


class McqLogprobBasicParams(BaseModel):
    """Boundary-validated parameter payload for Phase-1 MCQ logprob runner."""

    provider: str = Field(default="openrouter", min_length=1)
    model: str = Field(min_length=1)
    permutation_strategy: PermutationStrategy = "latin_squares_25"
    format_idx: int = Field(default=0, ge=0, le=2)
    prompt_template_version: str = Field(default="v1", min_length=1)
    concurrency_cap: int = Field(default=10, ge=1, le=100)
    top_logprobs: int = Field(default=20, ge=1, le=20)
    max_questions: int | None = Field(default=None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @field_validator("provider", "model", "prompt_template_version")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text


@dataclass(frozen=True)
class InferenceAttemptOutcome:
    """Normalized outcome for one prompt inference attempt chain."""

    inference_result: dict[str, Any] | None
    error: str | None
    rate_limit_retry_count: int
    rate_limit_total_wait_ms: int


@dataclass(frozen=True)
class ProbeCallResult:
    """One synthetic probe call outcome."""

    outcome: Literal["ok", "rate_limited", "hard_fail", "other_error"]
    retry_count: int = 0
    wait_ms: int = 0
    error: str | None = None


@dataclass(frozen=True)
class ProbeTierStats:
    """Aggregate stats for one tested probe tier."""

    concurrency: int
    total_calls: int
    rate_limited_calls: int
    total_retry_count: int
    total_wait_ms: int
    passed: bool


@dataclass(frozen=True)
class ProbeResult:
    """Resolved probe result for one model."""

    model: str
    resolved_concurrency: int
    excluded_reason: str | None
    reachability_outcome: str
    tier_stats: tuple[ProbeTierStats, ...] = field(default_factory=tuple)


class HardFailureError(RuntimeError):
    """Raised when an invocation must fail immediately."""


def full_120_indices() -> list[tuple[int, int, int, int, int]]:
    """Return canonical lexicographic 5! permutation enumeration."""
    return [tuple(p) for p in itertools.permutations((1, 2, 3, 4, 5))]


def _latin_square_rows_for_k(k: int) -> list[tuple[int, int, int, int, int]]:
    return [
        tuple(((i + (k * j)) % 5) + 1 for j in range(5))
        for i in range(5)
    ]


def _cyclic_baseline_rows() -> list[tuple[int, int, int, int, int]]:
    return [
        tuple(((i + j) % 5) + 1 for j in range(5))
        for i in range(5)
    ]


def latin_squares_25_indices() -> list[tuple[int, int, int, int, int]]:
    """Return 25 deterministic permutations from five Latin squares.

    Construction:
    - Four squares from ``cell(i,j) = (i + k*j) mod 5 + 1`` for ``k=1..4``.
    - One cyclic baseline from ``cell(i,j) = (i + j) mod 5 + 1`` with a fixed
      symbol relabel to avoid duplicating ``k=1`` rows while preserving Latin
      structure and per-position balance.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    for k in (1, 2, 3, 4):
        rows.extend(_latin_square_rows_for_k(k))

    remapped_baseline = [
        tuple(_BASELINE_REMAP[value] for value in row)
        for row in _cyclic_baseline_rows()
    ]
    rows.extend(remapped_baseline)

    distinct = {tuple(row) for row in rows}
    if len(distinct) != 25:
        raise RuntimeError(
            "latin_squares_25 construction must yield 25 unique permutations"
        )
    return rows


def single_latin_square_5_indices() -> list[tuple[int, int, int, int, int]]:
    """Return one deterministic five-row Latin-square slice."""
    return _cyclic_baseline_rows()


def resolve_permutation_sigmas(
    strategy: PermutationStrategy,
) -> list[tuple[int, int, int, int, int]]:
    """Resolve deterministic sigma list for one permutation strategy."""
    if strategy == "full_120":
        return full_120_indices()
    if strategy == "latin_squares_25":
        return latin_squares_25_indices()
    return single_latin_square_5_indices()


def _canonical_permutation_index_map() -> dict[tuple[int, int, int, int, int], int]:
    return {sigma: idx for idx, sigma in enumerate(full_120_indices())}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value)


def _has_picture_reference(row: pd.Series, columns: Sequence[str]) -> bool:
    for column in columns:
        if column not in row.index:
            continue
        if _PICTURE_REF_REGEX.search(_to_text(row.get(column))):
            return True
    return False


def _has_any_image_markup(row: pd.Series, all_columns: Sequence[str]) -> bool:
    for column in all_columns:
        text = _to_text(row.get(column))
        if not text:
            continue
        if _IMG_TAG_REGEX.search(text):
            return True
        if _MARKDOWN_IMAGE_REGEX.search(text):
            return True
        if _EXTERNAL_IMAGE_URL_REGEX.search(text):
            return True
    return False


def apply_midterm2_subset_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented midterm2 clean subset filter in-runner."""
    missing = [column for column in _REQUIRED_DATASET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"midterm_subset_missing_required_columns:{missing}")

    filtered = frame.copy()
    item_type = filtered["ItemType"].astype(str).str.strip().str.lower()
    filtered = filtered[item_type == "choice"].copy()

    picture_cols = [col for col in _MIDTERM_FILTER_COLUMNS if col in filtered.columns]
    image_scan_cols = list(filtered.columns)

    has_picture_refs = filtered.apply(
        lambda row: _has_picture_reference(row, picture_cols),
        axis=1,
    )
    has_images = filtered.apply(
        lambda row: _has_any_image_markup(row, image_scan_cols),
        axis=1,
    )
    filtered = filtered[~(has_picture_refs | has_images)].copy()
    filtered = filtered.reset_index(drop=True)
    return filtered


def _call_artifact_uri_by_id(context: MethodRunnerContext, artifact_id: int) -> str:
    artifact = (
        context.repository.session.query(CallArtifact)
        .filter(CallArtifact.id == int(artifact_id))
        .first()
    )
    if artifact is None:
        raise ValueError(f"CallArtifact id={artifact_id} not found")
    return str(artifact.uri)


def _safe_token(value: str) -> str:
    token = _SAFE_TOKEN_REGEX.sub("_", str(value or "").strip())
    return token or "value"


def _extract_status_code(exc: Exception) -> int | None:
    direct = getattr(exc, "status_code", None)
    if direct is not None:
        try:
            return int(direct)
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                return None
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    headers: Mapping[str, Any] | None = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not isinstance(headers, Mapping):
        return None

    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return True
    message = str(exc).lower()
    return "timeout" in message


async def _run_inference_with_429_retry(
    *,
    service: InferenceService,
    prompt: str,
    top_logprobs: int,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> InferenceAttemptOutcome:
    """Run one inference with explicit 429/5xx/timeout policy."""
    rate_limit_retries = 0
    total_wait_ms = 0
    server_error_retried = False
    timeout_retried = False

    while True:
        try:
            inference_result = await service.run_inference(
                prompt=prompt,
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=int(top_logprobs),
            )
            return InferenceAttemptOutcome(
                inference_result=inference_result,
                error=None,
                rate_limit_retry_count=rate_limit_retries,
                rate_limit_total_wait_ms=total_wait_ms,
            )
        except Exception as exc:
            status_code = _extract_status_code(exc)

            if status_code in {401, 403, 404}:
                raise HardFailureError(f"hard_fail_http_{status_code}") from exc

            if status_code == 429:
                if rate_limit_retries >= len(_RATE_LIMIT_BACKOFF_SECONDS):
                    return InferenceAttemptOutcome(
                        inference_result=None,
                        error="rate_limited",
                        rate_limit_retry_count=rate_limit_retries,
                        rate_limit_total_wait_ms=total_wait_ms,
                    )
                retry_after = _extract_retry_after_seconds(exc)
                wait_seconds = (
                    retry_after
                    if retry_after is not None
                    else _RATE_LIMIT_BACKOFF_SECONDS[rate_limit_retries]
                )
                rate_limit_retries += 1
                total_wait_ms += int(wait_seconds * 1000)
                await sleeper(wait_seconds)
                continue

            if status_code is not None and status_code >= 500:
                if not server_error_retried:
                    server_error_retried = True
                    await sleeper(1.0)
                    continue
                return InferenceAttemptOutcome(
                    inference_result=None,
                    error=f"http_{status_code}",
                    rate_limit_retry_count=rate_limit_retries,
                    rate_limit_total_wait_ms=total_wait_ms,
                )

            if _is_timeout_error(exc):
                if not timeout_retried:
                    timeout_retried = True
                    await sleeper(1.0)
                    continue
                return InferenceAttemptOutcome(
                    inference_result=None,
                    error="timeout",
                    rate_limit_retry_count=rate_limit_retries,
                    rate_limit_total_wait_ms=total_wait_ms,
                )

            if status_code is not None and 400 <= status_code < 500:
                return InferenceAttemptOutcome(
                    inference_result=None,
                    error=f"http_{status_code}",
                    rate_limit_retry_count=rate_limit_retries,
                    rate_limit_total_wait_ms=total_wait_ms,
                )

            return InferenceAttemptOutcome(
                inference_result=None,
                error=type(exc).__name__.lower(),
                rate_limit_retry_count=rate_limit_retries,
                rate_limit_total_wait_ms=total_wait_ms,
            )


def _format_prompt(
    row: Mapping[str, Any],
    sigma: tuple[int, int, int, int, int],
    format_idx: int,
) -> tuple[str, str]:
    options = [_to_text(row[f"Option{label}"]) for label in _OPTION_LABELS]
    original_correct_letter = _to_text(row.get("CorrectLetter")).strip().upper()
    if original_correct_letter not in _OPTION_LABELS:
        raise ValueError(f"invalid_correct_letter:{original_correct_letter!r}")

    original_correct_idx = _OPTION_LABELS.index(original_correct_letter)
    permuted_options = [options[idx - 1] for idx in sigma]
    new_correct_idx = sigma.index(original_correct_idx + 1)
    new_correct_letter = _OPTION_LABELS[new_correct_idx]

    options_block = "\n".join(
        f"{label}. {permuted_options[idx]}"
        for idx, label in enumerate(_OPTION_LABELS)
    )
    suffix = _FORMAT_SUFFIXES[int(format_idx)]
    prompt = f"{_to_text(row['ProblemBody'])}\n\n{options_block}\n\n{suffix}"
    return prompt, new_correct_letter


def _coerce_top_logprob_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        payload = dict(entry)
    elif hasattr(entry, "model_dump"):
        payload = dict(entry.model_dump())
    else:
        payload = {
            "token": getattr(entry, "token", None),
            "logprob": getattr(entry, "logprob", None),
            "token_id": getattr(entry, "token_id", None),
        }
    return {
        "token": payload.get("token"),
        "logprob": payload.get("logprob"),
        "token_id": payload.get("token_id"),
    }


def _extract_first_token_top_logprobs(inference_result: dict[str, Any]) -> list[dict[str, Any]]:
    provider_response = inference_result.get("provider_response")
    raw_response = getattr(provider_response, "raw_response", None)
    if raw_response is None:
        return []
    choices = getattr(raw_response, "choices", None)
    if not choices:
        return []
    first_choice = choices[0]
    logprobs_obj = getattr(first_choice, "logprobs", None)
    content = getattr(logprobs_obj, "content", None)
    if not content:
        return []
    first_token = content[0]
    top_logprobs = getattr(first_token, "top_logprobs", None)
    if not top_logprobs:
        return []
    return [_coerce_top_logprob_entry(item) for item in top_logprobs]


def _token_matches_letter(token: Any, letter: str) -> bool:
    normalized = _to_text(token).strip().upper()
    return normalized == letter


def _candidate_logprobs_from_top(
    top_logprobs: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {label: None for label in _OPTION_LABELS}
    for label in _OPTION_LABELS:
        values: list[float] = []
        for entry in top_logprobs:
            if not _token_matches_letter(entry.get("token"), label):
                continue
            raw = entry.get("logprob")
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        out[label] = max(values) if values else None
    return out


def _predict_letter(candidate_logprobs: Mapping[str, float | None]) -> str | None:
    best_letter: str | None = None
    best_value: float | None = None
    for label in _OPTION_LABELS:
        value = candidate_logprobs.get(label)
        if value is None:
            continue
        if best_value is None or float(value) > best_value:
            best_value = float(value)
            best_letter = label
    return best_letter


def _load_imported_dataframe(context: MethodRunnerContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    if context.imported_run_id is None:
        raise ValueError("mcq_logprob_basic_requires_imported_run_id")
    imported_run = context.repository.get_provenanced_run_by_id(int(context.imported_run_id))
    if imported_run is None:
        raise ValueError(f"imported_run_not_found:{context.imported_run_id}")
    result_ref = _to_text(imported_run.result_ref).strip()
    if not result_ref:
        raise ValueError(f"imported_run_missing_result_ref:{context.imported_run_id}")

    artifact_service = ArtifactService(repository=context.repository)
    parquet_bytes = artifact_service.storage.read_from_uri(result_ref)
    frame = pd.read_parquet(io.BytesIO(parquet_bytes), engine="pyarrow")
    metadata = dict(imported_run.metadata_json or {})
    return frame, metadata


def _permutation_plan(
    strategy: PermutationStrategy,
) -> list[tuple[int, tuple[int, int, int, int, int]]]:
    canonical_idx = _canonical_permutation_index_map()
    out: list[tuple[int, tuple[int, int, int, int, int]]] = []
    for sigma in resolve_permutation_sigmas(strategy):
        idx = canonical_idx.get(tuple(sigma))
        if idx is None:
            raise ValueError(f"unknown_permutation_sigma:{sigma}")
        out.append((int(idx), tuple(sigma)))
    return out


async def run_mcq_logprob_basic(
    parameters: Dict[str, Any],
    context: MethodRunnerContext,
) -> MethodRunnerResult:
    """Run the Phase-1 MCQ logprob skeleton and persist one parquet artifact."""
    try:
        parsed = McqLogprobBasicParams.model_validate(parameters or {})
    except ValidationError as exc:
        raise ValueError(f"invalid_mcq_logprob_basic_parameters: {exc}") from exc

    full_frame, imported_meta = _load_imported_dataframe(context)
    subset = apply_midterm2_subset_filter(full_frame)
    if parsed.max_questions is not None:
        subset = subset.head(int(parsed.max_questions)).reset_index(drop=True)
    if subset.empty:
        raise ValueError("midterm_subset_empty_after_filters")

    permutation_plan = _permutation_plan(parsed.permutation_strategy)

    provider = ProviderFactory().create_chat_provider(parsed.provider, parsed.model)
    service = InferenceService(provider=provider, repository=None, max_retries=1)
    semaphore = asyncio.Semaphore(int(parsed.concurrency_cap))
    output_rows: list[dict[str, Any]] = []

    async def _run_one(
        question_idx: int,
        row: Mapping[str, Any],
        permutation_idx: int,
        sigma: tuple[int, int, int, int, int],
    ) -> dict[str, Any]:
        prompt, correct_letter = _format_prompt(row, sigma, parsed.format_idx)
        async with semaphore:
            outcome = await _run_inference_with_429_retry(
                service=service,
                prompt=prompt,
                top_logprobs=int(parsed.top_logprobs),
            )

        if outcome.inference_result is None:
            return {
                "question_idx": int(question_idx),
                "item_id": int(row["ItemID"]),
                "temp_bank_id": int(row["temp_bank_id"]),
                "permutation_idx": int(permutation_idx),
                "permutation_sigma": list(sigma),
                "format_idx": int(parsed.format_idx),
                "prompt_template_version": parsed.prompt_template_version,
                "first_token_top_logprobs_json": "[]",
                "candidate_logprobs_json": json.dumps(
                    {label: None for label in _OPTION_LABELS},
                    sort_keys=True,
                ),
                "predicted_letter": None,
                "correct_letter": str(correct_letter),
                "is_correct": None,
                "error": str(outcome.error or "unknown_error"),
                "rate_limit_retry_count": int(outcome.rate_limit_retry_count),
                "rate_limit_total_wait_ms": int(outcome.rate_limit_total_wait_ms),
            }

        top_logprobs = _extract_first_token_top_logprobs(outcome.inference_result)
        candidate_logprobs = _candidate_logprobs_from_top(top_logprobs)
        predicted_letter = _predict_letter(candidate_logprobs)
        is_correct = bool(predicted_letter == correct_letter) if predicted_letter else None

        return {
            "question_idx": int(question_idx),
            "item_id": int(row["ItemID"]),
            "temp_bank_id": int(row["temp_bank_id"]),
            "permutation_idx": int(permutation_idx),
            "permutation_sigma": list(sigma),
            "format_idx": int(parsed.format_idx),
            "prompt_template_version": parsed.prompt_template_version,
            "first_token_top_logprobs_json": json.dumps(top_logprobs, sort_keys=True),
            "candidate_logprobs_json": json.dumps(candidate_logprobs, sort_keys=True),
            "predicted_letter": predicted_letter,
            "correct_letter": str(correct_letter),
            "is_correct": is_correct,
            "error": None,
            "rate_limit_retry_count": int(outcome.rate_limit_retry_count),
            "rate_limit_total_wait_ms": int(outcome.rate_limit_total_wait_ms),
        }

    try:
        tasks: list[asyncio.Task[dict[str, Any]]] = []
        for question_idx, row in enumerate(subset.to_dict(orient="records")):
            for permutation_idx, sigma in permutation_plan:
                tasks.append(
                    asyncio.create_task(
                        _run_one(
                            question_idx=question_idx,
                            row=row,
                            permutation_idx=permutation_idx,
                            sigma=sigma,
                        )
                    )
                )

        for completed in asyncio.as_completed(tasks):
            try:
                output_rows.append(await completed)
            except HardFailureError:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    finally:
        await service.close()

    artifact_frame = pd.DataFrame(output_rows)
    artifact_frame = artifact_frame.sort_values(
        by=["question_idx", "permutation_idx", "format_idx"],
        kind="stable",
    ).reset_index(drop=True)

    parquet_buffer = io.BytesIO()
    artifact_frame.to_parquet(parquet_buffer, engine="pyarrow", index=False)
    parquet_bytes = parquet_buffer.getvalue()

    dataset_name = _to_text(imported_meta.get("dataset_name") or "midterm2_questions")
    dataset_version = _to_text(imported_meta.get("dataset_version") or "v1")
    logical_filename = (
        f"mcq_logprob_{_safe_token(parsed.model)}_"
        f"{parsed.permutation_strategy}_{parsed.format_idx}.parquet"
    )
    artifact_service = ArtifactService(repository=context.repository)
    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=int(context.source_group_id),
        step_name="mcq_logprob_basic",
        logical_filename=logical_filename,
        data=parquet_bytes,
        artifact_type="mcq_logprob_rows_parquet",
        content_type="application/octet-stream",
        metadata={
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "imported_run_id": int(context.imported_run_id or 0),
            "model": parsed.model,
            "provider": parsed.provider,
            "permutation_strategy": parsed.permutation_strategy,
            "prompt_template_version": parsed.prompt_template_version,
            "row_count": int(len(artifact_frame)),
        },
    )
    artifact_uri = _call_artifact_uri_by_id(context, int(artifact_id))

    caller_meta = dict(parsed.metadata or {})
    runner_meta = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "imported_run_id": int(context.imported_run_id or 0),
        "provider": parsed.provider,
        "model": parsed.model,
        "permutation_strategy": parsed.permutation_strategy,
        "prompt_template_version": parsed.prompt_template_version,
    }
    # Caller keys win unless they collide with runner-protected keys (runner invariants win).
    merged_metadata: dict[str, Any] = {
        k: v for k, v in caller_meta.items() if k not in _MCQ_LOGPROB_RUNNER_METADATA_PROTECTED_KEYS
    }
    merged_metadata.update(runner_meta)

    return MethodRunnerResult(
        output_json={
            "artifact_id": int(artifact_id),
            "artifact_uri": artifact_uri,
            "row_count": int(len(artifact_frame)),
            "question_count": int(len(subset)),
            "permutation_count": int(len(permutation_plan)),
        },
        pipeline_stage_role="export",
        result_ref=artifact_uri,
        pipeline_stage_context={
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "imported_run_id": int(context.imported_run_id or 0),
            "model": parsed.model,
            "permutation_strategy": parsed.permutation_strategy,
            "format_idx": int(parsed.format_idx),
            "row_count": int(len(artifact_frame)),
        },
        metadata_json=merged_metadata,
    )


async def probe_rate_limits_per_model(
    models: Sequence[str],
    target_concurrencies: Mapping[str, int],
    *,
    provider: str = "openrouter",
    trial_repetitions: int = 2,
    prompt: str = "Answer with a single letter A-E.\n\nAnswer:",
    call_probe: Callable[[str], Awaitable[ProbeCallResult]] | None = None,
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, ProbeResult]:
    """Probe per-model rate-limit tolerance with reachability + ramp tiers."""
    managed_services: dict[str, InferenceService] = {}

    async def _default_call_probe(model: str) -> ProbeCallResult:
        service = managed_services.get(model)
        if service is None:
            chat_provider = ProviderFactory().create_chat_provider(provider, model)
            service = InferenceService(provider=chat_provider, repository=None, max_retries=1)
            managed_services[model] = service

        try:
            outcome = await _run_inference_with_429_retry(
                service=service,
                prompt=prompt,
                top_logprobs=5,
                sleeper=sleeper,
            )
        except HardFailureError as exc:
            return ProbeCallResult(outcome="hard_fail", error=str(exc))

        if outcome.inference_result is not None:
            return ProbeCallResult(
                outcome="ok",
                retry_count=int(outcome.rate_limit_retry_count),
                wait_ms=int(outcome.rate_limit_total_wait_ms),
            )
        if outcome.error == "rate_limited":
            return ProbeCallResult(
                outcome="rate_limited",
                retry_count=int(outcome.rate_limit_retry_count),
                wait_ms=int(outcome.rate_limit_total_wait_ms),
                error=outcome.error,
            )
        return ProbeCallResult(
            outcome="other_error",
            retry_count=int(outcome.rate_limit_retry_count),
            wait_ms=int(outcome.rate_limit_total_wait_ms),
            error=str(outcome.error or "unknown_error"),
        )

    probe_caller = call_probe or _default_call_probe
    results: dict[str, ProbeResult] = {}
    try:
        for model in models:
            ceiling = max(1, int(target_concurrencies.get(model, 1)))
            reachability = await probe_caller(model)
            if reachability.outcome == "hard_fail":
                results[str(model)] = ProbeResult(
                    model=str(model),
                    resolved_concurrency=1,
                    excluded_reason="EXCLUDED:auth/missing",
                    reachability_outcome=reachability.outcome,
                    tier_stats=tuple(),
                )
                continue
            if reachability.outcome == "rate_limited":
                results[str(model)] = ProbeResult(
                    model=str(model),
                    resolved_concurrency=1,
                    excluded_reason="EXCLUDED:rate_limited_at_concurrency_1",
                    reachability_outcome=reachability.outcome,
                    tier_stats=tuple(),
                )
                continue
            if reachability.outcome != "ok":
                results[str(model)] = ProbeResult(
                    model=str(model),
                    resolved_concurrency=1,
                    excluded_reason="EXCLUDED:reachability_failed",
                    reachability_outcome=reachability.outcome,
                    tier_stats=tuple(),
                )
                continue

            resolved_concurrency = 1
            tier_stats: list[ProbeTierStats] = []
            for tier in (2, 4, 8):
                if tier > ceiling:
                    break
                call_count = int(tier * trial_repetitions)
                call_results = await asyncio.gather(
                    *[probe_caller(model) for _ in range(call_count)]
                )

                if any(item.outcome == "hard_fail" for item in call_results):
                    results[str(model)] = ProbeResult(
                        model=str(model),
                        resolved_concurrency=max(1, resolved_concurrency),
                        excluded_reason="EXCLUDED:auth/missing",
                        reachability_outcome=reachability.outcome,
                        tier_stats=tuple(tier_stats),
                    )
                    break

                rate_limited_calls = sum(
                    1 for item in call_results if item.outcome == "rate_limited"
                )
                total_retry_count = sum(int(item.retry_count) for item in call_results)
                total_wait_ms = sum(int(item.wait_ms) for item in call_results)
                pass_ratio = 0.0 if call_count == 0 else rate_limited_calls / call_count
                passed = pass_ratio <= 0.25

                tier_stats.append(
                    ProbeTierStats(
                        concurrency=int(tier),
                        total_calls=int(call_count),
                        rate_limited_calls=int(rate_limited_calls),
                        total_retry_count=int(total_retry_count),
                        total_wait_ms=int(total_wait_ms),
                        passed=bool(passed),
                    )
                )

                if passed:
                    resolved_concurrency = int(tier)
                    continue
                break
            else:
                results[str(model)] = ProbeResult(
                    model=str(model),
                    resolved_concurrency=max(1, min(resolved_concurrency, ceiling)),
                    excluded_reason=None,
                    reachability_outcome=reachability.outcome,
                    tier_stats=tuple(tier_stats),
                )
                continue

            if str(model) not in results:
                results[str(model)] = ProbeResult(
                    model=str(model),
                    resolved_concurrency=max(1, min(resolved_concurrency, ceiling)),
                    excluded_reason=None,
                    reachability_outcome=reachability.outcome,
                    tier_stats=tuple(tier_stats),
                )
    finally:
        if call_probe is None:
            await asyncio.gather(
                *[service.close() for service in managed_services.values()],
                return_exceptions=True,
            )

    return results


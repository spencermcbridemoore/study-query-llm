"""Tests for Phase-1 MCQ logprob runner foundations."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import CallArtifact
from study_query_llm.db.raw_call_repository import RawCallRepository
from study_query_llm.providers.base import BaseLLMProvider, ProviderResponse
from study_query_llm.services.artifact_service import ArtifactService
from study_query_llm.services.method_runners.mcq_logprob_basic import (
    HardFailureError,
    ProbeCallResult,
    _token_matches_letter,
    _run_inference_with_429_retry,
    apply_midterm2_subset_filter,
    full_120_indices,
    latin_squares_25_indices,
    probe_rate_limits_per_model,
    resolve_permutation_sigmas,
    run_mcq_logprob_basic,
)
from study_query_llm.services.method_runtime_registry import MethodRunnerContext


@pytest.fixture
def db_connection():
    db = DatabaseConnectionV2("sqlite:///:memory:", enable_pgvector=False)
    db.init_db()
    return db


class _FakeHttpError(Exception):
    def __init__(self, status_code: int, headers: dict[str, Any] | None = None) -> None:
        super().__init__(f"http {status_code}")
        hdrs = dict(headers or {})
        self.status_code = int(status_code)
        self.response = SimpleNamespace(
            status_code=int(status_code),
            headers=hdrs,
        )


class _ScriptedInferenceService:
    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = list(outcomes)

    async def run_inference(self, **_kwargs: Any) -> dict[str, Any]:
        if not self._outcomes:
            raise AssertionError("scripted outcomes exhausted")
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return dict(item)


class _CapturingProvider(BaseLLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        self.calls.append({"prompt": prompt, "kwargs": dict(kwargs)})
        top_logprobs = [
            SimpleNamespace(token="A", logprob=-0.1, token_id=101),
            SimpleNamespace(token="B", logprob=-1.4, token_id=102),
        ]
        raw_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A"),
                    finish_reason="stop",
                    logprobs=SimpleNamespace(
                        content=[SimpleNamespace(top_logprobs=top_logprobs)]
                    ),
                )
            ],
            usage=SimpleNamespace(total_tokens=7, prompt_tokens=6, completion_tokens=1),
        )
        return ProviderResponse(
            text="A",
            provider="capturing",
            tokens=7,
            latency_ms=1.0,
            metadata={"finish_reason": "stop"},
            raw_response=raw_response,
        )

    def get_provider_name(self) -> str:
        return "capturing"


def _build_midterm_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ItemID": 101,
                "ItemType": "choice",
                "ProblemBody": "A block slides on a rough incline.",
                "ProblemBodyTemplate": "",
                "OptionA": "A",
                "OptionB": "B",
                "OptionC": "C",
                "OptionD": "D",
                "OptionE": "E",
                "CorrectLetter": "A",
                "temp_bank_id": 6445,
            },
            {
                "ItemID": 102,
                "ItemType": "multi-answer",
                "ProblemBody": "This row should be filtered by item type.",
                "ProblemBodyTemplate": "",
                "OptionA": "A",
                "OptionB": "B",
                "OptionC": "C",
                "OptionD": "D",
                "OptionE": "E",
                "CorrectLetter": "A",
                "temp_bank_id": 6445,
            },
            {
                "ItemID": 103,
                "ItemType": "choice",
                "ProblemBody": "Refer to Figure 1 for this force diagram.",
                "ProblemBodyTemplate": "",
                "OptionA": "A",
                "OptionB": "B",
                "OptionC": "C",
                "OptionD": "D",
                "OptionE": "E",
                "CorrectLetter": "B",
                "temp_bank_id": 6438,
            },
            {
                "ItemID": 104,
                "ItemType": "choice",
                "ProblemBody": "Text looks fine.",
                "ProblemBodyTemplate": "",
                "OptionA": "![img](https://example.com/a.png)",
                "OptionB": "B",
                "OptionC": "C",
                "OptionD": "D",
                "OptionE": "E",
                "CorrectLetter": "C",
                "temp_bank_id": 6438,
            },
        ]
    )


def _seed_imported_run(
    *,
    repo: RawCallRepository,
    request_group_id: int,
    artifact_dir: str,
    frame: pd.DataFrame,
) -> int:
    artifact_service = ArtifactService(repository=repo, artifact_dir=artifact_dir)
    buf = io.BytesIO()
    frame.to_parquet(buf, engine="pyarrow", index=False)
    artifact_id = artifact_service.store_group_blob_artifact(
        group_id=int(request_group_id),
        step_name="csv_parse",
        logical_filename="midterm2_questions_v1.parquet",
        data=buf.getvalue(),
        artifact_type="dataset_canonical_parquet",
        content_type="application/octet-stream",
        metadata={"dataset_name": "midterm2_questions", "dataset_version": "v1"},
    )
    artifact = (
        repo.session.query(CallArtifact).filter(CallArtifact.id == int(artifact_id)).first()
    )
    assert artifact is not None
    return int(
        repo.create_provenanced_run(
            run_kind="execution",
            run_status="completed",
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            run_key="imported-midterm2",
            result_ref=str(artifact.uri),
            metadata_json={
                "execution_role": "method_execution",
                "pipeline_stage_role": "imported",
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
    )


def test_permutation_helpers_are_deterministic_and_distinct():
    full = full_120_indices()
    assert len(full) == 120
    assert full[0] == (1, 2, 3, 4, 5)
    assert full[-1] == (5, 4, 3, 2, 1)

    latin = latin_squares_25_indices()
    assert len(latin) == 25
    assert len(set(tuple(p) for p in latin_squares_25_indices())) == 25
    assert len(set(tuple(p) for p in latin)) == 25
    assert latin == latin_squares_25_indices()

    single = resolve_permutation_sigmas("single_latin_square_5")
    assert len(single) == 5
    assert all(len(sigma) == 5 for sigma in single)


def test_apply_midterm_filter_excludes_non_choice_and_image_rows():
    frame = _build_midterm_like_frame()
    subset = apply_midterm2_subset_filter(frame)
    assert list(subset["ItemID"].astype(int)) == [101]
    assert set(subset["ItemType"].astype(str).str.lower()) == {"choice"}


def test_token_matches_letter_handles_supported_prefix_markers():
    assert _token_matches_letter("▁A", "A") is True
    assert _token_matches_letter("ĠA", "A") is True
    assert _token_matches_letter(" A", "A") is True
    assert _token_matches_letter("A", "A") is True
    assert _token_matches_letter("a", "A") is True
    assert _token_matches_letter("(A", "A") is False
    assert _token_matches_letter("AA", "A") is False


@pytest.mark.asyncio
async def test_runner_rejects_unknown_parameters():
    context = MethodRunnerContext(
        repository=None,  # type: ignore[arg-type]
        request_group_id=1,
        source_group_id=1,
        method_name="inference.mcq_logprob.basic",
        method_version="0.1",
        run_key="rk",
    )
    with pytest.raises(ValueError, match="invalid_mcq_logprob_basic_parameters"):
        await run_mcq_logprob_basic(
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "unknown_param": "boom",
            },
            context,
        )


@pytest.mark.asyncio
async def test_rate_limit_retry_uses_retry_after_header():
    waits: list[float] = []

    async def _sleep(seconds: float) -> None:
        waits.append(float(seconds))

    service = _ScriptedInferenceService(
        outcomes=[
            _FakeHttpError(429, headers={"Retry-After": "3"}),
            {"provider_response": SimpleNamespace(raw_response=None)},
        ]
    )
    out = await _run_inference_with_429_retry(
        service=service,  # type: ignore[arg-type]
        prompt="p",
        top_logprobs=5,
        sleeper=_sleep,
    )
    assert out.inference_result is not None
    assert out.error is None
    assert out.rate_limit_retry_count == 1
    assert out.rate_limit_total_wait_ms == 3000
    assert waits == [3.0]


@pytest.mark.asyncio
async def test_rate_limit_retry_stops_at_retry_cap():
    waits: list[float] = []

    async def _sleep(seconds: float) -> None:
        waits.append(float(seconds))

    service = _ScriptedInferenceService(
        outcomes=[_FakeHttpError(429) for _ in range(6)]
    )
    out = await _run_inference_with_429_retry(
        service=service,  # type: ignore[arg-type]
        prompt="p",
        top_logprobs=5,
        sleeper=_sleep,
    )
    assert out.inference_result is None
    assert out.error == "rate_limited"
    assert out.rate_limit_retry_count == 4
    assert out.rate_limit_total_wait_ms == int((1 + 2 + 4 + 8) * 1000)
    assert waits == [1.0, 2.0, 4.0, 8.0]


@pytest.mark.asyncio
async def test_probe_ramp_acceptance_uses_previous_passing_tier():
    scripted_outcomes = [
        ProbeCallResult(outcome="ok"),  # reachability
        # tier=2, repetitions=2 -> 4 calls (all pass)
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        # tier=4, repetitions=2 -> 8 calls, 3 rate-limited => fail (>25%)
        ProbeCallResult(outcome="rate_limited"),
        ProbeCallResult(outcome="rate_limited"),
        ProbeCallResult(outcome="rate_limited"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
        ProbeCallResult(outcome="ok"),
    ]

    async def _call_probe(_model: str) -> ProbeCallResult:
        if not scripted_outcomes:
            raise AssertionError("scripted outcomes exhausted")
        return scripted_outcomes.pop(0)

    result = await probe_rate_limits_per_model(
        models=["openai/gpt-4o-mini"],
        target_concurrencies={"openai/gpt-4o-mini": 8},
        trial_repetitions=2,
        call_probe=_call_probe,
    )
    model_result = result["openai/gpt-4o-mini"]
    assert model_result.excluded_reason is None
    assert model_result.resolved_concurrency == 2
    assert len(model_result.tier_stats) == 2
    assert model_result.tier_stats[0].passed is True
    assert model_result.tier_stats[1].passed is False
    assert model_result.tier_stats[1].rate_limited_calls == 3


@pytest.mark.asyncio
async def test_runner_minimal_end_to_end_writes_parquet_artifact(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())

    frame = _build_midterm_like_frame()

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="mcq-logprob-phase1-test",
                metadata_json={},
            )
        )
        imported_run_id = _seed_imported_run(
            repo=repo,
            request_group_id=request_group_id,
            artifact_dir=artifact_dir,
            frame=frame,
        )

        provider = _CapturingProvider()
        from study_query_llm.providers.factory import ProviderFactory

        monkeypatch.setattr(
            ProviderFactory,
            "create_chat_provider",
            lambda _self, provider_name, model: provider,
        )

        context = MethodRunnerContext(
            repository=repo,
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            method_name="inference.mcq_logprob.basic",
            method_version="0.1",
            run_key="rk",
            imported_run_id=int(imported_run_id),
            imported_run_metadata={
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
        out = await run_mcq_logprob_basic(
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "permutation_strategy": "single_latin_square_5",
                "format_idx": 0,
                "concurrency_cap": 2,
                "max_questions": 1,
            },
            context,
        )

        assert out.pipeline_stage_role == "export"
        assert out.result_ref is not None
        assert out.output_json["row_count"] == 5
        assert out.output_json["question_count"] == 1
        assert len(provider.calls) == 5
        assert all("system_prompt" not in call["kwargs"] for call in provider.calls)

        artifact_service = ArtifactService(repository=repo, artifact_dir=artifact_dir)
        artifact_bytes = artifact_service.storage.read_from_uri(str(out.result_ref))
        artifact_frame = pd.read_parquet(io.BytesIO(artifact_bytes), engine="pyarrow")
        assert len(artifact_frame) == 5
        assert set(artifact_frame["item_id"].astype(int)) == {101}
        assert set(artifact_frame["error"].dropna().astype(str)) == set()
        assert set(artifact_frame["rate_limit_retry_count"].astype(int)) == {0}


@pytest.mark.asyncio
async def test_runner_threads_system_prompt_when_present(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())

    frame = _build_midterm_like_frame()

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="mcq-logprob-system-prompt-test",
                metadata_json={},
            )
        )
        imported_run_id = _seed_imported_run(
            repo=repo,
            request_group_id=request_group_id,
            artifact_dir=artifact_dir,
            frame=frame,
        )

        provider = _CapturingProvider()
        from study_query_llm.providers.factory import ProviderFactory

        monkeypatch.setattr(
            ProviderFactory,
            "create_chat_provider",
            lambda _self, provider_name, model: provider,
        )

        context = MethodRunnerContext(
            repository=repo,
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            method_name="inference.mcq_logprob.basic",
            method_version="0.1",
            run_key="rk",
            imported_run_id=int(imported_run_id),
            imported_run_metadata={
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
        system_prompt = "Reply with exactly one letter."
        out = await run_mcq_logprob_basic(
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "permutation_strategy": "single_latin_square_5",
                "format_idx": 0,
                "concurrency_cap": 2,
                "max_questions": 1,
                "system_prompt": system_prompt,
            },
            context,
        )

        assert out.output_json["row_count"] == 5
        assert len(provider.calls) == 5
        assert all(call["kwargs"].get("system_prompt") == system_prompt for call in provider.calls)


@pytest.mark.asyncio
async def test_runner_metadata_merges_experiment_label_from_parameters_metadata(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    frame = _build_midterm_like_frame()

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="mcq-metadata-merge-test",
                metadata_json={},
            )
        )
        imported_run_id = _seed_imported_run(
            repo=repo,
            request_group_id=request_group_id,
            artifact_dir=artifact_dir,
            frame=frame,
        )

        provider = _CapturingProvider()
        from study_query_llm.providers.factory import ProviderFactory

        monkeypatch.setattr(
            ProviderFactory,
            "create_chat_provider",
            lambda _self, provider_name, model: provider,
        )

        context = MethodRunnerContext(
            repository=repo,
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            method_name="inference.mcq_logprob.basic",
            method_version="0.1",
            run_key="rk",
            imported_run_id=int(imported_run_id),
            imported_run_metadata={
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
        label = "mcq_logprob_format0_2026_05_09"
        out = await run_mcq_logprob_basic(
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "permutation_strategy": "single_latin_square_5",
                "format_idx": 0,
                "concurrency_cap": 2,
                "max_questions": 1,
                "metadata": {"experiment_label": label},
            },
            context,
        )

        assert out.metadata_json.get("experiment_label") == label
        assert out.metadata_json.get("dataset_name") == "midterm2_questions"


@pytest.mark.asyncio
async def test_runner_metadata_collision_runner_owned_keys_win(
    db_connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    monkeypatch.chdir(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    frame = _build_midterm_like_frame()

    with db_connection.session_scope() as session:
        repo = RawCallRepository(session)
        request_group_id = int(
            repo.create_group(
                group_type="analysis_request",
                name="mcq-metadata-collision-test",
                metadata_json={},
            )
        )
        imported_run_id = _seed_imported_run(
            repo=repo,
            request_group_id=request_group_id,
            artifact_dir=artifact_dir,
            frame=frame,
        )

        provider = _CapturingProvider()
        from study_query_llm.providers.factory import ProviderFactory

        monkeypatch.setattr(
            ProviderFactory,
            "create_chat_provider",
            lambda _self, provider_name, model: provider,
        )

        context = MethodRunnerContext(
            repository=repo,
            request_group_id=int(request_group_id),
            source_group_id=int(request_group_id),
            method_name="inference.mcq_logprob.basic",
            method_version="0.1",
            run_key="rk",
            imported_run_id=int(imported_run_id),
            imported_run_metadata={
                "dataset_name": "midterm2_questions",
                "dataset_version": "v1",
            },
        )
        out = await run_mcq_logprob_basic(
            {
                "provider": "openrouter",
                "model": "openai/gpt-4o-mini",
                "permutation_strategy": "single_latin_square_5",
                "format_idx": 0,
                "concurrency_cap": 2,
                "max_questions": 1,
                "metadata": {"dataset_name": "should_not_override", "experiment_label": "ok"},
            },
            context,
        )

        assert out.metadata_json.get("dataset_name") == "midterm2_questions"
        assert out.metadata_json.get("experiment_label") == "ok"


@pytest.mark.asyncio
async def test_hard_fail_status_raises_for_auth_or_missing():
    service = _ScriptedInferenceService(outcomes=[_FakeHttpError(401)])
    with pytest.raises(HardFailureError, match="hard_fail_http_401"):
        await _run_inference_with_429_retry(
            service=service,  # type: ignore[arg-type]
            prompt="p",
            top_logprobs=5,
        )


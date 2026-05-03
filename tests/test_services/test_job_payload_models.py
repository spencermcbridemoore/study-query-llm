"""Tests for job payload Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from study_query_llm.services.jobs import (
    FinalizeRunPayload,
    JobSnapshot,
    McqRunPayload,
    ReduceKPayload,
    RunKTryPayload,
    parse_analysis_run_payload,
    parse_finalize_run_payload,
    parse_job_snapshot,
    parse_mcq_run_payload,
    parse_reduce_k_payload,
    parse_run_k_try_payload,
)


def test_parse_job_snapshot_valid():
    raw = {
        "id": 1,
        "job_type": "run_k_try",
        "payload_json": {"embedding_engine": "e/a", "dataset": "dbpedia"},
        "job_key": "rk_1",
        "base_run_key": "base",
        "seed_value": 42,
    }
    snap = parse_job_snapshot(raw)
    assert snap.id == 1
    assert snap.job_type == "run_k_try"
    assert snap.payload_json["embedding_engine"] == "e/a"
    assert snap.job_key == "rk_1"


def test_parse_job_snapshot_minimal():
    raw = {"id": 1, "job_type": "reduce_k"}
    snap = parse_job_snapshot(raw)
    assert snap.id == 1
    assert snap.payload_json == {}
    assert snap.job_key == ""


def test_parse_job_snapshot_invalid():
    with pytest.raises(ValidationError):
        parse_job_snapshot({"job_type": "x"})  # missing id
    with pytest.raises(ValidationError):
        parse_job_snapshot({"id": "not_int", "job_type": "x"})  # id not int


def test_parse_run_k_try_payload_valid():
    raw = {
        "embedding_engine": "e/a",
        "dataset": "dbpedia",
        "summarizer": "None",
        "k_min": 2,
        "k_max": 20,
    }
    payload = parse_run_k_try_payload(raw)
    assert payload.embedding_engine == "e/a"
    assert payload.dataset == "dbpedia"
    assert payload.k_min == 2
    assert payload.k_max == 20


def test_parse_run_k_try_payload_defaults():
    raw = {"embedding_engine": "e", "dataset": "d"}
    payload = parse_run_k_try_payload(raw)
    assert payload.summarizer == "None"
    assert payload.k_min == 2
    assert payload.k_max == 20


def test_parse_run_k_try_payload_invalid():
    with pytest.raises(ValidationError):
        parse_run_k_try_payload({})  # missing embedding_engine, dataset
    with pytest.raises(ValidationError):
        parse_run_k_try_payload({"embedding_engine": "e"})  # missing dataset


def test_parse_mcq_run_payload_valid():
    payload = parse_mcq_run_payload(
        {
            "run_key": "rk1",
            "deployment": "gpt-4o-mini",
            "level": "high school",
            "subject": "physics",
            "options_per_question": 4,
            "questions_per_test": 20,
            "label_style": "upper",
            "spread_correct_answer_uniformly": False,
            "samples_per_combo": 50,
        }
    )
    assert payload.run_key == "rk1"
    assert payload.deployment == "gpt-4o-mini"
    assert payload.samples_per_combo == 50
    assert payload.determinism_class == "non_deterministic"


def test_parse_mcq_run_payload_invalid():
    with pytest.raises(ValidationError):
        parse_mcq_run_payload({"deployment": "gpt-4o-mini"})  # missing run_key/level/subject


def test_parse_analysis_run_payload_valid():
    payload = parse_analysis_run_payload(
        {
            "request_id": 42,
            "sweep_type": "mcq",
            "analysis_key": "mcq_compliance",
            "scope": "run",
            "method_name": "mcq_compliance_metrics",
            "method_version": "1.0",
            "required": True,
            "result_keys": ["format_compliance_rate"],
        }
    )
    assert payload.request_id == 42
    assert payload.sweep_type == "mcq"
    assert payload.analysis_key == "mcq_compliance"
    assert payload.required is True
    assert payload.run_key == ""
    assert payload.parameters == {}
    assert payload.force is False


def test_parse_analysis_run_payload_valid_clustering_fields():
    payload = parse_analysis_run_payload(
        {
            "request_id": 51,
            "sweep_type": "clustering",
            "analysis_key": "bundle_eval",
            "run_key": "rk_1",
            "method_name": "kmeans+normalize+pca+sweep",
            "method_version": "1.2",
            "parameters": {"top_n": 5},
            "force": True,
        }
    )
    assert payload.request_id == 51
    assert payload.sweep_type == "clustering"
    assert payload.analysis_key == "bundle_eval"
    assert payload.run_key == "rk_1"
    assert payload.parameters == {"top_n": 5}
    assert payload.force is True


def test_parse_analysis_run_payload_invalid():
    with pytest.raises(ValidationError):
        parse_analysis_run_payload({"request_id": 1})  # missing sweep_type/analysis_key


def test_parse_reduce_k_payload_valid():
    payload = parse_reduce_k_payload(
        {
            "run_key": "rk1",
            "dataset": "dbpedia",
            "embedding_engine": "engine/a",
            "summarizer": "None",
            "k_min": 2,
            "k_max": 3,
            "tries_per_k": 2,
        }
    )
    assert isinstance(payload, ReduceKPayload)
    assert payload.run_key == "rk1"
    assert payload.tries_per_k == 2


def test_parse_reduce_k_payload_invalid():
    with pytest.raises(ValidationError):
        parse_reduce_k_payload({"run_key": "rk1"})  # missing required fields


def test_parse_finalize_run_payload_valid():
    payload = parse_finalize_run_payload(
        {
            "run_key": "rk1",
            "dataset": "dbpedia",
            "embedding_engine": "engine/a",
            "summarizer": "None",
            "k_ranges": [[2, 3]],
            "tries_per_k": 2,
        }
    )
    assert isinstance(payload, FinalizeRunPayload)
    assert payload.run_key == "rk1"
    assert payload.k_ranges == [[2, 3]]


def test_parse_finalize_run_payload_invalid():
    with pytest.raises(ValidationError):
        parse_finalize_run_payload({"run_key": "rk1"})  # missing required fields

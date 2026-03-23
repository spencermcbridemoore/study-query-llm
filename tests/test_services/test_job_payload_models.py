"""Tests for job payload Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from study_query_llm.services.jobs import (
    JobSnapshot,
    RunKTryPayload,
    parse_job_snapshot,
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

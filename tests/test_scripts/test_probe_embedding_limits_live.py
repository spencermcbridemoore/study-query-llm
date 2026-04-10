"""Unit tests for probe_embedding_limits_live script helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "probe_embedding_limits_live.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("probe_embedding_limits_live_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load probe_embedding_limits_live script module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_int_csv_deduplicates_preserves_order() -> None:
    module = _load_script_module()
    out = module.parse_int_csv("1,8,8,32,1,64", arg_name="--vals")
    assert out == [1, 8, 32, 64]


def test_parse_int_csv_rejects_non_positive() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="positive integer"):
        module.parse_int_csv("1,0,4", arg_name="--vals")


def test_collect_limit_headers_filters_relevant_keys() -> None:
    module = _load_script_module()
    headers = {
        "x-ratelimit-remaining-tokens": "1000",
        "x-ratelimit-reset-seconds": "12",
        "content-type": "application/json",
        "x-request-id": "abc",
    }
    out = module.collect_limit_headers(headers)
    assert "x-ratelimit-remaining-tokens" in out
    assert "x-ratelimit-reset-seconds" in out
    assert "content-type" not in out
    assert "x-request-id" not in out


def test_summarize_step_computes_expected_rates() -> None:
    module = _load_script_module()
    step = module.StepConfig(
        step_index=1,
        platform="azure",
        batch_size=4,
        concurrency=2,
        duration_seconds=60,
    )
    events = [
        module.RequestEvent(
            timestamp_utc="2026-04-09T08:00:00.000Z",
            platform="azure",
            step_index=1,
            request_index=1,
            batch_size=4,
            concurrency=2,
            status_code=200,
            ok=True,
            latency_ms=100.0,
            est_input_tokens=40,
            usage_total_tokens=40,
            response_items_count=4,
            error_type="",
            error_message="",
            limit_headers={},
        ),
        module.RequestEvent(
            timestamp_utc="2026-04-09T08:01:00.000Z",
            platform="azure",
            step_index=1,
            request_index=2,
            batch_size=4,
            concurrency=2,
            status_code=429,
            ok=False,
            latency_ms=200.0,
            est_input_tokens=40,
            usage_total_tokens=None,
            response_items_count=None,
            error_type="HTTP_429",
            error_message="too many requests",
            limit_headers={"x-ratelimit-reset-seconds": "1"},
        ),
    ]
    summary = module.summarize_step(events, step, "completed")
    assert summary.attempted_requests == 2
    assert summary.success_requests == 1
    assert summary.throttled_requests == 1
    assert summary.error_requests == 1
    assert pytest.approx(summary.attempted_rpm, rel=1e-6) == 2.0
    assert pytest.approx(summary.success_rpm, rel=1e-6) == 1.0
    assert pytest.approx(summary.est_tpm_attempted, rel=1e-6) == 80.0
    assert pytest.approx(summary.est_tpm_success, rel=1e-6) == 40.0
    assert pytest.approx(summary.error_rate, rel=1e-6) == 0.5
    assert pytest.approx(summary.throttle_rate, rel=1e-6) == 0.5
    assert summary.sample_error.startswith("HTTP_429")


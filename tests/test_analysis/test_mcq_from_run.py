"""Unit tests for MCQ analysis helpers."""

from __future__ import annotations

from study_query_llm.analysis.mcq_from_run import (
    compliance_rates_from_probe,
    sweep_chi_square_vs_uniform,
    sweep_pooled_distribution,
)


def test_compliance_rates_from_probe():
    details = {
        "summary": {
            "samples_attempted": 10,
            "samples_with_successful_call": 9,
            "samples_with_valid_answer_key": 8,
            "heading_present_count": 7,
        }
    }
    r = compliance_rates_from_probe(details)
    assert r["answer_key_parse_rate"] == 0.8
    assert abs(r["question_count_compliance_rate"] - 8 / 9) < 1e-9
    assert abs(r["format_compliance_rate"] - 7 / 9) < 1e-9


def test_sweep_pooled_and_chi():
    s1 = {
        "pooled_distribution": {
            "A": {"count": 10, "pct": 0.25},
            "B": {"count": 10, "pct": 0.25},
            "C": {"count": 20, "pct": 0.5},
            "D": {"count": 0, "pct": 0.0},
        }
    }
    s2 = {
        "pooled_distribution": {
            "A": {"count": 0, "pct": 0.0},
            "B": {"count": 20, "pct": 0.5},
            "C": {"count": 10, "pct": 0.25},
            "D": {"count": 10, "pct": 0.25},
        }
    }
    agg = sweep_pooled_distribution([s1, s2])
    assert agg["total_answers"] == 80
    chi, p = sweep_chi_square_vs_uniform([s1, s2])
    assert chi > 0
    assert p is None

"""Tests for MCQ probe prompt construction."""

from __future__ import annotations

from study_query_llm.experiments.mcq_answer_position_probe import build_prompt


def test_build_prompt_includes_level_when_set():
    p = build_prompt("physics", 5, ["A", "B", "C", "D"], level="high school")
    assert "in physics at high school level" in p


def test_build_prompt_omits_level_phrase_when_empty():
    p = build_prompt("physics", 5, ["A", "B", "C", "D"], level=None)
    assert "Create a physics multiple-choice test" in p
    assert " at " not in p[:80]  # no "in X at Y level" intro


def test_build_prompt_spread_adds_instruction():
    p = build_prompt(
        "math",
        3,
        ["A", "B", "C", "D"],
        spread_correct_answer_uniformly=True,
    )
    assert "Spread the correct answers" in p
    assert "A,B,C,D" in p


def test_build_prompt_no_spread_line_when_false():
    p = build_prompt("math", 3, ["A", "B", "C", "D"], spread_correct_answer_uniformly=False)
    assert "Spread the correct answers" not in p

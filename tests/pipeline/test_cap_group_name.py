"""Tests for group-name capping helper."""

from __future__ import annotations

from study_query_llm.pipeline.runner import cap_group_name


def test_cap_group_name_keeps_short_names() -> None:
    name = "analyze:method:123:run_key_short"
    capped, full = cap_group_name(name)
    assert capped == name
    assert full is None


def test_cap_group_name_truncates_and_hashes() -> None:
    name = "x" * 220
    capped, full = cap_group_name(name)
    assert len(capped) == 180
    assert capped[171] == ":"
    assert len(capped.split(":")[-1]) == 8
    assert full == name


def test_cap_group_name_is_deterministic() -> None:
    name = "analyze:method:123:" + ("run_key_" * 40)
    capped_a, full_a = cap_group_name(name)
    capped_b, full_b = cap_group_name(name)
    assert capped_a == capped_b
    assert full_a == full_b == name

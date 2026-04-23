"""Unit tests for SemEval 2013 SRA five-way parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.datasets.source_specs.semeval2013_sra_5way import (
    SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION,
    parse_semeval2013_sra_5way_snapshot,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_fixture(tmp_path: Path, *, count_value: str) -> None:
    prefix = tmp_path / "semevalFormatProcessing-5way"
    _write(
        prefix / "answers.csv",
        "id1\tQuestion one\tReference one\tStudent answer one\n",
    )
    _write(
        prefix / "trainingGold.txt",
        (
            "id\tqid\ttestSet\tmodule\tcount\taccuracy\n"
            f"id1\tQ1\t\tScience\t{count_value}\tcorrect\n"
        ),
    )
    _write(prefix / "trainingGold-partial.txt", "id\tqid\ttestSet\tmodule\tcount\taccuracy\n")
    _write(prefix / "testGold-UA.txt", "id\tqid\ttestSet\tmodule\tcount\taccuracy\n")
    _write(prefix / "testGold-UQ.txt", "id\tqid\ttestSet\tmodule\tcount\taccuracy\n")
    _write(
        prefix / "partialEntailmentGold.txt",
        "id\tqid\ttestSet\tmodule\tcount\taccuracy\n",
    )


def _ctx(tmp_path: Path) -> ParserContext:
    return ParserContext(
        dataset_group_id=1,
        artifact_uris={},
        artifact_dir_local=tmp_path,
        source_metadata={"kind": "github_raw"},
        parser_id="semeval2013_sra_5way.default",
        parser_version=SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION,
    )


def test_semeval_parser_version_is_v2() -> None:
    assert SEMEVAL2013_SRA_5WAY_DEFAULT_PARSER_VERSION == "v2"


def test_parse_semeval_emits_int_gold_count_with_raw_companion(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, count_value="01")

    rows = list(parse_semeval2013_sra_5way_snapshot(_ctx(tmp_path)))
    assert len(rows) == 1
    row = rows[0]
    assert row.extra["gold_count"] == 1
    assert isinstance(row.extra["gold_count"], int)
    assert row.extra["gold_count_raw"] == "01"


def test_parse_semeval_rejects_non_integer_gold_count(tmp_path: Path) -> None:
    _seed_fixture(tmp_path, count_value="one")

    with pytest.raises(ValueError, match="non-integer gold count"):
        list(parse_semeval2013_sra_5way_snapshot(_ctx(tmp_path)))

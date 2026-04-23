"""Unit tests for 20 Newsgroups canonical parser."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.datasets.source_specs.twenty_newsgroups import (
    TWENTY_NEWSGROUPS_6CAT,
    TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE,
    TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH,
    TWENTY_NEWSGROUPS_RESEARCH_MAX_TEXT_LEN,
    TWENTY_NEWSGROUPS_RESEARCH_MIN_TEXT_LEN,
    parse_twenty_newsgroups_snapshot,
    twenty_newsgroups_6cat_subquery_spec,
    twenty_newsgroups_research_subquery_spec,
)
from study_query_llm.pipeline.types import SubquerySpec


def _write_archive(tmp_path: Path, members: dict[str, str]) -> None:
    archive_path = tmp_path / TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for member_name, text in sorted(members.items()):
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _parser_ctx(tmp_path: Path) -> ParserContext:
    return ParserContext(
        dataset_group_id=101,
        artifact_uris={
            TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH: f"file://{TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH}"
        },
        artifact_dir_local=tmp_path,
        source_metadata={"kind": "figshare_file"},
    )


def test_parse_twenty_newsgroups_snapshot_parses_train_and_test(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        {
            "20news-bydate-train/alt.atheism/1001": "Atheism discussion text that is long enough.",
            "20news-bydate-train/sci.space/2001": "Space exploration and orbit mechanics discussion.",
            "20news-bydate-test/sci.space/2002": "Shuttle mission update and satellite payload notes.",
            "20news-bydate-test/misc.forsale/3001": "Used bike for sale with delivery details and extras.",
            "20news-bydate-train/comp.graphics/4001": "short",
            "README.txt": "ignored control file",
        },
    )

    rows = list(parse_twenty_newsgroups_snapshot(_parser_ctx(tmp_path)))

    # v2: the parser emits the full archive (including the previously-filtered
    # "short" comp.graphics row); length-window selection is the snapshot
    # layer's job.
    assert len(rows) == 5
    assert [row.position for row in rows] == [0, 1, 2, 3, 4]
    assert {row.extra["split"] for row in rows} == {"train", "test"}
    assert {row.extra["subset_profile"] for row in rows} == {"all_categories"}
    assert all(row.extra["newsgroup"] == row.label_name for row in rows)

    label_by_name = {str(row.label_name): int(row.label) for row in rows}
    # Categories are sorted; with comp.graphics now present it slots in at 1.
    assert label_by_name["alt.atheism"] == 0
    assert label_by_name["comp.graphics"] == 1
    assert label_by_name["misc.forsale"] == 2
    assert label_by_name["sci.space"] == 3

    source_ids = {row.source_id for row in rows}
    assert "train:alt.atheism:1001" in source_ids
    assert "test:sci.space:2002" in source_ids
    assert "train:comp.graphics:4001" in source_ids

    rows_by_source = {row.source_id: row for row in rows}
    short_row = rows_by_source["train:comp.graphics:4001"]
    assert short_row.text == "short"
    assert short_row.extra["text_len_chars"] == len("short") == 5
    assert all(row.extra["text_len_chars"] == len(row.text) for row in rows)


def test_parse_twenty_newsgroups_snapshot_drops_only_empty_bodies(tmp_path: Path) -> None:
    _write_archive(
        tmp_path,
        {
            "20news-bydate-train/alt.atheism/1001": "Atheism discussion text that is long enough.",
            "20news-bydate-train/comp.graphics/4001": "   \x00\x00   ",
        },
    )

    rows = list(parse_twenty_newsgroups_snapshot(_parser_ctx(tmp_path)))

    assert len(rows) == 1
    assert rows[0].source_id == "train:alt.atheism:1001"


def test_parse_twenty_newsgroups_snapshot_missing_archive_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected file missing"):
        list(parse_twenty_newsgroups_snapshot(_parser_ctx(tmp_path)))


def test_twenty_newsgroups_6cat_subquery_spec_defaults_match_canonical_convention() -> None:
    spec = twenty_newsgroups_6cat_subquery_spec()

    assert isinstance(spec, SubquerySpec)
    assert spec.label_mode == TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE == "labeled"
    assert spec.filter_expr is None
    assert spec.sample_n is None
    assert spec.sample_fraction is None
    assert spec.sampling_seed is None

    assert spec.category_filter is not None
    assert tuple(spec.category_filter.keys()) == ("newsgroup",)
    assert tuple(spec.category_filter["newsgroup"]) == tuple(sorted(TWENTY_NEWSGROUPS_6CAT))


def test_twenty_newsgroups_6cat_subquery_spec_canonical_dict_matches_manual_construction() -> None:
    factory_dict = twenty_newsgroups_6cat_subquery_spec().to_canonical_dict()
    manual_dict = SubquerySpec(
        label_mode="labeled",
        category_filter={"newsgroup": list(TWENTY_NEWSGROUPS_6CAT)},
    ).to_canonical_dict()

    assert factory_dict == manual_dict


def test_twenty_newsgroups_6cat_subquery_spec_passes_sampling_kwargs_through() -> None:
    spec = twenty_newsgroups_6cat_subquery_spec(
        sample_n=600,
        sampling_seed=42,
    )
    canonical = spec.to_canonical_dict()
    assert canonical["sample_n"] == 600
    assert canonical["sampling_seed"] == 42
    assert canonical["sample_fraction"] is None
    assert canonical["label_mode"] == "labeled"


def test_twenty_newsgroups_6cat_subquery_spec_label_mode_override() -> None:
    spec = twenty_newsgroups_6cat_subquery_spec(label_mode="all")
    assert spec.label_mode == "all"
    assert spec.to_canonical_dict()["label_mode"] == "all"


def test_twenty_newsgroups_research_subquery_spec_defaults_match_v1_window() -> None:
    spec = twenty_newsgroups_research_subquery_spec()

    assert isinstance(spec, SubquerySpec)
    assert spec.label_mode == "labeled"
    assert spec.category_filter is None
    assert spec.sample_n is None
    assert spec.sample_fraction is None
    assert spec.sampling_seed is None

    assert spec.filter_expr is not None
    assert (
        spec.filter_expr
        == f"text.str.len() > {TWENTY_NEWSGROUPS_RESEARCH_MIN_TEXT_LEN}"
        f" and text.str.len() <= {TWENTY_NEWSGROUPS_RESEARCH_MAX_TEXT_LEN}"
    )


def test_twenty_newsgroups_research_subquery_spec_composes_with_6cat() -> None:
    spec = twenty_newsgroups_research_subquery_spec(newsgroups=TWENTY_NEWSGROUPS_6CAT)

    assert spec.category_filter is not None
    assert tuple(spec.category_filter.keys()) == ("newsgroup",)
    assert tuple(spec.category_filter["newsgroup"]) == tuple(sorted(TWENTY_NEWSGROUPS_6CAT))
    assert spec.filter_expr == (
        f"text.str.len() > {TWENTY_NEWSGROUPS_RESEARCH_MIN_TEXT_LEN}"
        f" and text.str.len() <= {TWENTY_NEWSGROUPS_RESEARCH_MAX_TEXT_LEN}"
    )


def test_twenty_newsgroups_research_subquery_spec_custom_bounds_and_kwargs() -> None:
    spec = twenty_newsgroups_research_subquery_spec(
        label_mode="all",
        min_chars=20,
        max_chars=500,
        sample_n=300,
        sampling_seed=7,
    )

    canonical = spec.to_canonical_dict()
    assert canonical["label_mode"] == "all"
    assert canonical["filter_expr"] == "text.str.len() > 20 and text.str.len() <= 500"
    assert canonical["sample_n"] == 300
    assert canonical["sampling_seed"] == 7
    # to_canonical_dict() drops None-valued keys; absence == "no category filter".
    assert canonical.get("category_filter") is None
    assert spec.category_filter is None

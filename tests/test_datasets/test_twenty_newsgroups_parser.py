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
    parse_twenty_newsgroups_snapshot,
    twenty_newsgroups_6cat_subquery_spec,
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

    assert len(rows) == 4
    assert [row.position for row in rows] == [0, 1, 2, 3]
    assert {row.extra["split"] for row in rows} == {"train", "test"}
    assert {row.extra["subset_profile"] for row in rows} == {"all_categories"}
    assert all(row.extra["newsgroup"] == row.label_name for row in rows)

    label_by_name = {str(row.label_name): int(row.label) for row in rows}
    assert label_by_name["alt.atheism"] == 0
    assert label_by_name["misc.forsale"] == 1
    assert label_by_name["sci.space"] == 2

    source_ids = {row.source_id for row in rows}
    assert "train:alt.atheism:1001" in source_ids
    assert "test:sci.space:2002" in source_ids


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

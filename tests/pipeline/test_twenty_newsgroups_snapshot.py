"""Integration coverage for 20 Newsgroups full parse and 6-cat slicing."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY
from study_query_llm.datasets.source_specs.twenty_newsgroups import (
    TWENTY_NEWSGROUPS_6CAT,
    TWENTY_NEWSGROUPS_DATASET_SLUG,
    twenty_newsgroups_6cat_subquery_spec,
)
from study_query_llm.db.connection_v2 import DatabaseConnectionV2
from study_query_llm.db.models_v2 import Group
from study_query_llm.pipeline.acquire import acquire
from study_query_llm.pipeline.parse import parse
from study_query_llm.pipeline.snapshot import snapshot
from study_query_llm.pipeline.types import SubquerySpec


def _db(tmp_path: Path) -> DatabaseConnectionV2:
    db_path = (tmp_path / "twenty_newsgroups_snapshot.sqlite3").resolve()
    db = DatabaseConnectionV2(f"sqlite:///{db_path.as_posix()}", enable_pgvector=False)
    db.init_db()
    return db


def _archive_bytes(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for member_name, text in sorted(members.items()):
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_twenty_newsgroups_full_parse_and_6cat_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_STORAGE_BACKEND", "local")
    db = _db(tmp_path)
    artifact_dir = str((tmp_path / "artifacts").resolve())
    spec = ACQUIRE_REGISTRY[TWENTY_NEWSGROUPS_DATASET_SLUG]

    all_categories = list(TWENTY_NEWSGROUPS_6CAT) + ["misc.forsale"]
    members: dict[str, str] = {}
    for index, category in enumerate(all_categories, start=1):
        split_prefix = "20news-bydate-train" if index % 2 else "20news-bydate-test"
        members[f"{split_prefix}/{category}/{1000 + index}"] = (
            f"{category} sample discussion with enough detail for parser acceptance."
        )
    members["20news-bydate-test/sci.space/9001"] = "short"

    payload = _archive_bytes(members)
    acquired = acquire(
        spec,
        db=db,
        artifact_dir=artifact_dir,
        fetch=lambda _url: payload,
    )
    parsed = parse(acquired.group_id, db=db, artifact_dir=artifact_dir)
    parsed_reuse = parse(acquired.group_id, db=db, artifact_dir=artifact_dir)
    full = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    full_reuse = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(label_mode="all"),
        db=db,
        artifact_dir=artifact_dir,
    )
    sixcat = snapshot(
        parsed.group_id,
        subquery_spec=twenty_newsgroups_6cat_subquery_spec(),
        db=db,
        artifact_dir=artifact_dir,
    )
    sixcat_reuse_factory = snapshot(
        parsed.group_id,
        subquery_spec=twenty_newsgroups_6cat_subquery_spec(),
        db=db,
        artifact_dir=artifact_dir,
    )
    sixcat_reuse_manual = snapshot(
        parsed.group_id,
        subquery_spec=SubquerySpec(
            label_mode="labeled",
            category_filter={"newsgroup": list(TWENTY_NEWSGROUPS_6CAT)},
        ),
        db=db,
        artifact_dir=artifact_dir,
    )

    assert parsed.group_id == parsed_reuse.group_id
    assert full.group_id == full_reuse.group_id
    assert sixcat.group_id == sixcat_reuse_factory.group_id
    assert sixcat.group_id == sixcat_reuse_manual.group_id
    assert full_reuse.metadata["reused"] is True
    assert sixcat_reuse_factory.metadata["reused"] is True
    assert sixcat_reuse_manual.metadata["reused"] is True
    assert full.metadata["row_count"] == len(all_categories)
    assert sixcat.metadata["row_count"] == len(TWENTY_NEWSGROUPS_6CAT)
    assert sixcat.group_id != full.group_id

    table = pq.read_table(parsed.artifact_uris["dataframe.parquet"])
    extras = [json.loads(value) for value in table.column("extra_json").to_pylist()]
    assert {extra["subset_profile"] for extra in extras} == {"all_categories"}
    assert {extra["newsgroup"] for extra in extras} == set(all_categories)
    assert {extra["split"] for extra in extras} == {"train", "test"}

    with db.session_scope() as session:
        dataframe_group = session.query(Group).filter(Group.id == parsed.group_id).first()
        assert dataframe_group is not None
        metadata = dict(dataframe_group.metadata_json or {})
        assert metadata["dataset_slug"] == TWENTY_NEWSGROUPS_DATASET_SLUG
        assert metadata["parser_id"] == "twenty_newsgroups.default"
        assert metadata["parser_version"] == "v1"

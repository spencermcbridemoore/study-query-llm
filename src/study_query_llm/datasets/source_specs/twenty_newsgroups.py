"""20 Newsgroups source spec with pinned archive URL and canonical parser."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec

TWENTY_NEWSGROUPS_DATASET_SLUG = "twenty_newsgroups"
TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH = "20news-bydate.tar.gz"
TWENTY_NEWSGROUPS_ARCHIVE_URL = "https://ndownloader.figshare.com/files/5975967"
TWENTY_NEWSGROUPS_DEFAULT_PARSER_ID = "twenty_newsgroups.default"
TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION = "v1"

TWENTY_NEWSGROUPS_6CAT: tuple[str, ...] = (
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.misc",
)

TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE = "labeled"

_MIN_TEXT_LEN = 10
_MAX_TEXT_LEN = 1000
_SPLIT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("20news-bydate-train/", "train"),
    ("20news-bydate-test/", "test"),
)


def twenty_newsgroups_6cat_subquery_spec(
    *,
    label_mode: str = TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE,
    sample_n: int | None = None,
    sample_fraction: float | None = None,
    sampling_seed: int | None = None,
) -> "SubquerySpec":
    """Canonical 6-category snapshot selection over the 20 Newsgroups dataframe.

    Centralizes the literature-convention 6-category subset
    (:data:`TWENTY_NEWSGROUPS_6CAT`) plus the canonical ``label_mode='labeled'``
    convention so every caller materializes the same ``dataset_snapshot``
    group (identical ``spec_hash``) for the unsampled canonical view. Optional
    sampling kwargs flow straight through to :class:`SubquerySpec`; the
    underlying spec still requires ``sampling_seed`` whenever ``sample_n`` or
    ``sample_fraction`` is set.
    """
    from study_query_llm.pipeline.types import SubquerySpec

    return SubquerySpec(
        label_mode=label_mode,
        category_filter={"newsgroup": list(TWENTY_NEWSGROUPS_6CAT)},
        sample_n=sample_n,
        sample_fraction=sample_fraction,
        sampling_seed=sampling_seed,
    )


def twenty_newsgroups_file_specs() -> List[FileFetchSpec]:
    """Pinned archive used by the canonical 20 Newsgroups parser."""
    return [
        FileFetchSpec(
            relative_path=TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH,
            url=TWENTY_NEWSGROUPS_ARCHIVE_URL,
        )
    ]


def twenty_newsgroups_source_metadata() -> Dict[str, Any]:
    """Acquisition metadata for provenance and content-fingerprint stability."""
    return {
        "kind": "figshare_file",
        "dataset": "20newsgroups-bydate",
        "archive_url": TWENTY_NEWSGROUPS_ARCHIVE_URL,
        "archive_path": TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH,
        "text_filter": {"min_len_gt": _MIN_TEXT_LEN, "max_len_le": _MAX_TEXT_LEN},
        "pinning_identity": {
            "archive_url": TWENTY_NEWSGROUPS_ARCHIVE_URL,
            "archive_path": TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH,
        },
    }


def _decode_message(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="replace")


def _clean_text(raw_text: str) -> str:
    text = str(raw_text).replace("\x00", "").strip()
    if len(text) <= _MIN_TEXT_LEN or len(text) > _MAX_TEXT_LEN:
        return ""
    return text


def _parse_member_name(member_name: str) -> tuple[str, str, str] | None:
    normalized = str(member_name or "").lstrip("/")
    for prefix, split in _SPLIT_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        tail = normalized[len(prefix) :]
        parts = tail.split("/", 1)
        if len(parts) != 2:
            return None
        newsgroup, doc_name = parts
        if not newsgroup or not doc_name:
            return None
        return split, newsgroup, doc_name
    return None


def _load_archive_records(
    archive_path: Path,
) -> list[tuple[str, str, str, str, str]]:
    records: list[tuple[str, str, str, str, str]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = sorted((member for member in archive.getmembers() if member.isfile()), key=lambda m: m.name)
        for member in members:
            parsed = _parse_member_name(member.name)
            if parsed is None:
                continue
            split, newsgroup, doc_name = parsed
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(
                    "twenty_newsgroups parser could not extract archive member "
                    f"{member.name!r}"
                )
            with extracted:
                payload = extracted.read()
            text = _clean_text(_decode_message(payload))
            if not text:
                continue
            records.append((split, newsgroup, doc_name, member.name, text))
    if not records:
        raise ValueError("twenty_newsgroups parser produced no rows after text filtering")
    return records


def parse_twenty_newsgroups_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parse full train+test 20 Newsgroups archive into labeled canonical rows."""
    from study_query_llm.pipeline.types import SnapshotRow

    archive_path = ctx.artifact_dir_local / TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH
    if not archive_path.is_file():
        raise ValueError(f"twenty_newsgroups parser expected file missing: {archive_path}")

    raw_records = _load_archive_records(archive_path)
    categories = sorted({newsgroup for _split, newsgroup, _doc_name, _member_name, _text in raw_records})
    label_by_category = {name: index for index, name in enumerate(categories)}
    rows: list[SnapshotRow] = []
    for split, newsgroup, doc_name, member_name, text in raw_records:
        rows.append(
            SnapshotRow(
                position=len(rows),
                source_id=f"{split}:{newsgroup}:{doc_name}",
                text=text,
                label=label_by_category[newsgroup],
                label_name=newsgroup,
                extra={
                    "split": split,
                    "newsgroup": newsgroup,
                    "doc_name": doc_name,
                    "archive_member": member_name,
                    "subset_profile": "all_categories",
                },
            )
        )
    return rows

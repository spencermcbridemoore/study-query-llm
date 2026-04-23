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
# v2 (2026-04-22): drop in-parser 10<len<=1000 char filter so the canonical
# dataframe reflects the full archive; re-apply the literature-convention range
# at snapshot time via :func:`twenty_newsgroups_research_subquery_spec`.
TWENTY_NEWSGROUPS_DEFAULT_PARSER_VERSION = "v2"

TWENTY_NEWSGROUPS_6CAT: tuple[str, ...] = (
    "alt.atheism",
    "soc.religion.christian",
    "comp.graphics",
    "rec.sport.hockey",
    "sci.space",
    "talk.politics.misc",
)

TWENTY_NEWSGROUPS_6CAT_DEFAULT_LABEL_MODE = "labeled"

# Literature-convention text-length window for the K-LLMmeans / replication
# regime. Applied at snapshot time via filter_expr; not enforced by the parser.
TWENTY_NEWSGROUPS_RESEARCH_MIN_TEXT_LEN = 10
TWENTY_NEWSGROUPS_RESEARCH_MAX_TEXT_LEN = 1000

_SPLIT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("20news-bydate-train/", "train"),
    ("20news-bydate-test/", "test"),
)


def _research_text_filter_expr(*, min_chars: int, max_chars: int) -> str:
    """Pandas ``.query()`` expression matching the v1-era length window.

    Evaluated by snapshot's python-engine query against the canonical ``text``
    column, so no parquet schema change is required.
    """
    return f"text.str.len() > {int(min_chars)} and text.str.len() <= {int(max_chars)}"


def twenty_newsgroups_research_subquery_spec(
    *,
    label_mode: str = "labeled",
    newsgroups: Iterable[str] | None = None,
    min_chars: int = TWENTY_NEWSGROUPS_RESEARCH_MIN_TEXT_LEN,
    max_chars: int = TWENTY_NEWSGROUPS_RESEARCH_MAX_TEXT_LEN,
    sample_n: int | None = None,
    sample_fraction: float | None = None,
    sampling_seed: int | None = None,
) -> "SubquerySpec":
    """Snapshot spec reproducing the v1 parser's text-length window.

    With ``newsgroups=None`` and default bounds this materializes the same
    row-set the v1 parser used to emit (10 < ``len(text)`` <= 1000) across all
    20 categories. Pass ``newsgroups=TWENTY_NEWSGROUPS_6CAT`` to compose with
    the canonical 6-category subset in a single ``SubquerySpec`` (identical
    ``spec_hash`` for identical inputs).
    """
    from study_query_llm.pipeline.types import SubquerySpec

    category_filter: Dict[str, Any] | None = None
    if newsgroups is not None:
        category_filter = {"newsgroup": list(newsgroups)}

    return SubquerySpec(
        label_mode=label_mode,
        filter_expr=_research_text_filter_expr(
            min_chars=min_chars, max_chars=max_chars
        ),
        category_filter=category_filter,
        sample_n=sample_n,
        sample_fraction=sample_fraction,
        sampling_seed=sampling_seed,
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

    Note: this spec applies *no* text-length filter. To reproduce the v1-era
    "6cat + 10<len<=1000" research view, use
    :func:`twenty_newsgroups_research_subquery_spec` with
    ``newsgroups=TWENTY_NEWSGROUPS_6CAT``.
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
    """Acquisition metadata for provenance and content-fingerprint stability.

    Note: ``content_fingerprint`` is derived only from ``dataset_slug``,
    ``pinning_identity``, and the sorted ``(relative_path, sha256)`` file
    pairs (see ``content_fingerprint`` in ``datasets.acquisition``). Top-level
    keys outside ``pinning_identity`` are informational only and do not affect
    dataset-group identity.
    """
    return {
        "kind": "figshare_file",
        "dataset": "20newsgroups-bydate",
        "archive_url": TWENTY_NEWSGROUPS_ARCHIVE_URL,
        "archive_path": TWENTY_NEWSGROUPS_ARCHIVE_RELATIVE_PATH,
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
    """Strip null bytes and surrounding whitespace; preserve full body length.

    v2 (2026-04-22): no length filtering. Empty strings are still dropped by
    the caller. Length-window selection lives in the snapshot layer (see
    :func:`twenty_newsgroups_research_subquery_spec`).
    """
    return str(raw_text).replace("\x00", "").strip()


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
        raise ValueError(
            "twenty_newsgroups parser produced no rows after cleaning empty bodies"
        )
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
                    "text_len_chars": len(text),
                },
            )
        )
    return rows

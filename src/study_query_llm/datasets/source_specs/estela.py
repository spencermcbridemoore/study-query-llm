"""Estela uncategorized prompt dictionary snapshot from pinned repository pickle."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.utils.text_utils import flatten_prompt_dict

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow, SubquerySpec

ESTELA_DATASET_SLUG = "estela"
ESTELA_GITHUB_ORG = "spencermcbridemoore"
ESTELA_GITHUB_REPO = "study-query-llm"
# Pinned commit containing notebooks/estela_prompt_data.pkl.
ESTELA_PINNED_GIT_REF = "b7238961d8ce0f30ca54059569c882d632732b4b"
ESTELA_PICKLE_RELATIVE_PATH = "notebooks/estela_prompt_data.pkl"
ESTELA_DEFAULT_PARSER_ID = "estela.default"
# v2 (2026-04-22): drop in-parser 10<len<=1000 char filter so the canonical
# dataframe reflects the full prompt dictionary; re-apply the legacy length
# window at snapshot time via :func:`estela_research_subquery_spec`.
ESTELA_DEFAULT_PARSER_VERSION = "v2"

# Legacy v1 text-length window, retained as the research-replication default.
ESTELA_RESEARCH_MIN_TEXT_LEN = 10
ESTELA_RESEARCH_MAX_TEXT_LEN = 1000


def estela_raw_url(relative_repo_path: str) -> str:
    rel = relative_repo_path.lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{ESTELA_GITHUB_ORG}/"
        f"{ESTELA_GITHUB_REPO}/{ESTELA_PINNED_GIT_REF}/{rel}"
    )


def estela_file_specs() -> List[FileFetchSpec]:
    return [
        FileFetchSpec(
            relative_path=ESTELA_PICKLE_RELATIVE_PATH,
            url=estela_raw_url(ESTELA_PICKLE_RELATIVE_PATH),
        )
    ]


def estela_source_metadata() -> Dict[str, Any]:
    return {
        "kind": "github_raw",
        "organization": ESTELA_GITHUB_ORG,
        "repository": ESTELA_GITHUB_REPO,
        "git_ref": ESTELA_PINNED_GIT_REF,
        "description": "Estela uncategorized prompt dictionary pickle snapshot",
        "files": [ESTELA_PICKLE_RELATIVE_PATH],
    }


def _clean_prompt_text(value: Any) -> str:
    """Strip null bytes and surrounding whitespace; preserve full text length.

    v2 (2026-04-22): no length filtering. Empty strings are still dropped by
    the caller. Length-window selection lives in the snapshot layer (see
    :func:`estela_research_subquery_spec`).
    """
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def estela_research_subquery_spec(
    *,
    label_mode: str = "all",
    min_chars: int = ESTELA_RESEARCH_MIN_TEXT_LEN,
    max_chars: int = ESTELA_RESEARCH_MAX_TEXT_LEN,
    sample_n: int | None = None,
    sample_fraction: float | None = None,
    sampling_seed: int | None = None,
) -> "SubquerySpec":
    """Snapshot spec reproducing the v1 parser's text-length window.

    With default bounds this matches the row-set the v1 parser used to emit
    (10 < ``len(text)`` <= 1000) over the unlabeled prompt dictionary.
    Default ``label_mode='all'`` mirrors estela's lack of categorical labels;
    ``'unlabeled'`` is equivalent in practice but makes the intent explicit.
    """
    from study_query_llm.pipeline.types import SubquerySpec

    return SubquerySpec(
        label_mode=label_mode,
        filter_expr=(
            f"text.str.len() > {int(min_chars)}"
            f" and text.str.len() <= {int(max_chars)}"
        ),
        sample_n=sample_n,
        sample_fraction=sample_fraction,
        sampling_seed=sampling_seed,
    )


def parse_estela_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    """Parse Estela prompt dictionary into unlabeled SnapshotRows."""
    from study_query_llm.pipeline.types import SnapshotRow

    pkl_path = ctx.artifact_dir_local / ESTELA_PICKLE_RELATIVE_PATH
    if not pkl_path.is_file():
        raise ValueError(f"estela parser expected file missing: {pkl_path}")

    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    flat = flatten_prompt_dict(payload)
    rows: list[SnapshotRow] = []
    ordered_items = sorted(
        flat.items(),
        key=lambda item: tuple(str(part) for part in item[0]),
    )
    for path_tuple, raw_text in ordered_items:
        text = _clean_prompt_text(raw_text)
        if not text:
            continue
        source_path = [str(part) for part in path_tuple]
        source_id = "/".join(source_path) or f"estela:{len(rows)}"
        rows.append(
            SnapshotRow(
                position=len(rows),
                source_id=source_id,
                text=text,
                label=None,
                label_name=None,
                extra={
                    "source_path": source_path,
                    "source_file": ESTELA_PICKLE_RELATIVE_PATH,
                    "subset_profile": "all_uncategorized",
                    "text_len_chars": len(text),
                },
            )
        )

    if not rows:
        raise ValueError(
            "estela parser produced no rows; check pickle contents or prompt filters"
        )
    return rows

"""Estela uncategorized prompt dictionary snapshot from pinned repository pickle."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext
from study_query_llm.utils.text_utils import flatten_prompt_dict

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow

ESTELA_DATASET_SLUG = "estela"
ESTELA_GITHUB_ORG = "spencermcbridemoore"
ESTELA_GITHUB_REPO = "study-query-llm"
# Pinned commit containing notebooks/estela_prompt_data.pkl.
ESTELA_PINNED_GIT_REF = "b7238961d8ce0f30ca54059569c882d632732b4b"
ESTELA_PICKLE_RELATIVE_PATH = "notebooks/estela_prompt_data.pkl"
ESTELA_DEFAULT_PARSER_ID = "estela.default"
ESTELA_DEFAULT_PARSER_VERSION = "v1"


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
    if value is None:
        return ""
    text = str(value).replace("\x00", "").strip()
    if len(text) <= 10 or len(text) > 1000:
        return ""
    return text


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
                },
            )
        )

    if not rows:
        raise ValueError(
            "estela parser produced no rows; check pickle contents or prompt filters"
        )
    return rows

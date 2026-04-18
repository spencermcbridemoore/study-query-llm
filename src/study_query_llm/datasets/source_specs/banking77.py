"""BANKING77 source spec with pinned HuggingFace resolve URLs and parquet parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import pyarrow.parquet as pq

from study_query_llm.datasets.acquisition import FileFetchSpec
from study_query_llm.datasets.source_specs.parser_protocol import ParserContext

if TYPE_CHECKING:
    from study_query_llm.pipeline.types import SnapshotRow

BANKING77_DATASET_SLUG = "banking77"
BANKING77_HF_DATASET = "mteb/banking77"
BANKING77_HF_REVISION = "18072d2685ea682290f7b8924d94c62acc19c0b2"

_BANKING77_SPLIT_FILES: tuple[tuple[str, str], ...] = (
    ("data/train-00000-of-00001.parquet", "train"),
    ("data/test-00000-of-00001.parquet", "test"),
)


def banking77_resolve_url(relative_path: str) -> str:
    normalized = relative_path.lstrip("/")
    return (
        f"https://huggingface.co/datasets/{BANKING77_HF_DATASET}/resolve/"
        f"{BANKING77_HF_REVISION}/{normalized}"
    )


def banking77_file_specs() -> List[FileFetchSpec]:
    return [
        FileFetchSpec(relative_path=relative_path, url=banking77_resolve_url(relative_path))
        for relative_path, _split in _BANKING77_SPLIT_FILES
    ]


def banking77_source_metadata() -> Dict[str, Any]:
    return {
        "kind": "huggingface_resolve",
        "dataset": BANKING77_HF_DATASET,
        "revision": BANKING77_HF_REVISION,
        "files": [relative_path for relative_path, _split in _BANKING77_SPLIT_FILES],
    }


def _normalize_label_name(raw_row: dict[str, Any], label: int | None) -> str | None:
    for key in ("label_text", "intent", "label_name"):
        raw_value = raw_row.get(key)
        if raw_value is None:
            continue
        text = str(raw_value).strip()
        if text:
            return text
    if label is None:
        return None
    return f"label_{label}"


def _coerce_extra(raw_row: dict[str, Any], *, split: str) -> dict[str, Any]:
    extra: dict[str, Any] = {"split": split}
    for key, value in raw_row.items():
        if key in {"text", "label", "label_text", "intent", "label_name"}:
            continue
        extra[str(key)] = value
    return extra


def parse_banking77_snapshot(ctx: ParserContext) -> Iterable["SnapshotRow"]:
    from study_query_llm.pipeline.types import SnapshotRow

    rows: list[SnapshotRow] = []
    position = 0
    for relative_path, split in _BANKING77_SPLIT_FILES:
        parquet_path = ctx.artifact_dir_local / relative_path
        if not parquet_path.is_file():
            raise ValueError(
                f"BANKING77 parser expected acquisition file missing: {parquet_path}"
            )
        table = pq.read_table(parquet_path)
        for split_index, raw_row in enumerate(table.to_pylist()):
            raw_text = raw_row.get("text")
            text = str(raw_text or "").replace("\x00", "").strip()
            if not text:
                raise ValueError(
                    f"BANKING77 row has empty text: split={split} index={split_index}"
                )
            raw_label = raw_row.get("label")
            label = int(raw_label) if raw_label is not None else None
            rows.append(
                SnapshotRow(
                    position=position,
                    source_id=f"{split}:{split_index}",
                    text=text,
                    label=label,
                    label_name=_normalize_label_name(raw_row, label),
                    extra=_coerce_extra(raw_row, split=split),
                )
            )
            position += 1
    return rows

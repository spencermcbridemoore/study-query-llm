"""Tests for scripts/living/prepare_mosart_midterm2_overlap15.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "living" / "prepare_mosart_midterm2_overlap15.py"

TARGET_QUESTION_IDS = (
    "5_8_astronomy_q02",
    "5_8_astronomy_q04",
    "5_8_astronomy_q06",
    "5_8_astronomy_q07",
    "5_8_astronomy_q10",
    "9_12_astronomy_q08",
    "9_12_astronomy_q11",
    "9_12_astronomy_q12",
    "9_12_astronomy_q15",
    "k_4_astronomy_q01",
    "k_4_astronomy_q04",
    "k_4_astronomy_q08",
    "k_4_astronomy_q13",
    "ls_mastery_q05",
    "ls_mastery_q06",
)


@pytest.fixture
def prep_mod():
    spec = importlib.util.spec_from_file_location("prepare_mosart_midterm2_overlap15", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample_questions_frame() -> pd.DataFrame:
    rows = []
    for idx, qid in enumerate(TARGET_QUESTION_IDS, start=1):
        doc = qid.rsplit("_q", 1)[0]
        rows.append(
            {
                "question_id": qid,
                "document_id": doc,
                "stem": f"Stem for {qid}",
                "opt_a_text": "A text",
                "opt_b_text": "B text",
                "opt_c_text": "C text",
                "opt_d_text": "D text",
                "opt_e_text": "E text",
                "correct_answer": "a",
            }
        )
    rows.append(
        {
            "question_id": "9_12_astronomy_q16",
            "document_id": "9_12_astronomy",
            "stem": "Outlier stem",
            "opt_a_text": "A",
            "opt_b_text": "B",
            "opt_c_text": "C",
            "opt_d_text": "D",
            "opt_e_text": "E",
            "correct_answer": "b",
        }
    )
    return pd.DataFrame(rows)


def _sample_links_frame() -> pd.DataFrame:
    rows = []
    for qid in TARGET_QUESTION_IDS:
        rows.append(
            {
                "question_id": qid,
                "misconception_id": f"MIS_{qid}",
                "option_letter": "b",
                "match_confidence": "likely",
            }
        )
    rows.append(
        {
            "question_id": "5_8_astronomy_q02",
            "misconception_id": "MIS_low_confidence",
            "option_letter": "c",
            "match_confidence": "possible",
        }
    )
    return pd.DataFrame(rows)


def test_map_questions_to_midterm2_excludes_q16_and_assigns_sequential_ids(prep_mod) -> None:
    mapped = prep_mod.map_questions_to_midterm2(_sample_questions_frame())
    assert len(mapped) == 15
    assert set(mapped["question_id"]) == set(TARGET_QUESTION_IDS)
    assert "9_12_astronomy_q16" not in set(mapped["question_id"])
    assert list(mapped["ItemID"]) == list(range(1, 16))
    assert set(mapped["temp_bank_id"]) == {1, 2, 3, 4}


def test_collapse_misconception_links_prefers_highest_confidence(prep_mod) -> None:
    collapsed = prep_mod.collapse_misconception_links(
        _sample_links_frame(),
        question_ids=TARGET_QUESTION_IDS,
    )
    winner = collapsed[collapsed["question_id"] == "5_8_astronomy_q02"].iloc[0]
    assert winner["misconception_id"] == "MIS_5_8_astronomy_q02"


def test_build_id_sidecar_has_one_row_per_question(prep_mod) -> None:
    mapped = prep_mod.map_questions_to_midterm2(_sample_questions_frame())
    collapsed = prep_mod.collapse_misconception_links(
        _sample_links_frame(),
        question_ids=[str(q) for q in mapped["question_id"]],
    )
    sidecar = prep_mod.build_id_sidecar(mapped, collapsed)
    assert len(sidecar) == 15
    assert list(sidecar.columns) == list(prep_mod.SIDECAR_COLUMNS)


def test_missing_links_file_raises_file_not_found(prep_mod, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prep_mod.load_misconception_links(tmp_path / "missing.csv")


def test_missing_links_columns_raises_value_error(prep_mod, tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("question_id,option_letter\nq1,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misconception_links_missing_columns"):
        prep_mod.load_misconception_links(bad)


def test_prepare_overlap15_end_to_end(prep_mod, tmp_path: Path) -> None:
    tables = tmp_path / "mosart_tables"
    tables.mkdir(parents=True)
    _sample_questions_frame().to_csv(tables / "questions.csv", index=False, encoding="utf-8")
    _sample_links_frame().to_csv(
        tables / "question_misconception_links.csv",
        index=False,
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    outputs = prep_mod.prepare_overlap15(mosart_tables_dir=tables, output_dir=out_dir)
    frame = pd.read_csv(outputs["csv"], encoding="utf-8")
    sidecar = pd.read_csv(outputs["sidecar"], encoding="utf-8")
    assert len(frame) == 15
    assert len(sidecar) == 15
    token_payload = json.loads(outputs["token_estimate_json"].read_text(encoding="utf-8"))
    assert float(token_payload["v1_mean"]) > 0
    assert float(token_payload["v2_chat_system_mean"]) > 0

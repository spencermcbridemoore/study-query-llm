"""Pure helpers for ``run_overnight_fivesnap_backfill`` (import via importlib)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "living" / "run_overnight_fivesnap_backfill.py"


@pytest.fixture(scope="module")
def overnight_mod():
    spec = importlib.util.spec_from_file_location("run_overnight_fivesnap_backfill", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pair_slug_normalizes_special_chars(overnight_mod):
    assert overnight_mod.pair_slug("Azure/OpenAI", "text-embedding-3-small") == (
        "Azure_OpenAI__text-embedding-3-small"
    )


def test_intersection_candidate_pairs_empty_snapshots(overnight_mod):
    class _Sess:
        pass

    assert overnight_mod.intersection_candidate_pairs(_Sess(), []) == []

"""Unit tests for per-pair orchestrator helper behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "living" / "run_per_pair_backfill_orchestrator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_per_pair_backfill_orchestrator", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_engine_slug_and_pair_id() -> None:
    mod = _load_module()
    assert mod.engine_slug("qwen/qwen3-embedding-4b") == "qwen_qwen3-embedding-4b"
    pair = mod.PairKey(snapshot_id=17, embedding_engine="qwen/qwen3-embedding-4b")
    assert pair.pair_id == "snap17__qwen_qwen3-embedding-4b"


def test_embedding_level_reason_detection() -> None:
    mod = _load_module()
    assert mod._pair_has_embedding_level_reason({"reason_code": "provider_unresolved"})
    assert mod._pair_has_embedding_level_reason({"reason_code": "still_no_embedding_after_phase0"})
    assert not mod._pair_has_embedding_level_reason({"reason_code": "phase1_failed"})


def test_load_price_overrides_filters_invalid(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "prices.json"
    path.write_text(
        '{"ok/a": 0.12, "bad": "x", "zero": 0, "neg": -1, "ok/b": "0.34"}',
        encoding="utf-8",
    )
    parsed = mod._load_price_overrides(path)
    assert parsed == {"ok/a": 0.12, "ok/b": 0.34}

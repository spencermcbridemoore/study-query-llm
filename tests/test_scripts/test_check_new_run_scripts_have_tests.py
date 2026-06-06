"""Unit tests for scripts/check_new_run_scripts_have_tests.py (plan 3.1 creation lint)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "check_new_run_scripts_have_tests.py"


def _mod():
    spec = importlib.util.spec_from_file_location("check_new_run_scripts_have_tests", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


class TestAddedRunScripts:
    def test_matches_root_run_scripts_only(self) -> None:
        mod = _mod()
        paths = [
            "scripts/run_foo.py",            # match
            "scripts/run_bar_baz.py",        # match
            "scripts/check_foo.py",          # no: not run_
            "scripts/living/run_qux.py",     # no: subdirectory
            "scripts/run_foo.txt",           # no: not .py
            "src/study_query_llm/run_x.py",  # no: not scripts/
            "scripts" + chr(92) + "run_win.py",  # windows separator normalised (built via chr to avoid escaping)
        ]
        assert mod.added_run_scripts(paths) == [
            "scripts/run_bar_baz.py",
            "scripts/run_foo.py",
            "scripts/run_win.py",
        ]

    def test_dedups_and_sorts(self) -> None:
        mod = _mod()
        assert mod.added_run_scripts(["scripts/run_a.py", "scripts/run_a.py"]) == ["scripts/run_a.py"]


class TestStemReference:
    def test_referenced_by_filename(self, tmp_path: Path) -> None:
        mod = _mod()
        root = _tree(tmp_path, {"test_run_foo.py": "def test_x():\n    pass\n"})
        assert mod.stem_is_referenced("run_foo", root) is True

    def test_referenced_by_content(self, tmp_path: Path) -> None:
        mod = _mod()
        root = _tree(tmp_path, {"sub/test_other.py": "import scripts.run_foo  # noqa\n"})
        assert mod.stem_is_referenced("run_foo", root) is True

    def test_not_referenced(self, tmp_path: Path) -> None:
        mod = _mod()
        root = _tree(tmp_path, {"test_unrelated.py": "def test_y():\n    pass\n"})
        assert mod.stem_is_referenced("run_foo", root) is False

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        mod = _mod()
        # filename mentions the stem but lives under __pycache__ -> ignored
        root = _tree(tmp_path, {"__pycache__/run_foo.cpython-312.pyc": "run_foo"})
        assert mod.stem_is_referenced("run_foo", root) is False

    def test_missing_tests_dir_is_unreferenced(self, tmp_path: Path) -> None:
        mod = _mod()
        assert mod.stem_is_referenced("run_foo", tmp_path / "does_not_exist") is False


class TestFindUntested:
    def test_flags_only_untested_added_run_scripts(self, tmp_path: Path) -> None:
        mod = _mod()
        _tree(tmp_path, {"test_run_foo.py": "references run_foo\n"})
        added = [
            "scripts/run_foo.py",         # referenced -> ok
            "scripts/run_bar.py",         # NOT referenced -> offender
            "scripts/check_x.py",         # not a run-script
            "scripts/living/run_z.py",    # subdirectory, not in scope
        ]
        assert mod.find_untested_run_scripts(added, tmp_path) == ["scripts/run_bar.py"]

    def test_empty_when_no_added_run_scripts(self, tmp_path: Path) -> None:
        mod = _mod()
        assert mod.find_untested_run_scripts(["scripts/check_x.py", "src/y.py"], tmp_path) == []


class TestMainSmoke:
    def test_help_exits_zero(self) -> None:
        mod = _mod()
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0

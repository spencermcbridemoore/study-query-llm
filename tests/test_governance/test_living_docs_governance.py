"""Unit + smoke tests for the living-docs-only governance enforcement."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.internal.living_docs_governance import (  # noqa: E402
    ANNOTATION_TOKEN,
    RESTRICTED_FILES,
    RESTRICTED_PREFIX_DIRS,
    find_restricted_paths,
    is_restricted_path,
    messages_contain_annotation,
)


class TestIsRestrictedPath:
    @pytest.mark.parametrize(
        "path",
        [
            "docs/history/foo.md",
            "docs/history/sub/dir/bar.md",
            "docs/deprecated/README.md",
            "docs/plans/STEP-1.md",
            "docs/experiments/CUSTOM_SWEEP_README.md",
            "scripts/history/one_offs/check_db_empty.py",
            "scripts/deprecated/migrate_v1_to_v2.py",
            "docs/IMPLEMENTATION_PLAN.md",
            "docs/ARCHITECTURE.md",
            "docs/API.md",
            "docs/MIGRATION_GUIDE.md",
            "docs/PHASE1_5_VERIFICATION.md",
            "docs/PLOT_ORGANIZATION.md",
        ],
    )
    def test_restricted_paths_match(self, path: str) -> None:
        assert is_restricted_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "docs/living/CURRENT_STATE.md",
            "docs/DATA_PIPELINE.md",
            "docs/STANDING_ORDERS.md",
            "docs/runbooks/README.md",
            "src/study_query_llm/services/study_service.py",
            "tests/test_services/test_inference.py",
            "scripts/check_persistence_contract.py",
            "scripts/internal/living_docs_governance.py",
            "notebooks/some_notebook.ipynb",
            "README.md",
            "AGENTS.md",
            ".cursorrules",
            "",
        ],
    )
    def test_allowed_paths_do_not_match(self, path: str) -> None:
        assert is_restricted_path(path) is False

    def test_windows_backslashes_normalise(self) -> None:
        assert is_restricted_path("docs\\history\\foo.md") is True
        assert is_restricted_path("scripts\\deprecated\\x.py") is True
        assert is_restricted_path("docs\\living\\CURRENT_STATE.md") is False

    def test_prefix_match_requires_trailing_slash_semantics(self) -> None:
        # 'docs/history2/' must NOT match the 'docs/history/' prefix.
        assert is_restricted_path("docs/history2/foo.md") is False
        # but the exact prefix dir membership IS matched
        assert is_restricted_path("docs/history/sub/foo.md") is True


class TestFindRestrictedPaths:
    def test_returns_only_restricted_subset_sorted(self) -> None:
        paths = [
            "src/study_query_llm/cli.py",
            "docs/history/old.md",
            "docs/living/CURRENT_STATE.md",
            "docs/IMPLEMENTATION_PLAN.md",
            "scripts/deprecated/old.py",
        ]
        assert find_restricted_paths(paths) == [
            "docs/IMPLEMENTATION_PLAN.md",
            "docs/history/old.md",
            "scripts/deprecated/old.py",
        ]

    def test_dedups_and_normalises(self) -> None:
        paths = [
            "docs\\history\\a.md",
            "docs/history/a.md",
            "docs/history/a.md",
        ]
        assert find_restricted_paths(paths) == ["docs/history/a.md"]

    def test_empty_in_empty_out(self) -> None:
        assert find_restricted_paths([]) == []
        assert find_restricted_paths(["", "  "]) == []


class TestMessagesContainAnnotation:
    def test_default_token_found(self) -> None:
        messages = [
            "feat(x): add y",
            "docs(history): record old workflow [restricted-doc-edit-ok]",
        ]
        assert messages_contain_annotation(messages) is True

    def test_default_token_absent(self) -> None:
        messages = ["feat(x): add y", "fix(z): adjust"]
        assert messages_contain_annotation(messages) is False

    def test_custom_token(self) -> None:
        messages = ["chore: archive [legacy-ok]"]
        assert messages_contain_annotation(messages, token="[legacy-ok]") is True
        assert messages_contain_annotation(messages, token="[other]") is False

    def test_empty_messages(self) -> None:
        assert messages_contain_annotation([]) is False
        assert messages_contain_annotation(["", None, ""]) is False  # type: ignore[list-item]

    def test_annotation_token_constant_is_documented_value(self) -> None:
        assert ANNOTATION_TOKEN == "[restricted-doc-edit-ok]"


class TestRestrictedSetMembership:
    def test_known_legacy_files_in_set(self) -> None:
        for legacy in (
            "docs/IMPLEMENTATION_PLAN.md",
            "docs/ARCHITECTURE.md",
            "docs/API.md",
            "docs/MIGRATION_GUIDE.md",
        ):
            assert legacy in RESTRICTED_FILES

    def test_lane_prefixes_in_set(self) -> None:
        for prefix in (
            "docs/history/",
            "docs/deprecated/",
            "scripts/history/",
            "scripts/deprecated/",
        ):
            assert prefix in RESTRICTED_PREFIX_DIRS


class TestCheckScriptSmoke:
    """End-to-end smoke tests for scripts/check_living_docs_drift.py.

    These run the actual script against the real repo with a zero-width
    diff range so they cannot accidentally fail the suite based on what is
    or isn't currently checked in.
    """

    def _run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return subprocess.run(
            [sys.executable, "scripts/check_living_docs_drift.py", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=REPO,
            env=env,
        )

    def test_help_runs_cleanly(self) -> None:
        result = self._run_script("--help")
        assert result.returncode == 0
        assert "living-docs" in result.stdout.lower() or "restricted" in result.stdout.lower()

    def test_zero_width_range_passes(self) -> None:
        # Diffing HEAD against itself produces no changed files; check must pass.
        result = self._run_script("--base", "HEAD", "--head", "HEAD")
        assert result.returncode == 0, (
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

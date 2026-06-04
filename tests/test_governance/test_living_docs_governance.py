"""Unit + smoke tests for the living-docs-only governance enforcement."""

from __future__ import annotations

import os
import re
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


def _parse_mdc_tombstoned_paths(mdc_text: str) -> tuple[set[str], set[str]]:
    """Parse the tombstoned-path list out of living-docs-only.mdc.

    Returns ``(prefix_dirs, files)``: ``prefix_dirs`` are ``docs/foo/``-style
    directory prefixes (parsed from ``docs/foo/**`` entries, trailing slash
    kept) and ``files`` are exact-path entries. Only the "Confirmed
    restricted/tombstoned paths today" block (up to "### Archive Retrieval")
    is scanned, so back-ticked prose elsewhere in the section is ignored.
    """
    lines = mdc_text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if "restricted/tombstoned paths today" in ln),
        None,
    )
    assert start is not None, "tombstoned-paths sentinel not found in .mdc"
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i].startswith("### Archive Retrieval")
        ),
        len(lines),
    )
    prefix_dirs: set[str] = set()
    files: set[str] = set()
    for line in lines[start + 1 : end]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        match = re.search(r"`([^`]+)`", stripped)
        if match is None:
            continue
        token = match.group(1)
        if token.endswith("/**"):
            prefix_dirs.add(token[:-2])  # drop trailing "**", keep the "/"
        else:
            files.add(token)
    return prefix_dirs, files


class TestMdcTableMatchesGovernanceConstants:
    """1.3: living-docs-only.mdc mirrors the governance constants.

    The .mdc tombstone list and the living_docs_governance constants must
    agree in BOTH directions, so editing one without the other fails here
    (and, via the suite, in CI). Closes the gap TestRestrictedSetMembership
    leaves open: that class only spot-checks a few entries one-directionally
    and never reads the .mdc at all.
    """

    MDC_PATH = REPO / ".cursor" / "rules" / "living-docs-only.mdc"

    def _parsed(self) -> tuple[set[str], set[str]]:
        text = self.MDC_PATH.read_text(encoding="utf-8")
        return _parse_mdc_tombstoned_paths(text)

    def test_mdc_file_exists(self) -> None:
        assert self.MDC_PATH.is_file(), f"missing {self.MDC_PATH}"

    def test_prefix_dirs_match_constants(self) -> None:
        mdc_prefixes, _ = self._parsed()
        assert mdc_prefixes == set(RESTRICTED_PREFIX_DIRS), (
            "living-docs-only.mdc prefix dirs disagree with "
            "RESTRICTED_PREFIX_DIRS; update both together.\n"
            f"  only in .mdc:      {sorted(mdc_prefixes - set(RESTRICTED_PREFIX_DIRS))}\n"
            f"  only in constants: {sorted(set(RESTRICTED_PREFIX_DIRS) - mdc_prefixes)}"
        )

    def test_files_match_constants(self) -> None:
        _, mdc_files = self._parsed()
        assert mdc_files == set(RESTRICTED_FILES), (
            "living-docs-only.mdc files disagree with RESTRICTED_FILES; "
            "update both together.\n"
            f"  only in .mdc:      {sorted(mdc_files - set(RESTRICTED_FILES))}\n"
            f"  only in constants: {sorted(set(RESTRICTED_FILES) - mdc_files)}"
        )

    def test_parser_found_nonempty_sets(self) -> None:
        # Fail loud if the parse comes back empty (heading text changed or the
        # list was reformatted as a markdown table) instead of letting the
        # equality checks vacuously compare the constants against empty sets.
        mdc_prefixes, mdc_files = self._parsed()
        assert mdc_prefixes, "no prefix dirs parsed from .mdc tombstone list"
        assert mdc_files, "no files parsed from .mdc tombstone list"

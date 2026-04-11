"""
SemEval-2013 Task 7 — Student Response Analysis, five-way gold labels (mirror).

Uses public GitHub mirror ashudeep/Student-Response-Analysis (not the official
LDC distribution). Pinned commit for reproducible raw.githubusercontent.com URLs.

Gold / label files under ``semevalFormatProcessing-5way/`` plus repository README.
"""

from __future__ import annotations

from typing import Any, Dict, List

from study_query_llm.datasets.acquisition import FileFetchSpec

SEMEVAL2013_SRA_5WAY_SLUG = "semeval2013_sra_5way"
GITHUB_ORG = "ashudeep"
GITHUB_REPO = "Student-Response-Analysis"
# Tip of master at integration time (reproducible mirror).
PINNED_GIT_REF = "1d6d30b265e6038fd6f6395d4cfd6686aef4b97f"

_PREFIX = "semevalFormatProcessing-5way"
_GOLD_FILES = (
    "trainingGold.txt",
    "trainingGold-partial.txt",
    "testGold-UA.txt",
    "testGold-UQ.txt",
    "partialEntailmentGold.txt",
)


def _raw_url(repo_relative_path: str) -> str:
    rel = repo_relative_path.lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{GITHUB_ORG}/"
        f"{GITHUB_REPO}/{PINNED_GIT_REF}/{rel}"
    )


def semeval2013_sra_5way_file_specs() -> List[FileFetchSpec]:
    specs: List[FileFetchSpec] = []
    for name in _GOLD_FILES:
        rel = f"{_PREFIX}/{name}"
        specs.append(FileFetchSpec(relative_path=rel, url=_raw_url(rel)))
    specs.append(FileFetchSpec(relative_path="README.md", url=_raw_url("README.md")))
    return specs


def semeval2013_sra_5way_source_metadata() -> Dict[str, Any]:
    return {
        "kind": "github_raw",
        "organization": GITHUB_ORG,
        "repository": GITHUB_REPO,
        "git_ref": PINNED_GIT_REF,
        "description": "SemEval-2013 Task 7 SRA five-way gold files (community mirror)",
        "note": "Official corpus may require LDC; this pin is the ashudeep/Student-Response-Analysis mirror.",
    }

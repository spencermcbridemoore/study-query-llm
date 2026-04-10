"""
AuSeM (Automating Sensemaking Measurements) — public CSVs from tufts-ml/AuSeM.

Pinned ref: commit on main at documentation time (reproducible raw.githubusercontent.com URLs).
"""

from __future__ import annotations

from typing import Any, Dict, List

from study_query_llm.datasets.acquisition import FileFetchSpec

AUSEM_DATASET_SLUG = "ausem"
AUSEM_GITHUB_ORG = "tufts-ml"
AUSEM_GITHUB_REPO = "AuSeM"
# Pinned commit (repo main as of integration); update intentionally when bumping data version.
AUSEM_PINNED_GIT_REF = "271b93baa0ab9aae70806c7364a4c4304f927143"

_STUDENT_EXPLANATION_CSV = (
    "problem1.csv",
    "problem2.csv",
    "problem3.csv",
    "problem4.csv",
)


def ausem_raw_url(relative_repo_path: str) -> str:
    rel = relative_repo_path.lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{AUSEM_GITHUB_ORG}/"
        f"{AUSEM_GITHUB_REPO}/{AUSEM_PINNED_GIT_REF}/{rel}"
    )


def ausem_file_specs() -> List[FileFetchSpec]:
    """Four Student_Explanations problem CSVs."""
    specs: List[FileFetchSpec] = []
    for name in _STUDENT_EXPLANATION_CSV:
        rel = f"Student_Explanations/{name}"
        specs.append(FileFetchSpec(relative_path=rel, url=ausem_raw_url(rel)))
    return specs


def ausem_source_metadata() -> Dict[str, Any]:
    """`source` block for acquisition.json."""
    return {
        "kind": "github_raw",
        "organization": AUSEM_GITHUB_ORG,
        "repository": AUSEM_GITHUB_REPO,
        "git_ref": AUSEM_PINNED_GIT_REF,
        "description": "AuSeM Student_Explanations problem CSVs",
    }

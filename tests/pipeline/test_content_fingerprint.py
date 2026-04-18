"""Tests for acquisition.content_fingerprint()."""

from __future__ import annotations

from copy import deepcopy

from study_query_llm.datasets.acquisition import content_fingerprint


def _manifest() -> dict:
    return {
        "schema_version": "1.0",
        "dataset_slug": "bank77",
        "acquired_at": "2026-04-17T00:00:00+00:00",
        "source": {
            "kind": "huggingface_resolve",
            "pinning_identity": {"dataset": "mteb/banking77", "commit": "abc123"},
        },
        "files": [
            {"relative_path": "train.parquet", "sha256": "a" * 64, "byte_size": 100},
            {"relative_path": "test.parquet", "sha256": "b" * 64, "byte_size": 80},
        ],
    }


def test_content_fingerprint_ignores_acquired_at() -> None:
    manifest_a = _manifest()
    manifest_b = deepcopy(manifest_a)
    manifest_b["acquired_at"] = "2026-04-17T23:59:59+00:00"

    fp_a = content_fingerprint(dataset_slug="bank77", manifest=manifest_a)
    fp_b = content_fingerprint(dataset_slug="bank77", manifest=manifest_b)
    assert fp_a == fp_b


def test_content_fingerprint_ignores_file_order() -> None:
    manifest_a = _manifest()
    manifest_b = deepcopy(manifest_a)
    manifest_b["files"] = list(reversed(manifest_b["files"]))

    fp_a = content_fingerprint(dataset_slug="bank77", manifest=manifest_a)
    fp_b = content_fingerprint(dataset_slug="bank77", manifest=manifest_b)
    assert fp_a == fp_b


def test_content_fingerprint_changes_when_sha_changes() -> None:
    manifest_a = _manifest()
    manifest_b = deepcopy(manifest_a)
    manifest_b["files"][0]["sha256"] = "c" * 64

    fp_a = content_fingerprint(dataset_slug="bank77", manifest=manifest_a)
    fp_b = content_fingerprint(dataset_slug="bank77", manifest=manifest_b)
    assert fp_a != fp_b

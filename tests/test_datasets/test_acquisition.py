"""Tests for layer-0 dataset acquisition (fetch, manifest, bundle write)."""

import json
from pathlib import Path

import pytest

from study_query_llm.datasets.acquisition import (
    ACQUISITION_SCHEMA_VERSION,
    FileFetchSpec,
    acquisition_manifest_sha256,
    build_acquisition_manifest,
    download_acquisition_files,
    sha256_hex,
    write_acquisition_bundle,
    zenodo_file_download_url,
)
from study_query_llm.datasets.source_specs.ausem import parse_ausem_snapshot
from study_query_llm.datasets.source_specs.registry import ACQUIRE_REGISTRY


def test_sha256_hex():
    assert sha256_hex(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_download_acquisition_files_with_injected_fetch():
    specs = [
        FileFetchSpec(relative_path="a/p1.csv", url="http://example.com/1"),
        FileFetchSpec(relative_path="a/p2.csv", url="http://example.com/2"),
    ]

    def fake_fetch(url: str) -> bytes:
        return url.encode("utf-8")

    got = download_acquisition_files(specs, fetch=fake_fetch)
    assert len(got) == 2
    assert got[0].relative_path == "a/p1.csv"
    assert got[0].data == b"http://example.com/1"
    assert got[0].sha256 == sha256_hex(b"http://example.com/1")
    assert got[0].byte_size == len(b"http://example.com/1")


def test_build_acquisition_manifest_sorts_files():
    specs = [
        FileFetchSpec("z.csv", "http://x/z"),
        FileFetchSpec("a.csv", "http://x/a"),
    ]
    fetched = download_acquisition_files(specs, fetch=lambda u: b"x")
    manifest = build_acquisition_manifest(
        dataset_slug="test_ds",
        source={"kind": "test"},
        files=fetched,
        runner_script="scripts/foo.py",
        extra_runner={"note": "unit"},
    )
    assert manifest["schema_version"] == ACQUISITION_SCHEMA_VERSION
    assert manifest["dataset_slug"] == "test_ds"
    paths = [f["relative_path"] for f in manifest["files"]]
    assert paths == ["a.csv", "z.csv"]


def test_write_acquisition_bundle_roundtrip(tmp_path: Path):
    specs = [FileFetchSpec("sub/x.txt", "http://example.com/x")]
    fetched = download_acquisition_files(specs, fetch=lambda u: b"hello")
    manifest = build_acquisition_manifest(
        dataset_slug="t",
        source={"kind": "x"},
        files=fetched,
        runner_script="scripts/x.py",
    )
    mp = write_acquisition_bundle(tmp_path, manifest, fetched)
    assert mp.name == "acquisition.json"
    loaded = json.loads(mp.read_text(encoding="utf-8"))
    assert loaded["dataset_slug"] == "t"
    assert (tmp_path / "files" / "sub" / "x.txt").read_bytes() == b"hello"


def test_acquisition_manifest_sha256_stable_for_identical_dict():
    m = {
        "schema_version": ACQUISITION_SCHEMA_VERSION,
        "dataset_slug": "s",
        "acquired_at": "fixed",
        "source": {"kind": "k"},
        "runner": {"script": "x.py"},
        "files": [{"relative_path": "a.csv", "url": "http://x", "sha256": "abc", "byte_size": 3}],
    }
    assert acquisition_manifest_sha256(m) == acquisition_manifest_sha256(m)


def test_acquire_registry_contains_ausem():
    assert "ausem" in ACQUIRE_REGISTRY
    cfg = ACQUIRE_REGISTRY["ausem"]
    assert cfg.default_parser is parse_ausem_snapshot
    files = cfg.file_specs()
    assert len(files) == 4
    assert all("Student_Explanations/problem" in f.relative_path for f in files)
    assert all("raw.githubusercontent.com" in f.url for f in files)
    meta = cfg.source_metadata()
    assert meta["kind"] == "github_raw"
    assert meta["git_ref"]


def test_zenodo_file_download_url():
    u = zenodo_file_download_url(16912394, "sources_v2.xlsx")
    assert u == "https://zenodo.org/records/16912394/files/sources_v2.xlsx?download=1"


def test_zenodo_file_download_url_rejects_path_traversal():
    with pytest.raises(ValueError):
        zenodo_file_download_url(1, "../etc/passwd")
    with pytest.raises(ValueError):
        zenodo_file_download_url(1, "a/b")


def test_acquire_registry_sources_uncertainty_qc():
    cfg = ACQUIRE_REGISTRY["sources_uncertainty_qc"]
    files = cfg.file_specs()
    assert len(files) == 1
    assert files[0].relative_path == "sources_v2.xlsx"
    assert "zenodo.org/records/16912394/files/sources_v2.xlsx" in files[0].url
    meta = cfg.source_metadata()
    assert meta["kind"] == "zenodo"
    assert meta["record_id"] == 16912394


def test_acquire_registry_semeval2013_sra_5way():
    cfg = ACQUIRE_REGISTRY["semeval2013_sra_5way"]
    files = cfg.file_specs()
    assert len(files) == 7
    paths = sorted(f.relative_path for f in files)
    assert paths[0] == "README.md"
    assert "semevalFormatProcessing-5way/answers.csv" in paths
    assert "semevalFormatProcessing-5way/partialEntailmentGold.txt" in paths
    assert all(
        "1d6d30b265e6038fd6f6395d4cfd6686aef4b97f" in f.url for f in files
    )
    meta = cfg.source_metadata()
    assert meta["kind"] == "github_raw"
    assert "ldc" in meta["note"].lower()

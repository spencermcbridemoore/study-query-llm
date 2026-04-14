"""Unit tests for verify_call_artifact_blob_lanes URL parsing helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "verify_call_artifact_blob_lanes.py"


@pytest.fixture(scope="module")
def lane_mod():
    spec = importlib.util.spec_from_file_location("verify_call_artifact_blob_lanes", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_azure_blob_container_from_uri_dev(lane_mod):
    u = (
        "https://acct.blob.core.windows.net/artifacts-dev/dev/42/step/file.json"
    )
    assert lane_mod.azure_blob_container_from_uri(u) == "artifacts-dev"


def test_azure_blob_container_from_uri_plain_artifacts(lane_mod):
    u = "https://acct.blob.core.windows.net/artifacts/dev/1/x.npy"
    assert lane_mod.azure_blob_container_from_uri(u) == "artifacts"


def test_azure_blob_container_from_uri_local_path(lane_mod):
    assert lane_mod.azure_blob_container_from_uri("/data/artifacts/1/x.json") is None


def test_azure_blob_path_after_container(lane_mod):
    u = "https://acct.blob.core.windows.net/artifacts-dev/dev/42/a.json"
    assert lane_mod.azure_blob_path_after_container(u) == "dev/42/a.json"


def test_azure_blob_path_after_container_no_key(lane_mod):
    u = "https://acct.blob.core.windows.net/artifacts-dev"
    assert lane_mod.azure_blob_path_after_container(u) == ""


@pytest.mark.skipif(not SCRIPT.is_file(), reason="script missing")
def test_non_https_not_azure_blob(lane_mod):
    assert lane_mod.azure_blob_container_from_uri("http://acct.blob.core.windows.net/x/y") is None

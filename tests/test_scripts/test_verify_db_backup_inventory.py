"""Unit tests for scripts/verify_db_backup_inventory.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts" / "verify_db_backup_inventory.py"


@pytest.fixture
def verify_mod():
    spec = importlib.util.spec_from_file_location("verify_db_backup_inventory", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_env(monkeypatch: pytest.MonkeyPatch, *, jet: str, local: str, azure: str) -> None:
    monkeypatch.setenv("JETSTREAM_DATABASE_URL", jet)
    monkeypatch.setenv("LOCAL_DATABASE_URL", local)
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azure)


def test_main_success_path_returns_zero(verify_mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify_mod, "load_dotenv", lambda *_a, **_k: None)
    _set_env(
        monkeypatch,
        jet="postgresql://jet.example:5432/db",
        local="postgresql://local.example:5432/db",
        azure="UseDevelopmentStorage=true",
    )
    monkeypatch.setattr(
        verify_mod,
        "_table_counts",
        lambda _url, _label: {
            "raw_calls": 1,
            "groups": 2,
            "group_members": 3,
            "group_links": 4,
            "embedding_vectors": 5,
            "call_artifacts": 6,
        },
    )
    monkeypatch.setattr(
        verify_mod,
        "_load_manifests",
        lambda: (
            [
                {
                    "backup_id": "b1",
                    "source": "jetstream",
                    "table_counts": {"groups": 2},
                    "blob_uri": "https://example.blob.core.windows.net/db-backups/b1.dump",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        verify_mod,
        "_list_db_backup_blobs",
        lambda _conn: [("b1.dump", 1024)],
    )

    assert verify_mod.main([]) == 0


def test_main_returns_nonzero_on_db_count_mismatch(
    verify_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_mod, "load_dotenv", lambda *_a, **_k: None)
    _set_env(
        monkeypatch,
        jet="postgresql://jet.example:5432/db",
        local="postgresql://local.example:5432/db",
        azure="",
    )

    def _fake_counts(_url: str, label: str):
        if "Jetstream" in label:
            return {"groups": 20}
        return {"groups": 19}

    monkeypatch.setattr(verify_mod, "_table_counts", _fake_counts)
    monkeypatch.setattr(verify_mod, "_load_manifests", lambda: ([], []))

    assert verify_mod.main([]) == 1


def test_main_returns_nonzero_on_azure_listing_exception(
    verify_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_mod, "load_dotenv", lambda *_a, **_k: None)
    _set_env(
        monkeypatch,
        jet="",
        local="",
        azure="UseDevelopmentStorage=true",
    )
    monkeypatch.setattr(verify_mod, "_table_counts", lambda _url, _label: None)
    monkeypatch.setattr(verify_mod, "_load_manifests", lambda: ([], []))

    def _raise_listing_error(_conn: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(verify_mod, "_list_db_backup_blobs", _raise_listing_error)

    assert verify_mod.main([]) == 1


def test_main_returns_nonzero_on_malformed_manifest(
    verify_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_mod, "load_dotenv", lambda *_a, **_k: None)
    _set_env(monkeypatch, jet="", local="", azure="")
    monkeypatch.setattr(verify_mod, "_table_counts", lambda _url, _label: None)
    monkeypatch.setattr(
        verify_mod,
        "_load_manifests",
        lambda: ([], ["bad.manifest.json: JSONDecodeError: bad json"]),
    )

    assert verify_mod.main([]) == 1


def test_main_returns_nonzero_on_manifest_blob_mismatch(
    verify_mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(verify_mod, "load_dotenv", lambda *_a, **_k: None)
    _set_env(monkeypatch, jet="", local="", azure="UseDevelopmentStorage=true")
    monkeypatch.setattr(verify_mod, "_table_counts", lambda _url, _label: None)
    monkeypatch.setattr(
        verify_mod,
        "_load_manifests",
        lambda: (
            [
                {
                    "backup_id": "b2",
                    "source": "jetstream",
                    "table_counts": {"groups": 2},
                    "blob_uri": "https://example.blob.core.windows.net/db-backups/missing.dump",
                }
            ],
            [],
        ),
    )
    monkeypatch.setattr(
        verify_mod,
        "_list_db_backup_blobs",
        lambda _conn: [("present.dump", 50)],
    )

    assert verify_mod.main([]) == 1

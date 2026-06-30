"""Tests for study_query_llm.vllm_serving.hf_download.snapshot_to_local.

Fully mocked: the only network call lives behind the injectable ``_downloader``
seam, so these tests pass a stub and never touch HuggingFace / the network.
"""

from pathlib import Path
from unittest.mock import MagicMock

from study_query_llm.vllm_serving.hf_download import (
    snapshot_to_local,
    default_models_dir,
)


# ---------------------------------------------------------------------------
# _downloader is called with the contract kwargs
# ---------------------------------------------------------------------------

def test_snapshot_calls_downloader_with_repo_and_dest(tmp_path):
    """_downloader is invoked with repo_id, local_dir, and ignore_patterns."""
    downloader = MagicMock(return_value=str(tmp_path))

    snapshot_to_local("Qwen/Qwen2.5-7B-Instruct-AWQ", str(tmp_path), _downloader=downloader)

    downloader.assert_called_once()
    kwargs = downloader.call_args.kwargs
    assert kwargs["repo_id"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert kwargs["local_dir"] == str(tmp_path)


def test_snapshot_uses_keyword_arguments_only(tmp_path):
    """_downloader is called with keyword args (no positional repo/dest)."""
    downloader = MagicMock(return_value=str(tmp_path))

    snapshot_to_local("org/model", str(tmp_path), _downloader=downloader)

    assert downloader.call_args.args == ()
    assert set(downloader.call_args.kwargs) == {"repo_id", "local_dir", "ignore_patterns"}


def test_snapshot_default_ignore_patterns(tmp_path):
    """Default ignore_patterns ('*.pt', '*.bin') are passed as a list."""
    downloader = MagicMock(return_value=str(tmp_path))

    snapshot_to_local("org/model", str(tmp_path), _downloader=downloader)

    assert downloader.call_args.kwargs["ignore_patterns"] == ["*.pt", "*.bin"]


def test_snapshot_custom_ignore_patterns_forwarded_as_list(tmp_path):
    """Custom ignore_patterns are forwarded, coerced to a list."""
    downloader = MagicMock(return_value=str(tmp_path))

    snapshot_to_local(
        "org/model",
        str(tmp_path),
        ignore_patterns=("*.gguf", "*.onnx"),
        _downloader=downloader,
    )

    forwarded = downloader.call_args.kwargs["ignore_patterns"]
    assert forwarded == ["*.gguf", "*.onnx"]
    assert isinstance(forwarded, list)


# ---------------------------------------------------------------------------
# return value is the resolved absolute dest dir
# ---------------------------------------------------------------------------

def test_snapshot_returns_resolved_absolute_dest(tmp_path):
    """Return value is the resolved (absolute) dest dir, not the downloader's."""
    # Give the downloader a different return value to prove the function returns
    # the *resolved dest_dir*, not whatever the downloader hands back.
    downloader = MagicMock(return_value="/some/other/path/from/downloader")

    result = snapshot_to_local("org/model", str(tmp_path), _downloader=downloader)

    assert result == str(Path(tmp_path).expanduser().resolve())
    assert Path(result).is_absolute()


def test_snapshot_resolves_relative_dest_to_absolute(monkeypatch, tmp_path):
    """A relative dest dir is resolved to an absolute path on return."""
    monkeypatch.chdir(tmp_path)
    downloader = MagicMock(return_value="models")

    result = snapshot_to_local("org/model", "models", _downloader=downloader)

    # local_dir is forwarded exactly as given (relative)...
    assert downloader.call_args.kwargs["local_dir"] == "models"
    # ...but the returned path is absolute.
    assert Path(result).is_absolute()
    assert result == str((tmp_path / "models").resolve())


# ---------------------------------------------------------------------------
# no network: importing / calling never reaches huggingface_hub
# ---------------------------------------------------------------------------

def test_snapshot_does_not_touch_network(monkeypatch, tmp_path):
    """With an injected _downloader, the real huggingface_hub is never imported."""
    import study_query_llm.vllm_serving.hf_download as mod

    sentinel = MagicMock(
        side_effect=AssertionError("default downloader must not be used in tests")
    )
    monkeypatch.setattr(mod, "_default_downloader", sentinel)

    downloader = MagicMock(return_value=str(tmp_path))
    snapshot_to_local("org/model", str(tmp_path), _downloader=downloader)

    sentinel.assert_not_called()
    downloader.assert_called_once()


# ---------------------------------------------------------------------------
# default_models_dir
# ---------------------------------------------------------------------------

def test_default_models_dir():
    """default_models_dir is ~/vllm_models."""
    assert default_models_dir() == str(Path.home() / "vllm_models")

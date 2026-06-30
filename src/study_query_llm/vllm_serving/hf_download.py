"""
Host-side HuggingFace snapshot download for the local-backend TLS workaround.

Why this module exists
~~~~~~~~~~~~~~~~~~~~~~~~
On the operator's LOCAL Windows network an intercepting TLS proxy presents a CA
that Windows trusts but that is *not* trusted inside the vLLM container.  As a
result the container cannot reach huggingface.co directly
(``CERTIFICATE_VERIFY_FAILED``).  The verified workaround is to download the
model snapshot on the HOST (where TLS works), then mount that directory into the
container read-only and run vLLM fully offline
(``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1``).

This is a PER-BACKEND concern: only the local Docker backend needs it.  The
vast.ai (Linux cloud) backend downloads in-container and does NOT use this
module.

Network policy
~~~~~~~~~~~~~~~
The only network call lives in the injected ``_downloader`` (defaulting to the
real ``huggingface_hub.snapshot_download``).  Tests pass their own
``_downloader`` so unit tests never touch the network.  ``huggingface_hub`` is
imported lazily inside the call so importing this module is side-effect free.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def _default_downloader() -> Callable[..., str]:
    """Return the real ``huggingface_hub.snapshot_download``.

    Imported lazily so that merely importing this module does not require
    ``huggingface_hub`` to be installed (e.g. in unit tests that inject their
    own downloader).
    """
    from huggingface_hub import snapshot_download  # local import: side-effect free module load

    return snapshot_download


def snapshot_to_local(
    repo: str,
    dest_dir: str,
    *,
    ignore_patterns=("*.pt", "*.bin"),
    _downloader: Optional[Callable[..., str]] = None,
) -> str:
    """Download a HuggingFace repo snapshot to ``dest_dir`` on the host.

    Used ONLY by the local Docker backend's TLS workaround: the host has working
    TLS, so the weights are fetched here and later mounted read-only into the
    offline container.

    Parameters
    ----------
    repo:
        HuggingFace repo id (e.g. ``"Qwen/Qwen2.5-7B-Instruct-AWQ"``).
    dest_dir:
        Target directory for the snapshot.  Created by ``snapshot_download`` if
        absent.  The resolved absolute path is returned.
    ignore_patterns:
        Glob patterns to skip.  Defaults to ``("*.pt", "*.bin")`` so we prefer
        the safetensors weights and avoid pulling redundant/pickled formats.
    _downloader:
        Injectable seam.  Defaults to ``huggingface_hub.snapshot_download``.
        Called as ``_downloader(repo_id=repo, local_dir=dest_dir,
        ignore_patterns=list(ignore_patterns))``.  Tests inject a stub so no
        network is touched.

    Returns
    -------
    str
        The resolved (absolute) ``dest_dir`` to mount into the container.
    """
    downloader = _downloader if _downloader is not None else _default_downloader()

    logger.info(
        "Downloading HF snapshot host-side (TLS workaround): repo=%s dest=%s ignore=%s",
        repo,
        dest_dir,
        ignore_patterns,
    )
    downloader(
        repo_id=repo,
        local_dir=dest_dir,
        ignore_patterns=list(ignore_patterns),
    )

    resolved = str(Path(dest_dir).expanduser().resolve())
    logger.info("HF snapshot ready at %s", resolved)
    return resolved


def default_models_dir() -> str:
    """Return the default host directory for downloaded vLLM model snapshots.

    Defaults to ``~/vllm_models``.  The local backend uses a per-model
    subdirectory beneath this.
    """
    return str(Path.home() / "vllm_models")

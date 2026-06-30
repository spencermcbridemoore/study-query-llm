"""
VRAM probing + display-safety headroom checks for the vLLM serving module.

The local-Docker backend serves vLLM on the operator's own workstation (an
RTX 4090, 24 GB).  vLLM is a VRAM hog, and a too-large
``--gpu-memory-utilization`` has crashed the operator's *display* in the past
(the GPU is shared with the desktop compositor / other apps).  To make local
launches safe this module provides two pieces:

* :func:`query_vram` -- read the current GPU memory state.  It prefers the
  ``pynvml`` (NVML) bindings and falls back to shelling out to ``nvidia-smi``
  when ``pynvml`` is unavailable or fails to initialise.  The NVML handle is
  injectable (``_nvml=``) so tests never touch a real GPU.

* :func:`assert_vram_headroom` -- pure arithmetic.  Given a
  :class:`VRAMStatus`, the requested ``gpu_memory_utilization``, a hard
  ``ceiling_mb`` and a ``headroom_mb`` buffer, it decides whether a launch is
  safe and either returns the required-MB figure or raises
  :class:`VRAMSafetyError` with a clear, number-bearing message.  Because it
  takes a constructed ``VRAMStatus`` it is trivially unit-testable with no GPU.

vLLM's ``--gpu-memory-utilization`` reserves the given *fraction of total* VRAM
for the model+KV-cache, so the bytes a launch will consume is
``int(gpu_memory_utilization * total_mb)`` -- independent of how much is free
right now.  The two refusal conditions are therefore:

1. the cap itself is larger than the configured ceiling (the operator asked for
   more than they decided is ever safe on this card), and
2. there is not enough *currently free* VRAM to satisfy the cap plus a headroom
   buffer (launching now would starve the display / other apps and could crash
   them).
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# The largest --gpu-memory-utilization-derived reservation we will ever permit
# on the local card by default.  On a 24 GB RTX 4090 this leaves a comfortable
# slice for the desktop/display.  Override per-backend via vram_ceiling_mb.
DEFAULT_VRAM_CEILING_MB = 16000

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class VRAMStatus:
    """Snapshot of a single GPU's memory state, in whole megabytes."""

    index: int
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


class VRAMSafetyError(RuntimeError):
    """Raised when launching vLLM would be unsafe for the local display/GPU."""


def _bytes_to_mb(num_bytes: int) -> int:
    """Convert a byte count to whole megabytes (floor)."""
    return int(num_bytes // _BYTES_PER_MB)


def _query_vram_pynvml(index: int, nvml) -> VRAMStatus:
    """Read VRAM via the NVML bindings (``pynvml`` or an injected stand-in).

    ``nvml`` is a module-like object exposing the standard ``nvml*`` functions.
    ``nvmlShutdown`` is always attempted in a ``finally`` so we never leak the
    NVML session.
    """
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        mem = nvml.nvmlDeviceGetMemoryInfo(handle)

        raw_name = nvml.nvmlDeviceGetName(handle)
        # NVML returns ``bytes`` on some versions and ``str`` on others.
        if isinstance(raw_name, bytes):
            name = raw_name.decode("utf-8", "replace")
        else:
            name = str(raw_name)

        return VRAMStatus(
            index=index,
            name=name,
            total_mb=_bytes_to_mb(int(mem.total)),
            free_mb=_bytes_to_mb(int(mem.free)),
            used_mb=_bytes_to_mb(int(mem.used)),
        )
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception as exc:  # pragma: no cover - shutdown best-effort
            logger.debug("[vram] nvmlShutdown failed (ignored): %s", exc)


def _query_vram_nvidia_smi(index: int) -> VRAMStatus:
    """Fallback: parse ``nvidia-smi`` CSV output (no pynvml available)."""
    cmd = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.total,memory.free,memory.used,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise VRAMSafetyError(
            "Could not read GPU VRAM: neither pynvml nor nvidia-smi is usable "
            f"(nvidia-smi invocation failed: {exc}). Refusing to launch vLLM "
            "without a VRAM safety check. Pass enable_vram_check=False to "
            "override (NOT recommended on a machine with an active display)."
        ) from exc

    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        raise VRAMSafetyError(
            f"Could not parse nvidia-smi output for VRAM check: {line!r}"
        )

    try:
        total_mb = int(float(parts[0]))
        free_mb = int(float(parts[1]))
        used_mb = int(float(parts[2]))
    except ValueError as exc:
        raise VRAMSafetyError(
            f"Could not parse nvidia-smi VRAM numbers from {line!r}: {exc}"
        ) from exc

    name = parts[3] if len(parts) >= 4 else "unknown"

    return VRAMStatus(
        index=index,
        name=name,
        total_mb=total_mb,
        free_mb=free_mb,
        used_mb=used_mb,
    )


def query_vram(index: int = 0, *, _nvml=None) -> VRAMStatus:
    """Return the current memory state of GPU ``index`` as a :class:`VRAMStatus`.

    Prefers NVML via ``pynvml``; if ``pynvml`` cannot be imported or
    ``nvmlInit`` fails, falls back to parsing ``nvidia-smi``.

    Args:
        index: Zero-based GPU index to query.
        _nvml: Test seam.  When provided it is used directly as the NVML
            module-like object (a real or fake ``pynvml``); ``pynvml`` is not
            imported.  Tests inject a stub here to avoid touching a real GPU.

    Raises:
        VRAMSafetyError: If neither NVML nor ``nvidia-smi`` can produce a
            reading.  (We treat an unreadable GPU as a refusal-to-launch rather
            than silently proceeding without the safety check.)
    """
    if _nvml is not None:
        return _query_vram_pynvml(index, _nvml)

    try:
        import pynvml  # type: ignore
    except Exception as exc:
        logger.debug(
            "[vram] pynvml unavailable (%s); falling back to nvidia-smi", exc
        )
        return _query_vram_nvidia_smi(index)

    try:
        return _query_vram_pynvml(index, pynvml)
    except VRAMSafetyError:
        raise
    except Exception as exc:
        logger.debug(
            "[vram] pynvml query failed (%s); falling back to nvidia-smi", exc
        )
        return _query_vram_nvidia_smi(index)


def assert_vram_headroom(
    *,
    gpu_memory_utilization: float,
    status: VRAMStatus,
    ceiling_mb: int,
    headroom_mb: int = 512,
) -> int:
    """Verify it is safe to launch vLLM, or raise :class:`VRAMSafetyError`.

    This is pure arithmetic over a constructed :class:`VRAMStatus`, so it is
    trivially unit-testable without a GPU.

    vLLM's ``--gpu-memory-utilization`` reserves a fraction of *total* VRAM, so
    the launch will consume ``int(gpu_memory_utilization * status.total_mb)``
    megabytes regardless of current free memory.

    Two refusal conditions:

    1. **Cap too large** -- the reservation exceeds ``ceiling_mb`` (the
       operator's hard limit for this card).  Launching would risk the display
       even on an otherwise-idle GPU.
    2. **Not enough free now** -- ``status.free_mb`` is below
       ``required_mb + headroom_mb``.  Launching would starve the display /
       other apps that are currently using the card and could crash them.

    Args:
        gpu_memory_utilization: The fraction passed to vLLM (0.0-1.0).
        status: A snapshot of GPU memory (from :func:`query_vram` or built in a
            test).
        ceiling_mb: Hard upper bound on the reservation we will ever permit.
        headroom_mb: Buffer of free VRAM to leave for the display / other apps.

    Returns:
        ``required_mb`` -- the megabytes the launch will reserve.

    Raises:
        VRAMSafetyError: If either refusal condition holds.
    """
    required_mb = int(gpu_memory_utilization * status.total_mb)

    if required_mb > ceiling_mb:
        raise VRAMSafetyError(
            f"Refusing to launch vLLM on GPU {status.index} ({status.name}): "
            f"requested gpu_memory_utilization={gpu_memory_utilization} of "
            f"{status.total_mb} MB total = {required_mb} MB, which exceeds the "
            f"configured VRAM ceiling of {ceiling_mb} MB. Lower "
            f"gpu_memory_utilization or raise the ceiling if you are certain "
            f"the display can tolerate it."
        )

    needed_free = required_mb + headroom_mb
    if status.free_mb < needed_free:
        raise VRAMSafetyError(
            f"Refusing to launch vLLM on GPU {status.index} ({status.name}): "
            f"need {required_mb} MB (cap) + {headroom_mb} MB headroom = "
            f"{needed_free} MB free, but only {status.free_mb} MB is free now "
            f"(total {status.total_mb} MB, used {status.used_mb} MB). "
            f"Close other GPU apps or lower gpu_memory_utilization; launching "
            f"now could crash the display or other GPU processes."
        )

    logger.info(
        "[vram] GPU %s (%s): cap=%d MB, free=%d MB, headroom=%d MB -- OK to launch",
        status.index,
        status.name,
        required_mb,
        status.free_mb,
        headroom_mb,
    )
    return required_mb

"""Tests for study_query_llm.vllm_serving.vram.

Fully mocked: no real GPU, no NVML, no nvidia-smi.  ``assert_vram_headroom``
is pure arithmetic over a constructed :class:`VRAMStatus`, and ``query_vram``
is exercised via the ``_nvml=`` injection seam with a fake NVML module-like
object, so nothing here touches a physical card.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from study_query_llm.vllm_serving.vram import (
    DEFAULT_VRAM_CEILING_MB,
    VRAMSafetyError,
    VRAMStatus,
    assert_vram_headroom,
    query_vram,
)


_MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _status(*, total_mb=24000, free_mb=20000, used_mb=4000, index=0, name="RTX 4090") -> VRAMStatus:
    """Build a VRAMStatus snapshot for the arithmetic tests."""
    return VRAMStatus(
        index=index,
        name=name,
        total_mb=total_mb,
        free_mb=free_mb,
        used_mb=used_mb,
    )


class _FakeNVML:
    """A module-like stand-in for pynvml that records calls and returns
    fixed memory numbers (in bytes), so query_vram never touches a GPU."""

    def __init__(self, *, total_mb=24000, free_mb=20000, used_mb=4000, name="NVIDIA GeForce RTX 4090"):
        self._total = total_mb * _MB
        self._free = free_mb * _MB
        self._used = used_mb * _MB
        self._name = name
        self.init_called = 0
        self.shutdown_called = 0
        self.handle_index = None

    def nvmlInit(self):
        self.init_called += 1

    def nvmlShutdown(self):
        self.shutdown_called += 1

    def nvmlDeviceGetHandleByIndex(self, index):
        self.handle_index = index
        return SimpleNamespace(index=index)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=self._total, free=self._free, used=self._used)

    def nvmlDeviceGetName(self, handle):
        return self._name


# ---------------------------------------------------------------------------
# assert_vram_headroom -- happy path
# ---------------------------------------------------------------------------

def test_assert_vram_headroom_passes_with_ample_free_and_small_cap():
    """Ample free VRAM + a cap well under the ceiling -> returns required_mb."""
    status = _status(total_mb=24000, free_mb=22000, used_mb=2000)
    required = assert_vram_headroom(
        gpu_memory_utilization=0.4,
        status=status,
        ceiling_mb=16000,
        headroom_mb=512,
    )
    # 0.4 * 24000 = 9600 MB reserved.
    assert required == 9600


def test_assert_vram_headroom_returns_int_required_mb():
    """The returned figure floors the fractional product to an int."""
    status = _status(total_mb=24000, free_mb=24000, used_mb=0)
    required = assert_vram_headroom(
        gpu_memory_utilization=0.123,
        status=status,
        ceiling_mb=DEFAULT_VRAM_CEILING_MB,
    )
    assert required == int(0.123 * 24000)  # 2952
    assert isinstance(required, int)


def test_assert_vram_headroom_passes_exactly_at_headroom_boundary():
    """free == required + headroom is acceptable (strict < is the refusal)."""
    status = _status(total_mb=24000, free_mb=9600 + 512, used_mb=0)
    required = assert_vram_headroom(
        gpu_memory_utilization=0.4,
        status=status,
        ceiling_mb=16000,
        headroom_mb=512,
    )
    assert required == 9600


def test_assert_vram_headroom_passes_at_ceiling_boundary():
    """required == ceiling is acceptable (strict > is the refusal)."""
    status = _status(total_mb=24000, free_mb=24000, used_mb=0)
    required = assert_vram_headroom(
        gpu_memory_utilization=0.5,
        status=status,
        ceiling_mb=12000,  # 0.5 * 24000 == 12000 exactly
        headroom_mb=0,
    )
    assert required == 12000


# ---------------------------------------------------------------------------
# assert_vram_headroom -- cap-too-large refusal
# ---------------------------------------------------------------------------

def test_assert_vram_headroom_raises_when_required_exceeds_ceiling():
    """required_mb > ceiling_mb -> VRAMSafetyError (cap itself too large)."""
    status = _status(total_mb=24000, free_mb=24000, used_mb=0)
    with pytest.raises(VRAMSafetyError) as exc_info:
        assert_vram_headroom(
            gpu_memory_utilization=0.9,  # 0.9 * 24000 = 21600 > 16000
            status=status,
            ceiling_mb=16000,
        )
    msg = str(exc_info.value)
    assert "21600" in msg  # required
    assert "16000" in msg  # ceiling


def test_assert_vram_headroom_cap_too_large_checked_before_free():
    """Cap-too-large fires even when there is plenty of free VRAM."""
    # free is huge, so only the ceiling condition can trip here.
    status = _status(total_mb=24000, free_mb=24000, used_mb=0)
    with pytest.raises(VRAMSafetyError):
        assert_vram_headroom(
            gpu_memory_utilization=0.8,  # 19200 > 1000
            status=status,
            ceiling_mb=1000,
        )


# ---------------------------------------------------------------------------
# assert_vram_headroom -- not-enough-free refusal
# ---------------------------------------------------------------------------

def test_assert_vram_headroom_raises_when_free_below_required_plus_headroom():
    """free_mb < required + headroom -> VRAMSafetyError (not enough free now)."""
    # required = 0.4 * 24000 = 9600; need 9600 + 512 = 10112 free, only 5000.
    status = _status(total_mb=24000, free_mb=5000, used_mb=19000)
    with pytest.raises(VRAMSafetyError) as exc_info:
        assert_vram_headroom(
            gpu_memory_utilization=0.4,
            status=status,
            ceiling_mb=16000,
            headroom_mb=512,
        )
    msg = str(exc_info.value)
    assert "9600" in msg     # required
    assert "10112" in msg    # required + headroom
    assert "5000" in msg     # free now


def test_assert_vram_headroom_raises_just_below_headroom_boundary():
    """One MB short of (required + headroom) still refuses."""
    status = _status(total_mb=24000, free_mb=9600 + 512 - 1, used_mb=0)
    with pytest.raises(VRAMSafetyError):
        assert_vram_headroom(
            gpu_memory_utilization=0.4,
            status=status,
            ceiling_mb=16000,
            headroom_mb=512,
        )


def test_assert_vram_headroom_default_headroom_is_512():
    """The default headroom_mb buffer is 512 MB."""
    # free == required + 511 -> below the 512 default buffer -> refuse.
    status = _status(total_mb=24000, free_mb=9600 + 511, used_mb=0)
    with pytest.raises(VRAMSafetyError):
        assert_vram_headroom(
            gpu_memory_utilization=0.4,
            status=status,
            ceiling_mb=16000,
        )
    # free == required + 512 -> exactly the default buffer -> pass.
    status_ok = _status(total_mb=24000, free_mb=9600 + 512, used_mb=0)
    assert (
        assert_vram_headroom(
            gpu_memory_utilization=0.4,
            status=status_ok,
            ceiling_mb=16000,
        )
        == 9600
    )


# ---------------------------------------------------------------------------
# query_vram -- injected fake NVML (no real GPU)
# ---------------------------------------------------------------------------

def test_query_vram_with_injected_nvml_returns_expected_status():
    """query_vram converts the fake NVML byte readings to whole-MB VRAMStatus."""
    fake = _FakeNVML(total_mb=24564, free_mb=20480, used_mb=4084, name="NVIDIA GeForce RTX 4090")
    status = query_vram(0, _nvml=fake)

    assert isinstance(status, VRAMStatus)
    assert status.index == 0
    assert status.name == "NVIDIA GeForce RTX 4090"
    assert status.total_mb == 24564
    assert status.free_mb == 20480
    assert status.used_mb == 4084


def test_query_vram_inits_and_shuts_down_nvml():
    """query_vram calls nvmlInit and always nvmlShutdown via the seam."""
    fake = _FakeNVML()
    query_vram(0, _nvml=fake)
    assert fake.init_called == 1
    assert fake.shutdown_called == 1


def test_query_vram_uses_requested_index():
    """The provided GPU index flows through to the handle lookup + status."""
    fake = _FakeNVML()
    status = query_vram(2, _nvml=fake)
    assert fake.handle_index == 2
    assert status.index == 2


def test_query_vram_decodes_bytes_name():
    """A bytes device name (older NVML) is decoded to str."""
    fake = _FakeNVML()
    fake.nvmlDeviceGetName = lambda handle: b"NVIDIA GeForce RTX 4090"
    status = query_vram(0, _nvml=fake)
    assert status.name == "NVIDIA GeForce RTX 4090"
    assert isinstance(status.name, str)


def test_query_vram_shuts_down_even_on_query_error():
    """nvmlShutdown still runs if a query call raises (finally cleanup)."""
    fake = _FakeNVML()

    def _boom(handle):
        raise RuntimeError("nvml query exploded")

    fake.nvmlDeviceGetMemoryInfo = _boom
    with pytest.raises(RuntimeError):
        query_vram(0, _nvml=fake)
    assert fake.shutdown_called == 1


def test_query_vram_result_feeds_assert_vram_headroom():
    """A status from the fake NVML composes with the headroom check."""
    fake = _FakeNVML(total_mb=24000, free_mb=22000, used_mb=2000)
    status = query_vram(0, _nvml=fake)
    required = assert_vram_headroom(
        gpu_memory_utilization=0.4,
        status=status,
        ceiling_mb=DEFAULT_VRAM_CEILING_MB,
    )
    assert required == 9600

"""
Vast.ai client seam for the vLLM-serving module.

This module defines the injectable abstraction over a cloud GPU provider so the
``VastAIBackend`` (in ``backends.py``) can provision, poll, and tear down a
remote instance without hard-wiring a specific vendor or transport.  The seam is
deliberately small and provider-agnostic: sibling clients for RunPod / Lambda /
etc. can subclass :class:`VastClient` without touching the backend.

Components
~~~~~~~~~~
* :class:`VastOffer`   -- a normalised search result (a rentable GPU offer).
* :class:`VastInstance`-- a normalised running/created instance with its SSH endpoint.
* :class:`VastClient`  -- the abstract seam (``search_offers`` / ``create_instance`` /
  ``get_instance`` / ``wait_until_running`` / ``destroy_instance``).
* :class:`VastCLIClient` -- the default implementation that shells out to the
  ``vastai`` CLI via ``subprocess`` and parses its JSON output.

IMPORTANT -- UNPROVEN END-TO-END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`VastCLIClient` has **not** been exercised against the live ``vastai`` CLI.
It is structured cleanly and mockably (the subprocess runner is injectable via
``_run``) so the calling code and tests can be developed without ever touching
the network or a paid cloud account.  Treat the exact CLI flag spellings and the
shape of the parsed JSON as best-effort guesses to be verified before any live
run.  Do not run this against a real account by default.

All side-effecting work (the CLI subprocess) goes through the injectable ``_run``
seam (default :func:`subprocess.run`); the polling sleep goes through ``_sleep``
(default :func:`time.sleep`).  Unit tests therefore never spawn a process or
block on a real clock.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Status vocabulary
# --------------------------------------------------------------------------- #
# vast.ai reports an instance's lifecycle via ``actual_status`` (and a coarser
# ``cur_state``).  We treat the instance as "running" once its status indicates a
# live container.  Kept as a module constant so tests and siblings can reuse it.
_RUNNING_STATUSES = frozenset({"running"})


# --------------------------------------------------------------------------- #
# Normalised value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VastOffer:
    """A rentable GPU offer returned by :meth:`VastClient.search_offers`.

    ``raw`` keeps the full provider payload so callers can inspect fields we do
    not normalise (region, reliability, inet speed, etc.) without a second call.
    """

    offer_id: str
    gpu_name: str
    num_gpus: int
    gpu_ram_mb: int
    dph: float  # dollars-per-hour for the whole offer (all GPUs)
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VastInstance:
    """A created / running instance returned by the lifecycle methods.

    ``ssh_host`` / ``ssh_port`` are the SSH endpoint used for local port
    forwarding; ``public_ip`` is set when the instance exposes mapped ports
    directly.  Any of the endpoint fields may be ``None`` before the instance is
    fully up (which is exactly what :meth:`VastClient.wait_until_running` waits
    for).  ``raw`` keeps the full provider payload.
    """

    instance_id: str
    status: str
    ssh_host: Optional[str]
    ssh_port: Optional[int]
    public_ip: Optional[str]
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Abstract seam
# --------------------------------------------------------------------------- #
class VastClient(ABC):
    """Injectable abstraction over a cloud GPU provider.

    The backend depends only on this interface, so a mock (in tests) or a
    sibling vendor implementation can be substituted freely.  Concrete clients
    must return the normalised :class:`VastOffer` / :class:`VastInstance` value
    objects above.
    """

    @abstractmethod
    def search_offers(
        self,
        *,
        gpu_name: str | None,
        min_gpu_ram_mb: int,
        num_gpus: int = 1,
        max_dph: float | None,
        limit: int = 10,
    ) -> list[VastOffer]:
        """Return matching rentable offers (newest/cheapest first is fine; the
        backend itself picks the cheapest)."""
        raise NotImplementedError

    @abstractmethod
    def create_instance(
        self,
        offer_id: str,
        *,
        image: str,
        env: dict | None = None,
        onstart: str | None = None,
        disk_gb: int = 40,
        label: str | None = None,
    ) -> VastInstance:
        """Provision an instance from ``offer_id`` running ``image`` with an
        optional ``onstart`` shell command and env vars.  Returns immediately
        after creation (the instance is typically still booting)."""
        raise NotImplementedError

    @abstractmethod
    def get_instance(self, instance_id: str) -> VastInstance:
        """Fetch the current state of an instance."""
        raise NotImplementedError

    def wait_until_running(
        self,
        instance_id: str,
        *,
        timeout: int = 600,
        interval: int = 10,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> VastInstance:
        """Poll :meth:`get_instance` until the instance is running with an SSH
        endpoint present, or ``timeout`` seconds elapse.

        "Running" requires BOTH a running status AND a reachable SSH endpoint
        (host + port), because the backend cannot establish its tunnel until the
        endpoint is published.  ``_sleep`` is injectable so tests advance the
        loop without a real clock.

        Raises:
            TimeoutError: if the instance does not reach a usable running state
                within ``timeout`` seconds.
        """
        deadline = time.monotonic() + timeout
        last: VastInstance | None = None
        while True:
            inst = self.get_instance(instance_id)
            last = inst
            if _is_running(inst):
                logger.info(
                    "[vast] Instance %s is running (status=%s, ssh=%s:%s)",
                    inst.instance_id,
                    inst.status,
                    inst.ssh_host,
                    inst.ssh_port,
                )
                return inst
            if time.monotonic() >= deadline:
                break
            logger.debug(
                "[vast] Waiting for instance %s to run (status=%s, ssh=%s:%s) ...",
                instance_id,
                inst.status,
                inst.ssh_host,
                inst.ssh_port,
            )
            _sleep(interval)

        status = last.status if last is not None else "unknown"
        raise TimeoutError(
            f"vast.ai instance {instance_id} did not reach a usable running state "
            f"(running status + SSH endpoint) within {timeout}s "
            f"(last status={status!r}). "
            f"Inspect manually with: vastai show instance {instance_id}"
        )

    @abstractmethod
    def destroy_instance(self, instance_id: str) -> None:
        """Tear down the instance.  Should be safe to call on an already-gone
        instance (the backend swallows errors, but implementations should not
        rely on that)."""
        raise NotImplementedError


def _is_running(inst: VastInstance) -> bool:
    """An instance is usable when it reports a running status AND publishes an
    SSH endpoint the backend can forward through."""
    status = (inst.status or "").strip().lower()
    has_ssh = bool(inst.ssh_host) and inst.ssh_port is not None
    return status in _RUNNING_STATUSES and has_ssh


# --------------------------------------------------------------------------- #
# Default implementation: subprocess over the "vastai" CLI
# --------------------------------------------------------------------------- #
class VastCLIClient(VastClient):
    """Default :class:`VastClient` that shells out to the ``vastai`` CLI.

    Each method maps to a ``vastai`` subcommand requesting JSON output where the
    CLI supports it (``--raw``), and parses the result into the normalised value
    objects.  The subprocess runner is injectable via ``_run`` (default
    :func:`subprocess.run`) so tests can stub the CLI entirely.

    UNPROVEN: the exact subcommand spellings, flags, and JSON field names below
    are best-effort and have NOT been validated against a live ``vastai`` CLI.
    They are isolated here so they can be corrected in one place once verified.
    """

    def __init__(
        self,
        *,
        bin: str = "vastai",
        api_key: str | None = None,
        _run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        """
        Args:
            bin: The CLI executable name / path (``"vastai"``).
            api_key: Optional API key; passed via ``--api-key`` when set so the
                CLI does not depend on ambient ``~/.vast_api_key`` state.
            _run: Injectable subprocess runner.  Must accept an argv list plus
                ``capture_output``/``text``/``check`` kwargs and return an object
                with ``.stdout`` / ``.stderr`` / ``.returncode``.
        """
        self.bin = bin
        self.api_key = api_key
        self._run = _run

    # -- subprocess plumbing ------------------------------------------------ #
    def _exec(self, args: list[str]) -> str:
        """Run ``vastai <args>`` and return stdout, raising on failure.

        Authentication (when ``api_key`` is set) and the JSON-output flag are the
        caller's responsibility per-subcommand; this only handles invocation and
        error surfacing.
        """
        argv = [self.bin, *args]
        if self.api_key:
            argv += ["--api-key", self.api_key]
        logger.debug("[vast] exec: %s", " ".join(argv))
        try:
            proc = self._run(
                argv,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"vast.ai CLI not found (looked for {self.bin!r}). "
                f"Install it with `pip install vastai` and ensure it is on PATH."
            ) from exc

        returncode = getattr(proc, "returncode", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        if returncode != 0:
            raise RuntimeError(
                f"vast.ai CLI command failed (exit {returncode}): "
                f"{' '.join(argv)}\nstderr: {stderr.strip()}"
            )
        return stdout

    def _exec_json(self, args: list[str]) -> Any:
        """Run a subcommand and parse its stdout as JSON."""
        stdout = self._exec(args)
        stripped = stdout.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"vast.ai CLI returned non-JSON output for {' '.join(args)!r}: "
                f"{stdout[:500]!r}"
            ) from exc

    # -- lifecycle ---------------------------------------------------------- #
    def search_offers(
        self,
        *,
        gpu_name: str | None,
        min_gpu_ram_mb: int,
        num_gpus: int = 1,
        max_dph: float | None,
        limit: int = 10,
    ) -> list[VastOffer]:
        # Build a vast.ai query string. gpu_ram is in GB on vast.ai; convert MB.
        min_gpu_ram_gb = max(0, int(min_gpu_ram_mb // 1024))
        query_parts = [
            "rentable=true",
            "verified=true",
            f"num_gpus={int(num_gpus)}",
            f"gpu_ram>={min_gpu_ram_gb}",
        ]
        if gpu_name:
            # vast.ai matches gpu_name with underscores (e.g. "RTX 4090" -> "RTX_4090").
            query_parts.append(f"gpu_name={gpu_name.replace(' ', '_')}")
        if max_dph is not None:
            query_parts.append(f"dph_total<={float(max_dph)}")

        args = [
            "search",
            "offers",
            " ".join(query_parts),
            "--raw",
            "--order",
            "dph_total",
            "--limit",
            str(int(limit)),
        ]
        payload = self._exec_json(args)
        rows = _as_row_list(payload)
        offers = [self._parse_offer(row) for row in rows]
        logger.info(
            "[vast] search_offers -> %d offer(s) (gpu=%s, >=%dMB, num_gpus=%d, max_dph=%s)",
            len(offers),
            gpu_name,
            min_gpu_ram_mb,
            num_gpus,
            max_dph,
        )
        return offers

    def create_instance(
        self,
        offer_id: str,
        *,
        image: str,
        env: dict | None = None,
        onstart: str | None = None,
        disk_gb: int = 40,
        label: str | None = None,
    ) -> VastInstance:
        args = [
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(int(disk_gb)),
            "--raw",
        ]
        if env:
            # vast.ai accepts a single env string of "-e KEY=VALUE ..." docker-run form.
            env_str = " ".join(f"-e {k}={v}" for k, v in env.items())
            args += ["--env", env_str]
        if onstart is not None:
            args += ["--onstart-cmd", onstart]
        if label is not None:
            args += ["--label", label]

        payload = self._exec_json(args)
        # `create instance` typically returns {"success": true, "new_contract": <id>}.
        instance_id = _extract_created_id(payload, fallback=offer_id)
        logger.info(
            "[vast] create_instance from offer %s -> instance %s (image=%s)",
            offer_id,
            instance_id,
            image,
        )
        # The create response carries no live endpoint; report it as created and
        # let the caller poll via wait_until_running.
        return VastInstance(
            instance_id=str(instance_id),
            status="created",
            ssh_host=None,
            ssh_port=None,
            public_ip=None,
            raw=payload if isinstance(payload, dict) else {"raw": payload},
        )

    def get_instance(self, instance_id: str) -> VastInstance:
        args = ["show", "instance", str(instance_id), "--raw"]
        payload = self._exec_json(args)
        row = _single_row(payload)
        return self._parse_instance(row, instance_id)

    def destroy_instance(self, instance_id: str) -> None:
        # No --raw needed; we only care that it exits 0.
        self._exec(["destroy", "instance", str(instance_id)])
        logger.info("[vast] destroy_instance %s requested", instance_id)

    # -- parsing helpers ---------------------------------------------------- #
    @staticmethod
    def _parse_offer(row: dict) -> VastOffer:
        offer_id = row.get("id") or row.get("ask_contract_id") or row.get("offer_id")
        num_gpus = _as_int(row.get("num_gpus"), default=1)
        # vast.ai reports per-GPU RAM in GB under "gpu_ram".
        gpu_ram_gb = _as_float(row.get("gpu_ram"), default=0.0)
        gpu_ram_mb = int(round(gpu_ram_gb * 1024))
        dph = _as_float(
            row.get("dph_total", row.get("dph_base", row.get("dph"))),
            default=0.0,
        )
        return VastOffer(
            offer_id=str(offer_id),
            gpu_name=str(row.get("gpu_name", "") or ""),
            num_gpus=num_gpus,
            gpu_ram_mb=gpu_ram_mb,
            dph=dph,
            raw=row,
        )

    @staticmethod
    def _parse_instance(row: dict, fallback_id: str) -> VastInstance:
        instance_id = (
            row.get("id")
            or row.get("instance_id")
            or row.get("new_contract")
            or fallback_id
        )
        # actual_status is the fine-grained lifecycle field; cur_state is coarser.
        status = (
            row.get("actual_status")
            or row.get("cur_state")
            or row.get("status")
            or "unknown"
        )
        ssh_host = row.get("ssh_host") or row.get("public_ipaddr") or None
        ssh_port = _as_optional_int(row.get("ssh_port"))
        public_ip = row.get("public_ipaddr") or None
        return VastInstance(
            instance_id=str(instance_id),
            status=str(status),
            ssh_host=str(ssh_host) if ssh_host else None,
            ssh_port=ssh_port,
            public_ip=str(public_ip) if public_ip else None,
            raw=row,
        )


# --------------------------------------------------------------------------- #
# JSON-shape coercion helpers (defensive against the unproven CLI output shape)
# --------------------------------------------------------------------------- #
def _as_row_list(payload: Any) -> list[dict]:
    """Coerce a CLI JSON payload into a list of row dicts.

    Accepts a bare list, or a dict wrapping the rows under a common key
    (``offers`` / ``instances`` / ``rows`` / ``data``).
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("offers", "instances", "rows", "data", "results"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        # A single object that *is* a row.
        return [payload]
    return []


def _single_row(payload: Any) -> dict:
    """Coerce a CLI JSON payload into a single row dict (the first if a list)."""
    if isinstance(payload, dict):
        # Some CLIs wrap a single instance under a list key.
        for key in ("instances", "rows", "data", "results"):
            inner = payload.get(key)
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return inner[0]
        return payload
    rows = _as_row_list(payload)
    return rows[0] if rows else {}


def _extract_created_id(payload: Any, *, fallback: str) -> str:
    """Pull the new instance id from a `create instance` response."""
    if isinstance(payload, dict):
        for key in ("new_contract", "instance_id", "id", "contract_id"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    if isinstance(payload, (int, str)):
        return str(payload)
    return str(fallback)


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

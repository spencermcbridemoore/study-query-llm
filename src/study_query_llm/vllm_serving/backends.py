"""
vLLM serving backends -- where a :class:`~study_query_llm.vllm_serving.config.VLLMServeConfig`
actually gets launched.

A backend turns a backend-agnostic serve config into a running, network-reachable
OpenAI-compatible vLLM server and returns its ``/v1`` base URL.  Two concrete
backends share one tiny abstract contract (:class:`VLLMBackend`):

* :class:`LocalDockerBackend` -- runs ``vllm/vllm-openai`` in Docker on the
  operator's own RTX 4090.  PROVEN live.  It carries the display-safety VRAM
  guard and the local-network TLS-interception workaround (host-side snapshot
  download + offline mounted container), mirroring the proven launch shape of
  ``providers/managers/local_docker_tei.py`` (that manager serves TEI on internal
  port 80; we serve vLLM on internal port 8000).

* :class:`VastAIBackend` -- provisions a paid cloud GPU via the injectable
  :class:`~study_query_llm.vllm_serving.vast_client.VastClient` seam.  UNPROVEN
  end-to-end (shape only -- never run live by default).  It mirrors the
  create/poll/health/teardown *structure* of ``providers/managers/aci_tei.py``
  but does NOT copy Azure provisioning specifics as if correct.  Its one hard
  guarantee: the instance id is recorded the instant the instance is created
  (before any wait), and :meth:`VastAIBackend.stop` always tries to destroy that
  paid instance and logs a manual ``vastai destroy instance <id>`` fallback if
  the automatic teardown fails.

Separation of concerns
~~~~~~~~~~~~~~~~~~~~~~~~
A backend's job is launch + teardown.  It deliberately does NOT health-wait:
:meth:`start` returns as soon as the server *should* be coming up, and the
owning manager (``manager.py``) drives ``probe.wait_for_models_ready`` against
the returned URL.  This keeps the backend small and lets the manager own the
health/idle/signal lifecycle.

TLS-interception note (LOCAL backend only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
On the operator's local Windows network an intercepting TLS proxy presents a CA
Windows trusts but the container does not, so the container cannot reach
huggingface.co (``CERTIFICATE_VERIFY_FAILED``).  The verified fix is to download
the snapshot on the HOST (TLS works there) and mount it read-only into an
``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` container.  The vast.ai (Linux
cloud) backend has no such proxy and downloads in-container, so it never uses the
offline workaround.

Imports
~~~~~~~
Per the module contract this file imports ONLY from sibling intra-module files
(``.config``, ``.vram``, ``.hf_download``, ``.vast_client``), stdlib, and the
third-party ``docker`` package (imported lazily so unit tests that inject a
``docker_client_factory`` never need it installed).  No other ``study_query_llm``
imports appear here.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

from . import hf_download as _hf_download_mod
from . import vram as _vram_mod
from .config import INTERNAL_PORT, VLLM_IMAGE, VLLMServeConfig
from .vast_client import VastClient, VastInstance
from .vram import DEFAULT_VRAM_CEILING_MB, VRAMStatus

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _sanitize_name(value: str) -> str:
    """Turn a model id / path into a Docker-safe container-name fragment.

    Docker container names must match ``[a-zA-Z0-9][a-zA-Z0-9_.-]*``.  We lower
    case, replace ``/`` and ``.`` (and any other illegal char) with ``-``, and
    strip leading separators so the result is always a legal name fragment.
    Mirrors ``local_docker_tei``'s ``model_id.replace("/", "-").replace(".", "-")``
    but is defensive about other characters (paths, ``:`` in repo revisions, …).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "-", value.strip().lower())
    cleaned = cleaned.strip("-_.")
    return cleaned or "model"


# --------------------------------------------------------------------------- #
# Abstract backend
# --------------------------------------------------------------------------- #
class VLLMBackend(ABC):
    """Launch + teardown contract for a single vLLM serve target.

    Concrete backends own *where* a config runs (local Docker, remote cloud) and
    nothing else: :meth:`start` brings the server up and returns its ``/v1`` base
    URL, :meth:`stop` frees everything the backend created.  Health-waiting is the
    manager's job, not the backend's.
    """

    #: Short, stable label for the backend (e.g. ``"local_docker"`` / ``"vast"``).
    #: The manager folds this into ``provider_label = f"vllm_{backend.name}"``.
    name: str = "base"

    @abstractmethod
    def start(self, config: VLLMServeConfig) -> str:
        """Launch the server described by ``config``.

        Returns:
            A network-reachable base URL ending in ``/v1`` (e.g.
            ``"http://localhost:8000/v1"``).  The server may still be loading the
            model when this returns -- the caller is responsible for the health
            wait (``probe.wait_for_models_ready``).
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Tear down everything :meth:`start` created.

        MUST be idempotent and MUST NOT raise: implementations swallow and log
        their own errors so teardown is safe to call from ``atexit`` / signal
        handlers / ``__exit__`` even after a partial or failed start.
        """
        raise NotImplementedError

    def is_remote(self) -> bool:
        """Whether this backend provisions remote (and likely *paid*) resources.

        ``False`` for local Docker; overridden to ``True`` by the vast backend so
        callers (e.g. the CLI cost-confirmation gate) can branch on it.
        """
        return False


# --------------------------------------------------------------------------- #
# Local Docker backend (PROVEN)
# --------------------------------------------------------------------------- #
class LocalDockerBackend(VLLMBackend):
    """Run ``vllm/vllm-openai`` in a local Docker container on the operator's GPU.

    Mirrors the proven launch shape of ``LocalDockerTEIManager`` (stale-container
    removal, ``device_requests`` GPU passthrough, host-port -> internal-port
    mapping, ``detach=True``/``remove=False``) but serves vLLM on internal port
    :data:`~study_query_llm.vllm_serving.config.INTERNAL_PORT` (8000) instead of
    TEI's port 80, and adds two vLLM-specific concerns:

    1. **Display-safety VRAM guard** -- before launching it probes VRAM and calls
       :func:`~study_query_llm.vllm_serving.vram.assert_vram_headroom`, which
       refuses (raises :class:`~study_query_llm.vllm_serving.vram.VRAMSafetyError`)
       if the requested reservation exceeds the ceiling or would not leave enough
       free VRAM for the display.  A too-greedy vLLM has crashed the operator's
       display before; this guard is the safety net.

    2. **TLS-interception offline workaround** -- on the local Windows network the
       container cannot reach HuggingFace, so the snapshot is downloaded on the
       host and mounted read-only into an offline container (see module
       docstring).

    All side-effecting dependencies are injectable so unit tests never touch
    Docker, the GPU, or the network: ``docker_client_factory`` (default
    ``docker.from_env``), ``vram_probe`` (default
    :func:`~study_query_llm.vllm_serving.vram.query_vram`), and ``hf_download``
    (default :func:`~study_query_llm.vllm_serving.hf_download.snapshot_to_local`).
    """

    name = "local_docker"

    def __init__(
        self,
        *,
        port: int = 8000,
        gpu_device: str = "all",
        container_name: Optional[str] = None,
        models_dir: Optional[str] = None,
        offline: Optional[bool] = None,
        hf_cache_dir: Optional[str] = None,
        enable_vram_check: bool = True,
        vram_ceiling_mb: int = DEFAULT_VRAM_CEILING_MB,
        gpu_index: int = 0,
        docker_client_factory: Optional[Callable[[], object]] = None,
        vram_probe: Optional[Callable[..., VRAMStatus]] = None,
        hf_download: Optional[Callable[..., str]] = None,
    ) -> None:
        """
        Args:
            port: HOST port to expose.  vLLM always listens on internal port
                8000; this maps ``host:8000`` -> ``container:8000``.
            gpu_device: Docker ``DeviceRequest`` count.  ``"all"`` -> ``count=-1``
                (all GPUs); a digit string passes that many GPUs.
            container_name: Explicit container name.  When ``None`` it is derived
                in :meth:`start` from ``config.served_model_name`` as
                ``"vllm-<sanitized-served-name>"``.
            models_dir: Host directory under which offline snapshots are stored
                (per-model subdir).  Defaults to
                :func:`~study_query_llm.vllm_serving.hf_download.default_models_dir`.
            offline: TLS workaround toggle.  ``None`` (default) auto-detects:
                offline on win32, or whenever ``config.model`` is a local dir.
                ``True`` forces the host-download+mount path; ``False`` forces
                in-container download.
            hf_cache_dir: Host HF cache to mount read-write at
                ``/root/.cache/huggingface`` for the in-container-download path
                (``offline=False``).  Defaults to ``~/.cache/huggingface``.
            enable_vram_check: When ``True`` (default) run the display-safety VRAM
                guard before launch.  Disable ONLY when you are certain no display
                shares the GPU.
            vram_ceiling_mb: Hard cap on the reservation the guard will permit.
            gpu_index: GPU index to probe for the VRAM guard.
            docker_client_factory: Test seam.  Returns a docker-client-like object
                exposing ``.containers``.  Defaults to ``docker.from_env``
                (imported lazily).
            vram_probe: Test seam.  Defaults to
                :func:`~study_query_llm.vllm_serving.vram.query_vram`.
            hf_download: Test seam.  Defaults to
                :func:`~study_query_llm.vllm_serving.hf_download.snapshot_to_local`.
        """
        self.port = port
        self.gpu_device = gpu_device
        self.container_name = container_name
        self.models_dir = models_dir
        self.offline = offline
        self.hf_cache_dir = hf_cache_dir
        self.enable_vram_check = enable_vram_check
        self.vram_ceiling_mb = vram_ceiling_mb
        self.gpu_index = gpu_index

        self._docker_client_factory = docker_client_factory
        self._vram_probe = vram_probe if vram_probe is not None else _vram_mod.query_vram
        self._hf_download = (
            hf_download if hf_download is not None else _hf_download_mod.snapshot_to_local
        )

        # Set during start() so stop() can tear down exactly what we created.
        self._container = None
        self._active_container_name: Optional[str] = None

    # -- docker plumbing ---------------------------------------------------- #
    def _get_docker_client(self):
        """Return a Docker client (injected factory or lazy ``docker.from_env``).

        ``docker`` is imported lazily so importing this module (and running unit
        tests that inject ``docker_client_factory``) does not require the
        ``docker`` package to be installed.
        """
        if self._docker_client_factory is not None:
            return self._docker_client_factory()
        import docker  # local import: keep ``docker`` an optional dependency

        return docker.from_env()

    def _device_requests(self):
        """Build the GPU ``DeviceRequest`` list (``count=-1`` for ``"all"``)."""
        import docker.types  # local import alongside docker client construction

        count = -1 if self.gpu_device == "all" else int(self.gpu_device)
        return [docker.types.DeviceRequest(count=count, capabilities=[["gpu"]])]

    def _remove_stale_container(self, client, name: str) -> None:
        """Stop+remove a pre-existing container of the same name (mirrors TEI).

        A leftover container from a crashed run would hold the host port and the
        GPU, so we clear it first.  ``docker.errors.NotFound`` (nothing to remove)
        is the normal case and is ignored.
        """
        try:
            import docker.errors  # local import: optional dependency

            not_found = docker.errors.NotFound
        except Exception:  # pragma: no cover - docker not installed in some tests
            not_found = ()  # type: ignore[assignment]

        try:
            old = client.containers.get(name)
        except Exception as exc:
            # NotFound is expected; anything else we log and continue (the run
            # below will surface a real conflict).
            if not_found and isinstance(exc, not_found):
                return
            logger.debug("[vllm/local] containers.get(%s) raised (ignored): %s", name, exc)
            return

        logger.info("[vllm/local] Removing stale container '%s' ...", name)
        try:
            old.stop()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("[vllm/local] stale stop failed (ignored): %s", exc)
        try:
            old.remove()
        except Exception as exc:  # pragma: no cover - best-effort
            logger.debug("[vllm/local] stale remove failed (ignored): %s", exc)

    # -- model resolution (TLS workaround) ---------------------------------- #
    def _resolve_model(self, config: VLLMServeConfig):
        """Decide ``(model_ref, volumes, environment)`` for the launch.

        Three cases, in priority order:

        1. ``config.model`` is an existing local directory -> mount it at
           ``/model`` read-only, serve ``--model /model`` fully offline.
        2. offline mode (auto on win32, or forced) -> host-download the snapshot
           into ``<models_dir>/<sanitized-model>``, mount it read-only at
           ``/model``, serve offline.
        3. otherwise -> in-container download: ``--model <repo>`` with the host HF
           cache mounted read-write; no offline env.

        Returns:
            ``(model_ref, volumes, environment)`` ready for ``containers.run``.
        """
        offline_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

        # Case 1: an explicit local model directory.
        if os.path.isdir(config.model):
            host_dir = str(Path(config.model).expanduser().resolve())
            logger.info(
                "[vllm/local] Serving local model directory %s (offline).", host_dir
            )
            volumes = {host_dir: {"bind": "/model", "mode": "ro"}}
            return "/model", volumes, dict(offline_env)

        # Decide whether to use the offline (host-download) path.
        use_offline = self.offline
        if use_offline is None:
            # Auto: the TLS-interception proxy is a LOCAL-Windows concern.
            use_offline = sys.platform.startswith("win")

        # Case 2: offline host-download + mount.
        if use_offline:
            models_root = self.models_dir or _hf_download_mod.default_models_dir()
            dest = str(Path(models_root).expanduser() / _sanitize_name(config.model))
            logger.info(
                "[vllm/local] Offline mode: downloading %s to host dir %s "
                "(TLS-interception workaround), then mounting read-only.",
                config.model,
                dest,
            )
            resolved = self._hf_download(config.model, dest)
            volumes = {resolved: {"bind": "/model", "mode": "ro"}}
            return "/model", volumes, dict(offline_env)

        # Case 3: in-container download (vLLM reaches HF directly).
        hf_cache = str(
            Path(self.hf_cache_dir or (Path.home() / ".cache" / "huggingface"))
            .expanduser()
            .resolve()
        )
        logger.info(
            "[vllm/local] Online mode: vLLM will download %s in-container "
            "(HF cache mounted at %s).",
            config.model,
            hf_cache,
        )
        volumes = {hf_cache: {"bind": "/root/.cache/huggingface", "mode": "rw"}}
        return config.model, volumes, {}

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, config: VLLMServeConfig) -> str:
        """Run the vLLM container and return its ``/v1`` base URL.

        Steps: VRAM guard -> resolve model/volumes/env -> remove stale container
        -> ``containers.run`` (detached) -> return URL.  Does NOT health-wait (the
        manager does).

        Raises:
            VRAMSafetyError: if the display-safety guard refuses the launch.  The
                container is never started in that case.
        """
        # 1. Display-safety VRAM guard (raise -> abort, never launch).
        if self.enable_vram_check:
            status = self._vram_probe(self.gpu_index)
            _vram_mod.assert_vram_headroom(
                gpu_memory_utilization=config.gpu_memory_utilization,
                status=status,
                ceiling_mb=self.vram_ceiling_mb,
            )

        # 2. Resolve model ref + volumes + env (TLS workaround lives here).
        model_ref, volumes, environment = self._resolve_model(config)

        # Derive the container name from the served model name at start time.
        name = self.container_name or ("vllm-" + _sanitize_name(config.served_model_name))
        self._active_container_name = name

        client = self._get_docker_client()

        # 3. Clear any stale container holding the name / port / GPU.
        self._remove_stale_container(client, name)

        command = config.to_serve_args(model_ref)
        logger.info(
            "[vllm/local] Starting container '%s' (model_ref=%s, host_port=%d->%d, gpu=%s)",
            name,
            model_ref,
            self.port,
            INTERNAL_PORT,
            self.gpu_device,
        )
        logger.debug("[vllm/local] serve args: %s", command)

        # 4. Launch detached.  vLLM listens on INTERNAL_PORT inside the container;
        #    we map the host ``port`` onto it.
        self._container = client.containers.run(
            image=VLLM_IMAGE,
            command=command,
            name=name,
            detach=True,
            ports={f"{INTERNAL_PORT}/tcp": self.port},
            volumes=volumes,
            environment=environment,
            device_requests=self._device_requests(),
            remove=False,
        )

        base_url = f"http://localhost:{self.port}/v1"
        logger.info(
            "[vllm/local] Container '%s' launched. Base URL: %s "
            "(server still loading; caller must health-wait).",
            name,
            base_url,
        )
        # 5. Return the network-reachable base URL.
        return base_url

    def stop(self) -> None:
        """Stop + remove the container.  Idempotent; swallows and logs errors.

        No health concerns here -- frees the host port and GPU VRAM and clears
        internal state so a subsequent :meth:`start` is clean.
        """
        container = self._container
        name = self._active_container_name
        if container is None:
            return
        try:
            logger.info("[vllm/local] Stopping container '%s' ...", name)
            container.stop()
            container.remove()
            logger.info("[vllm/local] Stopped '%s'.", name)
        except Exception as exc:
            logger.warning(
                "[vllm/local] Stop/remove failed for '%s' (may already be gone): %s",
                name,
                exc,
            )
        finally:
            self._container = None
            self._active_container_name = None


# --------------------------------------------------------------------------- #
# Vast.ai backend (UNPROVEN -- shape only)
# --------------------------------------------------------------------------- #
class VastAIBackend(VLLMBackend):
    """Provision a paid cloud GPU and run vLLM on it via the vast client seam.

    UNPROVEN end-to-end -- this mirrors the create/poll/health/teardown *shape* of
    ``ACITEIManager`` but is never run live by default.  The CLI gates provisioning
    behind an explicit cost confirmation.

    Cost-safety (the one hard guarantee)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A paid instance must never be orphaned.  Therefore ``self._instance_id`` is set
    the instant :meth:`VastClient.create_instance` returns -- *before* the
    potentially long :meth:`VastClient.wait_until_running` poll -- so that
    :meth:`stop` (called from ``__exit__`` / signal handler / atexit, even if the
    wait or tunnel raised) can always find and destroy the instance.  If automatic
    teardown fails, :meth:`stop` logs the instance id and the exact manual command
    (``vastai destroy instance <id>``) so the operator can clean up by hand.

    Network access
    ~~~~~~~~~~~~~~
    Two modes, selected by ``served_via``:

    * ``"ssh"`` (default) -- open an SSH local-forward ``localhost:port`` ->
      ``instance:8000`` through the injectable ``ssh_forwarder`` seam (default
      ``sshtunnel.SSHTunnelForwarder``, imported lazily).  No public port exposure.
    * ``"open_ports"`` -- talk straight to ``http://<public_ip>:<port>/v1`` (an
      ``api_key`` is required because the port is internet-reachable).

    The vast Linux host has no TLS-interception proxy, so the model downloads
    in-container and NO offline env is set (unlike the local backend).
    """

    name = "vast"

    def __init__(
        self,
        *,
        vast_client: Optional[VastClient] = None,
        port: int = 8000,
        image: str = VLLM_IMAGE,
        gpu_name: str = "RTX 4090",
        min_gpu_ram_mb: int = 24000,
        num_gpus: int = 1,
        max_dph: Optional[float] = None,
        disk_gb: int = 40,
        served_via: str = "ssh",
        api_key: Optional[str] = None,
        ssh_forwarder: Optional[Callable[..., object]] = None,
        label: str = "vllm-serving",
    ) -> None:
        """
        Args:
            vast_client: The injectable provider seam.  Prefer to inject one; when
                ``None`` a default :class:`VastCLIClient` is constructed (which is
                itself UNPROVEN and shells out to the ``vastai`` CLI).
            port: Local port the server is reachable on (the SSH-tunnel local end,
                or the public-IP port for ``open_ports``).
            image: Container image to provision (defaults to the vLLM image).
            gpu_name / min_gpu_ram_mb / num_gpus / max_dph: Offer search criteria.
            disk_gb: Instance disk size.
            served_via: ``"ssh"`` (tunnel) or ``"open_ports"`` (public IP).
            api_key: Required for ``served_via="open_ports"`` (the port is public).
            ssh_forwarder: Test seam.  A callable
                ``(remote_host, remote_port, local_port) -> obj`` where ``obj`` has
                ``.start()`` / ``.stop()``.  Defaults to a thin
                ``sshtunnel.SSHTunnelForwarder`` wrapper, imported lazily so tests
                never need ``sshtunnel`` installed.
            label: vast.ai instance label for easy identification.
        """
        if vast_client is None:
            # Default to the (unproven) CLI client.  Import locally to avoid a
            # hard module-level dependency surface; it lives in the same package.
            from .vast_client import VastCLIClient

            vast_client = VastCLIClient()
        self.vast_client = vast_client
        self.port = port
        self.image = image
        self.gpu_name = gpu_name
        self.min_gpu_ram_mb = min_gpu_ram_mb
        self.num_gpus = num_gpus
        self.max_dph = max_dph
        self.disk_gb = disk_gb
        self.served_via = served_via
        self.api_key = api_key
        self._ssh_forwarder_factory = ssh_forwarder
        self.label = label

        # Cost-safety state.  ``_instance_id`` is the single source of truth for
        # teardown and is set the moment the instance is created.
        self._instance_id: Optional[str] = None
        self._tunnel = None
        self._destroyed = False

    def is_remote(self) -> bool:
        return True

    # -- ssh tunnel seam ---------------------------------------------------- #
    def _make_tunnel(self, remote_host: str, remote_port: int, local_port: int):
        """Build (but do not start) the SSH local-forward object.

        Uses the injected ``ssh_forwarder`` factory when present; otherwise lazily
        wraps ``sshtunnel.SSHTunnelForwarder`` so ``sshtunnel`` is only required
        when an SSH tunnel is actually opened against a real instance.
        """
        if self._ssh_forwarder_factory is not None:
            return self._ssh_forwarder_factory(remote_host, remote_port, local_port)

        import sshtunnel  # local import: only needed for a live SSH tunnel

        return sshtunnel.SSHTunnelForwarder(
            (remote_host, int(remote_port)),
            remote_bind_address=("127.0.0.1", INTERNAL_PORT),
            local_bind_address=("127.0.0.1", int(local_port)),
        )

    # -- onstart command ---------------------------------------------------- #
    def _build_onstart(self, config: VLLMServeConfig) -> str:
        """Shell command run inside the instance to launch vLLM on ``0.0.0.0``.

        Downloads the model in-container (no offline env on the Linux cloud host)
        and binds to ``0.0.0.0`` so the SSH tunnel / public port can reach it.  The
        serve args come from the SAME ``config.to_serve_args`` the local backend
        uses, so the two lanes can never silently diverge.

        ``config.to_serve_args`` already emits ``--port 8000`` (the internal port);
        we add ``--host 0.0.0.0`` so the server is reachable from outside the
        container.
        """
        serve_args = config.to_serve_args(config.model)
        # Quote each arg defensively for the remote shell.
        quoted = " ".join(_shquote(a) for a in ["--host", "0.0.0.0", *serve_args])
        # The vLLM image's entrypoint accepts these as ``vllm serve`` args.
        return f"python -m vllm.entrypoints.openai.api_server {quoted}"

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, config: VLLMServeConfig) -> str:
        """Search -> create (record id immediately) -> wait -> expose -> URL.

        Returns the ``/v1`` base URL reachable from the local machine.  Does NOT
        health-wait (the manager does).
        """
        self._destroyed = False

        # 1. Find the cheapest matching offer.
        offers = self.vast_client.search_offers(
            gpu_name=self.gpu_name,
            min_gpu_ram_mb=self.min_gpu_ram_mb,
            num_gpus=self.num_gpus,
            max_dph=self.max_dph,
        )
        if not offers:
            raise RuntimeError(
                "vast.ai: no offers matched the search criteria "
                f"(gpu_name={self.gpu_name!r}, min_gpu_ram_mb={self.min_gpu_ram_mb}, "
                f"num_gpus={self.num_gpus}, max_dph={self.max_dph}). "
                "Relax the criteria (lower min_gpu_ram_mb, raise/clear max_dph, or "
                "broaden gpu_name) and retry."
            )
        offer = min(offers, key=lambda o: o.dph)
        logger.info(
            "[vllm/vast] Selected cheapest offer %s: %s x%d, %d MB, $%.4f/hr",
            offer.offer_id,
            offer.gpu_name,
            offer.num_gpus,
            offer.gpu_ram_mb,
            offer.dph,
        )

        # 2. Create the instance and RECORD THE ID IMMEDIATELY (before any wait),
        #    so teardown can always destroy this paid resource.
        onstart = self._build_onstart(config)
        inst: VastInstance = self.vast_client.create_instance(
            offer.offer_id,
            image=self.image,
            onstart=onstart,
            disk_gb=self.disk_gb,
            label=self.label,
        )
        self._instance_id = inst.instance_id
        logger.warning(
            "[vllm/vast] PAID instance %s created (offer %s, ~$%.4f/hr). "
            "Teardown command if anything goes wrong: vastai destroy instance %s",
            self._instance_id,
            offer.offer_id,
            offer.dph,
            self._instance_id,
        )

        # 3. Wait until it is running with an SSH endpoint published.
        inst = self.vast_client.wait_until_running(self._instance_id)

        # 4. Establish access.
        if self.served_via == "ssh":
            if not inst.ssh_host or inst.ssh_port is None:
                raise RuntimeError(
                    f"vast.ai instance {self._instance_id} is running but has no SSH "
                    f"endpoint (host={inst.ssh_host!r}, port={inst.ssh_port!r}); "
                    "cannot open a tunnel."
                )
            logger.info(
                "[vllm/vast] Opening SSH tunnel localhost:%d -> %s:%s (instance :%d)",
                self.port,
                inst.ssh_host,
                inst.ssh_port,
                INTERNAL_PORT,
            )
            self._tunnel = self._make_tunnel(inst.ssh_host, inst.ssh_port, self.port)
            self._tunnel.start()
            base = f"http://localhost:{self.port}/v1"
        elif self.served_via == "open_ports":
            if not self.api_key:
                raise RuntimeError(
                    "served_via='open_ports' exposes the vLLM port to the public "
                    "internet and therefore REQUIRES an api_key. Pass api_key=... "
                    "or use served_via='ssh'."
                )
            if not inst.public_ip:
                raise RuntimeError(
                    f"vast.ai instance {self._instance_id} has no public_ip for "
                    "served_via='open_ports'."
                )
            base = f"http://{inst.public_ip}:{self.port}/v1"
            logger.info("[vllm/vast] Using public endpoint %s", base)
        else:
            raise ValueError(
                f"Unknown served_via={self.served_via!r} (expected 'ssh' or 'open_ports')."
            )

        logger.info(
            "[vllm/vast] Instance %s reachable at %s (server still loading; caller "
            "must health-wait).",
            self._instance_id,
            base,
        )
        return base

    def stop(self) -> None:
        """Destroy the paid instance (guaranteed, at most once) and close any tunnel.

        Idempotent and never raises.  Order: close the tunnel (best-effort), then
        destroy the instance.  If destroy fails, the instance id and the manual
        ``vastai destroy instance <id>`` command are logged at WARNING so a paid
        instance is never silently orphaned.
        """
        # Close the tunnel first (best-effort).
        if self._tunnel is not None:
            try:
                logger.info("[vllm/vast] Closing SSH tunnel ...")
                self._tunnel.stop()
            except Exception as exc:
                logger.warning("[vllm/vast] Tunnel close failed (ignored): %s", exc)
            finally:
                self._tunnel = None

        # Destroy the instance at most once.
        if self._destroyed:
            return
        instance_id = self._instance_id
        if not instance_id:
            # Nothing was created (or already cleared) -- nothing to destroy.
            self._destroyed = True
            return

        self._destroyed = True  # guard so destroy fires at most once even on error
        try:
            logger.info("[vllm/vast] Destroying paid instance %s ...", instance_id)
            self.vast_client.destroy_instance(instance_id)
            logger.info("[vllm/vast] Destroy requested for instance %s.", instance_id)
        except Exception as exc:
            # The single most important failure to surface: a paid instance may
            # still be running and billing.  Log loudly with the manual fallback.
            logger.error(
                "[vllm/vast] AUTOMATIC TEARDOWN FAILED for paid instance %s: %s. "
                "The instance may STILL BE RUNNING AND BILLING. Destroy it manually: "
                "vastai destroy instance %s",
                instance_id,
                exc,
                instance_id,
            )
        finally:
            self._instance_id = None


def _shquote(value: str) -> str:
    """Minimal POSIX shell quote for the remote ``onstart`` command.

    ``shlex.quote`` targets POSIX shells, which is exactly the vast.ai Linux host,
    so use it directly.  Kept as a tiny wrapper for intent/readability at the call
    site.
    """
    import shlex

    return shlex.quote(value)

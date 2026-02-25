"""
ACI TEI Manager - Azure Container Instances lifecycle manager for HuggingFace TEI.

Creates, polls, health-checks, and tears down an Azure Container Instance running
the HuggingFace Text Embeddings Inference (TEI) Docker image. The instance exposes
an OpenAI-compatible /v1/embeddings endpoint.

Typical usage (context manager guarantees teardown even on crash):

    manager = ACITEIManager(
        subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
        resource_group="URAIT-USE1-NET-PROBLEMBANKGENERATOR-001-RGP",
        container_name="tei-bge-m3-sweep",
        model_id="BAAI/bge-m3",
        idle_timeout_seconds=1800,
    )
    with manager:
        provider = ACITEIEmbeddingProvider(manager)
        # ... run embeddings ...
"""

import os
import time
import threading
import logging
import urllib.request
import urllib.error
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.mgmt.containerinstance import ContainerInstanceManagementClient

logger = logging.getLogger(__name__)

# TEI CPU image -- no CUDA driver needed, works on all ACI SKUs.
# :1.9 (Ampere 80) is the generic GPU image used for ACI, which provisions
# A100-class hardware.  Swap to :86-1.9 if you use A10/A40 ACI SKUs.
_TEI_CPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:cpu-1.9"
_TEI_GPU_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:1.9"


class ACITEIManager:
    """
    Manages an Azure Container Instance running HuggingFace TEI.

    The class is intentionally synchronous so that it can be used from both
    synchronous scripts and as the underlying infrastructure for async providers.

    After ``create()`` completes, ``endpoint_url`` is set and the TEI server has
    passed its health check, so it is ready to serve requests immediately.

    The idle timer runs in a daemon thread; if the process exits the thread
    dies with it, but the ACI container will keep running on Azure until
    ``delete()`` is explicitly called.  Use the context manager to guarantee
    cleanup:

        with ACITEIManager(...) as manager:
            ...
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        container_name: str,
        model_id: str,
        location: str = "eastus",
        cpu: float = 4.0,
        memory_gb: float = 16.0,
        idle_timeout_seconds: int = 1800,
        port: int = 80,
        gpu_count: int = 0,
        gpu_sku: str = "V100",
        health_check_timeout: int = 600,
        health_check_interval: int = 10,
    ) -> None:
        """
        Args:
            subscription_id: Azure subscription ID.
            resource_group: Resource group name to create the container in.
            container_name: Name for the ACI container group (must be unique
                within the resource group).
            model_id: HuggingFace model ID, e.g. ``"BAAI/bge-m3"``.
            location: Azure region (must match resource group region).
            cpu: Number of vCPUs to allocate. 4 vCPUs is a good default for
                medium models.
            memory_gb: Memory in GB. 16 GB is a good default for medium models.
            idle_timeout_seconds: Seconds of inactivity before the container
                is automatically deleted. Defaults to 1800 (30 minutes).
                Call ``ping()`` on each embedding request to reset the timer.
            port: Container port that TEI listens on. TEI defaults to 80.
            gpu_count: Number of GPUs to attach (0 = CPU-only). Only available
                in certain regions and GPU SKUs.
            gpu_sku: GPU SKU string (e.g. ``"V100"``, ``"K80"``). Ignored when
                ``gpu_count == 0``.
            health_check_timeout: Seconds to wait for TEI to load the model
                before giving up.
            health_check_interval: Seconds between health check polls.
        """
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.container_name = container_name
        self.model_id = model_id
        self.location = location
        self.cpu = cpu
        self.memory_gb = memory_gb
        self.idle_timeout_seconds = idle_timeout_seconds
        self.port = port
        self.gpu_count = gpu_count
        self.gpu_sku = gpu_sku
        self.health_check_timeout = health_check_timeout
        self.health_check_interval = health_check_interval

        self.endpoint_url: Optional[str] = None
        self.provider_label: str = "aci_tei"

        self._client = None
        self._idle_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._deleted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-initialize the ACI management client."""
        if self._client is None:
            credential = DefaultAzureCredential()
            self._client = ContainerInstanceManagementClient(
                credential, self.subscription_id
            )
            logger.debug(
                "Initialized ContainerInstanceManagementClient for subscription %s",
                self.subscription_id,
            )
        return self._client

    def _build_container_group(self):
        """Construct the ContainerGroup model object for the TEI image."""
        from azure.mgmt.containerinstance.models import (  # noqa: PLC0415
            Container,
            ContainerGroup,
            ContainerGroupNetworkProtocol,
            ContainerPort,
            EnvironmentVariable,
            IpAddress,
            OperatingSystemTypes,
            Port,
            ResourceRequirements,
            ResourceRequests,
        )

        image = _TEI_GPU_IMAGE if self.gpu_count > 0 else _TEI_CPU_IMAGE

        resource_requests = ResourceRequests(
            memory_in_gb=self.memory_gb,
            cpu=self.cpu,
        )

        if self.gpu_count > 0:
            from azure.mgmt.containerinstance.models import GpuResource  # noqa: PLC0415

            resource_requests.gpu = GpuResource(
                count=self.gpu_count,
                sku=self.gpu_sku,
            )

        resources = ResourceRequirements(requests=resource_requests)

        env_vars = [
            EnvironmentVariable(name="MODEL_ID", value=self.model_id),
        ]

        container = Container(
            name=self.container_name,
            image=image,
            resources=resources,
            ports=[ContainerPort(port=self.port)],
            environment_variables=env_vars,
            command=["text-embeddings-router", "--model-id", self.model_id, "--port", str(self.port)],
        )

        ip_address = IpAddress(
            ports=[Port(protocol=ContainerGroupNetworkProtocol.tcp, port=self.port)],
            type="Public",
        )

        return ContainerGroup(
            location=self.location,
            containers=[container],
            os_type=OperatingSystemTypes.linux,
            ip_address=ip_address,
        )

    def _poll_for_ip(self, poll_interval: int = 5, timeout: int = 120) -> str:
        """Poll until the container group has a public IP. Returns the IP."""
        client = self._get_client()
        deadline = time.time() + timeout
        while time.time() < deadline:
            group = client.container_groups.get(
                self.resource_group, self.container_name
            )
            if (
                group.ip_address is not None
                and group.ip_address.ip is not None
                and group.provisioning_state in ("Succeeded", "Running")
            ):
                return group.ip_address.ip
            logger.debug(
                "[ACI] Waiting for IP (state=%s)...", group.provisioning_state
            )
            time.sleep(poll_interval)
        raise TimeoutError(
            f"ACI container '{self.container_name}' did not get a public IP "
            f"within {timeout}s."
        )

    def _wait_for_healthy(self) -> None:
        """Poll TEI /health until it returns 200 (model is loaded)."""
        health_url = f"{self.endpoint_url}/health"
        deadline = time.time() + self.health_check_timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(health_url, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info(
                            "[ACI] TEI endpoint is healthy: %s", self.endpoint_url
                        )
                        return
            except (urllib.error.URLError, OSError):
                pass
            logger.debug("[ACI] Waiting for TEI health check at %s ...", health_url)
            time.sleep(self.health_check_interval)
        raise TimeoutError(
            f"TEI endpoint at {health_url} did not become healthy within "
            f"{self.health_check_timeout}s. The model may be too large for the "
            f"allocated CPU/memory, or the container failed to start."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self) -> str:
        """
        Create the ACI container group, wait for an IP, and wait for TEI to
        finish loading the model.

        Returns:
            The base endpoint URL (e.g. ``"http://20.1.2.3:80/v1"``).

        Raises:
            TimeoutError: If the container does not get an IP or the TEI health
                check does not pass within the configured timeouts.
        """
        if self.endpoint_url is not None:
            logger.warning(
                "[ACI] Container '%s' is already running at %s",
                self.container_name,
                self.endpoint_url,
            )
            return self.endpoint_url

        self._deleted = False
        client = self._get_client()
        container_group = self._build_container_group()

        logger.info(
            "[ACI] Creating container group '%s' (model=%s, cpu=%s, mem=%sGB) ...",
            self.container_name,
            self.model_id,
            self.cpu,
            self.memory_gb,
        )

        poller = client.container_groups.begin_create_or_update(
            self.resource_group, self.container_name, container_group
        )
        poller.result()  # block until the ARM operation completes

        ip = self._poll_for_ip()
        self.endpoint_url = f"http://{ip}:{self.port}/v1"
        logger.info(
            "[ACI] Container is up. Endpoint: %s. Waiting for model load ...",
            self.endpoint_url,
        )

        self._wait_for_healthy()
        self._reset_idle_timer()

        logger.info("[ACI] Ready: %s  (model=%s)", self.endpoint_url, self.model_id)
        return self.endpoint_url

    def delete(self) -> None:
        """
        Delete the ACI container group and cancel the idle timer.

        Safe to call multiple times (subsequent calls are no-ops).
        """
        with self._lock:
            if self._deleted:
                return
            self._deleted = True
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

        try:
            logger.info("[ACI] Deleting container group '%s' ...", self.container_name)
            self._get_client().container_groups.begin_delete(
                self.resource_group, self.container_name
            ).result()
            logger.info("[ACI] Deleted '%s'.", self.container_name)
        except Exception as exc:
            logger.warning(
                "[ACI] Delete failed for '%s' (may already be gone): %s",
                self.container_name,
                exc,
            )
        finally:
            self.endpoint_url = None

    def ping(self) -> None:
        """
        Reset the idle shutdown timer.

        Call this on every embedding request so the container is not deleted
        while a sweep is actively in progress.
        """
        if not self._deleted:
            self._reset_idle_timer()

    def _reset_idle_timer(self) -> None:
        """Cancel any existing timer and start a fresh countdown."""
        with self._lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            self._idle_timer = threading.Timer(
                self.idle_timeout_seconds, self._idle_shutdown
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _idle_shutdown(self) -> None:
        """Called by the timer thread when idle timeout expires."""
        logger.warning(
            "[ACI] Idle timeout (%ss) reached — deleting '%s'",
            self.idle_timeout_seconds,
            self.container_name,
        )
        self.delete()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ACITEIManager":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.delete()


# ------------------------------------------------------------------
# Convenience factory that reads env vars
# ------------------------------------------------------------------

def manager_from_env(
    container_name: str,
    model_id: str,
    **kwargs,
) -> ACITEIManager:
    """
    Construct an ``ACITEIManager`` using env vars for credentials and defaults.

    Required env vars:
        AZURE_SUBSCRIPTION_ID

    Optional env vars (with defaults):
        AZURE_RESOURCE_GROUP    (default: URAIT-USE1-NET-PROBLEMBANKGENERATOR-001-RGP)
        ACI_TEI_LOCATION        (default: eastus)
        ACI_TEI_IDLE_TIMEOUT    (default: 1800)

    Additional keyword arguments are forwarded to ``ACITEIManager.__init__``.
    """
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
    if not subscription_id:
        raise ValueError(
            "AZURE_SUBSCRIPTION_ID environment variable is not set. "
            "Set it in your .env file or system environment."
        )

    resource_group = os.environ.get(
        "AZURE_RESOURCE_GROUP",
        "URAIT-USE1-NET-PROBLEMBANKGENERATOR-001-RGP",
    )
    location = os.environ.get("ACI_TEI_LOCATION", "eastus")
    idle_timeout = int(os.environ.get("ACI_TEI_IDLE_TIMEOUT", "1800"))

    return ACITEIManager(
        subscription_id=subscription_id,
        resource_group=resource_group,
        container_name=container_name,
        model_id=model_id,
        location=location,
        idle_timeout_seconds=idle_timeout,
        **kwargs,
    )

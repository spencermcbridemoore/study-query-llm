"""Tests for scripts.common.aci_tei_manager.ACITEIManager."""

import os
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

# The module is in scripts/common which is not a proper package install,
# so we import it directly. The conftest.py / pytest.ini setup should have
# sys.path configured so that `scripts/` is importable.
from scripts.common.aci_tei_manager import ACITEIManager, manager_from_env


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_manager(**kwargs) -> ACITEIManager:
    """Return a minimal ACITEIManager with safe defaults."""
    defaults = dict(
        subscription_id="sub-1234",
        resource_group="test-rg",
        container_name="tei-test",
        model_id="BAAI/bge-m3",
        location="eastus",
        cpu=2.0,
        memory_gb=8.0,
        idle_timeout_seconds=300,
        health_check_timeout=30,
        health_check_interval=1,
    )
    defaults.update(kwargs)
    return ACITEIManager(**defaults)


def _make_aci_client(ip="10.0.0.1", provisioning_state="Succeeded"):
    """Return a mock ContainerInstanceManagementClient."""
    client = MagicMock()

    # begin_create_or_update returns a poller
    poller = MagicMock()
    poller.result = MagicMock(return_value=None)
    client.container_groups.begin_create_or_update.return_value = poller

    # get returns a container group with an IP
    group = MagicMock()
    group.ip_address = MagicMock()
    group.ip_address.ip = ip
    group.provisioning_state = provisioning_state
    client.container_groups.get.return_value = group

    # begin_delete returns a poller
    delete_poller = MagicMock()
    delete_poller.result = MagicMock(return_value=None)
    client.container_groups.begin_delete.return_value = delete_poller

    return client


# ---------------------------------------------------------------------------
# _get_client
# ---------------------------------------------------------------------------

def test_get_client_creates_management_client():
    """_get_client() instantiates ContainerInstanceManagementClient."""
    manager = _make_manager()
    mock_client = MagicMock()

    with patch(
        "scripts.common.aci_tei_manager.DefaultAzureCredential"
    ) as MockCred, patch(
        "scripts.common.aci_tei_manager.ContainerInstanceManagementClient"
    ) as MockClient:
        MockClient.return_value = mock_client
        result = manager._get_client()

    MockCred.assert_called_once()
    MockClient.assert_called_once()
    assert result is mock_client


def test_get_client_is_cached():
    """_get_client() returns the same client on repeated calls."""
    manager = _make_manager()
    with patch("scripts.common.aci_tei_manager.DefaultAzureCredential"), \
         patch("scripts.common.aci_tei_manager.ContainerInstanceManagementClient") as MockClient:
        MockClient.return_value = MagicMock()
        c1 = manager._get_client()
        c2 = manager._get_client()

    assert c1 is c2
    MockClient.assert_called_once()


# ---------------------------------------------------------------------------
# _build_container_group
# ---------------------------------------------------------------------------

def test_build_container_group_cpu_image():
    """CPU image is used when gpu_count == 0."""
    from scripts.common.aci_tei_manager import _TEI_CPU_IMAGE

    manager = _make_manager(gpu_count=0)
    with _patch_aci_models():
        group = manager._build_container_group()

    container = group.containers[0]
    assert container.image == _TEI_CPU_IMAGE


def test_build_container_group_gpu_image():
    """GPU image is used when gpu_count > 0."""
    from scripts.common.aci_tei_manager import _TEI_GPU_IMAGE

    manager = _make_manager(gpu_count=1, gpu_sku="V100")
    with _patch_aci_models():
        group = manager._build_container_group()

    container = group.containers[0]
    assert container.image == _TEI_GPU_IMAGE


def test_build_container_group_model_in_env():
    """MODEL_ID environment variable is set for the TEI container."""
    manager = _make_manager(model_id="intfloat/e5-large-v2")
    with _patch_aci_models():
        group = manager._build_container_group()

    container = group.containers[0]
    env_names = [ev.name for ev in container.environment_variables]
    assert "MODEL_ID" in env_names
    model_value = next(ev.value for ev in container.environment_variables if ev.name == "MODEL_ID")
    assert model_value == "intfloat/e5-large-v2"


def test_build_container_group_cpu_and_memory():
    """Resource requests use the configured cpu and memory_gb values."""
    manager = _make_manager(cpu=4.0, memory_gb=16.0)
    with _patch_aci_models():
        group = manager._build_container_group()

    requests = group.containers[0].resources.requests
    assert requests.cpu == 4.0
    assert requests.memory_in_gb == 16.0


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------

def test_create_calls_begin_create_or_update():
    """create() calls begin_create_or_update with resource_group and container_name."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()

    call_args = aci_client.container_groups.begin_create_or_update.call_args
    assert call_args is not None, "begin_create_or_update was not called"
    positional = call_args[0]
    assert positional[0] == "test-rg"
    assert positional[1] == "tei-test"


def test_create_sets_endpoint_url():
    """create() sets endpoint_url from the container's public IP."""
    manager = _make_manager(port=80)
    aci_client = _make_aci_client(ip="20.1.2.3")

    with _patch_aci_sdk(aci_client), _patch_health():
        url = manager.create()

    assert url == "http://20.1.2.3:80/v1"
    assert manager.endpoint_url == "http://20.1.2.3:80/v1"


def test_create_is_idempotent():
    """Calling create() a second time returns cached endpoint without re-creating."""
    manager = _make_manager()
    aci_client = _make_aci_client(ip="5.6.7.8")

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        manager.create()

    assert aci_client.container_groups.begin_create_or_update.call_count == 1


def test_create_starts_idle_timer():
    """create() starts the idle timer."""
    manager = _make_manager(idle_timeout_seconds=9999)
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()

    assert manager._idle_timer is not None
    manager._idle_timer.cancel()


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

def test_delete_calls_begin_delete():
    """delete() calls begin_delete with correct resource_group and container_name."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        manager.delete()

    aci_client.container_groups.begin_delete.assert_called_once_with(
        "test-rg", "tei-test"
    )


def test_delete_cancels_idle_timer():
    """delete() cancels the idle timer."""
    manager = _make_manager(idle_timeout_seconds=9999)
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        timer = manager._idle_timer
        manager.delete()

    # threading.Timer.cancel() sets the internal finished event immediately
    assert timer.finished.is_set(), "timer should be cancelled after delete()"
    assert manager._idle_timer is None


def test_delete_clears_endpoint_url():
    """delete() sets endpoint_url back to None."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        manager.delete()

    assert manager.endpoint_url is None


def test_delete_is_idempotent():
    """Calling delete() multiple times is safe (no double-delete)."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        manager.delete()
        manager.delete()

    assert aci_client.container_groups.begin_delete.call_count == 1


def test_delete_handles_exception_gracefully():
    """delete() logs and swallows exceptions from the ACI API."""
    manager = _make_manager()
    aci_client = _make_aci_client()
    aci_client.container_groups.begin_delete.side_effect = RuntimeError("gone")

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        # Should not raise
        manager.delete()


# ---------------------------------------------------------------------------
# ping()
# ---------------------------------------------------------------------------

def test_ping_resets_idle_timer():
    """ping() cancels the existing timer and starts a new one."""
    manager = _make_manager(idle_timeout_seconds=9999)
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        first_timer = manager._idle_timer
        manager.ping()
        second_timer = manager._idle_timer

    # threading.Timer.cancel() sets the internal finished event immediately
    assert first_timer.finished.is_set(), "first timer should be cancelled after ping()"
    assert second_timer is not None
    assert second_timer is not first_timer
    second_timer.cancel()


def test_ping_noop_after_delete():
    """ping() does nothing after the container has been deleted."""
    manager = _make_manager(idle_timeout_seconds=9999)
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        manager.delete()
        manager.ping()  # should not raise or re-create timer

    assert manager._idle_timer is None


# ---------------------------------------------------------------------------
# _idle_shutdown()
# ---------------------------------------------------------------------------

def test_idle_shutdown_calls_delete():
    """_idle_shutdown() calls delete() when idle timeout expires."""
    manager = _make_manager(idle_timeout_seconds=0.01)
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        manager.create()
        # Wait for the very-short timer to fire
        time.sleep(0.2)

    assert aci_client.container_groups.begin_delete.call_count == 1


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

def test_context_manager_creates_and_deletes():
    """__enter__ calls create(), __exit__ calls delete()."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        with manager:
            assert manager.endpoint_url is not None

    assert aci_client.container_groups.begin_create_or_update.call_count == 1
    assert aci_client.container_groups.begin_delete.call_count == 1


def test_context_manager_deletes_on_exception():
    """__exit__ still calls delete() even if an exception is raised inside the block."""
    manager = _make_manager()
    aci_client = _make_aci_client()

    with _patch_aci_sdk(aci_client), _patch_health():
        with pytest.raises(RuntimeError):
            with manager:
                raise RuntimeError("sweep failed")

    aci_client.container_groups.begin_delete.assert_called_once()


# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------

def test_gpu_configuration_included_when_gpu_count_positive():
    """When gpu_count > 0, the container resources include a GpuResource."""
    manager = _make_manager(gpu_count=1, gpu_sku="V100")

    captured = {}

    def fake_aci_models():
        import types

        class GpuResource:
            def __init__(self, count, sku):
                self.count = count
                self.sku = sku

        class ResourceRequests:
            def __init__(self, memory_in_gb, cpu):
                self.memory_in_gb = memory_in_gb
                self.cpu = cpu
                self.gpu = None

        class ResourceRequirements:
            def __init__(self, requests):
                self.requests = requests

        class ContainerPort:
            def __init__(self, port):
                self.port = port

        class EnvironmentVariable:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        class Container:
            def __init__(self, name, image, resources, ports, environment_variables, command):
                self.name = name
                self.image = image
                self.resources = resources
                self.ports = ports
                self.environment_variables = environment_variables
                self.command = command
                captured["container"] = self

        class IpAddress:
            def __init__(self, ports, type):
                self.ports = ports
                self.type = type

        class Port:
            def __init__(self, protocol, port):
                self.protocol = protocol
                self.port = port

        class ContainerGroup:
            def __init__(self, location, containers, os_type, ip_address):
                self.location = location
                self.containers = containers
                self.os_type = os_type
                self.ip_address = ip_address

        return (
            GpuResource, ResourceRequests, ResourceRequirements,
            ContainerPort, EnvironmentVariable, Container,
            IpAddress, Port, ContainerGroup
        )

    (
        GpuResource, ResourceRequests, ResourceRequirements,
        ContainerPort, EnvironmentVariable, Container,
        IpAddress, Port, ContainerGroup
    ) = fake_aci_models()

    import azure.mgmt.containerinstance.models as aci_models

    with patch.object(aci_models, "GpuResource", GpuResource), \
         patch.object(aci_models, "ResourceRequests", ResourceRequests), \
         patch.object(aci_models, "ResourceRequirements", ResourceRequirements), \
         patch.object(aci_models, "ContainerPort", ContainerPort), \
         patch.object(aci_models, "EnvironmentVariable", EnvironmentVariable), \
         patch.object(aci_models, "Container", Container), \
         patch.object(aci_models, "IpAddress", IpAddress), \
         patch.object(aci_models, "Port", Port), \
         patch.object(aci_models, "ContainerGroup", ContainerGroup), \
         patch.object(aci_models, "ContainerGroupNetworkProtocol", MagicMock(tcp="tcp")), \
         patch.object(aci_models, "OperatingSystemTypes", MagicMock(linux="linux")):
        manager._build_container_group()

    container = captured.get("container")
    assert container is not None
    assert container.resources.requests.gpu is not None
    assert container.resources.requests.gpu.count == 1
    assert container.resources.requests.gpu.sku == "V100"


# ---------------------------------------------------------------------------
# manager_from_env()
# ---------------------------------------------------------------------------

def test_manager_from_env_reads_env_vars(monkeypatch):
    """manager_from_env() builds an ACITEIManager from environment variables."""
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-env")
    monkeypatch.setenv("AZURE_RESOURCE_GROUP", "rg-env")
    monkeypatch.setenv("ACI_TEI_LOCATION", "westus2")
    monkeypatch.setenv("ACI_TEI_IDLE_TIMEOUT", "600")

    mgr = manager_from_env("tei-test", "BAAI/bge-m3")

    assert mgr.subscription_id == "sub-env"
    assert mgr.resource_group == "rg-env"
    assert mgr.location == "westus2"
    assert mgr.idle_timeout_seconds == 600


def test_manager_from_env_uses_defaults(monkeypatch):
    """manager_from_env() uses sensible defaults when optional env vars are absent."""
    monkeypatch.setenv("AZURE_SUBSCRIPTION_ID", "sub-123")
    monkeypatch.delenv("AZURE_RESOURCE_GROUP", raising=False)
    monkeypatch.delenv("ACI_TEI_LOCATION", raising=False)
    monkeypatch.delenv("ACI_TEI_IDLE_TIMEOUT", raising=False)

    mgr = manager_from_env("tei-test", "BAAI/bge-m3")

    assert mgr.resource_group == "URAIT-USE1-NET-PROBLEMBANKGENERATOR-001-RGP"
    assert mgr.location == "eastus"
    assert mgr.idle_timeout_seconds == 1800


def test_manager_from_env_raises_without_subscription_id(monkeypatch):
    """manager_from_env() raises ValueError when AZURE_SUBSCRIPTION_ID is not set."""
    monkeypatch.delenv("AZURE_SUBSCRIPTION_ID", raising=False)

    with pytest.raises(ValueError, match="AZURE_SUBSCRIPTION_ID"):
        manager_from_env("tei-test", "BAAI/bge-m3")


# ---------------------------------------------------------------------------
# Private patch helpers
# ---------------------------------------------------------------------------

def _patch_aci_sdk(aci_client):
    """Patch DefaultAzureCredential and ContainerInstanceManagementClient."""
    return patch.multiple(
        "scripts.common.aci_tei_manager",
        DefaultAzureCredential=MagicMock(return_value=MagicMock()),
        ContainerInstanceManagementClient=MagicMock(return_value=aci_client),
    )


def _patch_health():
    """Patch _wait_for_healthy to be a no-op (avoid real HTTP calls)."""
    return patch.object(ACITEIManager, "_wait_for_healthy", return_value=None)


def _patch_aci_models():
    """Patch azure.mgmt.containerinstance.models with real imports."""
    try:
        import azure.mgmt.containerinstance.models  # noqa: F401
        return patch("builtins.open", MagicMock()) if False else _NullContext()
    except ImportError:
        pytest.skip("azure-mgmt-containerinstance not installed")


class _NullContext:
    """Context manager that does nothing (for when patching is not needed)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

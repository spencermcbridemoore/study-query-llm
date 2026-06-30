"""Fully-mocked unit tests for ``study_query_llm.vllm_serving.vast_client``.

These tests exercise :class:`VastCLIClient` with an injected ``_run`` (a fake
subprocess runner returning canned JSON) so the suite NEVER spawns the real
``vastai`` CLI, touches the network, or provisions a paid instance.  They cover:

* ``search_offers`` -> parsed :class:`VastOffer` list (gpu_ram GB->MB, dph,
  num_gpus) and that the right ``search offers`` subcommand / query string is
  issued with ``--raw``.
* ``create_instance`` -> parsed :class:`VastInstance` (id from ``new_contract``,
  status "created", endpoint fields None) and the ``create instance`` subcommand.
* ``get_instance`` / ``show instance`` -> parsed :class:`VastInstance`
  (actual_status, ssh_host/ssh_port, public_ip).
* ``destroy_instance`` -> the ``destroy instance <id>`` subcommand is issued.
* ``wait_until_running`` with an injected ``_sleep`` and a ``get_instance`` that
  flips from a not-ready state to a running state with an SSH endpoint, plus the
  timeout path.

The subprocess runner is the only side-effecting seam; we replace it with a
``MagicMock`` (mirroring the docker-mocking style in
``test_local_docker_tei_manager.py``).
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock

import pytest

from study_query_llm.vllm_serving.vast_client import (
    VastCLIClient,
    VastClient,
    VastInstance,
    VastOffer,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _completed(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
    """Build a stand-in for ``subprocess.CompletedProcess``."""
    return subprocess.CompletedProcess(
        args=["vastai"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _run_returning(payload, *, returncode: int = 0, stderr: str = ""):
    """An injectable ``_run`` that always returns ``json.dumps(payload)``."""
    runner = MagicMock(name="_run")
    runner.return_value = _completed(
        json.dumps(payload), returncode=returncode, stderr=stderr
    )
    return runner


def _last_argv(runner: MagicMock) -> list[str]:
    """The argv list passed to the most recent ``_run`` invocation."""
    return runner.call_args[0][0]


def _all_argvs(runner: MagicMock) -> list[list[str]]:
    return [c.args[0] for c in runner.call_args_list]


_OFFER_ROW = {
    "id": 778899,
    "gpu_name": "RTX_4090",
    "num_gpus": 2,
    "gpu_ram": 24.0,  # GB per GPU on vast.ai
    "dph_total": 0.534,
    "geolocation": "US",
}

_RUNNING_INSTANCE_ROW = {
    "id": 424242,
    "actual_status": "running",
    "cur_state": "running",
    "ssh_host": "ssh5.vast.ai",
    "ssh_port": 41123,
    "public_ipaddr": "203.0.113.7",
}

_LOADING_INSTANCE_ROW = {
    "id": 424242,
    "actual_status": "loading",
    "cur_state": "running",
    "ssh_host": None,
    "ssh_port": None,
    "public_ipaddr": None,
}


# ---------------------------------------------------------------------------
# Protocol / construction
# ---------------------------------------------------------------------------

def test_cli_client_is_a_vast_client():
    """VastCLIClient is a concrete VastClient (the injectable seam)."""
    client = VastCLIClient(_run=_run_returning([]))
    assert isinstance(client, VastClient)


def test_run_is_injected_not_real_subprocess():
    """The default _run is subprocess.run, but tests always inject a fake."""
    runner = _run_returning([])
    client = VastCLIClient(_run=runner)
    assert client._run is runner
    assert client._run is not subprocess.run


# ---------------------------------------------------------------------------
# search_offers
# ---------------------------------------------------------------------------

def test_search_offers_parses_offer():
    """search_offers parses canned JSON into a VastOffer (GB->MB, dph, gpus)."""
    runner = _run_returning([_OFFER_ROW])
    client = VastCLIClient(_run=runner)

    offers = client.search_offers(
        gpu_name="RTX 4090", min_gpu_ram_mb=24000, num_gpus=2, max_dph=0.9
    )

    assert len(offers) == 1
    offer = offers[0]
    assert isinstance(offer, VastOffer)
    assert offer.offer_id == "778899"
    assert offer.gpu_name == "RTX_4090"
    assert offer.num_gpus == 2
    assert offer.gpu_ram_mb == int(round(24.0 * 1024))  # 24576
    assert offer.dph == pytest.approx(0.534)
    assert offer.raw == _OFFER_ROW


def test_search_offers_issues_search_subcommand_with_raw():
    """search_offers shells out to `vastai search offers ... --raw`."""
    runner = _run_returning([_OFFER_ROW])
    client = VastCLIClient(_run=runner)

    client.search_offers(
        gpu_name="RTX 4090", min_gpu_ram_mb=24000, num_gpus=1, max_dph=0.9, limit=5
    )

    argv = _last_argv(runner)
    assert argv[0] == "vastai"
    assert argv[1:3] == ["search", "offers"]
    assert "--raw" in argv
    # limit is forwarded as a string.
    assert "--limit" in argv
    assert argv[argv.index("--limit") + 1] == "5"

    # The query string (argv[3]) encodes the search criteria.
    query = argv[3]
    assert "rentable=true" in query
    assert "num_gpus=1" in query
    # gpu_name spaces become underscores; min RAM MB->GB.
    assert "gpu_name=RTX_4090" in query
    assert "gpu_ram>=23" in query  # 24000 // 1024 == 23
    assert "dph_total<=0.9" in query


def test_search_offers_omits_optional_filters_when_unset():
    """No gpu_name / no max_dph -> those query terms are absent."""
    runner = _run_returning([])
    client = VastCLIClient(_run=runner)

    client.search_offers(gpu_name=None, min_gpu_ram_mb=16000, max_dph=None)

    query = _last_argv(runner)[3]
    assert "gpu_name=" not in query
    assert "dph_total" not in query


def test_search_offers_handles_wrapped_payload():
    """A dict payload wrapping rows under 'offers' is coerced to a row list."""
    runner = _run_returning({"offers": [_OFFER_ROW]})
    client = VastCLIClient(_run=runner)

    offers = client.search_offers(
        gpu_name=None, min_gpu_ram_mb=24000, max_dph=None
    )

    assert len(offers) == 1
    assert offers[0].offer_id == "778899"


def test_search_offers_empty_when_no_rows():
    runner = _run_returning([])
    client = VastCLIClient(_run=runner)

    assert client.search_offers(gpu_name=None, min_gpu_ram_mb=24000, max_dph=None) == []


# ---------------------------------------------------------------------------
# create_instance
# ---------------------------------------------------------------------------

def test_create_instance_parses_new_contract_id():
    """create_instance reads the new id from `new_contract` and reports created."""
    runner = _run_returning({"success": True, "new_contract": 999001})
    client = VastCLIClient(_run=runner)

    inst = client.create_instance(
        "778899",
        image="vllm/vllm-openai:latest",
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        onstart="vllm serve --model x --port 8000",
        disk_gb=50,
        label="vllm-serving",
    )

    assert isinstance(inst, VastInstance)
    assert inst.instance_id == "999001"
    assert inst.status == "created"
    # No endpoint is known yet at creation; that is what wait_until_running gets.
    assert inst.ssh_host is None
    assert inst.ssh_port is None
    assert inst.public_ip is None


def test_create_instance_issues_create_subcommand():
    """create_instance shells out to `vastai create instance <offer> ...`."""
    runner = _run_returning({"new_contract": 999001})
    client = VastCLIClient(_run=runner)

    client.create_instance(
        "778899",
        image="vllm/vllm-openai:latest",
        env={"K": "V"},
        onstart="run server",
        disk_gb=50,
        label="vllm-serving",
    )

    argv = _last_argv(runner)
    assert argv[1:3] == ["create", "instance"]
    assert "778899" in argv
    assert "--image" in argv
    assert argv[argv.index("--image") + 1] == "vllm/vllm-openai:latest"
    assert "--disk" in argv
    assert argv[argv.index("--disk") + 1] == "50"
    assert "--onstart-cmd" in argv
    assert argv[argv.index("--onstart-cmd") + 1] == "run server"
    assert "--label" in argv
    assert argv[argv.index("--label") + 1] == "vllm-serving"
    # env rendered docker-run style "-e K=V".
    assert "--env" in argv
    assert "-e K=V" in argv[argv.index("--env") + 1]
    assert "--raw" in argv


def test_create_instance_falls_back_to_offer_id_when_id_missing():
    """An unrecognised payload shape falls back to the offer id."""
    runner = _run_returning({"success": True})  # no id keys
    client = VastCLIClient(_run=runner)

    inst = client.create_instance("778899", image="img")
    assert inst.instance_id == "778899"


# ---------------------------------------------------------------------------
# get_instance
# ---------------------------------------------------------------------------

def test_get_instance_parses_running_instance():
    """get_instance parses status + ssh endpoint + public ip from `show`."""
    runner = _run_returning([_RUNNING_INSTANCE_ROW])
    client = VastCLIClient(_run=runner)

    inst = client.get_instance("424242")

    assert isinstance(inst, VastInstance)
    assert inst.instance_id == "424242"
    assert inst.status == "running"
    assert inst.ssh_host == "ssh5.vast.ai"
    assert inst.ssh_port == 41123
    assert inst.public_ip == "203.0.113.7"


def test_get_instance_issues_show_subcommand_with_raw():
    runner = _run_returning([_RUNNING_INSTANCE_ROW])
    client = VastCLIClient(_run=runner)

    client.get_instance("424242")

    argv = _last_argv(runner)
    assert argv[1:3] == ["show", "instance"]
    assert "424242" in argv
    assert "--raw" in argv


def test_get_instance_loading_has_no_endpoint():
    """A still-booting instance reports its status but no ssh endpoint."""
    runner = _run_returning([_LOADING_INSTANCE_ROW])
    client = VastCLIClient(_run=runner)

    inst = client.get_instance("424242")
    assert inst.status == "loading"
    assert inst.ssh_host is None
    assert inst.ssh_port is None


# ---------------------------------------------------------------------------
# destroy_instance
# ---------------------------------------------------------------------------

def test_destroy_instance_issues_destroy_subcommand():
    """destroy_instance shells out to `vastai destroy instance <id>`."""
    runner = MagicMock(name="_run")
    runner.return_value = _completed("destroyed\n")
    client = VastCLIClient(_run=runner)

    client.destroy_instance("424242")

    argv = _last_argv(runner)
    assert argv[1:3] == ["destroy", "instance"]
    assert "424242" in argv


def test_destroy_instance_returns_none():
    runner = MagicMock(name="_run")
    runner.return_value = _completed("ok")
    client = VastCLIClient(_run=runner)

    assert client.destroy_instance("424242") is None


# ---------------------------------------------------------------------------
# api_key plumbing
# ---------------------------------------------------------------------------

def test_api_key_appended_to_argv():
    """When api_key is set it is appended as --api-key to every invocation."""
    runner = _run_returning([])
    client = VastCLIClient(api_key="secret-key", _run=runner)

    client.search_offers(gpu_name=None, min_gpu_ram_mb=24000, max_dph=None)

    argv = _last_argv(runner)
    assert "--api-key" in argv
    assert argv[argv.index("--api-key") + 1] == "secret-key"


def test_no_api_key_means_no_flag():
    runner = _run_returning([])
    client = VastCLIClient(_run=runner)

    client.search_offers(gpu_name=None, min_gpu_ram_mb=24000, max_dph=None)

    assert "--api-key" not in _last_argv(runner)


# ---------------------------------------------------------------------------
# error surfacing
# ---------------------------------------------------------------------------

def test_nonzero_exit_raises_runtimeerror():
    """A non-zero CLI exit is surfaced as a RuntimeError with stderr."""
    runner = MagicMock(name="_run")
    runner.return_value = _completed("", returncode=2, stderr="boom: bad query")
    client = VastCLIClient(_run=runner)

    with pytest.raises(RuntimeError, match="boom: bad query"):
        client.get_instance("424242")


def test_missing_cli_raises_friendly_runtimeerror():
    """FileNotFoundError (CLI absent) is converted to an actionable RuntimeError."""
    runner = MagicMock(name="_run", side_effect=FileNotFoundError("vastai"))
    client = VastCLIClient(_run=runner)

    with pytest.raises(RuntimeError, match="vast.ai CLI not found"):
        client.get_instance("424242")


def test_non_json_output_raises_runtimeerror():
    """Non-JSON stdout from a JSON subcommand is surfaced clearly."""
    runner = MagicMock(name="_run")
    runner.return_value = _completed("not json at all")
    client = VastCLIClient(_run=runner)

    with pytest.raises(RuntimeError, match="non-JSON output"):
        client.search_offers(gpu_name=None, min_gpu_ram_mb=24000, max_dph=None)


# ---------------------------------------------------------------------------
# wait_until_running (inherited from VastClient)
# ---------------------------------------------------------------------------

def test_wait_until_running_returns_when_status_flips():
    """wait_until_running polls get_instance until running WITH an ssh endpoint."""
    client = VastCLIClient(_run=_run_returning([]))

    loading = VastInstance(
        instance_id="424242",
        status="loading",
        ssh_host=None,
        ssh_port=None,
        public_ip=None,
        raw={},
    )
    running = VastInstance(
        instance_id="424242",
        status="running",
        ssh_host="ssh5.vast.ai",
        ssh_port=41123,
        public_ip="203.0.113.7",
        raw={},
    )
    # First two polls: not ready; third poll: running with endpoint.
    client.get_instance = MagicMock(side_effect=[loading, loading, running])
    sleeper = MagicMock(name="_sleep")

    result = client.wait_until_running(
        "424242", timeout=600, interval=10, _sleep=sleeper
    )

    assert result is running
    assert client.get_instance.call_count == 3
    # Slept between the two not-ready polls, but not after success.
    assert sleeper.call_count == 2
    sleeper.assert_called_with(10)


def test_wait_until_running_requires_ssh_endpoint():
    """A 'running' status WITHOUT an ssh endpoint is not yet usable."""
    client = VastCLIClient(_run=_run_returning([]))

    running_no_ssh = VastInstance(
        instance_id="424242",
        status="running",
        ssh_host=None,  # endpoint not published yet
        ssh_port=None,
        public_ip=None,
        raw={},
    )
    running_ok = VastInstance(
        instance_id="424242",
        status="running",
        ssh_host="ssh5.vast.ai",
        ssh_port=41123,
        public_ip=None,
        raw={},
    )
    client.get_instance = MagicMock(side_effect=[running_no_ssh, running_ok])
    sleeper = MagicMock(name="_sleep")

    result = client.wait_until_running("424242", interval=5, _sleep=sleeper)

    assert result is running_ok
    assert client.get_instance.call_count == 2
    assert sleeper.call_count == 1


def test_wait_until_running_times_out():
    """wait_until_running raises TimeoutError if the instance never becomes ready."""
    client = VastCLIClient(_run=_run_returning([]))

    never = VastInstance(
        instance_id="424242",
        status="loading",
        ssh_host=None,
        ssh_port=None,
        public_ip=None,
        raw={},
    )
    client.get_instance = MagicMock(return_value=never)
    sleeper = MagicMock(name="_sleep")

    # timeout=0 -> deadline already passed: one poll, then raise.
    with pytest.raises(TimeoutError, match="424242"):
        client.wait_until_running("424242", timeout=0, interval=1, _sleep=sleeper)

    # Never sleeps because the deadline is already reached after the first poll.
    assert sleeper.call_count == 0

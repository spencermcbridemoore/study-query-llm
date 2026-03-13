"""
VastAI Execution Backend - run containerized jobs via the Vast.ai CLI.

Uses subprocess to call the vastai CLI (pip install vastai). No Python library
dependency -- shells out to the CLI. Constructor takes api_key or reads VASTAI_API_KEY.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from .protocol import (
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)


class VastAIExecution:
    """
    Execution backend that runs containers via the Vast.ai cloud GPU marketplace.

    Uses the vastai CLI: search offers, create instance, show instance, destroy instance, logs.
    """

    backend_type: str = "vastai"

    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the VastAI execution backend.

        Args:
            api_key: Vast.ai API key. If None, reads VASTAI_API_KEY from environment.
        """
        self.api_key = api_key or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "VastAIExecution requires api_key or VASTAI_API_KEY environment variable"
            )

    def _vastai_cmd(self, args: list[str]) -> str:
        """
        Run a vastai CLI command and return stdout.

        Args:
            args: Command arguments (e.g. ["search", "offers", "gpu_name=RTX_4090"])

        Returns:
            stdout from the command

        Raises:
            RuntimeError: If the command exits with non-zero status
        """
        cmd = ["vastai", "--api-key", self.api_key] + args + ["--raw"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"vastai command failed (exit {result.returncode}): {result.stderr}"
            )
        return result.stdout.strip()

    def _search_offers(self, resources: ResourceSpec) -> str:
        """Search for cheapest offer matching resource requirements. Returns offer ID."""
        query_parts = ["num_gpus>=1"]
        if resources.gpu_type:
            query_parts.append(f"gpu_name={resources.gpu_type}")
        if resources.gpu_count > 1:
            query_parts.append(f"num_gpus>={resources.gpu_count}")
        query = " ".join(query_parts)

        stdout = self._vastai_cmd(["search", "offers", query])
        offers = json.loads(stdout)
        if not offers:
            raise RuntimeError(f"No Vast.ai offers found for query: {query}")

        # Find cheapest (sort by dph - dollars per hour)
        if isinstance(offers, dict) and "offers" in offers:
            offers = offers["offers"]
        if isinstance(offers, dict):
            offers = list(offers.values()) if offers else []
        if not isinstance(offers, list):
            offers = [offers]
        sorted_offers = sorted(
            offers,
            key=lambda o: float(o.get("dph", o.get("dphtotal", float("inf")))),
        )
        first = sorted_offers[0]
        offer_id = str(first.get("id", first.get("offer_id", "")))
        if not offer_id:
            raise RuntimeError("Could not parse offer ID from search results")
        return offer_id

    def submit(self, spec: JobSpec) -> str:
        """
        Search for an offer, create an instance, and return the instance ID.

        Args:
            spec: Job specification (image, command, env, resources).

        Returns:
            Instance ID (job_ref) from vastai create instance.
        """
        offer_id = self._search_offers(spec.resources)

        cmd_parts = " ".join(spec.command)
        env_str = " ".join(f"-e {k}={v}" for k, v in spec.env.items()) if spec.env else ""

        args = [
            "create",
            "instance",
            offer_id,
            "--image",
            spec.image,
            "--onstart-cmd",
            cmd_parts,
        ]
        if env_str:
            args.extend(["--env", env_str])
        stdout = self._vastai_cmd(args)
        data = json.loads(stdout)
        instance_id = str(data.get("new_contract", data.get("id", stdout)))
        return instance_id

    def poll(self, job_ref: str) -> JobStatus:
        """
        Get the current status of a Vast.ai instance.

        Args:
            job_ref: Instance ID returned by submit().

        Returns:
            JobStatus with state mapped from vastai show instance output.
        """
        try:
            stdout = self._vastai_cmd(["show", "instance", job_ref])
        except RuntimeError as e:
            return JobStatus(
                state=JobState.FAILED,
                error_message=str(e),
            )

        data = json.loads(stdout)
        status_str = (
            data.get("status")
            or data.get("state")
            or data.get("actual_status", "unknown")
        )
        if isinstance(status_str, dict):
            status_str = status_str.get("status", "unknown")
        status_str = str(status_str).lower()

        state = self._map_vastai_status_to_job_state(status_str)
        return JobStatus(state=state)

    def _map_vastai_status_to_job_state(self, status: str) -> JobState:
        """Map Vast.ai instance status to JobState."""
        if status in ("running", "loading"):
            return JobState.RUNNING
        if status in ("created", "pending", "waiting"):
            return JobState.PENDING
        if status in ("exited", "done", "success"):
            return JobState.SUCCEEDED
        if status in ("failed", "error", "destroyed", "cancelled"):
            return JobState.FAILED
        if status == "cancelled":
            return JobState.CANCELLED
        return JobState.FAILED

    def cancel(self, job_ref: str) -> None:
        """
        Destroy a Vast.ai instance.

        Args:
            job_ref: Instance ID returned by submit().
        """
        try:
            self._vastai_cmd(["destroy", "instance", job_ref])
        except RuntimeError:
            pass  # Instance may already be gone

    def logs(self, job_ref: str, tail: int = 100) -> str:
        """
        Fetch instance logs from Vast.ai.

        Args:
            job_ref: Instance ID returned by submit().
            tail: Number of lines to fetch from the end.

        Returns:
            Log output as a string.
        """
        return self._vastai_cmd(["logs", job_ref, "--tail", str(tail)])

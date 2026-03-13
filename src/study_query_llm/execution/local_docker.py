"""
Local Docker Execution Backend - run containerized jobs on the local machine.

Wraps the Docker SDK to run arbitrary containers. Does NOT replace
LocalDockerTEIManager (which manages TEI model lifecycle). This backend
runs generic containerized jobs.
"""

from __future__ import annotations

import uuid
from typing import Optional

import docker
import docker.types

from .protocol import (
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)


class LocalDockerExecution:
    """
    Execution backend that runs containers via the local Docker daemon.

    Uses docker.containers.run(detach=True) to start containers and returns
    the container ID as the job reference.
    """

    backend_type: str = "local_docker"

    def __init__(self) -> None:
        """Initialize the local Docker execution backend."""
        self._client = None

    def _get_client(self):
        """Return a Docker client connected to the local daemon."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _device_requests(self, resources: ResourceSpec) -> list:
        """Build device_requests for GPU support."""
        if resources.gpu_count == 0:
            return []
        # -1 means all GPUs, positive N means N GPUs
        count = -1 if resources.gpu_count < 0 else resources.gpu_count
        return [
            docker.types.DeviceRequest(
                count=count,
                capabilities=[["gpu"]],
            )
        ]

    def submit(self, spec: JobSpec) -> str:
        """
        Run a container and return its ID as the job reference.

        Args:
            spec: Job specification (image, command, env, resources).

        Returns:
            Container ID (job_ref) for polling, canceling, or fetching logs.
        """
        client = self._get_client()
        name = spec.name or f"exec-{uuid.uuid4().hex[:12]}"
        device_requests = self._device_requests(spec.resources)

        env_list = [f"{k}={v}" for k, v in spec.env.items()] if spec.env else None

        container = client.containers.run(
            spec.image,
            spec.command,
            name=name,
            detach=True,
            environment=env_list,
            device_requests=device_requests,
            working_dir=spec.working_dir,
        )
        return container.id

    def poll(self, job_ref: str) -> JobStatus:
        """
        Get the current status of a container.

        Args:
            job_ref: Container ID returned by submit().

        Returns:
            JobStatus with state, exit_code (if exited), etc.
        """
        client = self._get_client()
        try:
            container = client.containers.get(job_ref)
        except Exception:
            return JobStatus(
                state=JobState.FAILED,
                error_message="Container not found (may have been removed)",
            )

        status_str = container.attrs.get("State", {}).get("Status", "unknown")
        state = self._map_docker_status_to_job_state(status_str, container)
        exit_code = container.attrs.get("State", {}).get("ExitCode")
        started_at = container.attrs.get("State", {}).get("StartedAt")
        finished_at = container.attrs.get("State", {}).get("FinishedAt")

        return JobStatus(
            state=state,
            exit_code=exit_code if exit_code is not None else None,
            started_at=started_at,
            ended_at=finished_at,
        )

    def _map_docker_status_to_job_state(self, status: str, container) -> JobState:
        """Map Docker container status to JobState."""
        if status == "running":
            return JobState.RUNNING
        if status == "created":
            return JobState.PENDING
        if status == "exited":
            exit_code = container.attrs.get("State", {}).get("ExitCode", -1)
            return JobState.SUCCEEDED if exit_code == 0 else JobState.FAILED
        if status in ("dead", "removing"):
            return JobState.FAILED
        if status in ("paused", "restarting"):
            return JobState.RUNNING
        return JobState.FAILED

    def cancel(self, job_ref: str) -> None:
        """
        Stop and remove a container.

        Args:
            job_ref: Container ID returned by submit().
        """
        client = self._get_client()
        try:
            container = client.containers.get(job_ref)
            container.stop()
            container.remove()
        except Exception:
            pass  # Container may already be gone

    def logs(self, job_ref: str, tail: int = 100) -> str:
        """
        Fetch container logs.

        Args:
            job_ref: Container ID returned by submit().
            tail: Number of lines to fetch from the end.

        Returns:
            Log output as a string.
        """
        client = self._get_client()
        container = client.containers.get(job_ref)
        output = container.logs(tail=tail)
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output)

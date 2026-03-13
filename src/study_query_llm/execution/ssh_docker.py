"""
SSH Docker Execution Backend - run containerized jobs on a remote host via SSH.

Uses subprocess to run `ssh user@host docker run ...` commands. No new pip
dependencies (uses system SSH).
"""

from __future__ import annotations

import subprocess
import uuid
from typing import Optional

from .protocol import (
    JobSpec,
    JobState,
    JobStatus,
    ResourceSpec,
)


class SSHDockerExecution:
    """
    Execution backend that runs Docker containers on a remote host via SSH.

    Uses `ssh user@host docker run ...` to submit, poll, cancel, and fetch logs.
    """

    backend_type: str = "ssh_docker"

    def __init__(
        self,
        host: str,
        user: str = "root",
        ssh_key_path: Optional[str] = None,
        docker_socket: Optional[str] = None,
    ) -> None:
        """
        Initialize the SSH Docker execution backend.

        Args:
            host: Remote host (IP or hostname)
            user: SSH user (default: root)
            ssh_key_path: Path to SSH private key (optional)
            docker_socket: Docker socket path on remote (optional, for custom setups)
        """
        self.host = host
        self.user = user
        self.ssh_key_path = ssh_key_path
        self.docker_socket = docker_socket

    def _ssh_run(self, cmd: str) -> str:
        """
        Run a command on the remote host via SSH.

        Args:
            cmd: Shell command to run on the remote host

        Returns:
            stdout from the command

        Raises:
            RuntimeError: If the command exits with non-zero status
        """
        ssh_args = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if self.ssh_key_path:
            ssh_args.extend(["-i", self.ssh_key_path])
        ssh_args.append(f"{self.user}@{self.host}")
        ssh_args.append(cmd)

        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (exit {result.returncode}): {result.stderr}"
            )
        return result.stdout.strip()

    def _docker_cmd(self, docker_args: str) -> str:
        """Build full docker command with optional socket override."""
        if self.docker_socket:
            return f"DOCKER_HOST={self.docker_socket} docker {docker_args}"
        return f"docker {docker_args}"

    def _gpus_flag(self, resources: ResourceSpec) -> str:
        """Build --gpus flag for docker run."""
        if resources.gpu_count == 0:
            return ""
        if resources.gpu_count < 0:
            return "--gpus all"
        if resources.gpu_type:
            return f'--gpus "device={resources.gpu_type}"'
        return f"--gpus {resources.gpu_count}"

    def submit(self, spec: JobSpec) -> str:
        """
        Run a container on the remote host and return its ID.

        Args:
            spec: Job specification (image, command, env, resources).

        Returns:
            Container ID (job_ref) from docker run -d stdout.
        """
        name = spec.name or f"exec-{uuid.uuid4().hex[:12]}"
        gpus = self._gpus_flag(spec.resources)
        gpus_part = f" {gpus}" if gpus else ""

        env_parts = []
        for k, v in spec.env.items():
            env_parts.append(f"-e {k}={v}")
        env_str = " ".join(env_parts) if env_parts else ""

        cmd_parts = " ".join(f'"{c}"' for c in spec.command)
        docker_args = (
            f"run -d --name {name}{gpus_part} {env_str} {spec.image} {cmd_parts}"
        )
        full_cmd = self._docker_cmd(docker_args)
        stdout = self._ssh_run(full_cmd)
        return stdout.strip()

    def poll(self, job_ref: str) -> JobStatus:
        """
        Get the current status of a container on the remote host.

        Args:
            job_ref: Container ID returned by submit().

        Returns:
            JobStatus with state mapped from docker inspect output.
        """
        docker_args = f"inspect --format '{{{{.State.Status}}}}' {job_ref}"
        full_cmd = self._docker_cmd(docker_args)
        try:
            status_str = self._ssh_run(full_cmd).strip().strip("'\"")
        except RuntimeError as e:
            return JobStatus(
                state=JobState.FAILED,
                error_message=str(e),
            )

        state = self._map_docker_status_to_job_state(status_str)
        return JobStatus(state=state)

    def _map_docker_status_to_job_state(self, status: str) -> JobState:
        """Map Docker container status string to JobState."""
        if status == "running":
            return JobState.RUNNING
        if status == "created":
            return JobState.PENDING
        if status == "exited":
            return JobState.SUCCEEDED  # Cannot distinguish 0 vs non-0 without more inspect
        if status in ("dead", "removing"):
            return JobState.FAILED
        if status in ("paused", "restarting"):
            return JobState.RUNNING
        return JobState.FAILED

    def cancel(self, job_ref: str) -> None:
        """
        Stop and remove a container on the remote host.

        Args:
            job_ref: Container ID returned by submit().
        """
        docker_args = f"stop {job_ref} && docker rm {job_ref}"
        full_cmd = self._docker_cmd(docker_args)
        try:
            self._ssh_run(full_cmd)
        except RuntimeError:
            pass  # Container may already be gone

    def logs(self, job_ref: str, tail: int = 100) -> str:
        """
        Fetch container logs from the remote host.

        Args:
            job_ref: Container ID returned by submit().
            tail: Number of lines to fetch from the end.

        Returns:
            Log output as a string.
        """
        docker_args = f"logs --tail {tail} {job_ref}"
        full_cmd = self._docker_cmd(docker_args)
        return self._ssh_run(full_cmd)

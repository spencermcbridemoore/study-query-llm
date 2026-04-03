#!/usr/bin/env python3
"""
Open an SSH local port forward so your PC can reach Jetstream Postgres on VM 127.0.0.1:5432.

Add to .env:

  JETSTREAM_SSH_HOST=<VM public IPv4 or DNS name>
  JETSTREAM_SSH_USER=exouser
  JETSTREAM_SSH_LOCAL_PORT=5433

Auth (pick one):

  - JETSTREAM_SSH_PASSWORD=...  (requires: pip install sshtunnel)
  - JETSTREAM_SSH_KEY=C:\\Users\\you\\.ssh\\id_ed25519  (passed to ssh -i; no extra package)
  - Neither: interactive ssh (key in agent or type password when prompted)

Optional: JETSTREAM_SSH_PORT=22

Then set DATABASE_URL to 127.0.0.1:JETSTREAM_SSH_LOCAL_PORT (see .env.example).

Storing SSH passwords in .env is convenient but risky; prefer SSH keys.

Usage (leave running in a dedicated terminal):
  python scripts/start_jetstream_postgres_tunnel.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def _run_password_tunnel(
    *,
    host: str,
    ssh_port: int,
    user: str,
    password: str,
    local_port: int,
) -> int:
    try:
        from sshtunnel import SSHTunnelForwarder  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: JETSTREAM_SSH_PASSWORD requires the sshtunnel package.\n"
            "  pip install sshtunnel",
            file=sys.stderr,
        )
        return 1

    print(
        f"Tunnel (password auth, leave running): local 127.0.0.1:{local_port} -> "
        f"{host}:{ssh_port} -> remote 127.0.0.1:5432",
        flush=True,
    )
    print(
        f"Point DATABASE_URL at postgresql://USER:PASS@127.0.0.1:{local_port}/DB?sslmode=prefer",
        flush=True,
    )
    tunnel = SSHTunnelForwarder(
        (host, ssh_port),
        ssh_username=user,
        ssh_password=password,
        remote_bind_address=("127.0.0.1", 5432),
        local_bind_address=("127.0.0.1", local_port),
    )
    tunnel.start()
    print("Tunnel up. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.stop()
    return 0


def _run_subprocess_ssh(
    *,
    host: str,
    user: str,
    local_port: int,
    identity: str | None,
    ssh_port: int,
) -> int:
    remote_bind = "127.0.0.1:5432"
    spec = f"{local_port}:{remote_bind}"
    cmd = ["ssh", "-N", "-p", str(ssh_port), "-L", spec]
    if identity:
        cmd.extend(["-i", identity])
    cmd.append(f"{user}@{host}")
    print("Tunnel (leave this running):", " ".join(cmd), flush=True)
    print(
        f"Point DATABASE_URL at postgresql://USER:PASS@127.0.0.1:{local_port}/DB?sslmode=prefer",
        flush=True,
    )
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("ERROR: ssh not found. Install OpenSSH client.", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env")

    host = (os.environ.get("JETSTREAM_SSH_HOST") or "").strip()
    user = (os.environ.get("JETSTREAM_SSH_USER") or "exouser").strip()
    local_port = int((os.environ.get("JETSTREAM_SSH_LOCAL_PORT") or "5433").strip())
    ssh_port = int((os.environ.get("JETSTREAM_SSH_PORT") or "22").strip())
    password = os.environ.get("JETSTREAM_SSH_PASSWORD")
    if password is not None:
        password = password.strip()
    identity = (os.environ.get("JETSTREAM_SSH_KEY") or "").strip() or None

    if not host:
        print(
            "ERROR: Set JETSTREAM_SSH_HOST in .env to your Jetstream VM hostname or IP.",
            file=sys.stderr,
        )
        return 1

    if password:
        return _run_password_tunnel(
            host=host,
            ssh_port=ssh_port,
            user=user,
            password=password,
            local_port=local_port,
        )

    return _run_subprocess_ssh(
        host=host,
        user=user,
        local_port=local_port,
        identity=identity,
        ssh_port=ssh_port,
    )


if __name__ == "__main__":
    sys.exit(main())

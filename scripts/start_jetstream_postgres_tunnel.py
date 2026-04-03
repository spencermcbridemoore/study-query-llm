#!/usr/bin/env python3
"""
Open an SSH local port forward so your PC can reach Jetstream Postgres on VM 127.0.0.1:5432.

Requires OpenSSH (`ssh` on PATH). Add to .env:

  JETSTREAM_SSH_HOST=your-instance.xxx000000.projects.jetstream-cloud.org
  JETSTREAM_SSH_USER=exouser
  JETSTREAM_SSH_LOCAL_PORT=5433

Then set DATABASE_URL to use 127.0.0.1:JETSTREAM_SSH_LOCAL_PORT (see .env.example).

Usage (leave running in a dedicated terminal):
  python scripts/start_jetstream_postgres_tunnel.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env")

    host = (os.environ.get("JETSTREAM_SSH_HOST") or "").strip()
    user = (os.environ.get("JETSTREAM_SSH_USER") or "exouser").strip()
    local_port = (os.environ.get("JETSTREAM_SSH_LOCAL_PORT") or "5433").strip()

    if not host:
        print(
            "ERROR: Set JETSTREAM_SSH_HOST in .env to your Jetstream VM SSH hostname or IP.\n"
            "  Example: JETSTREAM_SSH_HOST=sqllm-panel.xxx000000.projects.jetstream-cloud.org",
            file=sys.stderr,
        )
        return 1

    remote_bind = "127.0.0.1:5432"
    spec = f"{local_port}:{remote_bind}"
    cmd = ["ssh", "-N", "-L", spec, f"{user}@{host}"]
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


if __name__ == "__main__":
    sys.exit(main())

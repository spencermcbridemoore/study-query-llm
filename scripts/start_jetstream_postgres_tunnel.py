#!/usr/bin/env python3
"""
Open an SSH local port forward so your PC can reach Jetstream Postgres on VM 127.0.0.1:5432.

Add to .env:

  JETSTREAM_SSH_HOST=<VM public IPv4 or DNS name>
  JETSTREAM_SSH_USER=exouser
  JETSTREAM_SSH_LOCAL_PORT=5433

Auth (pick one):

  - JETSTREAM_SSH_PASSWORD=...  (requires: pip install sshtunnel)
  - If unset: JETSTREAM_POSTGRES_PASSWORD is used for SSH (same secret as DB; override with JETSTREAM_SSH_PASSWORD if they differ)
  - JETSTREAM_SSH_KEY=C:\\Users\\you\\.ssh\\id_ed25519
  - JETSTREAM_SSH_KEY_PASSPHRASE=...  (optional; if set, uses sshtunnel so the key passphrase is not typed — same security caveats as .env secrets)
  - Without passphrase in env: `ssh -i` with publickey-only (no fallback to Linux password prompt).
    "Enter passphrase for key" means the PEM file is still encrypted — re-export unencrypted or `ssh-keygen -p -f <key>`.
  - Neither password nor key: interactive ssh (will ask for account password)

Optional: JETSTREAM_SSH_PORT=22
Optional: JETSTREAM_SSH_DEBUG=1  (verbose paramiko/sshtunnel logs)

Then set DATABASE_URL to 127.0.0.1:JETSTREAM_SSH_LOCAL_PORT (see .env.example).

Storing SSH passwords or key passphrases in .env is convenient but risky; prefer ssh-agent and no secrets on disk.

Usage (leave running in a dedicated terminal):
  python scripts/start_jetstream_postgres_tunnel.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def _ssh_tunnel_troubleshooting() -> str:
    return (
        "SSH session failed (could not log in to the VM).\n"
        "  1. Test manually:  ssh -v -p PORT USER@HOST\n"
        "     You must get a shell or at least past authentication.\n"
        "  2. JETSTREAM_POSTGRES_PASSWORD is the *database* password. The Linux account "
        "(e.g. exouser) often has a *different* login password.\n"
        "     Set JETSTREAM_SSH_PASSWORD to your actual SSH password if it differs.\n"
        "  3. Horizon security group: allow TCP 22 from your current public IP.\n"
        "  4. If the VM allows only public keys, set JETSTREAM_SSH_KEY to your private key path.\n"
        "  5. Re-run with JETSTREAM_SSH_DEBUG=1 for detailed logs."
    )


def _maybe_enable_ssh_debug() -> None:
    v = (os.environ.get("JETSTREAM_SSH_DEBUG") or "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return
    logging.basicConfig(level=logging.DEBUG)
    for name in ("paramiko", "sshtunnel", "sshtunnel.SSHTunnelForwarder"):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _run_password_tunnel(
    *,
    host: str,
    ssh_port: int,
    user: str,
    password: str,
    local_port: int,
) -> int:
    _maybe_enable_ssh_debug()
    try:
        from sshtunnel import (  # type: ignore[import-untyped]
            BaseSSHTunnelForwarderError,
            SSHTunnelForwarder,
        )
    except ImportError:
        print(
            "ERROR: SSH password auth requires the sshtunnel package.\n"
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
    try:
        tunnel = SSHTunnelForwarder(
            (host, ssh_port),
            ssh_username=user,
            ssh_password=password,
            remote_bind_address=("127.0.0.1", 5432),
            local_bind_address=("127.0.0.1", local_port),
            allow_agent=False,
            host_pkey_directories=[],
        )
        tunnel.start()
    except AttributeError as e:
        if "DSSKey" in str(e) or "paramiko" in str(e).lower():
            print(
                "ERROR: sshtunnel is incompatible with paramiko 4.x (DSSKey removed).\n"
                "  pip install 'paramiko>=3.4,<4'\n"
                "  Or: pip install -e \".[jetstream-tunnel]\"",
                file=sys.stderr,
            )
            return 1
        raise
    except BaseSSHTunnelForwarderError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(_ssh_tunnel_troubleshooting(), file=sys.stderr)
        return 1
    print("Tunnel up. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.stop()
    return 0


def _run_pkey_tunnel(
    *,
    host: str,
    ssh_port: int,
    user: str,
    identity_path: str,
    key_passphrase: str,
    local_port: int,
) -> int:
    """Non-interactive tunnel using OpenSSH private key + passphrase (paramiko/sshtunnel)."""
    _maybe_enable_ssh_debug()
    p = Path(identity_path)
    if not p.is_file():
        print(f"ERROR: JETSTREAM_SSH_KEY file not found: {identity_path}", file=sys.stderr)
        return 1
    try:
        from sshtunnel import (  # type: ignore[import-untyped]
            BaseSSHTunnelForwarderError,
            SSHTunnelForwarder,
        )
    except ImportError:
        print(
            "ERROR: JETSTREAM_SSH_KEY_PASSPHRASE requires sshtunnel.\n"
            "  pip install sshtunnel  (and pip install 'paramiko>=3.4,<4')",
            file=sys.stderr,
        )
        return 1

    print(
        f"Tunnel (key + passphrase from env, leave running): local 127.0.0.1:{local_port} -> "
        f"{host}:{ssh_port} -> remote 127.0.0.1:5432",
        flush=True,
    )
    print(
        f"Point DATABASE_URL at postgresql://USER:PASS@127.0.0.1:{local_port}/DB?sslmode=prefer",
        flush=True,
    )
    try:
        tunnel = SSHTunnelForwarder(
            (host, ssh_port),
            ssh_username=user,
            ssh_pkey=str(p.resolve()),
            ssh_private_key_password=key_passphrase,
            remote_bind_address=("127.0.0.1", 5432),
            local_bind_address=("127.0.0.1", local_port),
            allow_agent=False,
            host_pkey_directories=[],
        )
        tunnel.start()
    except AttributeError as e:
        if "DSSKey" in str(e) or "paramiko" in str(e).lower():
            print(
                "ERROR: sshtunnel is incompatible with paramiko 4.x (DSSKey removed).\n"
                "  pip install 'paramiko>=3.4,<4'",
                file=sys.stderr,
            )
            return 1
        raise
    except BaseSSHTunnelForwarderError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(_ssh_tunnel_troubleshooting(), file=sys.stderr)
        return 1
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
        # Avoid falling back to password auth when the key fails (confusing prompt).
        # If the key file is still encrypted, OpenSSH will prompt for the key passphrase only.
        cmd.extend(
            [
                "-i",
                identity,
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "PreferredAuthentications=publickey",
            ]
        )
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
    ssh_pw = (os.environ.get("JETSTREAM_SSH_PASSWORD") or "").strip()
    pg_pw = (os.environ.get("JETSTREAM_POSTGRES_PASSWORD") or "").strip()
    password = ssh_pw or pg_pw
    identity = (os.environ.get("JETSTREAM_SSH_KEY") or "").strip() or None
    key_passphrase_raw = os.environ.get("JETSTREAM_SSH_KEY_PASSPHRASE")
    key_passphrase = (key_passphrase_raw or "").strip()

    if not host:
        print(
            "ERROR: Set JETSTREAM_SSH_HOST in .env to your Jetstream VM hostname or IP.",
            file=sys.stderr,
        )
        return 1

    if identity and not Path(identity).expanduser().is_file():
        print(f"ERROR: JETSTREAM_SSH_KEY file not found: {identity}", file=sys.stderr)
        return 1

    # Key file + passphrase in .env: use sshtunnel (OpenSSH `ssh` cannot read passphrase from env).
    if identity and key_passphrase:
        return _run_pkey_tunnel(
            host=host,
            ssh_port=ssh_port,
            user=user,
            identity_path=identity,
            key_passphrase=key_passphrase,
            local_port=local_port,
        )

    # Key file without passphrase in env: `ssh -i` (may prompt if key is encrypted).
    if identity:
        return _run_subprocess_ssh(
            host=host,
            user=user,
            local_port=local_port,
            identity=identity,
            ssh_port=ssh_port,
        )

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
        identity=None,
        ssh_port=ssh_port,
    )


if __name__ == "__main__":
    sys.exit(main())

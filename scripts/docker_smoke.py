#!/usr/bin/env python3
"""Utility script to smoke-test the Docker stack locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose_env(profile: Optional[str], database_url: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if profile:
        env["COMPOSE_PROFILES"] = profile
    if database_url:
        env["DATABASE_URL"] = database_url
    return env


def _run_compose(args: list[str], env: Dict[str, str]) -> None:
    cmd = ["docker", "compose", *args]
    print(f"[docker] {' '.join(args)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


def _wait_for_health(url: str, timeout: int) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"[health] {url} responded with 200")
                    return
        except Exception as exc:  # pragma: no cover - network timing dependent
            print(f"[health] waiting for {url}: {exc}")
        time.sleep(2)
    raise RuntimeError(f"Health check timed out after {timeout}s: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Docker stack (build + health check)."
    )
    parser.add_argument(
        "--profile",
        help="Value for COMPOSE_PROFILES (e.g., 'postgres').",
    )
    parser.add_argument(
        "--database-url",
        help="Override DATABASE_URL for the app container.",
    )
    parser.add_argument(
        "--health-url",
        default="http://127.0.0.1:5006/health",
        help="URL to poll for readiness (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Seconds to wait for health endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip docker compose --build (useful when image already built).",
    )

    args = parser.parse_args()
    env = _compose_env(args.profile, args.database_url)

    try:
        if not args.skip_build:
            _run_compose(["up", "--build", "-d"], env)
        else:
            _run_compose(["up", "-d"], env)
        _wait_for_health(args.health_url, args.timeout)
        print("[smoke] Success!")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[smoke] FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            _run_compose(["down"], env)
        except subprocess.CalledProcessError as exc:
            print(f"[docker] down failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())


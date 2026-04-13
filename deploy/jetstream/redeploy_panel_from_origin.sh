#!/usr/bin/env bash
# Jetstream2: sync repo from origin, optionally pin a new Panel image, pull, and restart compose.
#
# Panel code runs inside the container image (IMAGE_REF), not from the git checkout alone.
# Use NEW_IMAGE_REF after CI (or ./build-and-push.sh) publishes a new image digest/tag.
#
# Usage (on the VM, from deploy/jetstream/):
#   chmod +x redeploy_panel_from_origin.sh
#   ./redeploy_panel_from_origin.sh
#   NEW_IMAGE_REF='ghcr.io/org/study-query-llm@sha256:...' ./redeploy_panel_from_origin.sh
#
# Environment (all optional except NEW_IMAGE_REF when changing the image):
#   REPO               — git repo root (default: parent of this directory)
#   GIT_REMOTE         — default: origin
#   GIT_REF            — branch to pull (default: main)
#   SKIP_GIT=1         — do not run git fetch/pull
#   NEW_IMAGE_REF      — if set, replace the IMAGE_REF= line in .env.jetstream (UTF-8 backup first)
#   SKIP_COMPOSE_PULL=1 — skip "docker compose pull app" (only up -d; rare)
#
# Requires: git (unless SKIP_GIT=1), docker compose, curl, python3 (when NEW_IMAGE_REF is set)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_REF="${GIT_REF:-main}"
COMPOSE_FILE="docker-compose.jetstream.yml"
ENV_FILE="${ENV_FILE:-.env.jetstream}"
PROJECT="${PROJECT:-sqllm-jetstream}"
ENV_ABS="${SCRIPT_DIR}/${ENV_FILE}"

cd "$SCRIPT_DIR"

if [[ ! -f "$ENV_ABS" ]]; then
  echo "error: missing ${ENV_ABS} (copy from .env.jetstream.example and fill secrets)." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker not found" >&2
  exit 1
fi

compose() {
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" -p "$PROJECT" "$@"
}

if [[ "${SKIP_GIT:-0}" != "1" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "error: git not found; install git or set SKIP_GIT=1" >&2
    exit 1
  fi
  echo "=== git: fetch + pull ${GIT_REMOTE}/${GIT_REF} (repo: ${REPO}) ==="
  cd "$REPO"
  git fetch "$GIT_REMOTE" "$GIT_REF"
  git pull "$GIT_REMOTE" "$GIT_REF"
  cd "$SCRIPT_DIR"
else
  echo "=== SKIP_GIT=1 — skipping git fetch/pull ==="
fi

if [[ -n "${NEW_IMAGE_REF:-}" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 required to update IMAGE_REF in .env.jetstream" >&2
    exit 1
  fi
  echo "=== updating IMAGE_REF in ${ENV_FILE} (backup .bak.<timestamp>) ==="
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  cp -a "$ENV_ABS" "${ENV_ABS}.bak.${ts}"
  export NEW_IMAGE_REF
  python3 - "$ENV_ABS" <<'PY'
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
ref = os.environ.get("NEW_IMAGE_REF", "").strip()
if not ref:
    sys.exit("error: NEW_IMAGE_REF is empty")
# .env.jetstream may contain non-UTF-8 bytes (e.g. legacy editor); replace on decode.
text = path.read_bytes().decode("utf-8", errors="replace")
lines = text.splitlines()
out: list[str] = []
seen = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("IMAGE_REF=") and not stripped.startswith("#"):
        out.append(f"IMAGE_REF={ref}")
        seen = True
    else:
        out.append(line.rstrip("\n\r"))
if not seen:
    out.append(f"IMAGE_REF={ref}")
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
fi

if [[ "${SKIP_COMPOSE_PULL:-0}" == "1" ]]; then
  echo "=== SKIP_COMPOSE_PULL=1 — skipping docker compose pull app ==="
else
  echo "=== docker compose: pull app ==="
  compose pull app
fi

echo "=== docker compose: up -d ==="
compose up -d

echo "=== docker compose: ps ==="
compose ps

echo "=== GET http://127.0.0.1:5006/health ==="
if curl -fsS --max-time 20 http://127.0.0.1:5006/health; then
  echo ""
  echo "OK: Panel /health"
else
  echo "error: /health failed (see: docker compose ... logs app --tail 80)" >&2
  exit 1
fi

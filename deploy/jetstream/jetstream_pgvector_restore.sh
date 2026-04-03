#!/usr/bin/env bash
# Jetstream: wipe Compose Postgres volume, start db (pgvector image), pg_restore dump, start app.
#
# Requires docker-compose.jetstream.yml with image: pgvector/pgvector:pg17 for db.
# DESTRUCTIVE: docker compose down -v removes the named Postgres volume.
#
# Usage (from this directory):
#   chmod +x jetstream_pgvector_restore.sh
#   ./jetstream_pgvector_restore.sh /path/to/neon_for_jetstream.dump
#
# Optional env overrides:
#   COMPOSE_FILE=docker-compose.jetstream.yml
#   ENV_FILE=.env.jetstream
#   COMPOSE_PROJECT=sqllm-jetstream

set -euo pipefail

DUMP_PATH="${1:?Usage: $0 /path/to/file.dump}"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.jetstream.yml}"
ENV_FILE="${ENV_FILE:-.env.jetstream}"
PROJECT="${COMPOSE_PROJECT:-sqllm-jetstream}"
CONTAINER="${JETSTREAM_DB_CONTAINER:-sqllm-db}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$COMPOSE_FILE" ]] || [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Run from deploy/jetstream (need ${COMPOSE_FILE} and ${ENV_FILE})." >&2
  exit 1
fi
if [[ ! -f "$DUMP_PATH" ]]; then
  echo "ERROR: Dump not found: ${DUMP_PATH}" >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

: "${POSTGRES_USER:?POSTGRES_USER must be set in ${ENV_FILE}}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in ${ENV_FILE}}"
: "${POSTGRES_DB:?POSTGRES_DB must be set in ${ENV_FILE}}"

echo "=== Stopping stack and removing Postgres volume (DATA LOSS for this DB) ==="
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" -p "$PROJECT" down -v

echo "=== Starting db only ==="
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" -p "$PROJECT" up -d db

echo "=== Waiting for Postgres healthy (up to 120s) ==="
for _ in $(seq 1 60); do
  status="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "$CONTAINER" 2>/dev/null || true)"
  if [[ "$status" == "healthy" ]]; then
    echo "db health: $status"
    break
  fi
  sleep 2
done
status="$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "$CONTAINER" 2>/dev/null || true)"
if [[ "$status" != "healthy" ]]; then
  echo "ERROR: ${CONTAINER} not healthy (last status: ${status}). Check: docker compose ... logs db" >&2
  exit 1
fi

RESTORE_SCRIPT="$SCRIPT_DIR/restore_pg_dump_to_compose_db.sh"
if [[ ! -f "$RESTORE_SCRIPT" ]]; then
  echo "ERROR: Missing ${RESTORE_SCRIPT}" >&2
  exit 1
fi
# Windows uploads may add CRLF; strip so /usr/bin/env bash works.
if command -v dos2unix >/dev/null 2>&1; then
  dos2unix -q "$RESTORE_SCRIPT" 2>/dev/null || true
else
  sed -i 's/\r$//' "$RESTORE_SCRIPT" 2>/dev/null || sed -i '' 's/\r$//' "$RESTORE_SCRIPT" 2>/dev/null || true
fi
chmod +x "$RESTORE_SCRIPT"

echo "=== Restoring dump ==="
"$RESTORE_SCRIPT" "$(realpath "$DUMP_PATH")"

echo "=== Verifying pgvector extension ==="
docker exec -e "PGPASSWORD=${POSTGRES_PASSWORD}" "$CONTAINER" \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
  "SELECT extname FROM pg_extension WHERE extname = 'vector';" | grep -q vector \
  && echo "OK: extension vector is installed." \
  || echo "WARNING: extension vector not found — ensure db image is pgvector/pgvector:pg17 in ${COMPOSE_FILE}."

echo "=== Starting app ==="
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" -p "$PROJECT" up -d app

echo "=== Done. Neon-only extension pg_session_jwt may still error during restore; that is expected on self-hosted Postgres. ==="

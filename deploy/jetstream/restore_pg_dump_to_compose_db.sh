#!/usr/bin/env bash
# Restore a pg_dump custom-format file into the Jetstream Compose Postgres service.
#
# Prerequisites:
#   - Run from deploy/jetstream/ with .env.jetstream present (same dir as docker-compose.jetstream.yml).
#   - Stop the app first to avoid open connections:  docker compose ... stop app
#   - Container name matches compose: sqllm-db (see docker-compose.jetstream.yml).
#
# Usage:
#   chmod +x restore_pg_dump_to_compose_db.sh
#   ./restore_pg_dump_to_compose_db.sh /path/to/neon_for_jetstream_....dump
#
# Optional: set COMPOSE_PROJECT, compose file, env file:
#   COMPOSE_FILE=docker-compose.jetstream.yml \\
#   ENV_FILE=.env.jetstream \\
#   ./restore_pg_dump_to_compose_db.sh ./neon.dump

set -euo pipefail

DUMP_PATH="${1:?Usage: $0 /path/to/dump.dump}"

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.jetstream.yml}"
ENV_FILE="${ENV_FILE:-.env.jetstream}"
PROJECT="${COMPOSE_PROJECT:-sqllm-jetstream}"
CONTAINER="${JETSTREAM_DB_CONTAINER:-sqllm-db}"
REMOTE="/tmp/jetstream_pg_restore.dump"

if [[ ! -f "$DUMP_PATH" ]]; then
  echo "ERROR: file not found: $DUMP_PATH" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE not found (run this from deploy/jetstream/)." >&2
  exit 1
fi

# shellcheck disable=SC1090
set -a
source "$ENV_FILE"
set +a

: "${POSTGRES_USER:?POSTGRES_USER must be set in ${ENV_FILE}}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set in ${ENV_FILE}}"
: "${POSTGRES_DB:?POSTGRES_DB must be set in ${ENV_FILE}}"

echo "Copying dump into container ${CONTAINER}..."
docker cp "$DUMP_PATH" "${CONTAINER}:${REMOTE}"

echo "Running pg_restore (this may take a while)..."
docker exec \
  -e "PGPASSWORD=${POSTGRES_PASSWORD}" \
  "$CONTAINER" \
  pg_restore \
  -U "$POSTGRES_USER" \
  -d "$POSTGRES_DB" \
  --clean \
  --if-exists \
  --no-owner \
  --no-acl \
  --verbose \
  "$REMOTE"

echo "Removing dump from container..."
docker exec "$CONTAINER" rm -f "$REMOTE"

echo "Restore finished. Start app: docker compose -f ${COMPOSE_FILE} --env-file ${ENV_FILE} -p ${PROJECT} start app"

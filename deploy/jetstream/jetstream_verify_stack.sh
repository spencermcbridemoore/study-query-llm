#!/usr/bin/env bash
# Smoke checks after pg_restore / deploy.
#
# Usage (from deploy/jetstream/):
#   chmod +x jetstream_verify_stack.sh
#   ./jetstream_verify_stack.sh

set -u

ENV_FILE="${ENV_FILE:-.env.jetstream}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

fail=0
ok()  { echo "OK:  $*"; }
bad() { echo "FAIL: $*" >&2; fail=1; }

echo "=== docker: sqllm containers ==="
if docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E 'sqllm-|NAMES'; then
  ok "docker ps listing"
else
  bad "docker ps failed"
fi

echo ""
echo "=== app health (localhost:5006) ==="
if curl -fsS --max-time 10 http://127.0.0.1:5006/health; then
  echo ""
  ok "GET /health"
else
  bad "curl /health (is sqllm-app up and bound to 127.0.0.1:5006?)"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  bad "missing $ENV_FILE (run from deploy/jetstream/)"
  exit "$fail"
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${POSTGRES_USER:?set in $ENV_FILE}"
: "${POSTGRES_PASSWORD:?set in $ENV_FILE}"
: "${POSTGRES_DB:?set in $ENV_FILE}"

echo ""
echo "=== Postgres: pgvector extension ==="
if out="$(docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" sqllm-db \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc \
  "SELECT extname FROM pg_extension WHERE extname = 'vector';" 2>&1)"; then
  if echo "$out" | grep -q vector; then
    ok "extension vector: $out"
  else
    bad "extension vector missing or unexpected output: $out"
  fi
else
  bad "psql failed: $out"
fi

echo ""
echo "=== Postgres: row counts (sanity) ==="
for q in \
  "SELECT COUNT(*) AS raw_calls FROM raw_calls" \
  "SELECT COUNT(*) AS embedding_vectors FROM embedding_vectors"
do
  if out="$(docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" sqllm-db \
    psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "$q" 2>&1)"; then
    echo "  $q -> $out"
  else
    bad "query failed: $q ($out)"
  fi
done

echo ""
if [[ "$fail" -eq 0 ]]; then
  echo "All checks passed."
else
  echo "Some checks failed (see FAIL lines above)."
fi
exit "$fail"

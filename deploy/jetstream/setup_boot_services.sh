#!/usr/bin/env bash
# Install systemd so Panel + Postgres (Docker Compose) survive reboots.
# Optional: copy Caddyfile from this directory and enable Caddy.
#
# Run on the Jetstream/Ubuntu VM after `git pull` (e.g. from Guacamole SSH):
#   cd /path/to/repo/deploy/jetstream
#   chmod +x setup_boot_services.sh   # once, if needed
#   ./setup_boot_services.sh
#   ./setup_boot_services.sh --with-caddyfile   # also install /etc/caddy/Caddyfile
#
# Requires: sudo, Docker. Prerequisite: `.env.jetstream` in this directory.

set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${DEPLOY_DIR}/.env.jetstream"
INSTALL_CADDYFILE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-caddyfile) INSTALL_CADDYFILE=1 ;;
    -h|--help)
      head -n 18 "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: Missing ${ENV_FILE} (copy from .env.jetstream.example and edit)."
  exit 1
fi

sudo tee /etc/systemd/system/study-query-llm.service > /dev/null <<EOF
[Unit]
Description=Study Query LLM Panel App (Docker Compose)
Documentation=https://github.com/yourorg/study-query-llm
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${DEPLOY_DIR}
EnvironmentFile=${ENV_FILE}

ExecStart=/usr/bin/docker compose \\
    -f docker-compose.jetstream.yml \\
    --env-file .env.jetstream \\
    -p sqllm-jetstream \\
    up -d --remove-orphans

ExecStop=/usr/bin/docker compose \\
    -f docker-compose.jetstream.yml \\
    -p sqllm-jetstream \\
    down

ExecReload=/usr/bin/docker compose \\
    -f docker-compose.jetstream.yml \\
    --env-file .env.jetstream \\
    -p sqllm-jetstream \\
    up -d --remove-orphans

Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable docker
sudo systemctl enable --now study-query-llm

if [[ "$INSTALL_CADDYFILE" -eq 1 ]]; then
  if [[ ! -f "${DEPLOY_DIR}/Caddyfile" ]]; then
    echo "ERROR: ${DEPLOY_DIR}/Caddyfile not found (edit domain and bcrypt hash first)."
    exit 1
  fi
  sudo cp "${DEPLOY_DIR}/Caddyfile" /etc/caddy/Caddyfile
  sudo systemctl enable caddy
  sudo systemctl reload caddy 2>/dev/null || sudo systemctl restart caddy
elif systemctl list-unit-files caddy.service &>/dev/null; then
  sudo systemctl enable caddy || true
fi

echo ""
echo "OK. Check: sudo systemctl status study-query-llm docker"
echo "         sudo systemctl status caddy   # if installed"

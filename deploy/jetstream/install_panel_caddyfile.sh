#!/usr/bin/env bash
# Replace the default package Caddyfile with the Study Query LLM template (HTTPS + basic auth + Panel proxy).
# Run on the Jetstream VM after cloning the repo (e.g. ~/app).
#
#   cd /path/to/repo/deploy/jetstream
#   chmod +x install_panel_caddyfile.sh
#   ./install_panel_caddyfile.sh
#
# Then edit /etc/caddy/Caddyfile: set YOUR_DOMAIN to your Jetstream hostname.
# Either put a bcrypt hash from `caddy hash-password` in place of REPLACE_WITH_BCRYPT_HASH,
# or leave the placeholder and run ./rotate_caddy_basic_auth.sh --generate
#
#   sudo caddy validate --config /etc/caddy/Caddyfile
#   sudo systemctl reload caddy

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${DIR}/Caddyfile"
DST="${CADDYFILE:-/etc/caddy/Caddyfile}"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: Missing template ${SRC}" >&2
  exit 1
fi

if [[ -f "$DST" ]]; then
  BAK="${DST}.bak.$(date +%Y%m%d%H%M%S)"
  echo "Backing up ${DST} -> ${BAK}"
  sudo cp "$DST" "$BAK"
fi

echo "Installing ${SRC} -> ${DST}"
sudo cp "$SRC" "$DST"

echo ""
echo "Next:"
echo "  1. sudo nano ${DST}   # replace YOUR_DOMAIN with your Jetstream DNS name"
echo "  2. Set bcrypt hash: either run ./rotate_caddy_basic_auth.sh --generate"
echo "     or run: caddy hash-password --plaintext 'YourPassword' and paste into the file."
echo "  3. sudo caddy validate --config ${DST} && sudo systemctl reload caddy"
echo "See deploy/jetstream/README.md (First-time Caddy on the VM)."

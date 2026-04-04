#!/usr/bin/env bash
# Rotate Caddy HTTP Basic Auth password on the Jetstream VM (run via Guacamole SSH).
#
# Usage:
#   cd /path/to/repo/deploy/jetstream
#   chmod +x rotate_caddy_basic_auth.sh
#
# Generate a long random password, print it once, update Caddy (recommended):
#   ./rotate_caddy_basic_auth.sh --generate
#
# Type a password interactively:
#   ./rotate_caddy_basic_auth.sh
#
# Non-interactive (password may appear in process list / shell history):
#   CADDY_NEW_PASSWORD='your-secret' ./rotate_caddy_basic_auth.sh
#
# Optional: username (default admin)
#   CADDY_AUTH_USER=admin ./rotate_caddy_basic_auth.sh --generate
#
# Requires: sudo, caddy (for `caddy hash-password`), python3, systemd caddy.

set -euo pipefail

CADDYFILE="${CADDYFILE:-/etc/caddy/Caddyfile}"
USER_NAME="${CADDY_AUTH_USER:-admin}"
GENERATE=0

usage() {
  sed -n '1,28p' "$0" | tail -n +2
  exit 0
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate|-g) GENERATE=1 ;;
    -h|--help) usage ;;
    *) die "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

command -v caddy >/dev/null 2>&1 || die "caddy not found. Install Caddy first (see deploy/jetstream/README.md)."
command -v python3 >/dev/null 2>&1 || die "python3 is required."
[[ -f "$CADDYFILE" ]] || die "Missing $CADDYFILE — copy Caddyfile from this repo first."

if [[ -n "${CADDY_NEW_PASSWORD:-}" && "$GENERATE" -eq 1 ]]; then
  die "Use either CADDY_NEW_PASSWORD or --generate, not both."
fi

if [[ -n "${CADDY_NEW_PASSWORD:-}" ]]; then
  PW="$CADDY_NEW_PASSWORD"
  echo "Using password from environment CADDY_NEW_PASSWORD (ensure nobody can read your screen history)."
elif [[ "$GENERATE" -eq 1 ]]; then
  PW="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48), end="")')"
  [[ -n "$PW" ]] || die "Failed to generate password."
  echo ""
  echo "================================================================"
  echo "NEW BASIC AUTH PASSWORD (save it now; shown once):"
  echo ""
  echo "$PW"
  echo ""
  echo "================================================================"
  echo "User name: ${USER_NAME}"
  echo "(Copy the password line above into a password manager. Terminal scrollback may retain it.)"
  echo ""
else
  read -r -s -p "New Basic Auth password for user '${USER_NAME}': " PW
  echo
  read -r -s -p "Again: " PW2
  echo
  [[ "$PW" == "$PW2" ]] || die "Passwords do not match."
  [[ -n "$PW" ]] || die "Password is empty."
fi

# Hash (strip noise if caddy prints a label line)
RAW_HASH="$(caddy hash-password --plaintext "$PW" 2>/dev/null | tail -n 1 | tr -d '\r')"
[[ "$RAW_HASH" == \$2* ]] || die "Unexpected output from caddy hash-password: ${RAW_HASH:0:40}..."

BACKUP="${CADDYFILE}.bak.$(date +%Y%m%d%H%M%S)"
sudo cp "$CADDYFILE" "$BACKUP"
echo "Backed up to $BACKUP"

export CADDYFILE_PATH="$CADDYFILE"
export HASH_LINE="$RAW_HASH"
export USER_NAME

sudo -E python3 <<'PY'
import os
import re
from pathlib import Path

path = Path(os.environ["CADDYFILE_PATH"])
h = os.environ["HASH_LINE"].strip()
user = os.environ["USER_NAME"].strip()
text = path.read_text(encoding="utf-8")
if text.startswith("\ufeff"):
    text = text[1:]
text = text.replace("\r\n", "\n").replace("\r", "\n")


def _token_is_hash_or_placeholder(tok: str) -> bool:
    if tok == "REPLACE_WITH_BCRYPT_HASH":
        return True
    if tok.startswith("{$") and "}" in tok:
        return True
    if tok.startswith("$2") and tok.count("$") >= 3 and len(tok) >= 20:
        return True
    return False


def _process_line(line: str) -> tuple[str, bool]:
    if "#" in line:
        idx = line.index("#")
        main, tail = line[:idx], line[idx:]
    else:
        main, tail = line, ""
    base = main.rstrip()
    m = re.match(r"^(\s*)" + re.escape(user) + r"(\s+)(\S+)", base)
    if not m:
        m = re.match(r"^(\s*)[\"']" + re.escape(user) + r"[\"'](\s+)(\S+)", base)
    if not m:
        return line, False
    if not _token_is_hash_or_placeholder(m.group(3)):
        return line, False
    new_main = m.group(1) + user + m.group(2) + h
    return new_main + tail, True


lines = text.split("\n")
out_lines = []
n = 0
for line in lines:
    new_line, did = _process_line(line)
    if did:
        n += 1
    out_lines.append(new_line)

if n < 1:
    raise SystemExit(
        "No matching basic_auth line for user "
        + repr(user)
        + ". Expected one line like:\n"
        "        "
        + user
        + " $2a$...   (bcrypt from caddy hash-password)\n"
        "or REPLACE_WITH_BCRYPT_HASH or {$SOME_ENV} inside the basicauth / basic_auth block.\n"
        "Check CADDY_AUTH_USER matches the username on that line."
    )

new_text = "\n".join(out_lines)
if text.endswith("\n"):
    new_text += "\n"
path.write_text(new_text, encoding="utf-8")
print(f"Updated {n} hash line(s) for user {user!r}.")
PY

sudo caddy validate --config "$CADDYFILE"
sudo systemctl reload caddy

echo ""
echo "OK: Caddy reloaded. Log in with user '${USER_NAME}' and the new password."
echo "If you keep secrets in deploy/jetstream/.env.jetstream, set:"
echo "  CADDY_AUTH_USER=${USER_NAME}"
echo "  CADDY_AUTH_HASH=${RAW_HASH}"
if [[ "$GENERATE" -eq 1 ]]; then
  echo "(Plaintext password was printed in the box above — not repeated here.)"
fi

<#
.SYNOPSIS
  SSH to Jetstream: preflight disk, git pull, build on VM, atomically switch IMAGE_REF, restart app.

.NOTES
  Gitignored via .gitignore (scratch/local/*). Do not commit secrets. See scratch/local/README.md.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string] $JetstreamHost = "exouser@149.165.153.232",

    [Parameter(Mandatory = $false)]
    [string] $SshIdentity = "$env:USERPROFILE\.ssh\jetstream2_private_key_3",

    # Clone directory name under $HOME on the VM (default: ~/app)
    [Parameter(Mandatory = $false)]
    [string] $RemoteRepoDir = "app",

    [Parameter(Mandatory = $false)]
    [string] $GitRemote = "origin",

    [Parameter(Mandatory = $false)]
    [string] $GitRef = "main",

    # Local-only image tag on the VM (default: study-query-llm:jetstream-vm-<timestamp>)
    [Parameter(Mandatory = $false)]
    [string] $LocalImageTag = "",

    # Minimum free space required on remote root filesystem before build.
    [Parameter(Mandatory = $false)]
    [ValidateRange(1, 200000)]
    [int] $MinFreeRootMb = 3500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $SshIdentity)) {
    throw "SSH identity file not found: $SshIdentity"
}

if ([string]::IsNullOrWhiteSpace($LocalImageTag)) {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $LocalImageTag = "study-query-llm:jetstream-vm-$ts"
}

# Safe for bash single-quoted literals (no shell metacharacters)
$safeToken = '^[a-zA-Z0-9._/@:-]+$'
foreach ($pair in @(
        @{ Name = "RemoteRepoDir"; Value = $RemoteRepoDir },
        @{ Name = "GitRemote"; Value = $GitRemote },
        @{ Name = "GitRef"; Value = $GitRef },
        @{ Name = "LocalImageTag"; Value = $LocalImageTag })) {
    if ($pair.Value -notmatch $safeToken) {
        throw "$($pair.Name) contains unsupported characters; use only [a-zA-Z0-9._/@:-]"
    }
}

$remoteBash = @'
set -euo pipefail
REPO_ROOT="$HOME/__REPO_DIR__"
DEPLOY="$HOME/__REPO_DIR__/deploy/jetstream"
GIT_REMOTE='__GIT_REMOTE__'
GIT_REF='__GIT_REF__'
IMAGE_TAG='__IMAGE_TAG__'
MIN_FREE_MB='__MIN_FREE_MB__'

echo "=== preflight: free space on / ==="
free_kb=$(df -Pk / | awk 'NR==2 {print $4}')
free_mb=$((free_kb / 1024))
echo "Free on /: ${free_mb} MB (required: ${MIN_FREE_MB} MB)"
if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
    echo "ERROR: insufficient free disk for build; refusing to restart app."
    docker system df || true
    exit 42
fi

echo "=== git fetch + pull ==="
cd "$REPO_ROOT"
git fetch "$GIT_REMOTE" "$GIT_REF"
git pull "$GIT_REMOTE" "$GIT_REF"

echo "=== docker build on VM: ${IMAGE_TAG} ==="
docker build -t "$IMAGE_TAG" -f Dockerfile .

echo "=== backup .env.jetstream and set IMAGE_REF only (no file dump) ==="
cd "$DEPLOY"
ts=$(date -u +%Y%m%dT%H%M%SZ)
cp -a .env.jetstream ".env.jetstream.bak.${ts}"
PREV_IMAGE_REF="$(awk -F= '/^[[:space:]]*IMAGE_REF=/{print substr($0,index($0,"=")+1); exit}' .env.jetstream || true)"
export NEW_IMAGE_REF="$IMAGE_TAG"
python3 - <<'PY'
import os
import sys
from pathlib import Path

path = Path(".env.jetstream")
ref = os.environ.get("NEW_IMAGE_REF", "").strip()
if not ref:
    sys.exit("error: NEW_IMAGE_REF is empty")
# .env may contain legacy Windows-1252 bytes; never fail the deploy on decode.
text = path.read_bytes().decode("utf-8", errors="replace")
lines = text.splitlines()
out = []
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
echo "Previous IMAGE_REF: ${PREV_IMAGE_REF:-<unset>}"
echo "New IMAGE_REF: ${IMAGE_TAG}"
echo "Backup written: ${DEPLOY}/.env.jetstream.bak.${ts}"

echo "=== compose up app (--pull never --force-recreate) ==="
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d --pull never --force-recreate app

echo "=== health (retry up to 60s) ==="
health_ok=0
for _ in $(seq 1 30); do
    if curl -fsS --max-time 10 http://127.0.0.1:5006/health >/tmp/panel-health.json; then
        health_ok=1
        break
    fi
    sleep 2
done
if [ "$health_ok" -ne 1 ]; then
    echo "ERROR: new container failed health checks; rolling back to previous IMAGE_REF."
    if [ -n "${PREV_IMAGE_REF:-}" ]; then
        export NEW_IMAGE_REF="$PREV_IMAGE_REF"
        python3 - <<'PY'
import os
import sys
from pathlib import Path

path = Path(".env.jetstream")
ref = os.environ.get("NEW_IMAGE_REF", "").strip()
if not ref:
    sys.exit("error: NEW_IMAGE_REF is empty")
text = path.read_bytes().decode("utf-8", errors="replace")
lines = text.splitlines()
out = []
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
        docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d --pull never --force-recreate app || true
        sleep 3
        curl -fsS --max-time 20 http://127.0.0.1:5006/health || true
    else
        echo "No previous IMAGE_REF found for rollback."
    fi
    exit 1
fi
cat /tmp/panel-health.json
echo ""
echo "OK: Panel /health"

echo "=== docker compose ps ==="
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream ps
'@
$remoteBash = $remoteBash.Replace("__REPO_DIR__", $RemoteRepoDir).
    Replace("__GIT_REMOTE__", $GitRemote).
    Replace("__GIT_REF__", $GitRef).
    Replace("__IMAGE_TAG__", $LocalImageTag).
    Replace("__MIN_FREE_MB__", "$MinFreeRootMb")

$bytes = [System.Text.Encoding]::UTF8.GetBytes($remoteBash)
$b64 = [Convert]::ToBase64String($bytes)

$sshArgs = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=60",
    "-o", "IdentitiesOnly=yes",
    "-i", $SshIdentity,
    $JetstreamHost,
    "echo $b64 | base64 -d | bash"
)

Write-Host "Using image tag on VM: $LocalImageTag"
Write-Host "Remote: preflight disk -> git pull -> docker build -> IMAGE_REF switch -> health (+rollback)"
& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    if ($LASTEXITCODE -eq 42) {
        throw "Remote preflight failed: insufficient disk space for build (app was not restarted)."
    }
    throw "Remote script failed with exit code $LASTEXITCODE"
}

Write-Host "Done."

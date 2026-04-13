<#
.SYNOPSIS
  SSH to Jetstream: stop Panel app only, git pull, docker build on VM, set IMAGE_REF, start app (--pull never).

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
    [string] $LocalImageTag = ""
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

echo "=== stop app (db keeps running) ==="
cd "$DEPLOY"
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream stop app

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
echo "IMAGE_REF set to: ${IMAGE_TAG}"
echo "Backup written: ${DEPLOY}/.env.jetstream.bak.${ts}"

echo "=== compose up app (--pull never --force-recreate) ==="
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream up -d --pull never --force-recreate app

echo "=== docker compose ps ==="
docker compose -f docker-compose.jetstream.yml --env-file .env.jetstream -p sqllm-jetstream ps

echo "=== health ==="
curl -fsS --max-time 45 http://127.0.0.1:5006/health
echo ""
echo "OK: Panel /health"
'@
$remoteBash = $remoteBash.Replace("__REPO_DIR__", $RemoteRepoDir).
    Replace("__GIT_REMOTE__", $GitRemote).
    Replace("__GIT_REF__", $GitRef).
    Replace("__IMAGE_TAG__", $LocalImageTag)

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
Write-Host "Remote: stop app -> git pull -> docker build -> IMAGE_REF -> up --pull never"
& ssh @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "Remote script failed with exit code $LASTEXITCODE"
}

Write-Host "Done."

#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Restart the Panel app container on Jetstream over SSH (docker restart sqllm-app).

.DESCRIPTION
  Requires non-interactive SSH (BatchMode): use ssh-agent or an unencrypted key, or set
  JETSTREAM_SSH_KEY to an OpenSSH private key path.

  Environment (optional):
    JETSTREAM_SSH_HOST   - VM hostname or IPv4 (required unless -JetstreamHost)
    JETSTREAM_SSH_USER   - default: exouser
    JETSTREAM_SSH_KEY    - path to private key (optional; else tries ~/.ssh/id_ed25519, id_rsa)

.EXAMPLE
  $env:JETSTREAM_SSH_HOST = '149.165.153.232'
  ./scripts/restart_jetstream_panel_remote.ps1
#>
param(
    [string]$JetstreamHost = $env:JETSTREAM_SSH_HOST,
    [string]$SshUser = $(if ($env:JETSTREAM_SSH_USER) { $env:JETSTREAM_SSH_USER } else { "exouser" }),
    [string]$IdentityFile = $env:JETSTREAM_SSH_KEY
)

$ErrorActionPreference = "Stop"

if (-not $JetstreamHost) {
    Write-Error "Set JETSTREAM_SSH_HOST or pass -JetstreamHost (see deploy/jetstream/LOCAL_DEV_TUNNEL.md)."
}

$key = $IdentityFile
if (-not $key) {
    $candidates = @(
        (Join-Path $env:USERPROFILE ".ssh\id_ed25519"),
        (Join-Path $env:USERPROFILE ".ssh\id_rsa")
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $key = $candidates
}

$remote = @(
    "docker restart sqllm-app",
    "sleep 3",
    "curl -fsS http://127.0.0.1:5006/health"
) -join " && "

$ssh = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=25",
    "-o", "StrictHostKeyChecking=accept-new"
)
if ($key) {
    $ssh += "-i", $key
}
$ssh += "${SshUser}@${JetstreamHost}", $remote

Write-Host "ssh $($ssh[0..($ssh.Length-2)] -join ' ') ... $($ssh[-1])" -ForegroundColor DarkGray
& ssh @ssh
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
Write-Host "[OK] Panel container restarted; /health succeeded." -ForegroundColor Green

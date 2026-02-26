# Launcher for run_300_bigrun_sweep.py
# Auto-restarts up to $MaxAttempts times if the script fails mid-run.
# Subsequent restarts pass --force=false (default) so completed pkl files are skipped.

param(
    [int]$MaxAttempts = 5,
    [int]$RestartDelaySec = 30
)

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$Script    = Join-Path $RepoRoot "scripts\run_300_bigrun_sweep.py"
$LogFile   = Join-Path $RepoRoot "scripts\bigrun_300_launcher.log"
$EnvFile   = Join-Path $RepoRoot ".env"

# Load .env into the current process
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
        $name  = $Matches[1].Trim()
        $value = $Matches[2].Trim().Trim('"').Trim("'")
        [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
}

function Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

Log "=== 300-sample bigrun launcher starting (max $MaxAttempts attempts) ==="

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    Log "--- Attempt $attempt / $MaxAttempts ---"

    python $Script 2>&1 | Tee-Object -FilePath $LogFile -Append

    $exit = $LASTEXITCODE
    Log "Script exited with code $exit"

    if ($exit -eq 0) {
        Log "=== All runs completed successfully ==="
        exit 0
    }

    if ($attempt -lt $MaxAttempts) {
        Log "Script failed. Waiting $RestartDelaySec s before restart (completed runs will be skipped via pkl check)..."
        Start-Sleep -Seconds $RestartDelaySec
    }
}

Log "=== Launcher giving up after $MaxAttempts attempts ==="
exit 1

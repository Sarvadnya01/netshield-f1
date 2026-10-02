<#
.SYNOPSIS
    Restart the spark-stream container after a model swap.
.DESCRIPTION
    Restarts the spark-stream service. Clears streaming checkpoints
    only if --clear-checkpoints is specified (needed on schema changes).
.PARAMETER ClearCheckpoints
    Clear streaming checkpoints before restart.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\restart_stream.ps1
    powershell -ExecutionPolicy Bypass -File scripts\restart_stream.ps1 -ClearCheckpoints
#>
param(
    [switch]$ClearCheckpoints
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== Restart Spark Streaming ===" -ForegroundColor Cyan

if ($ClearCheckpoints) {
    $CkptDir = Join-Path $RepoRoot "checkpoints\streaming"
    if (Test-Path $CkptDir) {
        Write-Host "Clearing checkpoints at $CkptDir..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $CkptDir
    }
    Write-Host "Checkpoints cleared." -ForegroundColor Green
}

Push-Location $RepoRoot

Write-Host "Restarting spark-stream..." -ForegroundColor Yellow
docker compose --profile stream restart spark-stream

Write-Host "`nWaiting for spark-stream to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

$Status = docker compose --profile stream ps spark-stream --format json 2>$null | ConvertFrom-Json
if ($Status.State -eq "running") {
    Write-Host "spark-stream is running." -ForegroundColor Green
} else {
    Write-Host "WARNING: spark-stream may not be running. Check with: docker compose --profile stream ps" -ForegroundColor Red
}

Pop-Location

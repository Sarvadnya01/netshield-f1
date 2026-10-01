<#
.SYNOPSIS
    Run the federated learning experiment grid.
.DESCRIPTION
    Wrapper that activates the venv and runs the FL grid experiment.
    Uses NETSHIELD_PROFILE to select laptop or lab config.
.EXAMPLE
    .\scripts\run_fl_grid.ps1
    $env:NETSHIELD_PROFILE = "lab"; .\scripts\run_fl_grid.ps1
#>
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent

# Activate venv
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    & $VenvActivate
} else {
    Write-Error "Virtual environment not found at $VenvActivate"
    exit 1
}

$Profile = if ($env:NETSHIELD_PROFILE) { $env:NETSHIELD_PROFILE } else { "laptop" }
Write-Host "Running FL grid with profile: $Profile" -ForegroundColor Cyan

python -m netshield.federated.run_experiment

if ($LASTEXITCODE -ne 0) {
    Write-Error "FL grid experiment failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "FL grid complete." -ForegroundColor Green

<#
.SYNOPSIS
    Sets up the NetShield-FL Python environment on both laptop and lab machines.

.PARAMETER Profile
    Environment profile: "laptop" or "lab". Default: "laptop".

.PARAMETER TorchIndex
    PyTorch index URL for CUDA wheels. Default: CUDA 12.x index.

.PARAMETER Wheelhouse
    Path to a local wheelhouse directory for offline install.
#>
param(
    [ValidateSet("laptop", "lab")]
    [string]$Profile = "laptop",

    [string]$TorchIndex = "https://download.pytorch.org/whl/cu124",

    [string]$Wheelhouse = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== NetShield-FL Environment Setup ===" -ForegroundColor Cyan
Write-Host "Profile:    $Profile"
Write-Host "Repo root:  $RepoRoot"

# Create venv
$VenvDir = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
    py -3.11 -m venv $VenvDir
} else {
    Write-Host "`nVirtual environment already exists at $VenvDir" -ForegroundColor Green
}

# Activate
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
. $ActivateScript

Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch
Write-Host "`nInstalling PyTorch (CUDA)..." -ForegroundColor Yellow
if ($Wheelhouse -and (Test-Path $Wheelhouse)) {
    pip install torch torchvision --no-index --find-links $Wheelhouse
} else {
    pip install torch torchvision --index-url $TorchIndex
}

# Install requirements
Write-Host "`nInstalling ML requirements..." -ForegroundColor Yellow
if ($Wheelhouse -and (Test-Path $Wheelhouse)) {
    pip install --no-index --find-links $Wheelhouse -r (Join-Path $RepoRoot "requirements\ml.txt")
} else {
    pip install -r (Join-Path $RepoRoot "requirements\ml.txt")
}

Write-Host "`nInstalling App requirements..." -ForegroundColor Yellow
if ($Wheelhouse -and (Test-Path $Wheelhouse)) {
    pip install --no-index --find-links $Wheelhouse -r (Join-Path $RepoRoot "requirements\app.txt")
} else {
    pip install -r (Join-Path $RepoRoot "requirements\app.txt")
}

Write-Host "`nInstalling Dev requirements..." -ForegroundColor Yellow
if ($Wheelhouse -and (Test-Path $Wheelhouse)) {
    pip install --no-index --find-links $Wheelhouse -r (Join-Path $RepoRoot "requirements\dev.txt")
} else {
    pip install -r (Join-Path $RepoRoot "requirements\dev.txt")
}

# Install package in editable mode
Write-Host "`nInstalling netshield package (editable)..." -ForegroundColor Yellow
pip install -e $RepoRoot

# Set NETSHIELD_PROFILE for the user
Write-Host "`nSetting NETSHIELD_PROFILE=$Profile for current user..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable("NETSHIELD_PROFILE", $Profile, "User")
$env:NETSHIELD_PROFILE = $Profile

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Profile set to: $Profile"
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Verify with:   python scripts\verify_env.py"

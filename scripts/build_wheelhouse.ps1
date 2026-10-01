<#
.SYNOPSIS
    Download all Python wheels for offline installation on the lab PC.
.DESCRIPTION
    Uses pip download to fetch all ml/dev/app requirements + torch CUDA 12.x
    wheels for cp311 win_amd64 into transfer/to_lab/wheelhouse/.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build_wheelhouse.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Wheelhouse = Join-Path $RepoRoot "transfer\to_lab\wheelhouse"

Write-Host "=== Building Wheelhouse ===" -ForegroundColor Cyan
Write-Host "Destination: $Wheelhouse"

if (Test-Path $Wheelhouse) {
    Write-Host "Cleaning existing wheelhouse..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $Wheelhouse
}
New-Item -ItemType Directory -Path $Wheelhouse -Force | Out-Null

# Activate venv
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
} else {
    Write-Error "Virtual environment not found. Run setup_env.ps1 first."
    exit 1
}

$TorchIndex = "https://download.pytorch.org/whl/cu124"

# Download PyTorch CUDA wheels
Write-Host "`nDownloading PyTorch CUDA 12.x wheels..." -ForegroundColor Yellow
pip download torch torchvision `
    --index-url $TorchIndex `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --python-version 3.11 `
    --platform win_amd64

# Download ML requirements
Write-Host "`nDownloading ML requirements..." -ForegroundColor Yellow
pip download -r (Join-Path $RepoRoot "requirements\ml.txt") `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --python-version 3.11 `
    --platform win_amd64

# Download App requirements
Write-Host "`nDownloading App requirements..." -ForegroundColor Yellow
pip download -r (Join-Path $RepoRoot "requirements\app.txt") `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --python-version 3.11 `
    --platform win_amd64

# Download Dev requirements
Write-Host "`nDownloading Dev requirements..." -ForegroundColor Yellow
pip download -r (Join-Path $RepoRoot "requirements\dev.txt") `
    --dest $Wheelhouse `
    --only-binary=:all: `
    --python-version 3.11 `
    --platform win_amd64

# Print summary
$Files = Get-ChildItem $Wheelhouse
$TotalMB = ($Files | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n=== Wheelhouse Complete ===" -ForegroundColor Green
Write-Host "Files: $($Files.Count)"
Write-Host ("Total size: {0:N0} MB" -f $TotalMB)
Write-Host "Location: $Wheelhouse"

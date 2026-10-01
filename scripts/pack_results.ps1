<#
.SYNOPSIS
    Pack lab training results for transfer back to the laptop.
.DESCRIPTION
    Creates transfer/from_lab/results_<timestamp>.zip containing:
    - mlflow.db, mlruns/
    - reports/ (tables, figures, fl_runs)
    - models/serving/*.onnx, *.json (model cards, active.json)
    - logs/
    - RESULTS_MANIFEST.json
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\pack_results.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Host "=== Pack Results ===" -ForegroundColor Cyan

$OutDir = Join-Path $RepoRoot "transfer\from_lab"
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# Stage files into a temp directory
$Staging = Join-Path $OutDir "_staging_$Timestamp"
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

# mlflow.db
$MlflowDb = Join-Path $RepoRoot "mlflow.db"
if (Test-Path $MlflowDb) {
    Copy-Item $MlflowDb $Staging
    Write-Host "  mlflow.db" -ForegroundColor Green
}

# mlruns/
$MlrunsDir = Join-Path $RepoRoot "mlruns"
if (Test-Path $MlrunsDir) {
    Copy-Item -Path $MlrunsDir -Destination (Join-Path $Staging "mlruns") -Recurse
    Write-Host "  mlruns/" -ForegroundColor Green
}

# reports/
$ReportsDir = Join-Path $RepoRoot "reports"
if (Test-Path $ReportsDir) {
    Copy-Item -Path $ReportsDir -Destination (Join-Path $Staging "reports") -Recurse
    Write-Host "  reports/" -ForegroundColor Green
}

# models/serving/
$ServingDir = Join-Path $RepoRoot "models\serving"
if (Test-Path $ServingDir) {
    $DestServing = Join-Path $Staging "models\serving"
    New-Item -ItemType Directory -Path $DestServing -Force | Out-Null
    Get-ChildItem $ServingDir -File | Where-Object {
        $_.Extension -in ".onnx", ".json"
    } | ForEach-Object {
        Copy-Item $_.FullName $DestServing
        Write-Host "  models/serving/$($_.Name)" -ForegroundColor Green
    }
}

# logs/
$LogsDir = Join-Path $RepoRoot "logs"
if (Test-Path $LogsDir) {
    Copy-Item -Path $LogsDir -Destination (Join-Path $Staging "logs") -Recurse
    Write-Host "  logs/" -ForegroundColor Green
}

# Git hash
$GitHash = ""
try {
    Push-Location $RepoRoot
    $GitHash = (git rev-parse HEAD 2>$null).Trim()
    Pop-Location
} catch {
    $GitHash = "unknown"
}

# GPU name
$GpuName = "unknown"
try {
    $VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
    if (Test-Path $VenvActivate) {
        . $VenvActivate
        $GpuName = (python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>$null).Trim()
    }
} catch {}

# RESULTS_MANIFEST.json with checksums
$ManifestFiles = @()
Get-ChildItem $Staging -Recurse -File | ForEach-Object {
    $RelPath = $_.FullName.Substring($Staging.Length + 1).Replace("\", "/")
    $Hash = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    $ManifestFiles += @{
        path   = $RelPath
        size   = $_.Length
        sha256 = $Hash
    }
}

$ResultsManifest = @{
    git_hash   = $GitHash
    gpu_name   = $GpuName
    repo_root  = $RepoRoot
    created_at = (Get-Date -Format "o")
    profile    = if ($env:NETSHIELD_PROFILE) { $env:NETSHIELD_PROFILE } else { "unknown" }
    files      = $ManifestFiles
} | ConvertTo-Json -Depth 4

$ManifestPath = Join-Path $Staging "RESULTS_MANIFEST.json"
Set-Content -Path $ManifestPath -Value $ResultsManifest -Encoding UTF8

# Zip
$ZipPath = Join-Path $OutDir "results_$Timestamp.zip"
Compress-Archive -Path (Join-Path $Staging "*") -DestinationPath $ZipPath -Force

# Cleanup staging
Remove-Item -Recurse -Force $Staging

$ZipSize = (Get-Item $ZipPath).Length / 1MB
Write-Host "`n=== Results Packed ===" -ForegroundColor Green
Write-Host ("Zip: $ZipPath ({0:N1} MB)" -f $ZipSize)
Write-Host "Git hash: $GitHash"
Write-Host "GPU: $GpuName"
Write-Host "`nCopy this zip to the laptop and run unpack_results.ps1"

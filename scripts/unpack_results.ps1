<#
.SYNOPSIS
    Unpack lab results into the laptop repo.
.DESCRIPTION
    Verifies checksums and that the git hash matches a commit in this repo.
    Backs up existing reports/, models/serving/, mlflow.db.
    Extracts results into place. Rewrites MLflow paths if roots differ.
.PARAMETER ResultsZip
    Path to the results_<timestamp>.zip file.
.PARAMETER DryRun
    If set, only verify checksums and show what would be done.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\unpack_results.ps1 -ResultsZip transfer\from_lab\results_20261001.zip
    powershell -ExecutionPolicy Bypass -File scripts\unpack_results.ps1 -ResultsZip path\to\results.zip -DryRun
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ResultsZip,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== Unpack Lab Results ===" -ForegroundColor Cyan
Write-Host "Results zip: $ResultsZip"
Write-Host "Repo root:   $RepoRoot"
Write-Host "Mode:        $(if ($DryRun) { 'DRY RUN' } else { 'APPLY' })"

if (-not (Test-Path $ResultsZip)) {
    Write-Error "Results zip not found: $ResultsZip"
    exit 1
}

# Extract to temp dir
$TempDir = Join-Path $RepoRoot "transfer\_unpack_temp"
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
Expand-Archive -Path $ResultsZip -DestinationPath $TempDir

# Read manifest
$ManifestPath = Join-Path $TempDir "RESULTS_MANIFEST.json"
if (-not (Test-Path $ManifestPath)) {
    Write-Error "RESULTS_MANIFEST.json not found in zip"
    Remove-Item -Recurse -Force $TempDir
    exit 1
}
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
Write-Host "Lab git hash: $($Manifest.git_hash)"
Write-Host "Lab GPU:      $($Manifest.gpu_name)"
Write-Host "Lab root:     $($Manifest.repo_root)"

# Verify git hash exists in this repo
Write-Host "`n--- Verifying git hash ---" -ForegroundColor Yellow
Push-Location $RepoRoot
$HashExists = git cat-file -t $Manifest.git_hash 2>$null
Pop-Location
if ($HashExists -ne "commit") {
    Write-Host "WARNING: Git hash $($Manifest.git_hash) not found in this repo!" -ForegroundColor Red
    Write-Host "The lab may have been set up from a different commit." -ForegroundColor Red
    Write-Host "Continuing anyway..." -ForegroundColor Yellow
}
else {
    Write-Host "Git hash verified." -ForegroundColor Green
}

# Verify checksums
Write-Host "`n--- Verifying checksums ---" -ForegroundColor Yellow
$ChecksumOk = $true
foreach ($Entry in $Manifest.files) {
    $FilePath = Join-Path $TempDir $Entry.path
    if (-not (Test-Path $FilePath)) {
        Write-Host "  MISSING: $($Entry.path)" -ForegroundColor Red
        $ChecksumOk = $false
        continue
    }
    $Actual = (Get-FileHash $FilePath -Algorithm SHA256).Hash
    if ($Actual -ne $Entry.sha256) {
        Write-Host "  MISMATCH: $($Entry.path)" -ForegroundColor Red
        $ChecksumOk = $false
    }
}
if (-not $ChecksumOk) {
    Write-Error "Checksum verification failed!"
    Remove-Item -Recurse -Force $TempDir
    exit 1
}
Write-Host "All checksums verified ($($Manifest.files.Count) files)." -ForegroundColor Green

if ($DryRun) {
    Write-Host "`nDRY RUN: would extract $($Manifest.files.Count) files into $RepoRoot" -ForegroundColor Yellow
    Write-Host "  Would backup: reports/, models/serving/, mlflow.db" -ForegroundColor Yellow
    $LabRoot = $Manifest.repo_root.Replace("\", "/")
    $LocalRoot = $RepoRoot.Replace("\", "/")
    if ($LabRoot -ne $LocalRoot) {
        Write-Host "  Would rewrite MLflow paths: $LabRoot -> $LocalRoot" -ForegroundColor Yellow
    }
    Remove-Item -Recurse -Force $TempDir
    Write-Host "`nDry run complete." -ForegroundColor Green
    return
}

# Backup existing files
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupDir = Join-Path $RepoRoot "transfer\backup_$Timestamp"
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
Write-Host "`n--- Backing up existing files ---" -ForegroundColor Yellow

$ReportsDir = Join-Path $RepoRoot "reports"
if (Test-Path $ReportsDir) {
    Copy-Item -Path $ReportsDir -Destination (Join-Path $BackupDir "reports") -Recurse
    Write-Host "  Backed up reports/" -ForegroundColor Green
}

$ServingDir = Join-Path $RepoRoot "models\serving"
if (Test-Path $ServingDir) {
    $BackupServing = Join-Path $BackupDir "models\serving"
    New-Item -ItemType Directory -Path $BackupServing -Force | Out-Null
    Get-ChildItem $ServingDir -File | ForEach-Object { Copy-Item $_.FullName $BackupServing }
    Write-Host "  Backed up models/serving/" -ForegroundColor Green
}

$MlflowDb = Join-Path $RepoRoot "mlflow.db"
if (Test-Path $MlflowDb) {
    Copy-Item $MlflowDb (Join-Path $BackupDir "mlflow.db")
    Write-Host "  Backed up mlflow.db" -ForegroundColor Green
}

Write-Host "Backup: $BackupDir" -ForegroundColor Green

# Extract into repo
Write-Host "`n--- Extracting results ---" -ForegroundColor Yellow

# Copy each file from temp to repo
foreach ($Entry in $Manifest.files) {
    if ($Entry.path -eq "RESULTS_MANIFEST.json") { continue }
    $SrcPath = Join-Path $TempDir $Entry.path
    $DstPath = Join-Path $RepoRoot $Entry.path
    $DstDir = Split-Path $DstPath -Parent
    if (-not (Test-Path $DstDir)) {
        New-Item -ItemType Directory -Path $DstDir -Force | Out-Null
    }
    Copy-Item $SrcPath $DstPath -Force
}
Write-Host "Extracted $($Manifest.files.Count) files." -ForegroundColor Green

# Save manifest in reports
Copy-Item $ManifestPath (Join-Path $RepoRoot "reports\RESULTS_MANIFEST.json") -Force

# MLflow path rewrite if roots differ
$LabRoot = $Manifest.repo_root.Replace("\", "/")
$LocalRoot = $RepoRoot.Replace("\", "/")
$MlflowDb = Join-Path $RepoRoot "mlflow.db"

if (($LabRoot -ne $LocalRoot) -and (Test-Path $MlflowDb)) {
    Write-Host "`n--- Rewriting MLflow paths ---" -ForegroundColor Yellow
    Write-Host "  Lab root:   $LabRoot"
    Write-Host "  Local root: $LocalRoot"

    $VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
    . $VenvActivate
    python (Join-Path $RepoRoot "scripts\rewrite_mlflow_paths.py") `
        --db $MlflowDb --old-root $LabRoot --new-root $LocalRoot --apply
}

# Cleanup temp
Remove-Item -Recurse -Force $TempDir

Write-Host "`n=== Unpack Complete ===" -ForegroundColor Green
Write-Host "Results are in place. Review with:"
Write-Host "  - MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db"
Write-Host "  - Reports: reports/tables/, reports/fl_runs/"
Write-Host "  - Active model: models/serving/active.json"

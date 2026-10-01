<#
.SYNOPSIS
    Pack the repo, data, and serving files for transfer to the lab PC.
.DESCRIPTION
    Refuses if git tree is dirty. Creates transfer/to_lab/ with:
    - repo.bundle (git bundle --all)
    - data_bundle.zip (data/processed, data/stream, models/serving/feature_stats.json)
    - lab_setup.ps1 (copy)
    - MANIFEST.json (commit hash, file sizes, SHA-256 checksums)
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\pack_for_lab.ps1
#>
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== Pack for Lab ===" -ForegroundColor Cyan

# Check git status
Push-Location $RepoRoot
$Status = git status --porcelain
if ($Status) {
    Write-Error "Git tree is dirty. Commit or stash changes first.`n$Status"
    exit 1
}
$GitHash = (git rev-parse HEAD).Trim()
Write-Host "Git commit: $GitHash"
Pop-Location

$OutDir = Join-Path $RepoRoot "transfer\to_lab"
if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
}

# 1. Git bundle
Write-Host "`nCreating git bundle..." -ForegroundColor Yellow
$BundlePath = Join-Path $OutDir "repo.bundle"
Push-Location $RepoRoot
git bundle create $BundlePath --all
Pop-Location
Write-Host "Bundle: $BundlePath"

# 2. Data bundle
Write-Host "`nCreating data bundle..." -ForegroundColor Yellow
$DataZip = Join-Path $OutDir "data_bundle.zip"
if (Test-Path $DataZip) { Remove-Item $DataZip }

$TempStaging = Join-Path $OutDir "_staging"
if (Test-Path $TempStaging) { Remove-Item -Recurse -Force $TempStaging }
New-Item -ItemType Directory -Path $TempStaging -Force | Out-Null

# Copy processed data
$ProcessedDir = Join-Path $RepoRoot "data\processed"
if (Test-Path $ProcessedDir) {
    $DestProcessed = Join-Path $TempStaging "data\processed"
    Copy-Item -Path $ProcessedDir -Destination $DestProcessed -Recurse
}

# Copy stream data
$StreamDir = Join-Path $RepoRoot "data\stream"
if (Test-Path $StreamDir) {
    $DestStream = Join-Path $TempStaging "data\stream"
    Copy-Item -Path $StreamDir -Destination $DestStream -Recurse
}

# Copy feature_stats.json
$StatsFile = Join-Path $RepoRoot "models\serving\feature_stats.json"
if (Test-Path $StatsFile) {
    $DestStats = Join-Path $TempStaging "models\serving"
    New-Item -ItemType Directory -Path $DestStats -Force | Out-Null
    Copy-Item $StatsFile $DestStats
}

Compress-Archive -Path (Join-Path $TempStaging "*") -DestinationPath $DataZip -Force
Remove-Item -Recurse -Force $TempStaging
Write-Host "Data bundle: $DataZip"

# 3. Copy lab_setup.ps1
Write-Host "`nCopying lab_setup.ps1..." -ForegroundColor Yellow
$LabSetupSrc = Join-Path $RepoRoot "scripts\lab_setup.ps1"
$LabSetupDst = Join-Path $OutDir "lab_setup.ps1"
Copy-Item $LabSetupSrc $LabSetupDst
Write-Host "Lab setup script: $LabSetupDst"

# 4. MANIFEST.json
Write-Host "`nCreating MANIFEST.json..." -ForegroundColor Yellow
$ManifestFiles = @()
foreach ($File in (Get-ChildItem $OutDir -File | Where-Object { $_.Name -ne "MANIFEST.json" -and $_.Name -ne "_staging" })) {
    $Hash = (Get-FileHash $File.FullName -Algorithm SHA256).Hash
    $ManifestFiles += @{
        name     = $File.Name
        size     = $File.Length
        sha256   = $Hash
    }
}

$Manifest = @{
    git_hash    = $GitHash
    created_at  = (Get-Date -Format "o")
    repo_root   = $RepoRoot
    files       = $ManifestFiles
} | ConvertTo-Json -Depth 4

$ManifestPath = Join-Path $OutDir "MANIFEST.json"
Set-Content -Path $ManifestPath -Value $Manifest -Encoding UTF8
Write-Host "Manifest: $ManifestPath"

# Summary
$AllFiles = Get-ChildItem $OutDir -File
$TotalMB = ($AllFiles | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "`n=== Pack Complete ===" -ForegroundColor Green
Write-Host ("Total size: {0:N1} MB" -f $TotalMB)
Write-Host "Files:"
foreach ($F in $AllFiles) {
    Write-Host ("  {0,-30s} {1,10:N1} MB" -f $F.Name, ($F.Length / 1MB))
}
Write-Host "`nReminder: typical USB stick is 16-64 GB. Wheelhouse adds ~3 GB if included." -ForegroundColor Yellow

<#
.SYNOPSIS
    Set up the NetShield-FL repo on the lab PC from the transfer bundle.
.DESCRIPTION
    Standalone script that runs BEFORE the repo exists on the lab machine.
    Verifies checksums, clones from the git bundle, sets up the Python env,
    unpacks data, and runs environment verification.
.PARAMETER TransferDir
    Path to the transfer/to_lab directory (e.g. D:\transfer\to_lab).
.PARAMETER RepoRoot
    Where to clone the repo. Default: C:\dev\netshield-fl.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File lab_setup.ps1 -TransferDir D:\transfer\to_lab
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$TransferDir,

    [string]$RepoRoot = "C:\dev\netshield-fl"
)

$ErrorActionPreference = "Stop"

Write-Host "=== NetShield-FL Lab Setup ===" -ForegroundColor Cyan
Write-Host "Transfer dir: $TransferDir"
Write-Host "Repo root:    $RepoRoot"

# ---- 1. Verify MANIFEST checksums ----
Write-Host "`n--- Verifying checksums ---" -ForegroundColor Yellow
$ManifestPath = Join-Path $TransferDir "MANIFEST.json"
if (-not (Test-Path $ManifestPath)) {
    Write-Error "MANIFEST.json not found in $TransferDir"
    exit 1
}

$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json

foreach ($Entry in $Manifest.files) {
    $FilePath = Join-Path $TransferDir $Entry.name
    if (-not (Test-Path $FilePath)) {
        Write-Error "Missing file: $($Entry.name)"
        exit 1
    }
    $Actual = (Get-FileHash $FilePath -Algorithm SHA256).Hash
    if ($Actual -ne $Entry.sha256) {
        Write-Error "Checksum mismatch for $($Entry.name): expected $($Entry.sha256), got $Actual"
        exit 1
    }
    Write-Host "  OK: $($Entry.name)" -ForegroundColor Green
}
Write-Host "All checksums verified." -ForegroundColor Green

# ---- 2. Check Python 3.11 ----
Write-Host "`n--- Checking Python ---" -ForegroundColor Yellow
$PythonOk = $false
try {
    $PyVer = & py -3.11 --version 2>&1
    if ($PyVer -match "3\.11") {
        Write-Host "Found: $PyVer" -ForegroundColor Green
        $PythonOk = $true
    }
} catch {}

if (-not $PythonOk) {
    try {
        $PyVer = & python --version 2>&1
        if ($PyVer -match "3\.11") {
            Write-Host "Found: $PyVer (via python)" -ForegroundColor Green
            $PythonOk = $true
        }
    } catch {}
}

if (-not $PythonOk) {
    Write-Host @"

Python 3.11 not found. Install it per-user (no admin needed):
  1. Download from https://www.python.org/downloads/release/python-3119/
  2. Run the installer, CHECK 'Add Python to PATH', choose 'Install for current user'
  3. Reopen this terminal and rerun this script.

"@ -ForegroundColor Red
    exit 1
}

# ---- 3. Clone repo from bundle ----
Write-Host "`n--- Cloning repository ---" -ForegroundColor Yellow
$BundlePath = Join-Path $TransferDir "repo.bundle"
if (-not (Test-Path $BundlePath)) {
    Write-Error "repo.bundle not found in $TransferDir"
    exit 1
}

if (Test-Path $RepoRoot) {
    Write-Host "Repo directory already exists. Pulling from bundle..." -ForegroundColor Yellow
    Push-Location $RepoRoot
    git pull $BundlePath main
    Pop-Location
} else {
    $ParentDir = Split-Path $RepoRoot -Parent
    if (-not (Test-Path $ParentDir)) {
        New-Item -ItemType Directory -Path $ParentDir -Force | Out-Null
    }
    git clone $BundlePath $RepoRoot
    Push-Location $RepoRoot
    git checkout main
    Pop-Location
}
Write-Host "Repo ready at $RepoRoot" -ForegroundColor Green

# ---- 4. Set up Python environment ----
Write-Host "`n--- Setting up Python environment ---" -ForegroundColor Yellow
$SetupScript = Join-Path $RepoRoot "scripts\setup_env.ps1"
$WheelhouseDir = Join-Path $TransferDir "wheelhouse"
$SetupArgs = @("-Profile", "lab")
if (Test-Path $WheelhouseDir) {
    Write-Host "Wheelhouse found, using offline install." -ForegroundColor Green
    $SetupArgs += @("-Wheelhouse", $WheelhouseDir)
}

& powershell -ExecutionPolicy Bypass -File $SetupScript @SetupArgs

# ---- 5. Unpack data bundle ----
Write-Host "`n--- Unpacking data bundle ---" -ForegroundColor Yellow
$DataZip = Join-Path $TransferDir "data_bundle.zip"
if (Test-Path $DataZip) {
    Expand-Archive -Path $DataZip -DestinationPath $RepoRoot -Force
    Write-Host "Data unpacked into $RepoRoot" -ForegroundColor Green
} else {
    Write-Host "No data_bundle.zip found, skipping." -ForegroundColor Yellow
}

# ---- 6. Set profile ----
$env:NETSHIELD_PROFILE = "lab"
Write-Host "`nNETSHIELD_PROFILE set to 'lab' for this session." -ForegroundColor Green

# ---- 7. Verify environment ----
Write-Host "`n--- Verifying environment ---" -ForegroundColor Yellow
$VenvActivate = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
. $VenvActivate
python (Join-Path $RepoRoot "scripts\verify_env.py")

# ---- 8. Check for CUDA GPU ----
Write-Host "`n--- Checking CUDA GPU ---" -ForegroundColor Yellow
$CudaCheck = python -c "import torch; ok=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if ok else 'none'; print(f'{ok}|{name}')" 2>&1
$Parts = "$CudaCheck".Split("|")
if ($Parts[0] -ne "True") {
    Write-Error "No CUDA GPU detected. Cannot proceed with lab training."
    exit 1
}
Write-Host "GPU: $($Parts[1])" -ForegroundColor Green

Write-Host "`n=== Lab Setup Complete ===" -ForegroundColor Green
Write-Host "Next: run lab_train_all.ps1"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\lab_train_all.ps1"

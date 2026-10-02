<#
.SYNOPSIS
    Start the NetShield-FL demo (Kafka + API + Dashboard + Producer).
.PARAMETER ScoringMode
    "spark" (default) uses Spark Structured Streaming for scoring.
    "api" uses the API's built-in ONNX scorer (fallback if Spark misbehaves).
.PARAMETER Rate
    Producer events per second (default 200).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
    powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1 -ScoringMode api -Rate 100
#>
param(
    [ValidateSet("spark", "api")]
    [string]$ScoringMode = "spark",
    [int]$Rate = 200
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "`n=== NetShield-FL Demo Startup ===" -ForegroundColor Cyan
Write-Host "Scoring mode: $ScoringMode" -ForegroundColor Yellow
Write-Host "Producer rate: $Rate eps" -ForegroundColor Yellow

# ---- Pre-flight checks ----

# Docker running?
try {
    $null = docker info 2>&1
} catch {
    Write-Host "ERROR: Docker is not running. Start Docker Desktop first." -ForegroundColor Red
    exit 1
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker daemon not responding. Start Docker Desktop and wait." -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Docker is running" -ForegroundColor Green

# WSL memory config?
$WslConfig = Join-Path $env:USERPROFILE ".wslconfig"
if (Test-Path $WslConfig) {
    Write-Host "[OK] .wslconfig found" -ForegroundColor Green
} else {
    Write-Host "WARNING: No .wslconfig found at $WslConfig" -ForegroundColor Yellow
    Write-Host "  Copy docs\wslconfig.example to %UserProfile%\.wslconfig and restart WSL" -ForegroundColor Yellow
}

# active.json exists?
$ActiveJson = Join-Path $RepoRoot "models\serving\active.json"
if (-not (Test-Path $ActiveJson)) {
    Write-Host "ERROR: models\serving\active.json not found. Run smoke training first." -ForegroundColor Red
    exit 1
}

$Active = Get-Content $ActiveJson | ConvertFrom-Json
if ($Active.kind -eq "smoke") {
    Write-Host ""
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Yellow
    Write-Host "  WARNING: Active model is SMOKE (not production)" -ForegroundColor Yellow
    Write-Host "  Model: $($Active.model_file)" -ForegroundColor Yellow
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Yellow
    Write-Host ""
}
Write-Host "[OK] Active model: $($Active.model_file) (kind=$($Active.kind))" -ForegroundColor Green

# ---- Start services ----
Push-Location $RepoRoot

if ($ScoringMode -eq "api") {
    # API scoring mode: skip spark-stream
    Write-Host "`nStarting Kafka + API (scoring) + Dashboard..." -ForegroundColor Yellow
    $env:SCORING_MODE = "api"
    docker compose up -d kafka kafka-init
    # Wait for kafka
    Write-Host "Waiting for Kafka..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
    docker compose --profile app up -d --build
} else {
    # Normal: Spark does scoring
    Write-Host "`nStarting Kafka + Spark Streaming + API + Dashboard..." -ForegroundColor Yellow
    docker compose --profile stream --profile app up -d --build
}

# ---- Wait for readiness ----
Write-Host "`nWaiting for services..." -ForegroundColor Yellow

# Wait for API /health
$ApiReady = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        $R = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction Stop
        if ($R.status -eq "ok") {
            Write-Host "  API ready ($($R.model_version))" -ForegroundColor Green
            $ApiReady = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 3
}
if (-not $ApiReady) {
    Write-Host "  WARNING: API not responding yet. Check: docker compose --profile app logs api" -ForegroundColor Red
}

# Check dashboard
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "  Dashboard ready" -ForegroundColor Green
} catch {
    Write-Host "  WARNING: Dashboard not responding yet" -ForegroundColor Yellow
}

# ---- Start producer ----
$StreamData = Join-Path $RepoRoot "data\stream\stream.parquet"
if (Test-Path $StreamData) {
    Write-Host "`nStarting producer (scenario, $Rate eps) in new window..." -ForegroundColor Yellow
    $Cmd = "cd `"$RepoRoot`"; .\.venv\Scripts\Activate.ps1; python -m netshield.streaming.producer --mode scenario --rate $Rate"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $Cmd
} else {
    Write-Host "`nWARNING: stream.parquet not found - skipping producer" -ForegroundColor Yellow
    Write-Host "  Run ETL first: docker compose --profile etl up" -ForegroundColor Yellow
}

# Open dashboard
Start-Process "http://localhost:8501"

Write-Host "`n=== Demo is running ===" -ForegroundColor Green
Write-Host "  Dashboard:  http://localhost:8501" -ForegroundColor Cyan
Write-Host "  API docs:   http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  Stop:       powershell -ExecutionPolicy Bypass -File scripts\stop_demo.ps1" -ForegroundColor Yellow

Pop-Location

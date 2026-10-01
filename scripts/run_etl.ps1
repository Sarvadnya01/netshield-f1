<#
.SYNOPSIS
    Run the Spark ETL pipeline inside the spark-etl Docker container.

.DESCRIPTION
    Wraps docker compose to run the ETL. Pass any arguments through to run_etl.py.
    Example: .\scripts\run_etl.ps1 --max-files 5
             .\scripts\run_etl.ps1 --force ingest_clean
#>

$ErrorActionPreference = "Stop"

# Check free disk space
$drive = Get-PSDrive -Name (Split-Path -Qualifier $PSScriptRoot).TrimEnd(":")
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Write-Host "Free disk space: ${freeGB} GB" -ForegroundColor Cyan
if ($freeGB -lt 50) {
    Write-Host "WARNING: Less than 50 GB free disk space. ETL may need more!" -ForegroundColor Yellow
    Write-Host "Consider freeing space before running the full ETL." -ForegroundColor Yellow
}

Write-Host "Starting Spark ETL..." -ForegroundColor Green

# Pass all script arguments through to the ETL
docker compose --profile etl run --rm spark-etl python -m netshield.etl.run_etl @args

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host "`nETL completed successfully." -ForegroundColor Green
} else {
    Write-Host "`nETL failed with exit code $exitCode." -ForegroundColor Red
    exit $exitCode
}

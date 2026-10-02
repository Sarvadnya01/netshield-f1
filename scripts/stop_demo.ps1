<#
.SYNOPSIS
    Stop all NetShield-FL demo services and the producer.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== Stopping NetShield-FL Demo ===" -ForegroundColor Cyan

Push-Location $RepoRoot

# Stop producer (find Python processes running the producer module)
$Producers = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*netshield.streaming.producer*" }
if ($Producers) {
    Write-Host "Stopping producer process(es)..." -ForegroundColor Yellow
    $Producers | Stop-Process -Force
}

# Stop Docker services
Write-Host "Stopping Docker services..." -ForegroundColor Yellow
docker compose --profile stream --profile app --profile tools down 2>$null

# Clear scoring mode env
Remove-Item Env:\SCORING_MODE -ErrorAction SilentlyContinue

Write-Host "`nAll services stopped." -ForegroundColor Green
Write-Host "Kafka data is preserved. To fully clean: docker compose down -v" -ForegroundColor Gray

Pop-Location

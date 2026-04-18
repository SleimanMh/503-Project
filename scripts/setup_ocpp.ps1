# setup_ocpp.ps1
# One-shot script: build gateway, start all containers, open dashboard,
# then run the OCPP test simulator so you can see a live session immediately.
#
# Usage (single vehicle):
#   .\scripts\setup_ocpp.ps1 -NoBuild -SessionMinutes 10
#
# Usage (fleet with realistic random parameters):
#   .\scripts\setup_ocpp.ps1 -NoBuild -Fleet 4 -Realistic -SpeedFactor 30
#
# Usage (fleet with fixed stay duration, staggered arrivals):
#   .\scripts\setup_ocpp.ps1 -NoBuild -Fleet 3 -SessionMinutes 30 -SpreadMinutes 5
#
param(
    [int]   $SessionMinutes = 5,
    [int]   $Fleet          = 1,        # number of concurrent charge points
    [float] $SpreadMinutes  = 2,        # max arrival spread across fleet (minutes)
    [float] $SpeedFactor    = 1.0,      # time compression (60 = 1 real sec = 1 sim min)
    [int]   $MeterInterval  = 30,
    [switch]$Realistic,                 # sample random ACN-style parameters per vehicle
    [switch]$NoBuild                    # skip docker build
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot   # project root (parent of scripts/)

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "  EV Charging - OCPP Setup and Live Demo"       -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# -- 1. Check Docker is running
Write-Host "[1/5] Checking Docker..." -ForegroundColor Yellow
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }
    Write-Host "      Docker OK" -ForegroundColor Green
} catch {
    Write-Error "Docker is not running. Please start Docker Desktop first."
    exit 1
}

# -- 2. Build gateway (only changed container)
if (-not $NoBuild) {
    Write-Host "[2/5] Building gateway container (this takes ~4 min on first run)..." -ForegroundColor Yellow
    Push-Location $Root
    docker compose build --no-cache gateway
    if ($LASTEXITCODE -ne 0) { Write-Error "docker compose build failed"; exit 1 }
    Pop-Location
    Write-Host "      Build complete" -ForegroundColor Green
} else {
    Write-Host "[2/5] Skipping build (-NoBuild flag)" -ForegroundColor DarkGray
}

# -- 3. Start / restart containers
Write-Host "[3/5] Starting containers..." -ForegroundColor Yellow
Push-Location $Root
docker compose up -d --force-recreate gateway ml-service optimizer
if ($LASTEXITCODE -ne 0) { Write-Error "docker compose up failed"; exit 1 }
Pop-Location
Write-Host "      Containers started" -ForegroundColor Green

# -- 4. Wait for gateway to be healthy
Write-Host "[4/5] Waiting for gateway health check..." -ForegroundColor Yellow
$maxWait = 30
$waited  = 0
do {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $r = Invoke-RestMethod "http://localhost:8000/health" -TimeoutSec 2
        $healthy = $true
    } catch {
        $healthy = $false
    }
} while (-not $healthy -and $waited -lt $maxWait)

if (-not $healthy) {
    Write-Warning "Gateway did not become healthy in ${maxWait}s - proceeding anyway"
} else {
    Write-Host "      Gateway healthy after ${waited}s" -ForegroundColor Green
}

# -- 5. Open dashboard + run test session
Write-Host "[5/5] Opening dashboard and launching OCPP session..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Dashboard (main):        http://localhost:8501"                -ForegroundColor White
Write-Host "  OCPP Live Monitor page:  http://localhost:8501/OCPP_Monitor"   -ForegroundColor White
Write-Host "  REST API docs:           http://localhost:8000/docs"           -ForegroundColor White
Write-Host ""

# Open browser at the OCPP monitor page
Start-Process "http://localhost:8501/OCPP_Monitor"

# Short pause so the browser tab opens before the terminal fills with logs
Start-Sleep -Seconds 2

$Venv   = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "scripts\test_ocpp.py"

# Build simulator argument list
$SimArgs = @(
    $Script,
    "--url", "ws://localhost:8000",
    "--fleet", $Fleet,
    "--session-minutes", $SessionMinutes,
    "--spread-minutes", $SpreadMinutes,
    "--meter-interval", $MeterInterval,
    "--speed-factor", $SpeedFactor
)
if ($Realistic) { $SimArgs += "--realistic" }

if ($Realistic) {
    Write-Host "  Running fleet: $Fleet vehicle(s)  realistic mode  speed=${SpeedFactor}x" -ForegroundColor Cyan
} else {
    Write-Host "  Running fleet: $Fleet vehicle(s)  stay=${SessionMinutes} min  spread=${SpreadMinutes} min  speed=${SpeedFactor}x" -ForegroundColor Cyan
}
Write-Host "  Refresh the dashboard page to see the live sessions." -ForegroundColor Cyan
Write-Host ""

& $Venv @SimArgs

Write-Host ""
Write-Host "Done. Check the dashboard for the session summary." -ForegroundColor Green

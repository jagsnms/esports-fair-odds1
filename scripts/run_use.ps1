# Use Mode: build frontend and run backend serving it from /
# Run from anywhere; script cds to repo root. No reload.

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$NpmCmd = "C:\Program Files\nodejs\npm.cmd"
if (-not (Test-Path $NpmCmd)) {
    $NpmCmd = "npm.cmd"
}

# Build frontend
Push-Location (Join-Path $RepoRoot "frontend")
try {
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing frontend dependencies..."
        & $NpmCmd install
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
    Write-Host "Building frontend..."
    & $NpmCmd run build
    if ($LASTEXITCODE -ne 0) { exit 1 }
} finally {
    Pop-Location
}

# Start backend (no reload) on port 8000
$Python = Join-Path $RepoRoot ".venv311\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: .venv311 not found at repo root. Run .\scripts\bootstrap.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Use Mode running at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host ""

& $Python -m uvicorn backend.main:app --port 8000 --log-level info

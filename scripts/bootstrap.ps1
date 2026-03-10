# Bootstrap script for Windows: create venv, compile deps, install.
# Run from repo root: .\scripts\bootstrap.ps1

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location $RepoRoot

# Prefer Python 3.11
$Py311 = $null
try {
    $out = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { $Py311 = $out.Trim() }
} catch {}
if (-not $Py311) {
    try {
        $out = & python3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $Py311 = $out.Trim() }
    } catch {}
}
if (-not $Py311 -or -not (Test-Path $Py311)) {
    Write-Host "ERROR: Python 3.11 not found. Install Python 3.11 or add it to PATH (e.g. py -3.11)." -ForegroundColor Red
    exit 1
}

$VenvDir = ".venv311"
$VenvPath = Join-Path $RepoRoot $VenvDir
if (Test-Path $VenvPath) {
    Write-Host "Using existing $VenvDir at $VenvPath"
} else {
    Write-Host "Creating venv at $VenvDir (Python: $Py311)"
    & $Py311 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "ERROR: $VenvDir activation script not found at $Activate" -ForegroundColor Red
    exit 1
}
& $Activate

python -m pip install --upgrade pip
pip install pip-tools

$ReqIn = Join-Path $RepoRoot "requirements.in"
$ReqTxt = Join-Path $RepoRoot "requirements.txt"
$ReqDev = Join-Path $RepoRoot "requirements-dev.txt"
if (-not (Test-Path $ReqIn)) {
    Write-Host "ERROR: requirements.in not found at $ReqIn" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $ReqDev)) {
    Write-Host "ERROR: requirements-dev.txt not found at $ReqDev" -ForegroundColor Red
    exit 1
}
pip-compile $ReqIn -o $ReqTxt
pip install -r $ReqDev

try {
    python -m playwright install
    if ($LASTEXITCODE -ne 0) { throw "playwright install exited non-zero" }
} catch {
    Write-Host "WARNING: playwright install failed (some environments may not need it)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Run the app with: streamlit run app35_ml.py" -ForegroundColor Cyan

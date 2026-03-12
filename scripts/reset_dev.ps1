# Full dev reset: kill listeners on common ports, optionally clean runtime artifacts, optionally call backend reset.
# Run from anywhere; script cds to repo root. Safe: does not delete data/raw, data/processed, artifacts/reports, or the active BO3 corpus because that corpus now defaults outside the repo worktree.
# It also leaves any legacy in-worktree logs\bo3_backend_live_capture_contract.jsonl copy alone so older local copies are not silently destroyed.

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot

$Ports = @(8000, 5173, 3000, 4173)
$LegacyBo3WorktreePath = Join-Path $RepoRoot "logs\bo3_backend_live_capture_contract.jsonl"

# ---- A) Kill listeners on ports ----
function Get-PidsListeningOnPort($port) {
    $pids = @()
    try {
        $lines = netstat -ano | Select-String "LISTENING"
        foreach ($line in $lines) {
            if ($line.Line -match ":$port\s+") {
                if ($line.Line -match "LISTENING\s+(\d+)\s*$") {
                    $pids += [int]$matches[1]
                }
            }
        }
    } catch {}
    $pids | Sort-Object -Unique
}

$Killed = @()
foreach ($port in $Ports) {
    $pids = Get-PidsListeningOnPort -port $port
    foreach ($pid in $pids) {
        try {
            Write-Host "Killing PID $pid (port $port)..."
            taskkill /PID $pid /T /F 2>$null
            if ($LASTEXITCODE -eq 0) { $Killed += "port $port (PID $pid)" }
        } catch {}
    }
}
if ($Killed.Count -gt 0) {
    Write-Host "Killed: $($Killed -join ', ')" -ForegroundColor Yellow
} else {
    Write-Host "No listeners found on ports $($Ports -join ', ')." -ForegroundColor Gray
}

# ---- B) Optional runtime cleanup ----
$ToRemove = @(
    (Join-Path $RepoRoot "out"),
    (Join-Path $RepoRoot "logs\runtime\*"),
    (Join-Path $RepoRoot "logs\debug\*"),
    (Join-Path $RepoRoot "logs\*.log"),
    (Join-Path $RepoRoot "logs\*.json"),
    (Join-Path $RepoRoot "logs\*.jsonl"),
    (Join-Path $RepoRoot "pytest_fit_auc_data"),
    (Join-Path $RepoRoot "pytest_fit_midround_data")
)
$Deleted = @()
foreach ($item in $ToRemove) {
    if ($item -match "\*$") {
        $parent = Split-Path $item -Parent
        if (Test-Path $parent) {
            Get-ChildItem $item -ErrorAction SilentlyContinue | ForEach-Object {
                if ($_.FullName -eq $LegacyBo3WorktreePath) {
                    return
                }
                Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $Deleted += $_.FullName
            }
        }
    } elseif (Test-Path $item) {
        Remove-Item $item -Recurse -Force -ErrorAction SilentlyContinue
        $Deleted += $item
    }
}
if ($Deleted.Count -gt 0) {
    Write-Host "Deleted: $($Deleted.Count) path(s) (out/, logs/runtime/*, logs/debug/*, logs/*.log|*.json|*.jsonl except any legacy in-worktree BO3 corpus copy, pytest_fit_*)" -ForegroundColor Yellow
} else {
    Write-Host "No runtime artifacts to delete." -ForegroundColor Gray
}

# ---- C) Optional: call backend reset endpoint ----
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/v1/debug/reset" -Method POST -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
    if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
        Write-Host "Backend reset endpoint called." -ForegroundColor Green
    }
} catch {
    # Ignore (backend may not be running)
}

Write-Host "reset_dev.ps1 done." -ForegroundColor Green

# Esports Fair Odds

## Setup

1. Install dependencies:
   - Linux/macOS:
     ```bash
     ./scripts/bootstrap.sh
     ```
   - Windows PowerShell:
     ```powershell
     .\scripts\bootstrap.ps1
     ```
2. Run the legacy Streamlit app (dev/analysis only):
   ```bash
   streamlit run legacy/app/app35_ml.py
   ```

## Dev (migration branch)

Run both from the repo root in separate terminals:

- **Terminal 1** (backend):
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\run_backend_dev.ps1
  ```
- **Terminal 2** (frontend):
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\run_frontend_dev.ps1
  ```

URLs: backend **http://127.0.0.1:8000/health** and frontend **http://localhost:5173/**. Backend streams live state and history over WebSocket; frontend shows p_hat and bounds.

## Use Mode (single process)

Build frontend and run backend serving it at `/` (no separate Vite).

- **Double-click `run_use.cmd`** at repo root to start (no PowerShell needed).
- Or from PowerShell: `.\scripts\run_use.ps1`

Use Mode running at **http://127.0.0.1:8000**.

## Full dev reset

When ports or artifacts are stuck, reset listeners and optional runtime cleanup.

- **Double-click `reset_dev.cmd`** at repo root to hard reset (no PowerShell needed).
- Or from PowerShell: `.\scripts\reset_dev.ps1`

Kills listeners on ports 8000, 5173, 3000, 4173; optionally deletes `out/`, `logs/runtime/*`, `logs/debug/*`, `logs/*.log`, `logs/*.json`, `logs/*.jsonl`, `pytest_fit_auc_data/`, `pytest_fit_midround_data/`. Does not delete `data/raw`, `data/processed`, or `artifacts/reports`. Optionally calls POST `/api/v1/debug/reset` if the backend is running.

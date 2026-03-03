# Esports Fair Odds

## Setup

1. Run the bootstrap script (Windows PowerShell):
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

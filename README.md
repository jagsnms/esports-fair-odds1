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

## Use Mode (single process)

Build frontend and run backend serving it at `/` (no separate Vite).

- **Double-click `run_use.cmd`** at repo root to start (no PowerShell needed).
- Or from PowerShell: `.\scripts\run_use.ps1`

Use Mode running at **http://127.0.0.1:8000**.

## Full dev reset

When ports or artifacts are stuck, reset listeners and optional runtime cleanup.

- **Double-click `reset_dev.cmd`** at repo root to hard reset (no PowerShell needed).
- Or from PowerShell: `.\scripts\reset_dev.ps1`

Kills listeners on ports 8000, 5173, 3000, 4173; optionally deletes `out/`, `logs/runtime/*`, `logs/debug/*`, `logs/*.log`, `logs/*.json`, `logs/*.jsonl` except any legacy in-worktree `logs/bo3_backend_live_capture_contract.jsonl` copy, `pytest_fit_auc_data/`, `pytest_fit_midround_data/`. Does not delete `data/raw`, `data/processed`, or `artifacts/reports`. Optionally calls POST `/api/v1/debug/reset` if the backend is running.

Backend BO3 capture now accumulates first in a continuity-protected local store outside ordinary repo worktree hazards by default: `%LOCALAPPDATA%\EsportsFairOdds\corpus\bo3_backend_live_capture_contract.jsonl` (or `BO3_BACKEND_CAPTURE_PATH` if you explicitly override it). That active corpus is the thing being grown over time and normal git/worktree operations must not silently replace it with an older repo baseline.

Frozen evidence snapshots remain separate optional cuts under `automation/reports/`; they support bounded review or diagnostics, but they do not replace the active corpus. The old in-worktree `logs/bo3_backend_live_capture_contract.jsonl` path is no longer the continuity-protected active store.

If you collected rows before the external-path continuity fix landed, run this one-time alignment step before resuming normal collection:

```powershell
.\.venv311\Scripts\python.exe .\tools\align_backend_bo3_active_corpus.py
```

That command aligns any existing `logs/bo3_backend_live_capture_contract.jsonl` rows into the external active corpus path without blind overwrite or blind duplicate append. After one successful alignment, steady-state normal collection and the corpus-readiness analyzer should both use the external active corpus path.

If you need to recover a stronger divergent BO3 corpus branch, use the special recovery workflow instead of the continuity aligner:

```powershell
.\.venv311\Scripts\python.exe .\tools\recover_backend_bo3_divergent_corpus.py --source-b-stash-ref "stash@{N}" --dry-run
```

Only rerun that command with `--write-output` if the report says `safe_union_dry_run_only`, shows `conflict_row_count = 0`, and the output path is the active external corpus. This workflow unions unique rows, dedupes identical duplicate rows, and refuses same-identity content conflicts. It is a one-off recovery path only. It is not normal steady-state collection.




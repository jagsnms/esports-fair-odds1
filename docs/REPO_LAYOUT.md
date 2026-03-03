## Repository layout

- `engine/`: Core model types and pure compute modules (probability engine, rails, midround, diagnostics).
- `backend/`: FastAPI backend service, runner loop, in‑memory store, WebSocket broadcaster, and API routes.
- `frontend/`: React + Vite UI that connects to the backend over HTTP and WebSocket.
- `legacy/`: Historical Streamlit monolith (`legacy/app/app35_ml.py`) and its helpers (`legacy/fair_odds/`, old scrapers, archived ML scripts).
- `tools/`: Command‑line utilities and one‑off helpers for scraping, calibration, diagnostics, etc. (`tools/scrapers/` for scrapers).
- `config/`: Versioned configuration files (e.g. calibration JSON, in‑play strategy configs).
- `data/`:
  - `data/raw/`: Raw inputs and snapshots pulled from external systems.
  - `data/raw/bo3/`: Large BO3 API dumps and full‑pull snapshots.
  - `data/processed/`: Cleaned / feature‑ready parquet files and other derived datasets.
- `artifacts/`:
  - `artifacts/reports/`: Backtest outputs, calibration reports, and other analysis artifacts.
- `logs/`:
  - `logs/debug/`: Ad‑hoc debug logs.
  - `logs/runtime/`: Runtime logs for services and scripts (ignored by git).
  - Other structured logs that are still part of the engine/tooling contracts.
- `docs/`: Project documentation and reference material (including `docs/grid/` for GRID telemetry docs).

**Rules of thumb**

- Put **raw pulls** and big JSON/parquet dumps under `data/raw/` (never at repo root or directly under `logs/`).
- Put **cleaned / feature tables** under `data/processed/`.
- Put **backtests, calibration reports, and analysis outputs** under `artifacts/reports/`.
- Do **not** add one‑off scripts at the repo root; put them under `tools/` (or a subfolder like `tools/scrapers/`).


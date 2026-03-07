# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Esports Fair Odds is a CS2 live probability/trading terminal. The architecture is:
- **Backend** (`backend/`): FastAPI + Uvicorn, in-memory state, polls live feeds or replays
- **Engine** (`engine/`): Pure computation pipeline (no I/O), unit-testable
- **Frontend** (`frontend/`): React + Vite + TradingView Lightweight Charts

No databases or external infrastructure required — the backend uses an in-memory store.

### Running services

See `README.md` for standard commands. On Linux (cloud agent), use these directly instead of the PowerShell scripts referenced in the README:

- **Backend**: `python3 -m uvicorn backend.main:app --reload --port 8000 --reload-dir backend --reload-dir engine`
- **Frontend**: `cd frontend && npm run dev`
- **Health check**: `curl http://localhost:8000/health` → `{"ok": true}`
- Frontend proxies `/api/*` and `/health` to the backend via Vite config.

### Testing

- `python3 -m pytest tests/ -v` — runs all unit tests (pytest, asyncio_mode=auto per `pytest.ini`)
- `npx tsc -b --noEmit` in `frontend/` — TypeScript type check (no dedicated ESLint config)
- `npm run build` in `frontend/` — full production build (tsc + vite build)

### Gotchas

- Python packages install to `~/.local/bin` (user install). Ensure `$HOME/.local/bin` is on PATH.
- The README references PowerShell scripts (`scripts/*.ps1`); those are Windows-only. Use the direct commands above.
- Replay JSONL files in `logs/` may be test fixtures with minimal data; full replays require real match recordings.
- External API keys (`GRID_API_KEY` in `.env`) are optional — the app runs without them (BO3 source polls publicly, GRID requires a key).

# AGENTS.md

## Engine Bible — mandatory reference for all code changes

**Every code change MUST reference `docs/ENGINE_BIBLE_WITH_TOC_CH1-12_TOC_FIXED.md` (the "Engine Bible").** Read it before making any change. Each change must show a direct correlation to the Bible — either as a prerequisite for a Bible-described behavior, or a direct fix to align the codebase with the Bible's specification.

The Engine Bible defines the finished-product vision across 12 chapters:
- **Part I (Ch 1–3):** Core model architecture — p_hat, rails, q, the identity `p_hat = rail_low + q * (rail_high - rail_low)`, rail construction from carryover signals only.
- **Part II (Ch 4–5):** Intra-round mechanics — p_hat movement/confidence curves, round resolution signals (combat, objective, timer pressure).
- **Part III (Ch 6–7):** Engine operation — the 9-step canonical calculation flow (round state → combat → objective → timer → q → rails → target p_hat → movement → output), structural and behavioral invariants.
- **Part IV (Ch 8–9):** Calibration philosophy and testing framework.
- **Part V (Ch 10):** AI implementation contract for Cursor automations.
- **Part VI (Ch 11–12):** Automation governance — execution policy and rule block.

**Workflow:** Read the Bible → identify the gap between current code and Bible spec → implement in small, structured, testable steps → cite the relevant Bible chapter/section in commit messages or PR descriptions.

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

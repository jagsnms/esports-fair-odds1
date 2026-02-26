# Migration: Streamlit Monolith → FastAPI Backend + React Frontend

## End-goal architecture

- **Backend:** Long-lived Python service (FastAPI) that continuously polls live feeds (BO3, later GRID), normalizes raw payloads, maintains authoritative state, computes derived outputs (p_hat, rails/envelopes, state bounds, kappa/bands, midround mixtures, diagnostics), and emits append-only history points for charting.
- **Frontend:** Separate web UI (React + Vite + TypeScript) using TradingView Lightweight Charts for zoom/pan and a trading-terminal feel. The UI subscribes via WebSocket for incremental updates and does **not** recompute the engine.

## Core principles

1. **Backend is source of truth** — Authoritative State and Config live in the backend. The UI only sends config/control commands.
2. **Pure functions** — Ingest/normalize, state reducer, and compute modules are pure and unit-testable (no I/O, no UI dependencies).
3. **Append-only history / event log** — History points are produced by the backend tick loop; replay is possible for debugging.
4. **Explicit boundaries:**
   - `raw feed → normalize → Frame`
   - `Frame + State + Config → reducer → State'`
   - `State' + Config → compute → Derived`
   - `Derived + State' → HistoryPoint`
   - Publish via REST + WebSocket
5. **Identity and locks** — Team A/B mapping and other user overrides must not be overwritten by feed updates when locked. This is handled centrally in the reducer/identity layer.
6. **Troubleshooting-first** — Smaller modules, strong invariant checks, easy replay.

## Module boundaries

| Layer        | Location        | Responsibility |
|-------------|-----------------|-----------------|
| Models      | `engine/models.py` | Config, Frame, State, Derived, HistoryPoint |
| Config      | `engine/config.py` | Defaults, safe merge of partial config updates |
| Normalize   | `engine/normalize/` | Raw feed → Frame (to be extracted from app35_ml) |
| State       | `engine/state/`    | Reducer: Frame + State + Config → State' |
| Compute     | `engine/compute/`  | State' + Config → Derived |
| Diagnostics | `engine/diagnostics/` | Invariant checks (prob range, monotonic time, team identity) |
| Replay      | `engine/replay/`   | Replay event log for debugging (future) |
| Backend     | `backend/`         | FastAPI app, runner loop, store, WebSocket broadcaster, API routes |
| Frontend    | `frontend/`        | React + Vite + Lightweight Charts, WS client, panels |

## Milestone plan

1. **Milestone 1** — Define engine models, config, and invariants (no behavior changes). ✅ Scaffolding in place.
2. **Milestone 2** — Implement normalize + reducer + compute pipeline with tests; runnable via scripts (no FastAPI required to exercise).
3. **Milestone 3** — FastAPI backend with tick loop, REST + WebSocket, in-memory store; optional Redis later.
4. **Milestone 4** — React frontend with TradingView Lightweight Charts consuming WebSocket updates.
5. **Milestone 5** — Replay tooling and deterministic debugging logs.

## Replay and debugging

- The backend emits append-only history points. Logging these (e.g. to a file or ring buffer) allows deterministic replay.
- Invariant checks in `engine/diagnostics/invariants.py` validate prob ranges, monotonic time, and team identity consistency before publish.

## Current scaffold (no app35_ml changes)

- **Engine:** `engine/models.py`, `engine/config.py`, `engine/diagnostics/invariants.py` — minimal stubs.
- **Backend:** `backend/main.py` (FastAPI, `/health`, `/api/v1/state/current`, `/api/v1/stream` WebSocket stub), `backend/api/routes_state.py`, `backend/api/ws.py`.
- **Frontend:** Vite React TS app under `frontend/` with `lightweight-charts`, placeholder page that connects to the stream WebSocket and renders a dummy line series.

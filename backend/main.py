"""
FastAPI application entrypoint.

Long-lived service: polls feeds, maintains state, computes derived outputs,
emits history points; exposes REST and WebSocket for the frontend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_state import router as state_router
from backend.api.ws import router as ws_router

app = FastAPI(
    title="ESports Fair Odds API",
    description="Backend for live trading/probability terminal",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(state_router, prefix="/api/v1", tags=["state"])
app.include_router(ws_router, prefix="/api/v1", tags=["stream"])


@app.get("/health")
def health() -> dict[str, bool]:
    """Liveness/readiness check."""
    return {"ok": True}

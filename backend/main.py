"""
FastAPI application entrypoint.

Long-lived service: polls feeds, maintains state, computes derived outputs,
emits history points; exposes REST and WebSocket for the frontend.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes_bo3 import router as bo3_router
from backend.api.routes_debug import router as debug_router
from backend.api.routes_market import router as market_router
from backend.api.routes_prematch import router as prematch_router
from backend.api.routes_replay import router as replay_router
from backend.api.routes_state import config_router, router as state_router
from backend.api.ws import router as ws_router
from backend.deps import set_broadcaster, set_runner, set_store
from backend.services.broadcaster import Broadcaster
from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = MemoryStore(max_history=2000)
    broadcaster = Broadcaster()
    runner = Runner(store=store, broadcaster=broadcaster)
    set_store(store)
    set_broadcaster(broadcaster)
    set_runner(runner)
    runner.start()
    yield
    runner.stop()


app = FastAPI(
    title="ESports Fair Odds API",
    description="Backend for live trading/probability terminal",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(state_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(bo3_router, prefix="/api/v1")
app.include_router(debug_router, prefix="/api/v1")
app.include_router(market_router, prefix="/api/v1")
app.include_router(prematch_router, prefix="/api/v1")
app.include_router(replay_router, prefix="/api/v1")
app.include_router(ws_router, prefix="/api/v1")


@app.get("/health")
def health() -> dict[str, bool]:
    """Liveness/readiness check."""
    return {"ok": True}

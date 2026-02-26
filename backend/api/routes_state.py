"""
State API: current State/Derived, history, config update.
"""
from fastapi import APIRouter, Body

from backend.deps import get_store

router = APIRouter(prefix="/state", tags=["state"])


@router.get("/current")
async def get_state_current() -> dict:
    """Return current authoritative state and derived outputs."""
    store = get_store()
    return await store.get_current()


@router.get("/history")
async def get_state_history(limit: int = 2000) -> list[dict]:
    """Return last `limit` history points (wire format: t, p, lo, hi, m)."""
    store = get_store()
    return await store.get_history(limit=limit)


# Config update at /api/v1/config (not under /state)
config_router = APIRouter(tags=["config"])


@config_router.post("/config")
async def post_config(partial: dict = Body(...)) -> dict:
    """Merge partial config update; return current state."""
    store = get_store()
    await store.update_config(partial)
    return await store.get_current()

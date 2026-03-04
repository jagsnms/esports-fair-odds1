"""
State API: current State/Derived, history, config update.
"""
from fastapi import APIRouter, Body, HTTPException

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


@router.post("/clear")
async def post_state_clear() -> dict:
    """Clear in-memory history and reset current state/derived (chart + scoreboard). Keeps config. Returns new current."""
    store = get_store()
    await store.clear_display()
    return await store.get_current()


# Config update at /api/v1/config (not under /state)
config_router = APIRouter(tags=["config"])


@config_router.post("/config")
async def post_config(partial: dict = Body(...)) -> dict:
    """Merge partial config update; return current state. Returns 400 if match_id invalid."""
    store = get_store()
    try:
        await store.update_config(partial)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return await store.get_current()

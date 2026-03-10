"""
Prematch API: set prematch series (derives prematch_map), lock, unlock, clear.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from backend.deps import get_store
from engine.compute.series_iid import derive_p_map_from_p_series

router = APIRouter(prefix="/prematch", tags=["prematch"])

# BO3 best-of for derivation
PREMATCH_BEST_OF = 3


@router.post("/set")
async def post_prematch_set(body: dict = Body(...)) -> dict:
    """
    Set prematch_series; derive prematch_map; lock by default.
    Body: { prematch_series: float, prematch_locked?: bool }
    Validates 0.01 < prematch_series < 0.99.
    """
    store = get_store()
    p_series = body.get("prematch_series")
    if p_series is None:
        raise HTTPException(status_code=400, detail="prematch_series required")
    try:
        p_series = float(p_series)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="prematch_series must be a number")
    if not (0.01 < p_series < 0.99):
        raise HTTPException(status_code=400, detail="prematch_series must be between 0.01 and 0.99")
    p_map = derive_p_map_from_p_series(PREMATCH_BEST_OF, p_series)
    locked = body.get("prematch_locked")
    if locked is None:
        locked = True
    else:
        locked = bool(locked)
    partial = {
        "prematch_series": p_series,
        "prematch_map": p_map,
        "prematch_locked": locked,
    }
    await store.update_config(partial)
    return await store.get_current()


@router.post("/unlock")
async def post_prematch_unlock() -> dict:
    """Set prematch_locked=False (values unchanged until next /set or /clear)."""
    store = get_store()
    await store.update_config({"prematch_locked": False})
    return await store.get_current()


@router.post("/clear")
async def post_prematch_clear() -> dict:
    """Set prematch_series=None, prematch_map=None, prematch_locked=False."""
    store = get_store()
    await store.update_config({
        "prematch_series": None,
        "prematch_map": None,
        "prematch_locked": False,
    })
    return await store.get_current()

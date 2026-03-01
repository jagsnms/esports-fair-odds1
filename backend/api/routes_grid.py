"""
GRID API: list CS2 series candidates (Central Data, titleId=28). Cached.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from engine.ingest.grid_central_data import (
    DEFAULT_ORDER_DIRECTION,
    get_cs2_series_candidates,
)

router = APIRouter(prefix="/grid", tags=["grid"])

MAX_LIMIT = 100
DEFAULT_LIMIT = 25


@router.get("/candidates")
async def get_grid_candidates(
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    order_direction: str = Query(
        default=DEFAULT_ORDER_DIRECTION,
        description="ASC = soonest first, DESC = latest first",
    ),
) -> list[dict]:
    """
    Return GRID CS2 series candidates (Central Data, titleId=28).
    Default order: soonest first (ASC). Cached for 30–60s.
    """
    return get_cs2_series_candidates(limit=limit, order_direction=order_direction)

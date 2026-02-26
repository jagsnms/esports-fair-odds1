"""
Market API: Kalshi URL resolve (options), select side (set config), return current state.
"""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from backend.deps import get_store

router = APIRouter(prefix="/market", tags=["market"])


@router.post("/resolve")
async def post_market_resolve(body: dict = Body(...)) -> dict:
    """
    Resolve Kalshi URL to list of selectable teams/sides and their YES tickers.
    Body: { "kalshi_url": "https://..." }
    Returns: { "options": [{ "key", "label", "ticker_yes" }], "suggested": first key or null }
    """
    url = (body.get("kalshi_url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="kalshi_url required")
    try:
        from engine.market.kalshi_client import resolve_kalshi_match_url
        result = resolve_kalshi_match_url(url)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/select")
async def post_market_select(body: dict = Body(...)) -> dict:
    """
    Set config.kalshi_url and config.kalshi_ticker from selection; return current state.
    Body: { "kalshi_url": "https://...", "market_side_key": "TICKER-..." }
    """
    store = get_store()
    url = (body.get("kalshi_url") or "").strip()
    side_key = (body.get("market_side_key") or "").strip()
    if not side_key:
        raise HTTPException(status_code=400, detail="market_side_key required")
    partial = {"kalshi_ticker": side_key}
    if url:
        partial["kalshi_url"] = url
    await store.update_config(partial)
    return await store.get_current()

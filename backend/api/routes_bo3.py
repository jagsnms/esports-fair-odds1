"""
BO3 API: list live matches (normalized for frontend).
"""
from fastapi import APIRouter, HTTPException

from engine.ingest.bo3_client import get_snapshot, list_live_matches

router = APIRouter(prefix="/bo3", tags=["bo3"])


@router.get("/live_matches")
async def get_live_matches() -> list[dict]:
    """Return list of live BO3 matches: id, team1_name, team2_name, bo_type."""
    return await list_live_matches()


@router.get("/snapshot/{match_id}")
async def bo3_snapshot(match_id: int) -> dict:
    """Fetch raw BO3 snapshot for match_id. Returns ok, match_id, snapshot (or None). On fetch error returns 502."""
    try:
        snap = await get_snapshot(int(match_id))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BO3 snapshot fetch failed: {e}")
    if snap is None:
        return {"ok": False, "match_id": match_id, "snapshot": None}
    return {"ok": True, "match_id": match_id, "snapshot": snap}

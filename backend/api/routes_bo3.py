"""
BO3 API: list live matches (normalized for frontend); candidates (current+upcoming) and readiness probe.
"""
import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from engine.ingest.bo3_client import fetch_candidates, get_snapshot, list_live_matches, probe_snapshot_readiness

router = APIRouter(prefix="/bo3", tags=["bo3"])

READINESS_CONCURRENCY = 4


@router.get("/live_matches")
async def get_live_matches() -> list[dict]:
    """Return list of live BO3 matches: id, team1_name, team2_name, bo_type."""
    return await list_live_matches()


@router.get("/candidates")
async def get_candidates() -> list[dict[str, Any]]:
    """Fetch current + upcoming from BO3 public matches API; normalize and dedupe by id. Does not trust BO3 live/current."""
    try:
        return await fetch_candidates()
    except Exception:
        return []


@router.post("/readiness")
async def post_readiness(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Probe snapshot for each match_id; return telemetry_ready, status_code, reason, last_probe_ts per id. Concurrency limited."""
    match_ids = body.get("match_ids") or []
    if not isinstance(match_ids, list):
        match_ids = []
    match_ids = [int(x) for x in match_ids if isinstance(x, (int, float)) and int(x) == x]
    if not match_ids:
        return []
    sem = asyncio.Semaphore(READINESS_CONCURRENCY)

    async def probe_one(mid: int) -> dict[str, Any]:
        async with sem:
            return await probe_snapshot_readiness(mid)

    results = await asyncio.gather(*(probe_one(mid) for mid in match_ids), return_exceptions=True)
    out: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            mid = match_ids[i] if i < len(match_ids) else 0
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            out.append({
                "match_id": mid,
                "telemetry_ready": False,
                "status_code": 502,
                "reason": "upstream_error",
                "last_probe_ts": ts,
            })
        else:
            out.append(r)
    return out


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

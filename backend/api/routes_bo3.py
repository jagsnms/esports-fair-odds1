"""
BO3 API: list live matches (normalized for frontend); candidates (current+upcoming) and readiness probe.
When BO3 primary is pinned, readiness and snapshot are gated to the pinned id only (stops frontend probe storms).
"""
import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.deps import get_store
from engine.ingest.bo3_client import fetch_candidates, get_snapshot, list_live_matches, probe_snapshot_readiness

router = APIRouter(prefix="/bo3", tags=["bo3"])
logger = logging.getLogger(__name__)
BO3_RATE_DEBUG = os.environ.get("BO3_RATE_DEBUG", "") == "1"

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
    """Probe snapshot for each match_id; return telemetry_ready, status_code, reason, last_probe_ts per id. When pinned to BO3, only probe pinned id."""
    config = await get_store().get_config()
    cfg_src = (str(getattr(config, "primary_session_source", None) or "").strip().upper()) or None
    cfg_id = (str(getattr(config, "primary_session_id", None) or "").strip() or None) or None
    pinned_id: int | None = None
    if cfg_src == "BO3" and cfg_id:
        try:
            pinned_id = int(cfg_id)
        except (TypeError, ValueError):
            pass

    match_ids = body.get("match_ids") or []
    if not isinstance(match_ids, list):
        match_ids = []
    match_ids = [int(x) for x in match_ids if isinstance(x, (int, float)) and int(x) == x]
    ids_requested = len(match_ids)

    if pinned_id is not None:
        ids_prevented = max(0, ids_requested - 1)
        match_ids = [pinned_id]
        if BO3_RATE_DEBUG and ids_prevented > 0:
            logger.info(
                "BO3 pinned guard readiness: ids_requested=%s ids_prevented=%s pinned_id=%s",
                ids_requested,
                ids_prevented,
                pinned_id,
            )

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
    """Fetch raw BO3 snapshot for match_id. When BO3 primary is pinned, only pinned id allowed (403 otherwise)."""
    config = await get_store().get_config()
    cfg_src = (str(getattr(config, "primary_session_source", None) or "").strip().upper()) or None
    cfg_id = (str(getattr(config, "primary_session_id", None) or "").strip() or None) or None
    if cfg_src == "BO3" and cfg_id:
        try:
            pinned_id = int(cfg_id)
            if int(match_id) != pinned_id:
                if BO3_RATE_DEBUG:
                    logger.info("BO3 pinned guard snapshot: id_requested=%s pinned_id=%s blocked", match_id, pinned_id)
                raise HTTPException(
                    status_code=403,
                    detail="Pinned to another BO3 match; only pinned id allowed.",
                )
        except ValueError:
            pass
    try:
        snap = await get_snapshot(int(match_id))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BO3 snapshot fetch failed: {e}")
    if snap is None:
        return {"ok": False, "match_id": match_id, "snapshot": None}
    return {"ok": True, "match_id": match_id, "snapshot": snap}

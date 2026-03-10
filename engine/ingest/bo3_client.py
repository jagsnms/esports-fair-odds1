"""
Async BO3 client using cs2api. No asyncio.run; caller is already async.
Public matches API (candidates) fetched via aiohttp; current + upcoming merged and deduped.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    from cs2api import CS2
except ImportError:
    CS2 = None  # type: ignore

BO3_RATE_DEBUG = os.environ.get("BO3_RATE_DEBUG", "") == "1"
logger = logging.getLogger(__name__)

BO3_MATCHES_URL = "https://api.bo3.gg/api/v1/matches"
BO3_MATCHES_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://bo3.gg",
    "referer": "https://bo3.gg/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
}
BO3_MATCHES_PARAMS_BASE = {
    "scope": "widget-matches",
    "page[offset]": 0,
    "page[limit]": 100,
    "sort": "tier_rank,-start_date",
    "filter[matches.discipline_id][eq]": 1,
    "with": "teams,tournament,games,streams",
}


def _normalize_match_candidate(m: dict[str, Any]) -> dict[str, Any] | None:
    """Extract id, team1_name, team2_name, bo_type, tier, start_date, parsed_status, live_coverage."""
    mid = m.get("id") or m.get("match_id") or m.get("matchId")
    if mid is None:
        return None
    bet = m.get("bet_updates") or {}
    t1 = bet.get("team_1") if isinstance(bet, dict) else {}
    t2 = bet.get("team_2") if isinstance(bet, dict) else {}
    t1 = t1 if isinstance(t1, dict) else {}
    t2 = t2 if isinstance(t2, dict) else {}
    name1 = (t1.get("name") if isinstance(t1.get("name"), str) else None) or "Team 1"
    name2 = (t2.get("name") if isinstance(t2.get("name"), str) else None) or "Team 2"
    bo_type = m.get("bo_type", 3)
    try:
        bo_type = int(bo_type) if bo_type is not None else 3
    except (TypeError, ValueError):
        bo_type = 3
    tier = m.get("tier")
    start_date = m.get("start_date")
    parsed_status = m.get("parsed_status")
    live_coverage = m.get("live_coverage", False)
    return {
        "id": int(mid) if not isinstance(mid, int) else mid,
        "team1_name": str(name1),
        "team2_name": str(name2),
        "bo_type": bo_type,
        "tier": tier,
        "start_date": start_date,
        "parsed_status": str(parsed_status) if parsed_status is not None else None,
        "live_coverage": bool(live_coverage),
    }


async def fetch_candidates() -> list[dict[str, Any]]:
    """Fetch current + upcoming from BO3 public matches API; normalize and dedupe by id."""
    try:
        import aiohttp
    except ImportError:
        return []
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for status_bucket in ("current", "upcoming"):
        params = {**BO3_MATCHES_PARAMS_BASE, "filter[matches.status][in]": status_bucket}
        t0 = time.perf_counter()
        status_code: int | None = None
        retry_after: str | None = None
        data: Any = None
        try:
            async with aiohttp.ClientSession(headers=BO3_MATCHES_HEADERS) as session:
                async with session.get(BO3_MATCHES_URL, params=params) as resp:
                    status_code = getattr(resp, "status", None)
                    retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception:
            if status_code is None:
                status_code = 0
            data = None
        receive_ts = time.time()
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)
        if BO3_RATE_DEBUG:
            logger.info(
                "BO3 request endpoint_type=list status_code=%s duration_ms=%s retry_after=%s receive_ts=%.3f",
                status_code,
                duration_ms,
                retry_after,
                receive_ts,
                extra={"endpoint_type": "list", "status_code": status_code, "duration_ms": duration_ms, "retry_after": retry_after, "receive_ts": receive_ts},
            )
        if data is None:
            continue
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("results", data.get("matches", data.get("data", [])))
            if not isinstance(items, list):
                items = []
        else:
            items = []
        for m in items:
            if not isinstance(m, dict):
                continue
            norm = _normalize_match_candidate(m)
            if norm is None:
                continue
            mid = norm["id"]
            if mid in seen:
                continue
            seen.add(mid)
            out.append(norm)
    return out


async def list_live_matches() -> list[dict[str, Any]]:
    """Return list of live matches: id, team1_name, team2_name, bo_type."""
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")
    t0 = time.perf_counter()
    status_code = 200
    try:
        async with CS2() as cs2:
            raw = await cs2.get_live_matches()
    except Exception as e:
        status_code = 0
        msg = str(e).lower()
        if "429" in msg or "rate" in msg:
            status_code = 429
        elif "404" in msg or "not found" in msg:
            status_code = 404
        raise
    finally:
        if BO3_RATE_DEBUG:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            receive_ts = time.time()
            logger.info(
                "BO3 request endpoint_type=list status_code=%s duration_ms=%s receive_ts=%.3f",
                status_code,
                duration_ms,
                receive_ts,
                extra={"endpoint_type": "list", "status_code": status_code, "duration_ms": duration_ms, "receive_ts": receive_ts},
            )
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("results", raw.get("matches", raw.get("data", [])))
        if not isinstance(items, list):
            items = []
    else:
        items = []
    out: list[dict[str, Any]] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        if mid is None:
            continue
        bet = m.get("bet_updates") or {}
        t1 = bet.get("team_1") if isinstance(bet, dict) else {}
        t2 = bet.get("team_2") if isinstance(bet, dict) else {}
        t1 = t1 if isinstance(t1, dict) else {}
        t2 = t2 if isinstance(t2, dict) else {}
        name1 = t1.get("name", "Team 1") if isinstance(t1.get("name"), str) else "Team 1"
        name2 = t2.get("name", "Team 2") if isinstance(t2.get("name"), str) else "Team 2"
        bo_type = m.get("bo_type", 3)
        live_coverage = m.get("live_coverage", False)
        parsed_status = m.get("parsed_status")
        out.append({
            "id": mid,
            "team1_name": str(name1),
            "team2_name": str(name2),
            "bo_type": int(bo_type) if bo_type is not None else 3,
            "live_coverage": bool(live_coverage),
            "parsed_status": str(parsed_status) if parsed_status is not None else None,
        })
    return out


async def get_snapshot(
    match_id: int,
    *,
    _rate_debug_retry_count: int = 0,
    _rate_debug_backoff_s: float | None = None,
) -> dict[str, Any] | None:
    """Fetch live match snapshot for match_id. Returns raw snapshot dict or None.
    Use candidate \"id\" from fetch_candidates() as match_id (same identifier)."""
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")
    t0 = time.perf_counter()
    status_code = 200
    result: dict[str, Any] | None = None
    try:
        async with CS2() as cs2:
            snap = await cs2.get_live_match_snapshot(match_id)
        result = snap if isinstance(snap, dict) else None
    except Exception as e:
        status_code = 0
        msg = str(e).lower()
        if "429" in msg or "rate" in msg:
            status_code = 429
        elif "404" in msg or "not found" in msg:
            status_code = 404
        raise
    finally:
        if BO3_RATE_DEBUG:
            duration_ms = round((time.perf_counter() - t0) * 1000, 1)
            receive_ts = time.time()
            logger.info(
                "BO3 request endpoint_type=snapshot match_id=%s status_code=%s duration_ms=%s retry_count=%s backoff_s=%s receive_ts=%.3f",
                match_id,
                status_code,
                duration_ms,
                _rate_debug_retry_count,
                _rate_debug_backoff_s,
                receive_ts,
                extra={
                    "endpoint_type": "snapshot",
                    "match_id": match_id,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "retry_count": _rate_debug_retry_count,
                    "backoff_s": _rate_debug_backoff_s,
                    "receive_ts": receive_ts,
                },
            )
    return result


async def probe_snapshot_readiness(match_id: int) -> dict[str, Any]:
    """Probe snapshot endpoint for one match; return telemetry_ready, status_code, reason, last_probe_ts (best-effort)."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    if CS2 is None:
        return {
            "match_id": match_id,
            "telemetry_ready": False,
            "status_code": 0,
            "reason": "cs2api_not_available",
            "last_probe_ts": ts,
        }
    try:
        snap = await get_snapshot(match_id)
        if isinstance(snap, dict) and (snap.get("team_one") or snap.get("team_two")):
            return {
                "match_id": match_id,
                "telemetry_ready": True,
                "status_code": 200,
                "reason": "ok",
                "last_probe_ts": ts,
            }
        return {
            "match_id": match_id,
            "telemetry_ready": False,
            "status_code": 404,
            "reason": "not_ready_404",
            "last_probe_ts": ts,
        }
    except Exception as e:
        msg = str(e).lower()
        status_code = 502
        reason = "upstream_error"
        if "404" in msg or "not found" in msg:
            status_code = 404
            reason = "not_ready_404"
        elif "429" in msg or "rate" in msg:
            status_code = 429
            reason = "rate_limited"
        return {
            "match_id": match_id,
            "telemetry_ready": False,
            "status_code": status_code,
            "reason": reason,
            "last_probe_ts": ts,
        }

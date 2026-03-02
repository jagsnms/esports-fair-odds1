"""
Debug API: raw BO3 snapshot dump for field-name / reliability inspection.
Temporary; no impact on model logic.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time

from fastapi import APIRouter, HTTPException, Query

from backend.deps import get_runner, get_store

BO3_WS_PROBE_WAIT_S = 3.0
BO3_WS_FIRST_NON_PING_MAX_CHARS = 500

router = APIRouter(prefix="/debug", tags=["debug"])

MAX_RAW_JSON_CHARS = 200_000


def _snapshot_ts(snap: dict) -> str | None:
    """Extract a snapshot timestamp from common keys."""
    for key in ("created_at", "updated_at", "sent_time", "ts"):
        if key in snap and snap[key] is not None:
            return str(snap[key])
    return None


def _truncate_raw(snap: dict) -> tuple[dict | str, bool]:
    """
    Return (payload, truncated). If JSON would be > MAX_RAW_JSON_CHARS,
    first try pruning player_states / team players; else return string slice.
    """
    try:
        js = json.dumps(snap, default=str)
    except Exception:
        js = json.dumps({"error": "could not serialize snapshot"}, default=str)
    if len(js) <= MAX_RAW_JSON_CHARS:
        return (snap, False)
    # Try pruning heavy arrays
    try:
        pruned = copy.deepcopy(snap)
        truncated = False
        for team_key in ("team_one", "team_two"):
            if team_key not in pruned or not isinstance(pruned[team_key], dict):
                continue
            team = pruned[team_key]
            for list_key in ("players", "player_states", "player_states_flat"):
                if list_key in team and isinstance(team[list_key], list) and len(team[list_key]) > 10:
                    team[list_key] = team[list_key][:10]
                    truncated = True
        if "player_states" in pruned and isinstance(pruned["player_states"], list) and len(pruned["player_states"]) > 20:
            pruned["player_states"] = pruned["player_states"][:20]
            truncated = True
        if truncated:
            pruned["_truncated_players"] = True
        js2 = json.dumps(pruned, default=str)
        if len(js2) <= MAX_RAW_JSON_CHARS:
            return (pruned, True)
    except Exception:
        pass
    return (js[:MAX_RAW_JSON_CHARS], True)


@router.get("/bo3/ws_probe/{match_id}")
async def get_debug_bo3_ws_probe(match_id: int) -> dict:
    """
    Test-only: connect to BO3 live WS, wait up to 3s, count ping/pong vs data messages.
    Returns connected, total_messages, ping_pong_messages, non_ping_messages,
    first_non_ping_truncated, error_if_any. No API keys; read-only probe.
    """
    uri = f"wss://live.bo3.gg/ws/live_game/{match_id}"
    out = {
        "match_id": match_id,
        "connected": False,
        "total_messages": 0,
        "ping_pong_messages": 0,
        "non_ping_messages": 0,
        "first_non_ping_truncated": None,
        "error_if_any": None,
    }
    try:
        import websockets
    except ImportError:
        out["error_if_any"] = "websockets not installed"
        return out
    try:
        async with websockets.connect(
            uri,
            open_timeout=5.0,
            close_timeout=1.0,
        ) as ws:
            out["connected"] = True
            deadline = time.monotonic() + BO3_WS_PROBE_WAIT_S
            first_non_ping: str | None = None
            while time.monotonic() < deadline:
                remaining = max(0.1, deadline - time.monotonic())
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(0.5, remaining))
                except asyncio.TimeoutError:
                    break
                out["total_messages"] += 1
                is_ping_pong = False
                if isinstance(msg, str):
                    low = msg.strip().lower()
                    if low in ("ping", "pong"):
                        is_ping_pong = True
                elif isinstance(msg, bytes):
                    try:
                        low = msg.decode("utf-8", errors="replace").strip().lower()
                        if low in ("ping", "pong"):
                            is_ping_pong = True
                    except Exception:
                        pass
                if is_ping_pong:
                    out["ping_pong_messages"] += 1
                else:
                    out["non_ping_messages"] += 1
                    if first_non_ping is None:
                        first_non_ping = msg if isinstance(msg, str) else msg.decode("utf-8", errors="replace")
            if first_non_ping is not None:
                out["first_non_ping_truncated"] = (
                    first_non_ping[:BO3_WS_FIRST_NON_PING_MAX_CHARS]
                    if len(first_non_ping) > BO3_WS_FIRST_NON_PING_MAX_CHARS
                    else first_non_ping
                )
    except Exception as e:
        out["error_if_any"] = str(e)
    return out


@router.get("/bo3/last_snapshot")
async def get_debug_bo3_last_snapshot() -> dict:
    """
    Return last raw BO3 snapshot from runner (for debugging field names).
    If no snapshot yet, 404. Raw may be truncated if > 200k chars or players pruned.
    """
    runner = get_runner()
    snap = runner._bo3_last_raw_snapshot
    if not snap or not isinstance(snap, dict):
        raise HTTPException(status_code=404, detail="No BO3 snapshot yet; run live BO3 first.")
    config = await get_store().get_config()
    match_id = getattr(config, "match_id", None)
    raw_payload, truncated = _truncate_raw(snap)
    if truncated and isinstance(raw_payload, str):
        return {
            "match_id": match_id,
            "snapshot_ts": _snapshot_ts(snap),
            "raw_keys_top": list(snap.keys()),
            "raw": raw_payload,
            "truncated": True,
        }
    return {
        "match_id": match_id,
        "snapshot_ts": _snapshot_ts(snap),
        "raw_keys_top": list(snap.keys()),
        "raw": raw_payload,
        "truncated": truncated,
    }


@router.get("/bo3/last_snapshot/players")
async def get_debug_bo3_last_snapshot_players(
    team: str = Query("one", description="one or two"),
    n: int = Query(10, ge=1, le=50),
) -> dict:
    """Return first n player entries for the given team from last snapshot."""
    runner = get_runner()
    snap = runner._bo3_last_raw_snapshot
    if not snap or not isinstance(snap, dict):
        raise HTTPException(status_code=404, detail="No BO3 snapshot yet; run live BO3 first.")
    team_key = "team_one" if team.lower() in ("one", "1") else "team_two"
    team_obj = snap.get(team_key)
    if not isinstance(team_obj, dict):
        return {"team": team_key, "n": n, "players": [], "note": f"{team_key} missing or not a dict"}
    players = team_obj.get("players") or team_obj.get("player_states") or []
    if not isinstance(players, list):
        return {"team": team_key, "n": n, "players": [], "note": "players not a list"}
    subset = players[:n]
    return {"team": team_key, "n": n, "total": len(players), "players": subset}


@router.get("/telemetry")
async def get_debug_telemetry() -> dict:
    """
    Return source-agnostic match context diagnostics (accepted/rejected counts,
    last_accepted_key, last_reject_reason, per_source_health for BO3).
    """
    runner = get_runner()
    return getattr(runner, "get_match_context_diag", lambda: {"match_id": None, "context": {}})()


@router.get("/telemetry/sessions")
async def get_debug_telemetry_sessions() -> dict:
    """
    Return multi-session hub diagnostics: now_ts and list of sessions with
    session_key, source, id, last_update_ts, age_s, ctx, grid_schedule for GRID,
    and grid_auto_track_* when grid_auto_track is enabled.
    """
    runner = get_runner()
    config = await get_store().get_config()
    return getattr(runner, "get_sessions_diag", lambda c=None: {"now_ts": time.time(), "sessions": []})(config)


@router.post("/telemetry/clear_sessions")
async def post_debug_telemetry_clear_sessions() -> dict:
    """
    Clear runtime session state only: sessions registry, auto-track lists, readiness cache, grid schedule.
    Does NOT modify persistent config. Returns { "ok": true }.
    """
    runner = get_runner()
    getattr(runner, "clear_sessions", lambda: None)()
    return {"ok": True}

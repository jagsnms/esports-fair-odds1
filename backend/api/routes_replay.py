"""
Replay API: load JSONL replay, stop, status, list matches.
"""
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from backend.deps import get_store
from backend.store.memory_store import MemoryStore

router = APIRouter(prefix="/replay", tags=["replay"])


def _team_name_from_side(payload: dict, side: str) -> str:
    """Extract team name: payload[side]['fixture']['team_name'] or payload[side]['name']."""
    team = payload.get(side) if isinstance(payload.get(side), dict) else None
    if not team:
        return ""
    fixture = team.get("fixture") if isinstance(team.get("fixture"), dict) else None
    if fixture and isinstance(fixture.get("team_name"), str) and fixture["team_name"].strip():
        return fixture["team_name"].strip()
    if isinstance(team.get("name"), str) and team["name"].strip():
        return team["name"].strip()
    return ""


@router.get("/matches")
async def replay_matches(
    path: str = Query(..., description="JSONL file path (e.g. logs/bo3_pulls.jsonl)"),
) -> Any:
    """
    List matches in the JSONL: match_id, team1, team2, count.
    Sorted by count desc, then match_id asc. 400 if path missing or file not found.
    """
    if not path or not path.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "Query parameter 'path' is required (e.g. path=logs/bo3_pulls.jsonl)"},
        )
    path = path.strip()
    from engine.replay.bo3_jsonl import load_bo3_jsonl_entries, group_by_match

    entries = load_bo3_jsonl_entries(path)
    if not entries:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Path not found or no valid BO3 entries: {path!r}"},
        )
    by_match = group_by_match(entries)
    result: list[dict[str, Any]] = []
    for match_id, match_entries in by_match.items():
        team1 = ""
        team2 = ""
        for e in match_entries:
            payload = e.get("payload") if isinstance(e, dict) else None
            if isinstance(payload, dict):
                team1 = _team_name_from_side(payload, "team_one")
                team2 = _team_name_from_side(payload, "team_two")
                if team1 or team2:
                    break
        result.append({
            "match_id": match_id,
            "team1": team1 or "—",
            "team2": team2 or "—",
            "count": len(match_entries),
        })
    result.sort(key=lambda x: (-x["count"], x["match_id"]))
    return result


@router.post("/load")
async def replay_load(
    body: dict[str, Any],
    store: MemoryStore = Depends(get_store),
) -> dict[str, Any]:
    """
    POST body: path?, match_id?, speed?, loop?
    Set source=REPLAY and replay options; return current state.
    """
    path = body.get("path")
    match_id = body.get("match_id")
    speed = body.get("speed")
    loop = body.get("loop")
    partial: dict[str, Any] = {"source": "REPLAY"}
    if path is not None:
        partial["replay_path"] = str(path)
    if match_id is not None:
        partial["match_id"] = match_id
    if speed is not None:
        try:
            partial["replay_speed"] = float(speed)
        except (TypeError, ValueError):
            pass
    if loop is not None:
        partial["replay_loop"] = bool(loop)
    await store.update_config(partial)
    return await store.get_current()


@router.post("/stop")
async def replay_stop(
    store: MemoryStore = Depends(get_store),
) -> dict[str, Any]:
    """Set source=DUMMY to stop replay."""
    await store.update_config({"source": "DUMMY"})
    return await store.get_current()


@router.get("/status")
async def replay_status(
    store: MemoryStore = Depends(get_store),
) -> dict[str, Any]:
    """Current replay config fields plus runner progress (index/total) if available."""
    config = await store.get_config()
    out: dict[str, Any] = {
        "source": getattr(config, "source", None),
        "replay_path": getattr(config, "replay_path", None),
        "replay_loop": getattr(config, "replay_loop", True),
        "replay_speed": getattr(config, "replay_speed", 1.0),
        "replay_index": getattr(config, "replay_index", 0),
        "match_id": getattr(config, "match_id", None),
    }
    try:
        from backend.deps import get_runner
        runner = get_runner()
        if hasattr(runner, "get_replay_progress"):
            progress = runner.get_replay_progress()
            if progress is not None:
                out["replay_progress"] = progress
    except Exception:
        pass
    return out

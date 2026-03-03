"""
Replay API: load JSONL replay, stop, status, list matches, list sources.
"""
import os
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from backend.deps import get_store
from backend.store.memory_store import MemoryStore

router = APIRouter(prefix="/replay", tags=["replay"])

DEFAULT_REPLAY_PATH = "logs/bo3_pulls.jsonl"


def _discover_replay_sources() -> dict[str, Any]:
    """Scan logs/ for replay JSONL files. Only paths under logs/. Returns {default_path, sources}."""
    cwd = Path(os.getcwd())
    logs_dir = (cwd / "logs").resolve()
    default_path = DEFAULT_REPLAY_PATH
    sources: list[dict[str, Any]] = []

    if not logs_dir.is_dir():
        return {"default_path": default_path, "sources": []}

    try:
        logs_real = logs_dir.resolve()
        if not str(logs_real).startswith(str(cwd.resolve())):
            return {"default_path": default_path, "sources": []}
    except (OSError, ValueError):
        return {"default_path": default_path, "sources": []}

    raw_match_re = re.compile(r"^bo3_raw_match_(\d+)\.jsonl$", re.IGNORECASE)
    fixed: list[tuple[str, str, str]] = []  # (rel_path, label, kind)
    raw_matches: list[tuple[str, str, float, int]] = []  # (rel_path, label, mtime, size)
    history_points: list[tuple[str, str, float, int]] = []

    for f in logs_dir.iterdir():
        if not f.is_file() or f.suffix.lower() != ".jsonl":
            continue
        name = f.name
        try:
            stat = f.stat()
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            mtime = 0.0
            size = 0
        rel_str = "logs/" + name

        if name == "bo3_pulls.jsonl":
            fixed.append((rel_str, "BO3 pulls (legacy)", "raw"))
        elif name == "bo3_raw.jsonl":
            fixed.append((rel_str, "BO3 raw", "raw"))
        elif raw_match_re.match(name):
            mid = raw_match_re.match(name).group(1)
            raw_matches.append((rel_str, f"BO3 raw match {mid}", mtime, size))
        elif name == "history_points.jsonl":
            history_points.append((rel_str, "History points", mtime, size))
        elif name.startswith("history_points") and name.endswith(".jsonl") and name != "history_points.jsonl":
            history_points.append((rel_str, name, mtime, size))

    for rel_str, label, kind in fixed:
        try:
            p = (logs_dir / rel_str.split("/")[-1]).resolve()
            stat = p.stat()
            sources.append({
                "label": label,
                "path": rel_str,
                "kind": kind,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
        except OSError:
            sources.append({"label": label, "path": rel_str, "kind": kind, "mtime": 0, "size": 0})

    raw_matches.sort(key=lambda x: -x[2])
    for rel_str, label, _mtime, size in raw_matches:
        sources.append({"label": label, "path": rel_str, "kind": "raw", "mtime": _mtime, "size": size})

    for rel_str, label, _mtime, size in sorted(history_points, key=lambda x: -x[2]):
        sources.append({"label": label, "path": rel_str, "kind": "point", "mtime": _mtime, "size": size})

    return {"default_path": default_path, "sources": sources}


@router.get("/sources")
async def replay_sources() -> dict[str, Any]:
    """
    List available replay input files under logs/ (bo3_pulls.jsonl, bo3_raw_match_*.jsonl, history_points*.jsonl).
    Returns default_path and sources with label, path, kind (raw|point), mtime, size for UI dropdown.
    """
    return _discover_replay_sources()


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


def _matches_for_path(path: str) -> list[dict[str, Any]]:
    """Load matches from a single JSONL path. Returns list of {match_id, team1, team2, count}."""
    from engine.replay.bo3_jsonl import load_bo3_jsonl_entries, load_generic_jsonl, group_by_match

    entries = load_bo3_jsonl_entries(path)
    if not entries:
        generic = load_generic_jsonl(path)
        entries = [
            e for e in generic
            if isinstance(e.get("payload"), dict) and e.get("match_id") is not None
        ]
    if not entries:
        return []
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
    return result


@router.get("/matches")
async def replay_matches(
    path: str = Query(None, description="JSONL file path (e.g. logs/bo3_pulls.jsonl). Omit if all_sources=1."),
    all_sources: int = Query(0, description="If 1, return matches from every discovered replay file (path ignored)."),
) -> Any:
    """
    List matches: match_id, team1, team2, count. If all_sources=1, merge from all logs/*.jsonl replay files
    and include path per match so the client can load that file. Otherwise require path and return matches from that file.
    Sorted by count desc, then match_id asc. 400 if path missing when all_sources=0.
    """
    if all_sources == 1:
        # Discover all replay sources and merge matches; each match keeps the path with highest count
        discovered = _discover_replay_sources()
        sources = discovered.get("sources") or []
        merged: dict[int, dict[str, Any]] = {}  # match_id -> { match_id, team1, team2, count, path }
        cwd = Path(os.getcwd())
        for s in sources:
            rel_path = s.get("path") or ""
            if not rel_path or not rel_path.endswith(".jsonl"):
                continue
            # Resolve path: backend often runs with cwd = repo root
            abs_path = str((cwd / rel_path).resolve()) if not Path(rel_path).is_absolute() else rel_path
            for row in _matches_for_path(abs_path):
                mid = row["match_id"]
                if mid not in merged or row["count"] > merged[mid]["count"]:
                    merged[mid] = {**row, "path": rel_path}
        result = list(merged.values())
        result.sort(key=lambda x: (-x["count"], x["match_id"]))
        return result

    if not path or not path.strip():
        return JSONResponse(
            status_code=400,
            content={"detail": "Query parameter 'path' is required (e.g. path=logs/bo3_pulls.jsonl)"},
        )
    path = path.strip()
    cwd = Path(os.getcwd())
    abs_path = str((cwd / path).resolve()) if not Path(path).is_absolute() else path
    result = _matches_for_path(abs_path)
    if not result:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Path not found or no valid BO3 entries: {path!r}"},
        )
    for row in result:
        row["path"] = path
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
    if path is not None and str(path).strip():
        partial["replay_path"] = str(path).strip()
    if match_id is not None:
        partial["match_id"] = match_id
    if speed is not None:
        try:
            partial["replay_speed"] = float(speed)
        except (TypeError, ValueError):
            pass
    if loop is not None:
        partial["replay_loop"] = bool(loop)
    profile = body.get("midround_v2_weight_profile")
    if isinstance(profile, str) and profile.strip().lower() in ("current", "learned_v1", "learned_v2", "learned_fit"):
        partial["midround_v2_weight_profile"] = profile.strip().lower()
    await store.update_config(partial)
    return await store.get_current()


@router.post("/stop")
async def replay_stop(
    store: MemoryStore = Depends(get_store),
) -> dict[str, Any]:
    """Set source=BO3 to stop replay (idle telemetry)."""
    await store.update_config({"source": "BO3"})
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

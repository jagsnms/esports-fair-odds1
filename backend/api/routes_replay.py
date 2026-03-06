"""
Replay API: load JSONL replay, stop, status, list matches, list sources.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from backend.deps import get_store
from backend.store.memory_store import MemoryStore
from backend.services.runner import (
    REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE,
    REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY,
    REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED,
    REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_MISSING,
    REPLAY_POINT_REJECT_REASON_UNSUPPORTED_POLICY,
 )

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
async def replay_sources(
    store: MemoryStore = Depends(get_store),
) -> dict[str, Any]:
    """
    List available replay input files under logs/ (bo3_pulls.jsonl, bo3_raw_match_*.jsonl, history_points*.jsonl).
    Returns default_path and sources with label, path, kind (raw|point), mtime, size for UI dropdown.
    """
    discovered = _discover_replay_sources()
    sources = discovered.get("sources") or []
    config = await store.get_config()
    out_sources: list[dict[str, Any]] = []
    for s in sources:
        row = dict(s) if isinstance(s, dict) else {}
        kind = row.get("kind")
        row.update(_replay_source_contract_signaling(kind=kind, config=config))
        out_sources.append(row)
    return {"default_path": discovered.get("default_path", DEFAULT_REPLAY_PATH), "sources": out_sources}


def _coerce_replay_contract_policy(config: Any) -> str:
    raw = getattr(config, "replay_contract_policy", REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE)
    policy = str(raw).strip().lower() if raw is not None else REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE
    if policy != REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE:
        return REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE
    return policy


def _coerce_transition_sunset_epoch(config: Any) -> float | None:
    raw = getattr(config, "replay_point_transition_sunset_epoch", None)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _replay_source_contract_signaling(kind: Any, config: Any) -> dict[str, Any]:
    """
    Stage 1.5: signal replay contract eligibility without enforcing it.
    Runner remains final authority; this mirrors the Stage 1 gate surface for UI/discovery transparency.
    """
    k = str(kind).strip().lower() if isinstance(kind, str) else ""
    if k != "point":
        return {
            "selectable": True,
            "contract_class": "canonical_raw",
            "reason_code": None,
            "requires_transition_mode": False,
            "transition_window_valid": False,
        }

    policy = _coerce_replay_contract_policy(config)
    transition_enabled = bool(getattr(config, "replay_point_transition_enabled", False))
    sunset_epoch = _coerce_transition_sunset_epoch(config)
    now = time.time()
    transition_window_valid = bool(transition_enabled and sunset_epoch is not None and sunset_epoch > now)

    if policy != REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE:
        reason = REPLAY_POINT_REJECT_REASON_UNSUPPORTED_POLICY
        selectable = False
    elif not transition_enabled:
        reason = REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY
        selectable = False
    elif sunset_epoch is None:
        reason = REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_MISSING
        selectable = False
    elif sunset_epoch <= now:
        reason = REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED
        selectable = False
    else:
        reason = None
        selectable = True

    return {
        "selectable": bool(selectable),
        "contract_class": "non_canonical_point",
        "reason_code": reason,
        "requires_transition_mode": True,
        "transition_window_valid": bool(transition_window_valid),
    }


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
    POST body: path?, match_id?, speed?, loop?, replay contract gate fields?
    Set source=REPLAY and replay options; return current state.
    """
    path = body.get("path")
    match_id = body.get("match_id")
    speed = body.get("speed")
    loop = body.get("loop")
    replay_contract_policy = body.get("replay_contract_policy")
    replay_point_transition_enabled = body.get("replay_point_transition_enabled")
    replay_point_transition_sunset_epoch = body.get("replay_point_transition_sunset_epoch")
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
    if replay_contract_policy is not None:
        partial["replay_contract_policy"] = replay_contract_policy
    if replay_point_transition_enabled is not None:
        partial["replay_point_transition_enabled"] = bool(replay_point_transition_enabled)
    if replay_point_transition_sunset_epoch is not None:
        partial["replay_point_transition_sunset_epoch"] = replay_point_transition_sunset_epoch
    profile = body.get("midround_v2_weight_profile")
    if isinstance(profile, str) and profile.strip().lower() in ("current", "learned_v1", "learned_v2", "learned_fit"):
        partial["midround_v2_weight_profile"] = profile.strip().lower()
    await store.update_config(partial)
    config = await store.get_config()
    kind = _infer_replay_kind_from_path(getattr(config, "replay_path", None))
    preflight = _replay_source_contract_signaling(kind=kind, config=config)
    cur = await store.get_current()
    if not isinstance(cur, dict):
        cur = {}
    cur["replay_load_preflight"] = preflight
    return cur


def _infer_replay_kind_from_path(path: Any) -> str:
    """
    Lightweight kind inference for /replay/load preflight.
    Prefer discovered sources; fallback to filename heuristic.
    """
    if not isinstance(path, str) or not path.strip():
        return "raw"
    p = path.strip()
    # Match discovered sources first (rel paths).
    discovered = _discover_replay_sources()
    for s in (discovered.get("sources") or []):
        if isinstance(s, dict) and s.get("path") == p:
            k = s.get("kind")
            return str(k).strip().lower() if isinstance(k, str) else "raw"
    name = Path(p).name.lower()
    if "history_points" in name:
        return "point"
    return "raw"


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
        "replay_contract_policy": getattr(config, "replay_contract_policy", "reject_point_like"),
        "replay_point_transition_enabled": bool(getattr(config, "replay_point_transition_enabled", False)),
        "replay_point_transition_sunset_epoch": getattr(config, "replay_point_transition_sunset_epoch", None),
    }
    try:
        from backend.deps import get_runner
        runner = get_runner()
        if hasattr(runner, "get_replay_progress"):
            progress = runner.get_replay_progress()
            if progress is not None:
                out["replay_progress"] = progress
        if hasattr(runner, "get_replay_contract_status"):
            out["replay_contract_status"] = runner.get_replay_contract_status()
    except Exception:
        pass
    return out

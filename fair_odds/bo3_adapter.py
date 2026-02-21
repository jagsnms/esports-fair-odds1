"""
BO3.gg / cs2api adapter: sync wrappers for live matches and snapshot, and normalizer to app state shape.
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any, Dict, List, Optional

# Optional: only import cs2api when used
try:
    from cs2api import CS2
except ImportError:
    CS2 = None  # type: ignore


def _run_async(coro):
    """Run async coroutine from sync context. Uses new event loop per call."""
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    return asyncio.run(coro)


def fetch_bo3_live_matches() -> List[Dict[str, Any]]:
    """
    Fetch live matches from BO3.gg. Returns list of dicts with id, team1_name, team2_name, bo_type, etc.
    """
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")

    async def _fetch():
        async with CS2() as cs2:
            raw = await cs2.get_live_matches()
            return raw

    raw = _run_async(_fetch())

    # Normalize to list of match dicts
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("results", raw.get("matches", raw.get("data", [])))
        if not isinstance(items, list):
            items = []
    else:
        items = []

    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        if mid is None:
            continue
        # Team names: from bet_updates or from snapshot later
        bet = m.get("bet_updates") or {}
        t1 = bet.get("team_1") or {}
        t2 = bet.get("team_2") or {}
        name1 = (t1.get("name") if isinstance(t1, dict) else None) or "Team 1"
        name2 = (t2.get("name") if isinstance(t2, dict) else None) or "Team 2"
        bo_type = m.get("bo_type") or 3
        out.append({
            "id": mid,
            "match_id": mid,
            "team1_name": str(name1),
            "team2_name": str(name2),
            "bo_type": int(bo_type),
            "team1_score": m.get("team1_score"),
            "team2_score": m.get("team2_score"),
        })
    return out


def fetch_bo3_snapshot(match_id: int) -> Optional[Dict[str, Any]]:
    """Fetch live match snapshot for match_id. Returns raw snapshot dict or None on failure."""
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")

    async def _fetch():
        async with CS2() as cs2:
            return await cs2.get_live_match_snapshot(match_id)

    try:
        return _run_async(_fetch())
    except Exception:
        return None


# Map BO3 map_name (e.g. de_inferno) to app CS2_MAP_CT_RATE key (e.g. Inferno)
def _bo3_map_name_to_app(bo3_map: Optional[str], valid_keys: Optional[List[str]] = None) -> str:
    if not bo3_map or not isinstance(bo3_map, str):
        return "Average (no map)"
    s = str(bo3_map).strip().lower()
    if s.startswith("de_"):
        s = s[3:]
    # Title-case: inferno -> Inferno
    name = s.replace("_", " ").title().replace(" ", "")
    if not name:
        return "Average (no map)"
    if valid_keys and name in valid_keys:
        return name
    # Allow returning name even if not in valid_keys (app can use "Average (no map)" for unknown)
    return name


def normalize_bo3_snapshot_to_app(
    snapshot: Dict[str, Any],
    team_a_is_team_one: bool,
    valid_map_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Normalize BO3 snapshot to app state shape.
    team_a_is_team_one: True if Team A in the app is BO3 team_one, False if team_two.
    Returns dict with: team_a, team_b, rounds_a, rounds_b, maps_a_won, maps_b_won,
    map_name (app form), a_side ("CT" or "T"), series_fmt (e.g. "BO3"), round_number, game_number,
    team_a_econ, team_b_econ (sum of balance+equipment_value per team).
    """
    if not snapshot or not isinstance(snapshot, dict):
        return {}

    t1 = snapshot.get("team_one") or {}
    t2 = snapshot.get("team_two") or {}
    if not isinstance(t1, dict):
        t1 = {}
    if not isinstance(t2, dict):
        t2 = {}

    name1 = str(t1.get("name") or t1.get("team_name") or "Team 1")
    name2 = str(t2.get("name") or t2.get("team_name") or "Team 2")
    score1 = int(t1.get("score", 0)) if t1.get("score") is not None else 0
    score2 = int(t2.get("score", 0)) if t2.get("score") is not None else 0
    match_score1 = int(t1.get("match_score", 0)) if t1.get("match_score") is not None else 0
    match_score2 = int(t2.get("match_score", 0)) if t2.get("match_score") is not None else 0
    side1 = str(t1.get("side") or "").strip().upper()
    side2 = str(t2.get("side") or "").strip().upper()
    # BO3 uses "TERRORIST" -> app uses "T"
    if side1 == "TERRORIST":
        side1 = "T"
    if side2 == "TERRORIST":
        side2 = "T"

    if team_a_is_team_one:
        team_a, team_b = name1, name2
        rounds_a, rounds_b = score1, score2
        maps_a_won, maps_b_won = match_score1, match_score2
        a_side = side1 if side1 in ("CT", "T") else "Unknown"
    else:
        team_a, team_b = name2, name1
        rounds_a, rounds_b = score2, score1
        maps_a_won, maps_b_won = match_score2, match_score1
        a_side = side2 if side2 in ("CT", "T") else "Unknown"

    map_name_raw = snapshot.get("map_name") or ""
    map_name = _bo3_map_name_to_app(map_name_raw, valid_map_keys)
    # If normalized name not in valid keys, use "Average (no map)" so app logic works
    if valid_map_keys and map_name not in valid_map_keys:
        map_name = "Average (no map)"

    game_number = snapshot.get("game_number")
    if game_number is not None:
        try:
            game_number = int(game_number)
        except (TypeError, ValueError):
            game_number = None
    round_number = snapshot.get("round_number")
    if round_number is not None:
        try:
            round_number = int(round_number)
        except (TypeError, ValueError):
            round_number = None

    # series_fmt: BO3 from bo_type if we had it; snapshot doesn't have bo_type, so default BO3
    series_fmt = "BO3"

    # Team economy: sum of (balance + equipment_value) per player for each team
    def _team_econ(team_obj: dict) -> float:
        players = team_obj.get("player_states") if isinstance(team_obj, dict) else None
        if not isinstance(players, list):
            return 0.0
        total = 0.0
        for p in players:
            if not isinstance(p, dict):
                continue
            b = p.get("balance")
            e = p.get("equipment_value")
            total += float(b if b is not None else 0) + float(e if e is not None else 0)
        return total

    econ1 = _team_econ(t1)
    econ2 = _team_econ(t2)
    if team_a_is_team_one:
        team_a_econ, team_b_econ = econ1, econ2
    else:
        team_a_econ, team_b_econ = econ2, econ1

    return {
        "team_a": team_a,
        "team_b": team_b,
        "rounds_a": rounds_a,
        "rounds_b": rounds_b,
        "maps_a_won": maps_a_won,
        "maps_b_won": maps_b_won,
        "map_name": map_name,
        "a_side": a_side,
        "series_fmt": series_fmt,
        "round_number": round_number,
        "game_number": game_number,
        "team_a_econ": team_a_econ,
        "team_b_econ": team_b_econ,
    }

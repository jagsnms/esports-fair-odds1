"""
Convert raw BO3 snapshot + team_a_is_team_one into engine.Frame.
Defensive: handle missing fields. Uses live-updating fields for teams/scores/maps.
"""
from __future__ import annotations

import time
from typing import Any

from engine.models import Frame


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if not isinstance(obj, dict):
        return default
    return obj.get(key, default)


def _team_id(team: Any) -> int | None:
    """Extract numeric id from team dict if present. Defensive."""
    if not isinstance(team, dict):
        return None
    val = team.get("id")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _team_provider_id(team: Any) -> str | None:
    """Extract provider_id from team dict if present. Defensive."""
    if not isinstance(team, dict):
        return None
    val = team.get("provider_id")
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def _team_name(team: Any) -> str:
    """Prefer fixture.team_name or fixture.name, then team name. Defensive."""
    if not isinstance(team, dict):
        return "Team"
    fixture = _get(team, "fixture")
    if isinstance(fixture, dict):
        tn = _get(fixture, "team_name") or _get(fixture, "name")
        if isinstance(tn, str) and tn.strip():
            return tn.strip()
    name = _get(team, "name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return "Team"


def _normalize_side(side: Any) -> str | None:
    """Return 'T' or 'CT' if side is present and recognized; otherwise None."""
    if side is None:
        return None
    s = str(side).strip().upper()
    if s == "T":
        return "T"
    if s == "CT":
        return "CT"
    return None


def bo3_snapshot_to_frame(raw: dict[str, Any], team_a_is_team_one: bool = True) -> Frame:
    """
    Normalize BO3 snapshot to Frame.
    team_a_is_team_one: True => team A = team_one, team B = team_two; False => swapped.
    Uses team_one/team_two .score (rounds), .match_score (maps); fallback to match_fixture for maps.
    """
    t1 = _get(raw, "team_one") or {}
    t2 = _get(raw, "team_two") or {}
    name1 = _team_name(t1) or "Team 1"
    name2 = _team_name(t2) or "Team 2"
    team_one_id = _team_id(t1)
    team_two_id = _team_id(t2)
    team_one_provider_id = _team_provider_id(t1)
    team_two_provider_id = _team_provider_id(t2)

    # Round score (this map): team_one.score, team_two.score (live)
    score1 = int(t1.get("score", 0)) if t1.get("score") is not None else 0
    score2 = int(t2.get("score", 0)) if t2.get("score") is not None else 0

    # Series score (maps won): team_one.match_score, team_two.match_score; fallback to match_fixture
    match_score1 = t1.get("match_score")
    match_score2 = t2.get("match_score")
    if match_score1 is None or match_score2 is None:
        mf = _get(raw, "match_fixture") or {}
        match_score1 = mf.get("team_one_score", 0) if match_score1 is None else match_score1
        match_score2 = mf.get("team_two_score", 0) if match_score2 is None else match_score2
    match_score1 = int(match_score1) if match_score1 is not None else 0
    match_score2 = int(match_score2) if match_score2 is not None else 0

    if team_a_is_team_one:
        team_a_name, team_b_name = name1, name2
        rounds_a, rounds_b = score1, score2
        maps_a, maps_b = match_score1, match_score2
        team_a_side_raw = t1.get("side")
    else:
        team_a_name, team_b_name = name2, name1
        rounds_a, rounds_b = score2, score1
        maps_a, maps_b = match_score2, match_score1
        team_a_side_raw = t2.get("side")
    a_side = _normalize_side(team_a_side_raw)

    timestamp = time.time()
    map_name = raw.get("map_name", "") if isinstance(raw.get("map_name"), str) else ""
    game_number = raw.get("game_number", 1)
    try:
        map_index = max(0, int(game_number) - 1)
    except (TypeError, ValueError):
        map_index = 0
    bo_type = _get(raw, "bo_type", 3)
    series_fmt = f"bo{int(bo_type)}" if bo_type is not None else "bo3"

    alive1 = alive2 = 0
    hp1 = hp2 = 0.0
    cash1 = cash2 = 0.0
    # Alive-only sums for first-class microstate (cash_totals, loadout_totals, armor_totals)
    cash_alive_1 = cash_alive_2 = 0.0
    loadout_alive_1 = loadout_alive_2 = 0.0
    armor_alive_1 = armor_alive_2 = 0.0
    has_armor_1 = has_armor_2 = False
    ps1 = (t1.get("player_states") or []) if isinstance(t1, dict) else []
    ps2 = (t2.get("player_states") or []) if isinstance(t2, dict) else []
    player_states_present = (
        isinstance(t1, dict) and "player_states" in t1 and isinstance(t2, dict) and "player_states" in t2
    )
    for p in ps1:
        if isinstance(p, dict) and p.get("is_alive"):
            alive1 += 1
            cash_alive_1 += float(p.get("balance", 0) or 0)
            loadout_alive_1 += float(p.get("equipment_value", 0) or 0)
            if p.get("armor") is not None:
                armor_alive_1 += float(p.get("armor", 0) or 0)
                has_armor_1 = True
        if isinstance(p, dict) and p.get("health") is not None:
            hp1 += float(p.get("health", 0))
        if isinstance(p, dict) and p.get("balance") is not None:
            cash1 += float(p.get("balance", 0))
        if isinstance(p, dict) and p.get("equipment_value") is not None:
            cash1 += float(p.get("equipment_value", 0))
    for p in ps2:
        if isinstance(p, dict) and p.get("is_alive"):
            alive2 += 1
            cash_alive_2 += float(p.get("balance", 0) or 0)
            loadout_alive_2 += float(p.get("equipment_value", 0) or 0)
            if p.get("armor") is not None:
                armor_alive_2 += float(p.get("armor", 0) or 0)
                has_armor_2 = True
        if isinstance(p, dict) and p.get("health") is not None:
            hp2 += float(p.get("health", 0))
        if isinstance(p, dict) and p.get("balance") is not None:
            cash2 += float(p.get("balance", 0))
        if isinstance(p, dict) and p.get("equipment_value") is not None:
            cash2 += float(p.get("equipment_value", 0))

    if not team_a_is_team_one:
        alive1, alive2 = alive2, alive1
        hp1, hp2 = hp2, hp1
        cash1, cash2 = cash2, cash1
        cash_alive_1, cash_alive_2 = cash_alive_2, cash_alive_1
        loadout_alive_1, loadout_alive_2 = loadout_alive_2, loadout_alive_1
        armor_alive_1, armor_alive_2 = armor_alive_2, armor_alive_1
        has_armor_1, has_armor_2 = has_armor_2, has_armor_1

    # First-class microstate: set only when player_states were present in raw snapshot
    cash_totals = (cash_alive_1, cash_alive_2) if player_states_present else None
    loadout_totals = (loadout_alive_1, loadout_alive_2) if player_states_present else None
    armor_totals = (armor_alive_1, armor_alive_2) if (player_states_present and (has_armor_1 or has_armor_2)) else None

    # Optional live fields in bomb_phase_time_remaining
    bomb_phase: dict[str, Any] = {}
    for key in ("round_time", "round_time_remaining", "round_phase", "round_number", "is_bomb_planted"):
        val = raw.get(key)
        if val is not None:
            bomb_phase[key] = val
    if not bomb_phase:
        bomb_phase = None

    return Frame(
        timestamp=timestamp,
        teams=(team_a_name, team_b_name),
        scores=(rounds_a, rounds_b),
        alive_counts=(alive1, alive2),
        hp_totals=(hp1, hp2),
        cash_loadout_totals=(cash1, cash2),
        cash_totals=cash_totals,
        loadout_totals=loadout_totals,
        armor_totals=armor_totals,
        bomb_phase_time_remaining=bomb_phase,
        map_index=map_index,
        series_score=(maps_a, maps_b),
        map_name=map_name,
        series_fmt=series_fmt,
        a_side=a_side,
        team_one_id=team_one_id,
        team_two_id=team_two_id,
        team_one_provider_id=team_one_provider_id,
        team_two_provider_id=team_two_provider_id,
    )

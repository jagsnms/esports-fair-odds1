"""
Convert raw BO3 snapshot + team_a_is_team_one into engine.Frame.
Defensive: handle missing fields. Uses live-updating fields for teams/scores/maps.
Wealth decomposition: cash (balance) vs loadout (equipment_value or weapon-price estimate).
"""
from __future__ import annotations

import time
from typing import Any

from engine.models import Frame, PlayerRow
from engine.normalize.time_norm import normalize_round_time

# CS2 approximate prices for loadout fallback when equipment_value is 0 or missing
_WEAPON_PRICES: dict[str, int] = {
    "ak-47": 2700, "m4a4": 3100, "m4a1-s": 3100, "m4a1": 3100,
    "awp": 4750, "ssg08": 1700, "scar-20": 5000, "g3sg1": 5000,
    "galil ar": 1800, "famas": 2050,
    "mp9": 1250, "mac-10": 1050, "mp7": 1500, "mp5-sd": 1500,
    "ump-45": 1200, "p90": 2350, "pp-bizon": 1400,
    "nova": 1050, "xm1014": 2000, "mag-7": 1300, "sawedoff": 1100,
    "glock-18": 200, "usp-s": 200, "p2000": 200, "p250": 300,
    "five-seven": 500, "tec-9": 500, "cz75-auto": 500,
    "dual berettas": 300, "deagle": 700, "revolver": 600,
    "knife": 0, "knife_t": 0, "knife_ct": 0,
}
_ARMOR_KEVLAR = 650
_ARMOR_VESTHELM = 1000
_DEFUSE_KIT = 400


def _estimate_loadout_value(p: dict[str, Any]) -> float:
    """Estimate loadout value from primary_weapon, secondary_weapon, armor/helmet/defuse flags."""
    total = 0.0
    prim = (p.get("primary_weapon") or p.get("primary")) or ""
    if isinstance(prim, str) and prim.strip():
        total += float(_WEAPON_PRICES.get(prim.strip().lower(), 2500))
    sec = (p.get("secondary_weapon") or p.get("secondary")) or ""
    if isinstance(sec, str) and sec.strip():
        total += float(_WEAPON_PRICES.get(sec.strip().lower(), 200))
    if p.get("has_kevlar") or p.get("has_armor"):
        total += _ARMOR_KEVLAR
    if p.get("has_helmet"):
        total += 350 if (p.get("has_kevlar") or p.get("has_armor")) else _ARMOR_VESTHELM
    if p.get("has_defuse_kit"):
        total += _DEFUSE_KIT
    return total


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
    compact = s.replace("-", "_").replace(" ", "_")
    if compact == "TERRORIST":
        return "T"
    if compact in ("COUNTER_TERRORIST", "COUNTERTERRORIST"):
        return "CT"
    return None


def _players_from_states(states: list[dict[str, Any]] | Any) -> list[PlayerRow]:
    """Build PlayerRow list from raw player_states: alive first (source order), then dead."""
    if not isinstance(states, list):
        return []
    alive_rows: list[PlayerRow] = []
    dead_rows: list[PlayerRow] = []
    for p in states:
        if not isinstance(p, dict):
            continue
        name = p.get("nickname") or p.get("player_nickname") or p.get("steam_profile_nickname") or p.get("name")
        name_str = str(name).strip() if isinstance(name, str) else None
        alive_val = p.get("is_alive")
        alive = bool(alive_val) if alive_val is not None else None
        hp = p.get("health")
        hp_f = float(hp) if isinstance(hp, (int, float)) else None
        armor = p.get("armor")
        armor_f = float(armor) if isinstance(armor, (int, float)) else None
        helmet = p.get("has_helmet") if isinstance(p.get("has_helmet"), bool) else None
        cash = p.get("balance")
        cash_f = float(cash) if isinstance(cash, (int, float)) else None
        loadout = p.get("equipment_value")
        loadout_f = float(loadout) if isinstance(loadout, (int, float)) else None
        prim = p.get("primary_weapon")
        sec = p.get("secondary_weapon")
        weapons: list[str] = []
        for w in (prim, sec):
            if isinstance(w, str) and w.strip():
                weapons.append(w.strip())
        inv = p.get("inventory")
        if isinstance(inv, list):
            for w in inv:
                if isinstance(w, str) and w.strip():
                    weapons.append(w.strip())
        weapons_val = weapons if weapons else None
        has_bomb_val = p.get("has_bomb")
        has_bomb = bool(has_bomb_val) if has_bomb_val is not None else None
        has_kit_val = p.get("has_defuse_kit")
        has_kit = bool(has_kit_val) if has_kit_val is not None else None
        row = PlayerRow(
            name=name_str,
            alive=alive,
            hp=hp_f,
            armor=armor_f,
            helmet=helmet,
            cash=cash_f,
            loadout=loadout_f,
            weapons=weapons_val,
            has_bomb=has_bomb,
            has_kit=has_kit,
        )
        if alive is False:
            dead_rows.append(row)
        else:
            alive_rows.append(row)
    return alive_rows + dead_rows


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
    # Alive-only sums: cash = balance, loadout = equipment_value or weapon estimate
    cash_alive_1 = cash_alive_2 = 0.0
    loadout_alive_1 = loadout_alive_2 = 0.0
    n_ev_1 = n_ev_2 = 0
    n_est_1 = n_est_2 = 0
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
            ev = p.get("equipment_value")
            try:
                ev_f = float(ev) if ev is not None else 0.0
            except (TypeError, ValueError):
                ev_f = 0.0
            if ev_f > 0:
                loadout_alive_1 += ev_f
                n_ev_1 += 1
            else:
                loadout_alive_1 += _estimate_loadout_value(p)
                n_est_1 += 1
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
            ev = p.get("equipment_value")
            try:
                ev_f = float(ev) if ev is not None else 0.0
            except (TypeError, ValueError):
                ev_f = 0.0
            if ev_f > 0:
                loadout_alive_2 += ev_f
                n_ev_2 += 1
            else:
                loadout_alive_2 += _estimate_loadout_value(p)
                n_est_2 += 1
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
        n_ev_1, n_ev_2, n_est_1, n_est_2 = n_ev_2, n_ev_1, n_est_2, n_est_1
        armor_alive_1, armor_alive_2 = armor_alive_2, armor_alive_1
        has_armor_1, has_armor_2 = has_armor_2, has_armor_1

    # First-class microstate: set only when player_states were present in raw snapshot
    cash_totals = (cash_alive_1, cash_alive_2) if player_states_present else None
    loadout_totals = (loadout_alive_1, loadout_alive_2) if player_states_present else None
    wealth_totals = (
        (cash_alive_1 + loadout_alive_1, cash_alive_2 + loadout_alive_2)
        if player_states_present
        else None
    )
    total_ev = n_ev_1 + n_ev_2
    total_est = n_est_1 + n_est_2
    if player_states_present and (total_ev + total_est) > 0:
        if total_est == 0:
            loadout_source = "ev"
        elif total_ev == 0:
            loadout_source = "weapon_est"
        else:
            loadout_source = "mixed"
    else:
        loadout_source = None
    armor_totals = (armor_alive_1, armor_alive_2) if (player_states_present and (has_armor_1 or has_armor_2)) else None

    # Round time: normalize at ingest (ms -> seconds), canonical on Frame
    rtr_norm = normalize_round_time(raw.get("round_time_remaining"))
    rt_norm = normalize_round_time(raw.get("round_time"))
    round_time_remaining_s = rtr_norm.get("seconds")
    round_time_s = rt_norm.get("seconds")
    round_time_remaining_raw = raw.get("round_time_remaining") if rtr_norm.get("raw") is not None else None
    round_time_raw = raw.get("round_time") if rt_norm.get("raw") is not None else None

    # Optional live fields in bomb_phase_time_remaining (use normalized seconds for time)
    bomb_phase: dict[str, Any] = {}
    if round_time_remaining_s is not None:
        bomb_phase["round_time_remaining"] = round_time_remaining_s
    if round_time_s is not None:
        bomb_phase["round_time"] = round_time_s
    for key in ("round_phase", "round_number", "is_bomb_planted"):
        val = raw.get(key)
        if val is not None:
            bomb_phase[key] = val
    if not bomb_phase:
        bomb_phase = None

    # Player rows for HUD: map raw team_one/team_two player_states into Team A/B slots
    players_team_one = _players_from_states(ps1)
    players_team_two = _players_from_states(ps2)
    if team_a_is_team_one:
        players_a = players_team_one
        players_b = players_team_two
    else:
        players_a = players_team_two
        players_b = players_team_one

    # Flags for round_time_remaining normalization
    rtr_invalid = rtr_norm.get("invalid_reason")
    rtr_is_ms = bool(rtr_norm.get("is_ms"))
    rtr_raw = rtr_norm.get("raw")
    rtr_was_negative = bool(rtr_norm.get("was_negative"))
    round_time_remaining_was_ms = rtr_is_ms if rtr_raw is not None else None
    round_time_remaining_was_negative = rtr_was_negative if rtr_raw is not None else None
    round_time_remaining_was_out_of_range = (rtr_invalid == "out_of_range") if rtr_raw is not None else None
    round_time_remaining_was_missing = rtr_raw is None

    return Frame(
        timestamp=timestamp,
        teams=(team_a_name, team_b_name),
        scores=(rounds_a, rounds_b),
        alive_counts=(alive1, alive2),
        hp_totals=(hp1, hp2),
        cash_loadout_totals=(cash1, cash2),
        cash_totals=cash_totals,
        loadout_totals=loadout_totals,
        wealth_totals=wealth_totals,
        armor_totals=armor_totals,
        loadout_source=loadout_source,
        loadout_ev_count_a=n_ev_1 if player_states_present else None,
        loadout_ev_count_b=n_ev_2 if player_states_present else None,
        loadout_est_count_a=n_est_1 if player_states_present else None,
        loadout_est_count_b=n_est_2 if player_states_present else None,
        bomb_phase_time_remaining=bomb_phase,
        round_time_remaining_s=round_time_remaining_s,
        round_time_s=round_time_s,
        round_time_remaining_raw=round_time_remaining_raw,
        round_time_raw=round_time_raw,
        round_time_remaining_was_ms=round_time_remaining_was_ms,
        round_time_remaining_was_negative=round_time_remaining_was_negative,
        round_time_remaining_was_out_of_range=round_time_remaining_was_out_of_range,
        round_time_remaining_was_missing=round_time_remaining_was_missing,
        map_index=map_index,
        series_score=(maps_a, maps_b),
        map_name=map_name,
        series_fmt=series_fmt,
        a_side=a_side,
        team_one_id=team_one_id,
        team_two_id=team_two_id,
        team_one_provider_id=team_one_provider_id,
        team_two_provider_id=team_two_provider_id,
        players_a=players_a,
        players_b=players_b,
    )

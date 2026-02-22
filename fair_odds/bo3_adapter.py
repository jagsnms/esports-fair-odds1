"""
BO3.gg / cs2api adapter: sync wrappers for live matches and snapshot, and normalizer to app state shape.
"""
from __future__ import annotations

import asyncio
import re
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


# BO3 LOADOUT VALUE FIX — Live CS2 BO3 helper: price tables + alias normalization
# (equipment_value from feed often 0; we derive loadout value from weapons/armor/kit)
CS2_PRIMARY_WEAPON_PRICES = {
    "ak47": 2700,
    "m4a4": 2900,
    "m4a1_s": 2900,
    "m4a1_silencer": 2900,
    "famas": 1950,
    "galilar": 1800,
    "galil": 1800,
    "aug": 3300,
    "sg553": 3000,
    "krieg": 3000,
    "awp": 4750,
    "ssg08": 1700,
    "scout": 1700,
    "scar20": 5000,
    "g3sg1": 5000,
    "mac10": 1050,
    "mp9": 1250,
    "mp7": 1500,
    "mp5sd": 1500,
    "ump45": 1200,
    "p90": 2350,
    "bizon": 1400,
    "ppbizon": 1400,
    "nova": 1050,
    "xm1014": 2000,
    "mag7": 1300,
    "sawedoff": 1100,
    "negev": 1700,
    "m249": 5200,
}

CS2_SECONDARY_WEAPON_PRICES = {
    "glock": 0,
    "glock18": 0,
    "usp_s": 0,
    "usp_silencer": 0,
    "usp": 0,
    "hkp2000": 0,
    "p2000": 0,
    "elite": 300,
    "dualberettas": 300,
    "p250": 300,
    "fiveseven": 500,
    "five_seven": 500,
    "tec9": 500,
    "cz75a": 500,
    "cz75_auto": 500,
    "deagle": 700,
    "deserteagle": 700,
    "revolver": 600,
    "r8revolver": 600,
}

CS2_EQUIPMENT_PRICES = {
    "kevlar": 650,
    "helmet": 350,
    "defuse_kit": 400,
    "zeus": 200,
}

CS2_GRENADE_PRICES = {
    "flashbang": 200,
    "hegrenade": 300,
    "smokegrenade": 300,
    "molotov": 400,
    "incgrenade": 500,
    "decoy": 50,
}

CS2_ITEM_ALIASES = {
    "m4a1s": "m4a1_s",
    "m4a1_silencer_off": "m4a1_s",
    "m4a1_silencer": "m4a1_s",
    "m4a1": "m4a4",
    "galil_ar": "galilar",
    "sg_553": "sg553",
    "ssg_08": "ssg08",
    "mp5_sd": "mp5sd",
    "ump_45": "ump45",
    "pp_bizon": "ppbizon",
    "dual_berettas": "dualberettas",
    "five_seven": "five_seven",
    "cz75_auto": "cz75_auto",
    "desert_eagle": "deserteagle",
    "r8_revolver": "r8revolver",
    "usp-s": "usp_s",
    "usp_s": "usp_s",
    "usp_silencer": "usp_s",
    "glock_18": "glock18",
    "he_grenade": "hegrenade",
    "smoke_grenade": "smokegrenade",
    "molotov_grenade": "molotov",
    "incendiary": "incgrenade",
    "incendiary_grenade": "incgrenade",
    "defusekit": "defuse_kit",
    "kit": "defuse_kit",
    "taser": "zeus",
    "zeus_x27": "zeus",
    # BO3 LOADOUT VALUE FIX — underscore variants (e.g. "AK-47" -> ak_47; need to map to ak47)
    "ak_47": "ak47",
    "awp_": "awp",
    "scar_20": "scar20",
    "mag_7": "mag7",
    "bizon": "ppbizon",
}


def _normalize_cs2_item_name(name: Any) -> str:
    # BO3 LOADOUT VALUE FIX — normalize weapon/item name for price lookup
    if name is None or (isinstance(name, str) and not name.strip()):
        return ""
    s = str(name).strip().lower()
    s = s.replace("-", "_").replace(" ", "_").replace(".", "_")
    if s.startswith("weapon_"):
        s = s[7:]
    s = re.sub(r"_+", "_", s).strip("_")
    return CS2_ITEM_ALIASES.get(s, s)


def _player_weapon_name_strings(player: dict) -> List[str]:
    # BO3 LOADOUT VALUE FIX — collect all weapon/item names from any field the feed might use
    names: List[str] = []
    # Single-weapon fields
    for key in ("primary_weapon", "secondary_weapon", "primary", "secondary", "weapon", "pistol", "current_weapon", "active_weapon", "weapon_1", "weapon_2"):
        v = player.get(key)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            names.append(v.strip())
        elif isinstance(v, dict):
            n = v.get("name") or v.get("weapon_name") or v.get("id") or v.get("weapon")
            if n and str(n).strip():
                names.append(str(n).strip())
    # List fields: weapons, inventory, equipment
    for key in ("weapons", "inventory", "equipment", "weapon_list"):
        lst = player.get(key)
        if not isinstance(lst, list):
            continue
        for item in lst:
            if isinstance(item, str) and item.strip():
                names.append(item.strip())
            elif isinstance(item, dict):
                n = item.get("name") or item.get("weapon_name") or item.get("id") or item.get("weapon")
                if n and str(n).strip():
                    names.append(str(n).strip())
    return names


def _compute_cs2_player_loadout_est(player: dict) -> dict:
    # BO3 LOADOUT VALUE FIX — estimate buy value from weapons + armor/kit (do not use equipment_value)
    out = {
        "loadout_est_total": 0.0,
        "primary_value": 0.0,
        "secondary_value": 0.0,
        "armor_value": 0.0,
        "kit_value": 0.0,
        "unknown_items": [],
    }
    if not isinstance(player, dict):
        return out
    # Collect all weapon names from every possible feed field, then sum values
    for raw_name in _player_weapon_name_strings(player):
        key = _normalize_cs2_item_name(raw_name)
        if not key:
            continue
        val = CS2_PRIMARY_WEAPON_PRICES.get(key)
        if val is not None:
            out["primary_value"] += float(val)
            continue
        val = CS2_SECONDARY_WEAPON_PRICES.get(key)
        if val is not None:
            out["secondary_value"] += float(val)
            continue
        # Equipment (kevlar/helmet/defuse/zeus) usually come from booleans; optional name lookup
        val = CS2_EQUIPMENT_PRICES.get(key)
        if val is not None:
            if key in ("kevlar", "helmet"):
                out["armor_value"] += float(val)
            else:
                out["kit_value"] += float(val)
            continue
        out["unknown_items"].append(key)
    # Armor: has_kevlar + helmet or full 1000 (in case feed doesn't list "kevlar" by name)
    has_kevlar = bool(player.get("has_kevlar"))
    has_helmet = bool(player.get("has_helmet"))
    if has_kevlar:
        out["armor_value"] += float(CS2_EQUIPMENT_PRICES.get("kevlar", 650))
        if has_helmet:
            out["armor_value"] += float(CS2_EQUIPMENT_PRICES.get("helmet", 350))
    elif has_helmet:
        out["armor_value"] = 1000.0
    # Kit
    if bool(player.get("has_defuse_kit")):
        out["kit_value"] += float(CS2_EQUIPMENT_PRICES.get("defuse_kit", 400))
    if bool(player.get("has_zeus")):
        out["kit_value"] += float(CS2_EQUIPMENT_PRICES.get("zeus", 200))
    out["loadout_est_total"] = out["primary_value"] + out["secondary_value"] + out["armor_value"] + out["kit_value"]
    return out


def _compute_cs2_team_resource_totals(players: list) -> dict:
    # BO3 LOADOUT VALUE FIX — team totals: cash, loadout est, total resources; alive-only variants
    result = {
        "cash_total": 0.0,
        "loadout_est_total": 0.0,
        "total_resources": 0.0,
        "alive_cash_total": 0.0,
        "alive_loadout_est_total": 0.0,
        "alive_total_resources": 0.0,
        "alive_count": 0,
        "hp_alive_total": 0.0,
        "unknown_items": [],
    }
    if not isinstance(players, list):
        return result
    for p in players:
        if not isinstance(p, dict):
            continue
        cash = 0.0
        for k in ("balance", "cash", "money"):
            v = p.get(k)
            if v is not None:
                try:
                    cash = float(v)
                    break
                except (TypeError, ValueError):
                    pass
        loadout = _compute_cs2_player_loadout_est(p)
        is_alive = bool(p.get("is_alive") if p.get("is_alive") is not None else (p.get("health", 0) or 0) > 0)
        hp = 0.0
        try:
            hp = float(p.get("health") if p.get("health") is not None else 0)
        except (TypeError, ValueError):
            pass
        result["cash_total"] += cash
        result["loadout_est_total"] += loadout["loadout_est_total"]
        result["total_resources"] += cash + loadout["loadout_est_total"]
        result["unknown_items"].extend(loadout["unknown_items"])
        if is_alive:
            result["alive_cash_total"] += cash
            result["alive_loadout_est_total"] += loadout["loadout_est_total"]
            result["alive_total_resources"] += cash + loadout["loadout_est_total"]
            result["alive_count"] += 1
            result["hp_alive_total"] += hp
    return result


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

    # Team economy: sum of (cash + equipment value) per player for each team.
    # Locked-at-round-start is applied by the app; here we just compute from raw snapshot.
    def _player_num(p: dict, *keys: str) -> float:
        for k in keys:
            v = p.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _team_econ(team_obj: dict) -> float:
        players = team_obj.get("player_states") if isinstance(team_obj, dict) else None
        if not isinstance(players, list):
            return 0.0
        total = 0.0
        for p in players:
            if not isinstance(p, dict):
                continue
            total += _player_num(p, "balance", "cash", "money") + _player_num(p, "equipment_value", "equipment")
        return total

    econ1 = _team_econ(t1)
    econ2 = _team_econ(t2)
    if team_a_is_team_one:
        team_a_econ, team_b_econ = econ1, econ2
    else:
        team_a_econ, team_b_econ = econ2, econ1

    # BO3 LOADOUT VALUE FIX — derive team cash/loadout/total resources (and alive-only) from player_states
    players1 = t1.get("player_states") if isinstance(t1.get("player_states"), list) else []
    players2 = t2.get("player_states") if isinstance(t2.get("player_states"), list) else []
    tot1 = _compute_cs2_team_resource_totals(players1)
    tot2 = _compute_cs2_team_resource_totals(players2)
    loadout_derived_valid = bool(players1 or players2)  # only treat as valid when we have at least one team with players
    if snapshot.get("match_ended") or snapshot.get("game_ended"):
        loadout_derived_valid = False
    gs = snapshot.get("game_state")
    if gs is not None and str(gs).lower() in ("ended", "finished", "postgame"):
        loadout_derived_valid = False
    if team_a_is_team_one:
        team_a_cash_total, team_b_cash_total = tot1["cash_total"], tot2["cash_total"]
        team_a_loadout_est_total, team_b_loadout_est_total = tot1["loadout_est_total"], tot2["loadout_est_total"]
        team_a_total_resources, team_b_total_resources = tot1["total_resources"], tot2["total_resources"]
        team_a_alive_cash_total, team_b_alive_cash_total = tot1["alive_cash_total"], tot2["alive_cash_total"]
        team_a_alive_loadout_est_total, team_b_alive_loadout_est_total = tot1["alive_loadout_est_total"], tot2["alive_loadout_est_total"]
        team_a_alive_total_resources, team_b_alive_total_resources = tot1["alive_total_resources"], tot2["alive_total_resources"]
        team_a_alive_count, team_b_alive_count = tot1["alive_count"], tot2["alive_count"]
        team_a_hp_alive_total, team_b_hp_alive_total = tot1["hp_alive_total"], tot2["hp_alive_total"]
        unknown_items = list(dict.fromkeys(tot1["unknown_items"] + tot2["unknown_items"]))
    else:
        team_a_cash_total, team_b_cash_total = tot2["cash_total"], tot1["cash_total"]
        team_a_loadout_est_total, team_b_loadout_est_total = tot2["loadout_est_total"], tot1["loadout_est_total"]
        team_a_total_resources, team_b_total_resources = tot2["total_resources"], tot1["total_resources"]
        team_a_alive_cash_total, team_b_alive_cash_total = tot2["alive_cash_total"], tot1["alive_cash_total"]
        team_a_alive_loadout_est_total, team_b_alive_loadout_est_total = tot2["alive_loadout_est_total"], tot1["alive_loadout_est_total"]
        team_a_alive_total_resources, team_b_alive_total_resources = tot2["alive_total_resources"], tot1["alive_total_resources"]
        team_a_alive_count, team_b_alive_count = tot2["alive_count"], tot1["alive_count"]
        team_a_hp_alive_total, team_b_hp_alive_total = tot2["hp_alive_total"], tot1["hp_alive_total"]
        unknown_items = list(dict.fromkeys(tot2["unknown_items"] + tot1["unknown_items"]))

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
        # BO3 LOADOUT VALUE FIX
        "loadout_derived_valid": loadout_derived_valid,
        "team_a_cash_total": team_a_cash_total,
        "team_b_cash_total": team_b_cash_total,
        "team_a_loadout_est_total": team_a_loadout_est_total,
        "team_b_loadout_est_total": team_b_loadout_est_total,
        "team_a_total_resources": team_a_total_resources,
        "team_b_total_resources": team_b_total_resources,
        "team_a_alive_cash_total": team_a_alive_cash_total,
        "team_b_alive_cash_total": team_b_alive_cash_total,
        "team_a_alive_loadout_est_total": team_a_alive_loadout_est_total,
        "team_b_alive_loadout_est_total": team_b_alive_loadout_est_total,
        "team_a_alive_total_resources": team_a_alive_total_resources,
        "team_b_alive_total_resources": team_b_alive_total_resources,
        "team_a_alive_count": team_a_alive_count,
        "team_b_alive_count": team_b_alive_count,
        "team_a_hp_alive_total": team_a_hp_alive_total,
        "team_b_hp_alive_total": team_b_hp_alive_total,
        "unknown_items": unknown_items,
        # BO3 MIDROUND V1 — optional intraround features (from snapshot when available)
        "bomb_planted": bool(snapshot.get("bomb_planted") or snapshot.get("bomb_planted_at") is not None),
        "round_time_remaining_s": snapshot.get("round_time_remaining") or snapshot.get("round_time_remaining_s"),
        "round_phase": snapshot.get("round_phase") or snapshot.get("phase"),
        # BO3 terminal flags (for live terminal override when scoreboard lags)
        "game_ended": bool(snapshot.get("game_ended")),
        "match_ended": bool(snapshot.get("match_ended")),
        "match_status": str(snapshot.get("match_status") or "").strip().lower(),
    }

# GRID PROBE V1 — Normalization probe: live Central → seriesState → app-friendly normalized preview.
"""
Run from project root. Single run, no retries, no polling.
  python -m adapters.grid_probe.grid_normalize_series_state_probe

1) Central allSeries(filter: {live: {games: {}}}, first=10)
2) First series ID where seriesState(id) has data and title.nameShortened == "cs2"
3) Save raw Central + raw Series State; build normalized dict; save and print preview.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
RAW_CENTRAL_PATH = PROBE_DIR / "raw_grid_central_data.json"
RAW_SERIES_STATE_PATH = PROBE_DIR / "raw_grid_series_state.json"
NORMALIZED_PREVIEW_PATH = PROBE_DIR / "raw_grid_series_state_normalized_preview.json"

# GRID PROBE V2 — use V2 seriesState query (docs/GRID_CS2_NORMALIZED_FEATURE_CONTRACT.md)
USE_RICH_QUERY: bool = True

# GRID PROBE V1 — Central query vars (fixed for this probe)
CENTRAL_VARS = {
    "orderBy": "StartTimeScheduled",
    "orderDirection": "DESC",
    "first": 10,
    "filter": {"live": {"games": {}}},
}


def _save_json(path: Path, data: dict, pretty: bool = True) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


# V2 contract required fields for completeness_score (count present / total)
_V2_REQUIRED_KEYS = (
    "source",
    "series_id",
    "valid",
    "updated_at",
    "started",
    "finished",
    "game_index",
    "game_started",
    "game_finished",
    "rounds_a",
    "rounds_b",
    "map_name",
    "team_a_id",
    "team_b_id",
)


def _parse_updated_at(updated_at: Any) -> datetime | None:
    """Parse updated_at (ISO datetime); return None if missing/invalid. Fallback-safe."""
    if updated_at is None:
        return None
    if isinstance(updated_at, datetime):
        return updated_at
    if not isinstance(updated_at, str):
        return None
    try:
        s = updated_at.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _build_v2_contract(ss: dict) -> dict[str, Any]:
    """Build V2 normalized contract from raw seriesState. Fallback-safe; never crash on missing fields."""
    v2: dict[str, Any] = {}
    games = ss.get("games")
    if not isinstance(games, list) or not games:
        v2["source"] = "grid"
        v2["series_id"] = ss.get("id")
        v2["valid"] = ss.get("valid")
        v2["updated_at"] = ss.get("updatedAt")
        v2["started"] = ss.get("started")
        v2["finished"] = ss.get("finished")
        v2["game_index"] = None
        v2["game_started"] = None
        v2["game_finished"] = None
        v2["rounds_a"] = None
        v2["rounds_b"] = None
        v2["map_name"] = None
        v2["team_a_id"] = None
        v2["team_b_id"] = None
        v2["team_a_side"] = None
        v2["team_b_side"] = None
        v2["players_a"] = []
        v2["players_b"] = []
        v2["team_a_money"] = None
        v2["team_b_money"] = None
        v2["team_a_loadout_value"] = None
        v2["team_b_loadout_value"] = None
        v2["clock_seconds"] = None
        v2["clock_ticking"] = None
        v2["clock_type"] = None
        v2["clock_ticks_backwards"] = None
        v2["alive_count_a"] = None
        v2["alive_count_b"] = None
        v2["hp_total_a"] = None
        v2["hp_total_b"] = None
        v2["armor_total_a"] = None
        v2["armor_total_b"] = None
        v2["has_players"] = False
        v2["has_clock"] = False
        v2["completeness_score"] = 0.0
        v2["staleness_seconds"] = None
        v2["maps_a_won"] = None
        v2["maps_b_won"] = None
        return v2

    # Use current game: started and not finished (map 2+); else last game; else first
    current_game: dict[str, Any] = {}
    for g in reversed(games) if isinstance(games, list) else []:
        if isinstance(g, dict) and g.get("started") and not g.get("finished"):
            current_game = g
            break
    if not current_game and isinstance(games, list) and games:
        current_game = games[-1] if isinstance(games[-1], dict) else (games[0] if isinstance(games[0], dict) else {})
    if not current_game and isinstance(games, list) and games and isinstance(games[0], dict):
        current_game = games[0]
    first_game = current_game

    v2["source"] = "grid"
    v2["series_id"] = ss.get("id")
    v2["valid"] = ss.get("valid")
    v2["updated_at"] = ss.get("updatedAt")
    v2["started"] = ss.get("started")
    v2["finished"] = ss.get("finished")
    v2["game_index"] = first_game.get("sequenceNumber")
    v2["game_started"] = first_game.get("started")
    v2["game_finished"] = first_game.get("finished")

    # Series-level maps won (from seriesState.teams[].score)
    series_teams = ss.get("teams")
    if isinstance(series_teams, list) and len(series_teams) >= 2:
        st0 = series_teams[0] if isinstance(series_teams[0], dict) else {}
        st1 = series_teams[1] if isinstance(series_teams[1], dict) else {}
        v2["maps_a_won"] = st0.get("score") if st0.get("score") is not None else None
        v2["maps_b_won"] = st1.get("score") if st1.get("score") is not None else None
    else:
        v2["maps_a_won"] = None
        v2["maps_b_won"] = None

    map_node = first_game.get("map")
    v2["map_name"] = map_node.get("name") if isinstance(map_node, dict) else None

    game_teams = first_game.get("teams")
    if isinstance(game_teams, list) and len(game_teams) >= 2:
        t0, t1 = game_teams[0] if isinstance(game_teams[0], dict) else {}, game_teams[1] if isinstance(game_teams[1], dict) else {}
        v2["team_a_id"] = t0.get("id")
        v2["team_b_id"] = t1.get("id")
        v2["team_a_name"] = t0.get("name")
        v2["team_b_name"] = t1.get("name")
        v2["team_a_side"] = t0.get("side")
        v2["team_b_side"] = t1.get("side")
        v2["rounds_a"] = t0.get("score") if t0.get("score") is not None else None
        v2["rounds_b"] = t1.get("score") if t1.get("score") is not None else None
        v2["team_a_money"] = t0.get("money")
        v2["team_b_money"] = t1.get("money")
        v2["team_a_loadout_value"] = t0.get("loadoutValue")
        v2["team_b_loadout_value"] = t1.get("loadoutValue")

        players_a: list[dict[str, Any]] = []
        players_b: list[dict[str, Any]] = []
        for p in t0.get("players") or []:
            if isinstance(p, dict):
                players_a.append({
                    "id": p.get("id"),
                    "alive": p.get("alive"),
                    "current_health": p.get("currentHealth"),
                    "current_armor": p.get("currentArmor"),
                    "money": p.get("money"),
                    "loadout_value": p.get("loadoutValue"),
                })
        for p in t1.get("players") or []:
            if isinstance(p, dict):
                players_b.append({
                    "id": p.get("id"),
                    "alive": p.get("alive"),
                    "current_health": p.get("currentHealth"),
                    "current_armor": p.get("currentArmor"),
                    "money": p.get("money"),
                    "loadout_value": p.get("loadoutValue"),
                })
        v2["players_a"] = players_a
        v2["players_b"] = players_b
    else:
        v2["team_a_id"] = None
        v2["team_b_id"] = None
        v2["team_a_name"] = None
        v2["team_b_name"] = None
        v2["rounds_a"] = None
        v2["rounds_b"] = None
        v2["team_a_money"] = None
        v2["team_b_money"] = None
        v2["team_a_loadout_value"] = None
        v2["team_b_loadout_value"] = None
        v2["players_a"] = []
        v2["players_b"] = []

    # Most recent pistol winner (round 1 or round 13 in MR12) for UI
    v2["pistol_a_won"] = None
    v2["pistol_b_won"] = None
    segments = first_game.get("segments") if isinstance(first_game, dict) else None
    if isinstance(segments, list) and v2.get("team_a_id") is not None and v2.get("team_b_id") is not None:
        def _segment_winner(seg_list: list, seq_num: int) -> Any:
            for seg in seg_list:
                if not isinstance(seg, dict):
                    continue
                if seg.get("sequenceNumber") == seq_num:
                    for st in seg.get("teams") or []:
                        if isinstance(st, dict) and st.get("won") is True:
                            return st.get("id")
                    return None
            return None
        r1_winner = _segment_winner(segments, 1)
        r13_winner = _segment_winner(segments, 13)
        ra = v2.get("rounds_a")
        rb = v2.get("rounds_b")
        total_rounds = (int(ra) + int(rb)) if (ra is not None and rb is not None) else None
        if total_rounds is not None:
            if total_rounds < 13 and r1_winner is not None:
                v2["pistol_a_won"] = str(r1_winner) == str(v2["team_a_id"])
                v2["pistol_b_won"] = str(r1_winner) == str(v2["team_b_id"])
            elif total_rounds >= 13 and r13_winner is not None:
                v2["pistol_a_won"] = str(r13_winner) == str(v2["team_a_id"])
                v2["pistol_b_won"] = str(r13_winner) == str(v2["team_b_id"])

    cl = first_game.get("clock")
    if isinstance(cl, dict):
        v2["clock_seconds"] = cl.get("currentSeconds")
        v2["clock_ticking"] = cl.get("ticking")
        v2["clock_type"] = cl.get("type")
        v2["clock_ticks_backwards"] = cl.get("ticksBackwards")
    else:
        v2["clock_seconds"] = None
        v2["clock_ticking"] = None
        v2["clock_type"] = None
        v2["clock_ticks_backwards"] = None

    # Derived: alive_count_a, alive_count_b
    pa = v2.get("players_a") or []
    pb = v2.get("players_b") or []
    v2["alive_count_a"] = sum(1 for p in pa if p.get("alive") is True) if pa else None
    v2["alive_count_b"] = sum(1 for p in pb if p.get("alive") is True) if pb else None

    # Derived: hp_total_a, hp_total_b (sum current_health for alive players only)
    v2["hp_total_a"] = sum(p.get("current_health") or 0 for p in pa if p.get("alive") is True) if pa else None
    v2["hp_total_b"] = sum(p.get("current_health") or 0 for p in pb if p.get("alive") is True) if pb else None

    # Derived: armor_total_a, armor_total_b (sum current_armor for alive players only; missing -> 0)
    v2["armor_total_a"] = sum(p.get("current_armor") or 0 for p in pa if p.get("alive") is True) if pa else None
    v2["armor_total_b"] = sum(p.get("current_armor") or 0 for p in pb if p.get("alive") is True) if pb else None

    # Derived: has_players, has_clock
    v2["has_players"] = bool(pa and pb)
    v2["has_clock"] = v2.get("clock_seconds") is not None or v2.get("clock_type") is not None

    # Derived: completeness_score (fraction of required V2 fields present and non-null)
    present = sum(1 for k in _V2_REQUIRED_KEYS if v2.get(k) is not None)
    v2["completeness_score"] = round(present / len(_V2_REQUIRED_KEYS), 4) if _V2_REQUIRED_KEYS else 0.0

    # Derived: staleness_seconds (now - updated_at); None if updated_at missing/invalid
    updated_dt = _parse_updated_at(v2.get("updated_at"))
    if updated_dt is not None:
        now = datetime.now(timezone.utc)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=timezone.utc)
        v2["staleness_seconds"] = max(0, int((now - updated_dt).total_seconds()))
    else:
        v2["staleness_seconds"] = None

    return v2


def _extract_series_ids(payload: dict) -> list[str]:
    # GRID PROBE V1 — data.allSeries.edges[].node.id
    ids: list[str] = []
    data = payload.get("data") or {}
    all_series = data.get("allSeries")
    if not isinstance(all_series, dict):
        return ids
    edges = all_series.get("edges")
    if not isinstance(edges, list):
        return ids
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node = edge.get("node")
        if isinstance(node, dict) and node.get("id") is not None:
            ids.append(str(node["id"]))
    return ids


def _normalize_series_state(ss: dict) -> dict[str, Any]:
    # GRID PROBE V1 — extraction per docs/GRID_API_SPEC.md (gridapi.docx). Defensive .get(); never crash.
    out: dict[str, Any] = {}
    unavailable: list[str] = []

    out["series_id"] = ss.get("id")
    out["version"] = ss.get("version")
    out["startedAt"] = ss.get("startedAt")
    out["duration"] = ss.get("duration")
    title = ss.get("title")
    if isinstance(title, dict):
        out["title_short"] = title.get("nameShortened")
    else:
        out["title_short"] = title if isinstance(title, str) else None
    out["valid"] = ss.get("valid")
    out["updated_at"] = ss.get("updatedAt")
    out["started"] = ss.get("started")
    out["finished"] = ss.get("finished")
    fmt = ss.get("format")
    if isinstance(fmt, dict):
        out["format"] = {"_keys": list(fmt.keys())}
        for k in ("name", "value", "slug", "id"):
            if k in fmt and isinstance(fmt[k], str):
                out["format"]["_str"] = fmt[k]
                break
    else:
        out["format"] = fmt

    teams = ss.get("teams")
    games = ss.get("games")
    out["teams_count"] = len(teams) if isinstance(teams, list) else None
    out["games_count"] = len(games) if isinstance(games, list) else None
    team_ids: list[Any] = []
    team_names: list[Any] = []
    round_scores: list[Any] = []
    # GRID PROBE V1 — series-level teams (SeriesTeamStateCsgo): id, name, score
    series_teams_normalized: list[dict[str, Any]] = []
    if isinstance(teams, list):
        for t in teams:
            if isinstance(t, dict) and t.get("id") is not None:
                team_ids.append(t["id"])
            if isinstance(t, dict):
                st = {"id": t.get("id"), "name": t.get("name"), "score": t.get("score"), "won": t.get("won")}
                series_teams_normalized.append(st)
                if t.get("name") is not None:
                    team_names.append(t.get("name"))
                if t.get("score") is not None:
                    round_scores.append(t.get("score"))
    out["team_ids"] = team_ids
    out["team_names"] = team_names
    out["round_scores"] = round_scores if round_scores else None
    out["series_teams_normalized"] = series_teams_normalized

    game_ids: list[Any] = []
    games_normalized: list[dict[str, Any]] = []
    map_name: Any = None
    clock_phase: Any = None
    hp_totals: list[Any] = []
    money_totals: list[Any] = []
    alive_counts: list[Any] = []

    if isinstance(games, list):
        for g in games:
            if isinstance(g, dict) and g.get("id") is not None:
                game_ids.append(g["id"])
        if games and isinstance(games[0], dict):
            first_game = games[0]
            m = first_game.get("map")
            if isinstance(m, dict):
                map_name = m.get("name")
            cl = first_game.get("clock")
            if isinstance(cl, dict):
                clock_phase = {
                    "currentSeconds": cl.get("currentSeconds"),
                    "ticking": cl.get("ticking"),
                    "type": cl.get("type"),
                }

    out["map_name"] = map_name
    out["clock_phase"] = clock_phase
    out["raw_top_level_keys"] = list(ss.keys()) if isinstance(ss, dict) else []
    raw_team_keys: list[str] = []
    if isinstance(teams, list) and teams and isinstance(teams[0], dict):
        raw_team_keys = list(teams[0].keys())
    out["raw_team_keys_sample"] = raw_team_keys
    raw_game_keys: list[str] = []
    if isinstance(games, list) and games and isinstance(games[0], dict):
        raw_game_keys = list(games[0].keys())
    out["raw_game_keys_sample"] = raw_game_keys

    # GRID PROBE V1 — games[].teams[].players: alive, HP, money, side, score, clock/phase
    for gi, g in enumerate(games if isinstance(games, list) else []):
        if not isinstance(g, dict):
            continue
        gn: dict[str, Any] = {
            "id": g.get("id"),
            "sequenceNumber": g.get("sequenceNumber"),
            "map_name": None,
            "started": g.get("started"),
            "finished": g.get("finished"),
            "paused": g.get("paused"),
            "forfeited": g.get("forfeited"),
            "clock_seconds": None,
            "clock_ticking": None,
            "clock_type": None,
            "teams": [],
        }
        gm = g.get("map")
        if isinstance(gm, dict):
            gn["map_name"] = gm.get("name")
        gc = g.get("clock")
        if isinstance(gc, dict):
            gn["clock_seconds"] = gc.get("currentSeconds")
            gn["clock_ticking"] = gc.get("ticking")
            gn["clock_type"] = gc.get("type")
        game_teams = g.get("teams")
        if isinstance(game_teams, list):
            for gt in game_teams:
                if not isinstance(gt, dict):
                    continue
                gtn: dict[str, Any] = {
                    "id": gt.get("id"),
                    "name": gt.get("name"),
                    "score": gt.get("score"),
                    "won": gt.get("won"),
                    "side": gt.get("side"),
                    "money": gt.get("money"),
                    "loadoutValue": gt.get("loadoutValue"),
                    "netWorth": gt.get("netWorth"),
                    "players": [],
                }
                players = gt.get("players")
                if isinstance(players, list):
                    for p in players:
                        if not isinstance(p, dict):
                            continue
                        pn: dict[str, Any] = {
                            "id": p.get("id"),
                            "name": p.get("name"),
                            "alive": p.get("alive"),
                            "currentHealth": p.get("currentHealth"),
                            "currentArmor": p.get("currentArmor"),
                            "money": p.get("money"),
                            "loadoutValue": p.get("loadoutValue"),
                            "netWorth": p.get("netWorth"),
                            "kills": p.get("kills"),
                            "deaths": p.get("deaths"),
                        }
                        gtn["players"].append(pn)
                        if p.get("currentHealth") is not None:
                            hp_totals.append(p.get("currentHealth"))
                        if p.get("money") is not None:
                            money_totals.append(p.get("money"))
                        if p.get("alive") is not None:
                            alive_counts.append(1 if p.get("alive") else 0)
                gtn["alive_count"] = sum(1 for x in gtn["players"] if x.get("alive") is True) if gtn["players"] else None
                gn["teams"].append(gtn)
        games_normalized.append(gn)

    out["games_normalized"] = games_normalized
    out["player_data_summaries"] = []
    for gn in games_normalized:
        for t in gn.get("teams") or []:
            for p in t.get("players") or []:
                out["player_data_summaries"].append({"id": p.get("id"), "name": p.get("name"), "alive": p.get("alive"), "currentHealth": p.get("currentHealth"), "money": p.get("money"), "loadoutValue": p.get("loadoutValue"), "netWorth": p.get("netWorth")})
    out["alive_counts"] = alive_counts if alive_counts else None
    out["hp_totals"] = hp_totals if hp_totals else None
    out["money_totals"] = money_totals if money_totals else None
    out["bomb_phase_time"] = None

    # GRID PROBE V1 — record field paths we expected (from rich query) but did not find in payload
    if series_teams_normalized and all(st.get("name") is None for st in series_teams_normalized):
        unavailable.append("teams[].name")
    if series_teams_normalized and all(st.get("score") is None for st in series_teams_normalized):
        unavailable.append("teams[].score")
    has_any_players = False
    for gn in games_normalized:
        for gtn in gn.get("teams") or []:
            pl = gtn.get("players")
            if isinstance(pl, list) and len(pl) > 0:
                has_any_players = True
                break
        if has_any_players:
            break
    if games_normalized and not has_any_players:
        unavailable.append("games[].teams[].players")
    out["unavailable_fields"] = list(dict.fromkeys(unavailable))

    # GRID PROBE V2 — normalized contract (docs/GRID_CS2_NORMALIZED_FEATURE_CONTRACT.md)
    out["v2"] = _build_v2_contract(ss)
    return out


def main() -> None:
    # GRID PROBE V1
    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import (
        CENTRAL_DATA_GRAPHQL_URL,
        SERIES_STATE_GRAPHQL_URL,
        load_api_key,
        post_graphql,
    )
    from grid_queries import QUERY_FIND_CS2_SERIES_MINIMAL, QUERY_SERIES_STATE, QUERY_SERIES_STATE_RICH

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    # GRID PROBE V1 — Central allSeries (single request)
    print("[GRID PROBE V1] Central allSeries (live filter) ...")
    central = post_graphql(
        CENTRAL_DATA_GRAPHQL_URL,
        QUERY_FIND_CS2_SERIES_MINIMAL.strip(),
        variables=CENTRAL_VARS,
        api_key=api_key,
    )
    _save_json(RAW_CENTRAL_PATH, central)

    if central.get("errors"):
        print("[GRID PROBE V1] Central GraphQL errors:", central.get("errors"))
        sys.exit(1)

    ids = _extract_series_ids(central)
    if not ids:
        print("[GRID PROBE V1] No series IDs from Central.")
        sys.exit(1)

    # GRID PROBE V1 — first ID with seriesState data and title.nameShortened == "cs2"
    series_state_query = QUERY_SERIES_STATE_RICH.strip() if USE_RICH_QUERY else QUERY_SERIES_STATE.strip()
    selected_id: str | None = None
    series_state_raw: dict | None = None
    for sid in ids:
        state_resp = post_graphql(
            SERIES_STATE_GRAPHQL_URL,
            series_state_query,
            variables={"id": sid},
            api_key=api_key,
        )
        if state_resp.get("errors"):
            print("[GRID PROBE V1] Series State GraphQL errors (exact):", state_resp.get("errors"))
            continue
        data = state_resp.get("data") or {}
        ss = data.get("seriesState")
        if not isinstance(ss, dict):
            continue
        title = ss.get("title")
        title_short = None
        if isinstance(title, dict):
            title_short = title.get("nameShortened")
        elif isinstance(title, str):
            title_short = title
        if title_short != "cs2":
            continue
        selected_id = sid
        series_state_raw = ss
        # Keep raw response as full state response for save
        _save_json(RAW_SERIES_STATE_PATH, state_resp)
        break

    if not selected_id or not series_state_raw:
        print("[GRID PROBE V1] No series with seriesState data and title.nameShortened == 'cs2' in first 10.")
        sys.exit(1)

    print(f"[GRID PROBE V1] Selected series ID: {selected_id}")

    # GRID PROBE V1 — build normalized preview
    normalized = _normalize_series_state(series_state_raw)
    _save_json(NORMALIZED_PREVIEW_PATH, normalized)
    print("[GRID PROBE V1] Normalized preview (pretty JSON):")
    print(json.dumps(normalized, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

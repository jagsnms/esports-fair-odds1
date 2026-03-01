"""
GRID reducer: seriesState snapshot -> GridState -> canonical frame (Frame-compatible).
MVP: one snapshot = one state update; no event stream. No FastAPI deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.models import Frame, PlayerRow


@dataclass
class GridState:
    """Reduced state from GRID seriesState (team ids, round, clock, players, bomb)."""

    series_id: str | None = None
    game_index: int | None = None
    game_started: bool = False
    game_finished: bool = False
    map_name: str | None = None
    rounds_a: int | None = None
    rounds_b: int | None = None
    team_a_id: str | None = None
    team_b_id: str | None = None
    team_a_side: str | None = None
    team_b_side: str | None = None
    team_a_money: int | None = None
    team_b_money: int | None = None
    team_a_loadout_value: int | None = None
    team_b_loadout_value: int | None = None
    clock_seconds: int | None = None
    clock_ticking: bool | None = None
    clock_type: str | None = None
    clock_ticks_backwards: bool | None = None
    round_index: int | None = None
    round_phase: str | None = None
    maps_a_won: int | None = None
    maps_b_won: int | None = None
    players_a: list[dict[str, Any]] = field(default_factory=list)
    players_b: list[dict[str, Any]] = field(default_factory=list)
    bomb_planted: bool = False
    updated_at: str | None = None
    valid: bool | None = None


def reduce_event(state: GridState, event: dict[str, Any]) -> GridState:
    """
    Apply one seriesState snapshot (full payload) to state. Returns new GridState.
    MVP: event is data.seriesState from GraphQL response.
    """
    ss = event if isinstance(event, dict) else {}
    games = ss.get("games")
    if not isinstance(games, list) or not games:
        state.series_id = ss.get("id")
        state.valid = ss.get("valid")
        state.updated_at = ss.get("updatedAt")
        return state

    # Current game: started and not finished; else last; else first
    current_game: dict[str, Any] = {}
    for g in reversed(games):
        if isinstance(g, dict) and g.get("started") and not g.get("finished"):
            current_game = g
            break
    if not current_game:
        current_game = games[-1] if isinstance(games[-1], dict) else (games[0] if isinstance(games[0], dict) else {})

    state.series_id = ss.get("id")
    state.valid = ss.get("valid")
    state.updated_at = ss.get("updatedAt")
    state.game_index = current_game.get("sequenceNumber")
    state.game_started = bool(current_game.get("started"))
    state.game_finished = bool(current_game.get("finished"))
    map_node = current_game.get("map")
    state.map_name = map_node.get("name") if isinstance(map_node, dict) else None

    series_teams = ss.get("teams")
    if isinstance(series_teams, list) and len(series_teams) >= 2:
        st0 = series_teams[0] if isinstance(series_teams[0], dict) else {}
        st1 = series_teams[1] if isinstance(series_teams[1], dict) else {}
        state.maps_a_won = st0.get("score")
        state.maps_b_won = st1.get("score")

    game_teams = current_game.get("teams")
    if isinstance(game_teams, list) and len(game_teams) >= 2:
        t0 = game_teams[0] if isinstance(game_teams[0], dict) else {}
        t1 = game_teams[1] if isinstance(game_teams[1], dict) else {}
        state.team_a_id = t0.get("id") and str(t0.get("id"))
        state.team_b_id = t1.get("id") and str(t1.get("id"))
        state.team_a_side = t0.get("side") and str(t0.get("side")).upper()
        state.team_b_side = t1.get("side") and str(t1.get("side")).upper()
        state.rounds_a = t0.get("score") if t0.get("score") is not None else None
        state.rounds_b = t1.get("score") if t1.get("score") is not None else None
        state.team_a_money = t0.get("money")
        state.team_b_money = t1.get("money")
        state.team_a_loadout_value = t0.get("loadoutValue")
        state.team_b_loadout_value = t1.get("loadoutValue")
        state.players_a = [p for p in (t0.get("players") or []) if isinstance(p, dict)]
        state.players_b = [p for p in (t1.get("players") or []) if isinstance(p, dict)]
    else:
        state.players_a = []
        state.players_b = []

    cl = current_game.get("clock")
    if isinstance(cl, dict):
        state.clock_seconds = cl.get("currentSeconds")
        state.clock_ticking = cl.get("ticking")
        state.clock_type = cl.get("type")
        state.clock_ticks_backwards = cl.get("ticksBackwards")
    else:
        state.clock_seconds = None
        state.clock_ticking = None
        state.clock_type = None
        state.clock_ticks_backwards = None

    segments = current_game.get("segments")
    if isinstance(segments, list) and segments:
        last_seg = segments[-1] if isinstance(segments[-1], dict) else (segments[0] if isinstance(segments[0], dict) else {})
        state.round_index = last_seg.get("sequenceNumber")
    else:
        state.round_index = None
    state.round_phase = state.clock_type or "gameClock"
    state.bomb_planted = state.round_phase and "bomb" in (state.round_phase or "").lower()
    return state


def _alive_count(players: list[dict]) -> int:
    return sum(1 for p in players if p.get("alive") is True)


def _hp_total(players: list[dict]) -> float:
    return sum(float(p.get("currentHealth") or p.get("current_health") or 0) for p in players if p.get("alive") is True)


def _loadout_total(players: list[dict]) -> float:
    return sum(float(p.get("loadoutValue") or p.get("loadout_value") or 0) for p in players)


def _money_total(players: list[dict]) -> float:
    return sum(float(p.get("money") or 0) for p in players)


def grid_state_to_canonical_frame(state: GridState, team_a_is_team_one: bool = True) -> dict[str, Any]:
    """
    Build Frame-compatible dict from GridState. Keys match Frame attributes for reduce/compute path.
    """
    scores = (state.rounds_a or 0, state.rounds_b or 0)
    alive_a = _alive_count(state.players_a)
    alive_b = _alive_count(state.players_b)
    hp_a = _hp_total(state.players_a)
    hp_b = _hp_total(state.players_b)
    loadout_a = state.team_a_loadout_value or _loadout_total(state.players_a)
    loadout_b = state.team_b_loadout_value or _loadout_total(state.players_b)
    cash_a = state.team_a_money if state.team_a_money is not None else _money_total(state.players_a)
    cash_b = state.team_b_money if state.team_b_money is not None else _money_total(state.players_b)
    round_time_s: float | None = None
    if state.clock_seconds is not None:
        round_time_s = float(state.clock_seconds)
        if state.clock_ticks_backwards:
            round_time_s = 115.0 - round_time_s if round_time_s <= 115 else 0.0  # best-effort round remaining
    game_index = state.game_index if state.game_index is not None else 0
    series_score = (state.maps_a_won or 0, state.maps_b_won or 0)
    a_side = state.team_a_side
    team_one_id = state.team_a_id if team_a_is_team_one else state.team_b_id
    team_two_id = state.team_b_id if team_a_is_team_one else state.team_a_id
    team_one_provider_id = str(team_one_id) if team_one_id is not None else None
    team_two_provider_id = str(team_two_id) if team_two_id is not None else None

    players_a_rows = [
        PlayerRow(
            alive=p.get("alive"),
            hp=float(p.get("currentHealth") or p.get("current_health") or 0),
            armor=float(p.get("currentArmor") or p.get("current_armor") or 0),
            cash=float(p.get("money") or 0),
            loadout=float(p.get("loadoutValue") or p.get("loadout_value") or 0),
        )
        for p in state.players_a
    ]
    players_b_rows = [
        PlayerRow(
            alive=p.get("alive"),
            hp=float(p.get("currentHealth") or p.get("current_health") or 0),
            armor=float(p.get("currentArmor") or p.get("current_armor") or 0),
            cash=float(p.get("money") or 0),
            loadout=float(p.get("loadoutValue") or p.get("loadout_value") or 0),
        )
        for p in state.players_b
    ]

    return {
        "timestamp": 0.0,
        "teams": (state.team_a_id or "", state.team_b_id or ""),
        "scores": scores,
        "alive_counts": (alive_a, alive_b),
        "hp_totals": (hp_a, hp_b),
        "cash_loadout_totals": (cash_a + loadout_a, cash_b + loadout_b),
        "cash_totals": (cash_a, cash_b),
        "loadout_totals": (loadout_a, loadout_b),
        "map_index": (game_index - 1) if game_index else 0,
        "series_score": series_score,
        "map_name": state.map_name or "",
        "a_side": a_side,
        "round_time_remaining_s": round_time_s,
        "round_time_s": round_time_s,
        "team_one_id": None,
        "team_two_id": None,
        "team_one_provider_id": team_one_provider_id,
        "team_two_provider_id": team_two_provider_id,
        "players_a": players_a_rows,
        "players_b": players_b_rows,
        "bomb_phase_time_remaining": {"round_phase": state.round_phase} if state.round_phase else None,
    }


def grid_state_to_frame(state: GridState, team_a_is_team_one: bool = True, timestamp: float = 0.0) -> Frame:
    """Build Frame from GridState for reduce/compute path."""
    d = grid_state_to_canonical_frame(state, team_a_is_team_one)
    d["timestamp"] = timestamp
    valid = {f for f in Frame.__dataclass_fields__}
    kwargs = {k: v for k, v in d.items() if k in valid}
    return Frame(**kwargs)

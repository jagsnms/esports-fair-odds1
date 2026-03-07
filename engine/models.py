"""
Canonical data models for the live trading/probability pipeline.

Uses dataclasses (no pydantic dependency). All types are minimal stubs
for the migration; fields will be refined as logic is extracted from app35_ml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# --- Config (backend-owned; UI sends partial updates) ---


@dataclass(frozen=False)
class Config:
    """Runtime configuration. Backend is source of truth."""

    source: Literal["BO3", "GRID", "REPLAY"] = "BO3"
    match_id: Optional[int] = None
    grid_series_id: Optional[str] = None  # GRID series id when source=GRID
    bo3_match_ids: Optional[list[int]] = None  # multi-session: list of BO3 match ids
    grid_series_ids: Optional[list[str]] = None  # multi-session: list of GRID series ids
    # Primary session (which session drives the store in multi-session); None = first in list
    primary_session_source: Optional[str] = None  # "BO3" | "GRID"
    primary_session_id: Optional[str] = None  # match_id as string for BO3, series_id for GRID
    # BO3 auto-track (runtime only; manual bo3_match_ids overrides)
    bo3_auto_track: Optional[bool] = False
    bo3_auto_track_limit: Optional[int] = 5
    bo3_auto_track_refresh_s: Optional[float] = 30.0
    bo3_auto_track_probe_budget: Optional[int] = 40
    # GRID auto-track (runtime only; manual grid_series_ids overrides)
    grid_auto_track: Optional[bool] = False
    grid_auto_track_limit: Optional[int] = 5
    grid_auto_track_refresh_s: Optional[float] = 60.0
    poll_interval_s: float = 5.0
    contract_scope: str = ""
    series_fmt: str = ""
    prematch_series: Optional[float] = None  # manual SERIES input (0..1)
    prematch_map: Optional[float] = None  # derived from prematch_series
    prematch_locked: bool = False
    lock_team_mapping: bool = False  # identity lock: do not overwrite team mapping from feed
    market_delay_s: float = 0.0
    team_a_is_team_one: bool = True  # BO3: team A = team_one (True) or team_two (False)
    # Replay (source=REPLAY)
    replay_path: Optional[str] = "logs/bo3_pulls.jsonl"
    replay_loop: bool = True
    replay_speed: float = 1.0
    replay_index: int = 0  # runner-owned cursor
    # Replay contract gate (Stage 1): canonical replay defaults to rejecting point-like payloads.
    replay_contract_policy: str = "reject_point_like"
    # Optional temporary transition mode: explicit opt-in and sunset-bound.
    replay_point_transition_enabled: bool = False
    replay_point_transition_sunset_epoch: Optional[float] = None
    context_widening_enabled: bool = False  # gate context_risk widening + width cap (default OFF)
    # Context widening tuning knobs (defaults preserve current behavior).
    context_widen_beta: float = 0.25
    uncertainty_mult_min: float = 1.0
    uncertainty_mult_max: float = 1.35
    context_risk_weight_leverage: float = 0.4
    context_risk_weight_fragility: float = 0.4
    context_risk_weight_missingness: float = 0.2
    # Kalshi market
    market_enabled: bool = True
    kalshi_url: Optional[str] = None
    kalshi_ticker: Optional[str] = None  # resolved YES ticker for selected side
    market_delay_sec: int = 120
    market_poll_sec: int = 5
    market_side: Optional[str] = None  # e.g. "A" or "B" or team key
    # Midround V2 term weights: "current" | "learned_v1" | "learned_v2" | "learned_fit" (fitted suggested_coef)
    midround_v2_weight_profile: str = "current"
    # Runner/source: when True, resolve emits contract_diagnostics (testing-mode behavioral invariants).
    invariant_diagnostics: bool = False


# --- Frame (normalized live snapshot from feed) ---


@dataclass(frozen=True)
class PlayerRow:
    """Lightweight HUD player row snapshot, derived directly from BO3 player_states."""

    name: Optional[str] = None
    alive: Optional[bool] = None
    hp: Optional[float] = None
    armor: Optional[float] = None
    helmet: Optional[bool] = None
    cash: Optional[float] = None
    loadout: Optional[float] = None
    weapons: list[str] | None = None
    has_bomb: Optional[bool] = None
    has_kit: Optional[bool] = None


@dataclass(frozen=True)
class Frame:
    """Canonical live snapshot produced by normalize step."""

    timestamp: float = 0.0
    teams: tuple[str, str] = ("", "")
    scores: tuple[int, int] = (0, 0)
    alive_counts: tuple[int, int] = (0, 0)
    hp_totals: tuple[float, float] = (0.0, 0.0)
    cash_loadout_totals: tuple[float, float] = (0.0, 0.0)
    # First-class microstate (alive-only sums); None when player_states missing
    cash_totals: tuple[float, float] | None = None
    loadout_totals: tuple[float, float] | None = None
    wealth_totals: tuple[float, float] | None = None  # cash_totals + loadout_totals
    armor_totals: tuple[float, float] | None = None
    # Reliability debug: how loadout was sourced per team
    loadout_source: Optional[str] = None  # "ev" | "weapon_est" | "mixed"
    loadout_ev_count_a: Optional[int] = None
    loadout_ev_count_b: Optional[int] = None
    loadout_est_count_a: Optional[int] = None
    loadout_est_count_b: Optional[int] = None
    bomb_phase_time_remaining: Any = None  # structured per map/game
    # Canonical round time (normalized at ingest: ms -> seconds; use only *_s downstream)
    round_time_remaining_s: float | None = None
    round_time_s: float | None = None
    round_time_remaining_raw: float | int | None = None
    round_time_raw: float | int | None = None
    # Flags for round_time_remaining normalization (ingest-time diagnostics)
    round_time_remaining_was_ms: Optional[bool] = None
    round_time_remaining_was_negative: Optional[bool] = None
    round_time_remaining_was_out_of_range: Optional[bool] = None
    round_time_remaining_was_missing: Optional[bool] = None
    map_index: int = 0
    series_score: tuple[int, int] = (0, 0)
    map_name: str = ""
    series_fmt: str = ""
    # Team A side this map (T/CT); None when unknown
    a_side: Optional[str] = None
    # Stable identifiers from feed (team_one/team_two); use names as fallback if None
    team_one_id: Optional[int] = None
    team_two_id: Optional[int] = None
    team_one_provider_id: Optional[str] = None
    team_two_provider_id: Optional[str] = None
    # HUD player rows: Team A (left) and Team B (right). Snapshot-only; not persisted in history.
    players_a: list[PlayerRow] = field(default_factory=list)
    players_b: list[PlayerRow] = field(default_factory=list)


# --- State (authoritative running state after reducer) ---


@dataclass
class State:
    """Authoritative running state: config + last frame + identity + map index + last_total_rounds + segment."""

    config: Config = field(default_factory=Config)
    last_frame: Frame | None = None
    # Keys: a_is_team_one (bool), team_one_key (str), team_two_key (str). team_*_key = provider_id or name.
    team_mapping: dict[str, Any] = field(default_factory=dict)
    map_index: int = 0
    last_total_rounds: int = 0
    segment_id: int = 0
    last_series_score: Optional[tuple[int, int]] = None
    last_map_index: Optional[int] = None


# --- Derived (computed outputs from compute step) ---


@dataclass
class Derived:
    """Computed outputs: p_hat, corridors (series/map), kappa, debug bundle.

    Semantic layers (for series-winner contract):
    - Series corridor: bound_low/bound_high (aliases: series_low/series_high)
    - Map corridor: rail_low/rail_high (aliases: map_low/map_high)
    """

    p_hat: float = 0.5
    rail_low: float = 0.0
    rail_high: float = 1.0
    bound_low: float = 0.0
    bound_high: float = 1.0
    kappa: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    # Alias properties for semantic clarity (used by callers/serialization helpers).
    @property
    def series_low(self) -> float:
        return self.bound_low

    @property
    def series_high(self) -> float:
        return self.bound_high

    @property
    def map_low(self) -> float:
        return self.rail_low

    @property
    def map_high(self) -> float:
        return self.rail_high


# --- HistoryPoint (append-only chart point) ---


@dataclass
class HistoryPoint:
    """Single append-only point for charting and replay."""

    time: float = 0.0
    p_hat: float = 0.5
    bound_low: float = 0.0
    bound_high: float = 1.0
    rail_low: float = 0.0
    rail_high: float = 1.0
    market_mid: float | None = None
    segment_id: int = 0
    map_index: int | None = None  # 0-based map; for calibration join
    round_number: int | None = None  # round within map; for calibration join
    game_number: int | None = None  # 1-based game number if present
    explain: dict | None = None  # per-tick decomposition for calibration/ML (phase, rails, q_terms, micro_adj, etc.)
    event: dict | None = None  # outcome label: round_result | segment_result (round_number, winner_team_id, winner_is_team_a, etc.)
    # Team identity for score_diag_v2 / witness CSV (canonical: Team A == team_one when team_a_is_team_one True)
    team_one_id: int | None = None
    team_two_id: int | None = None
    team_one_provider_id: str | None = None
    team_two_provider_id: str | None = None
    team_a_is_team_one: bool | None = None
    a_side: str | None = None

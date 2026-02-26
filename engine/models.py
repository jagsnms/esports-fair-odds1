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

    source: Literal["BO3", "GRID", "REPLAY", "DUMMY"] = "BO3"
    match_id: Optional[int] = None
    poll_interval_s: float = 5.0
    contract_scope: str = ""
    series_fmt: str = ""
    prematch_map: Any = None
    prematch_series: Any = None
    lock_team_mapping: bool = False  # identity lock: do not overwrite team mapping from feed
    market_delay_s: float = 0.0
    team_a_is_team_one: bool = True  # BO3: team A = team_one (True) or team_two (False)
    # Replay (source=REPLAY)
    replay_path: Optional[str] = "logs/bo3_pulls.jsonl"
    replay_loop: bool = True
    replay_speed: float = 1.0
    replay_index: int = 0  # runner-owned cursor
    midround_enabled: bool = False


# --- Frame (normalized live snapshot from feed) ---


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
    armor_totals: tuple[float, float] | None = None
    bomb_phase_time_remaining: Any = None  # structured per map/game
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
    """Computed outputs: p_hat, rails, bounds, kappa, debug bundle."""

    p_hat: float = 0.5
    rail_low: float = 0.0
    rail_high: float = 1.0
    bound_low: float = 0.0
    bound_high: float = 1.0
    kappa: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)


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

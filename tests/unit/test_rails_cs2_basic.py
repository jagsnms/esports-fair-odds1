"""
Unit tests for CS2 round-state rails (compute_rails_cs2).
Scenarios: early 1-0 (wide), late 13-3 (narrower), bomb/low alive (adjust).
"""
from __future__ import annotations

import unittest
from typing import Any

from engine.compute.bounds import compute_bounds
from engine.compute.rails_cs2 import compute_rails_cs2
from engine.models import Config, Frame, State


def _frame(
    scores: tuple[int, int] = (0, 0),
    series_score: tuple[int, int] = (0, 0),
    map_index: int = 0,
    alive_counts: tuple[int, int] = (5, 5),
    hp_totals: tuple[float, float] = (0.0, 0.0),
    cash_loadout_totals: tuple[float, float] = (0.0, 0.0),
    bomb_phase_time_remaining: Any = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=scores,
        series_score=series_score,
        series_fmt="bo3",
        map_index=map_index,
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=cash_loadout_totals,
        bomb_phase_time_remaining=bomb_phase_time_remaining,
    )


def _config(prematch_map: float | None = None) -> Config:
    c = Config()
    if prematch_map is not None:
        c.prematch_map = prematch_map
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_rails_in_01_and_within_bounds() -> None:
    """Rails in [0,1], rail_lo <= rail_hi, and within bounds."""
    frame = _frame(scores=(5, 5))
    config = _config()
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    b_lo, b_hi = bounds
    assert 0 <= rail_lo <= 1
    assert 0 <= rail_hi <= 1
    assert rail_lo <= rail_hi
    assert b_lo <= rail_lo <= b_hi
    assert b_lo <= rail_hi <= b_hi


def test_early_score_wide_rails() -> None:
    """A) Early score 1-0 with neutral microstate -> rails wide-ish."""
    frame = _frame(scores=(1, 0), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    width_a = rail_hi - rail_lo
    assert width_a > 0.05
    assert 0 <= rail_lo <= rail_hi <= 1


def test_late_score_narrow_rails() -> None:
    """B) Late score 13-3 -> rails narrower and more decisive."""
    frame = _frame(scores=(13, 3), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    width_b = rail_hi - rail_lo
    assert 0 <= rail_lo <= rail_hi <= 1
    assert width_b >= 0.01


def test_late_narrower_than_early() -> None:
    """Scenario B (13-3) has narrower rails than A (1-0)."""
    config = _config()
    state = _state()
    frame_early = _frame(scores=(1, 0))
    frame_late = _frame(scores=(13, 3))
    bounds_early = compute_bounds(frame_early, config, state)
    bounds_late = compute_bounds(frame_late, config, state)
    r_early = compute_rails_cs2(frame_early, config, state, bounds_early)
    r_late = compute_rails_cs2(frame_late, config, state, bounds_late)
    width_early = r_early[1] - r_early[0]
    width_late = r_late[1] - r_late[0]
    assert width_late < width_early


def test_bomb_planted_or_low_alive_adjusts() -> None:
    """C) Bomb planted / low alive counts -> rails adjust (even if small)."""
    config = _config()
    state = _state()
    frame_neutral = _frame(scores=(5, 5), alive_counts=(5, 5), bomb_phase_time_remaining=None)
    frame_bomb = _frame(
        scores=(5, 5),
        alive_counts=(5, 5),
        bomb_phase_time_remaining={"is_bomb_planted": True},
    )
    frame_low_alive = _frame(scores=(5, 5), alive_counts=(1, 1))
    bounds = compute_bounds(frame_neutral, config, state)
    r_neutral = compute_rails_cs2(frame_neutral, config, state, bounds)
    r_bomb = compute_rails_cs2(frame_bomb, config, state, bounds)
    r_low = compute_rails_cs2(frame_low_alive, config, state, bounds)
    w_neutral = r_neutral[1] - r_neutral[0]
    w_bomb = r_bomb[1] - r_bomb[0]
    w_low = r_low[1] - r_low[0]
    assert w_bomb <= w_neutral + 0.001
    assert w_low <= w_neutral + 0.001
    assert 0 <= r_bomb[0] <= r_bomb[1] <= 1
    assert 0 <= r_low[0] <= r_low[1] <= 1


class TestRailsCs2Basic(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_rails_in_01_and_within_bounds(self) -> None:
        test_rails_in_01_and_within_bounds()

    def test_early_score_wide_rails(self) -> None:
        test_early_score_wide_rails()

    def test_late_score_narrow_rails(self) -> None:
        test_late_score_narrow_rails()

    def test_late_narrower_than_early(self) -> None:
        test_late_narrower_than_early()

    def test_bomb_planted_or_low_alive_adjusts(self) -> None:
        test_bomb_planted_or_low_alive_adjusts()


if __name__ == "__main__":
    unittest.main()

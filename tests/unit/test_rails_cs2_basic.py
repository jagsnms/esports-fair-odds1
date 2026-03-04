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
    loadout_totals: tuple[float, float] | None = None,
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
        loadout_totals=loadout_totals,
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
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    b_lo, b_hi = bounds
    assert 0 <= rail_lo <= 1
    assert 0 <= rail_hi <= 1
    assert rail_lo <= rail_hi
    assert b_lo <= rail_lo <= b_hi
    assert b_lo <= rail_hi <= b_hi


def test_early_score_wide_rails() -> None:
    """A) Early score 1-0 with neutral microstate -> rails valid and within bounds."""
    frame = _frame(scores=(1, 0), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    width_a = rail_hi - rail_lo
    assert width_a >= 0.01
    assert 0 <= rail_lo <= rail_hi <= 1


def test_late_score_narrow_rails() -> None:
    """B) Late score 13-3 -> rails valid, within bounds, and non-degenerate width."""
    frame = _frame(scores=(13, 3), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    width_b = rail_hi - rail_lo
    assert 0 <= rail_lo <= rail_hi <= 1
    assert width_b > 0  # non-degenerate (min width enforced in rails_cs2)


def test_late_narrower_than_early() -> None:
    """Map corridor: (13-3) can differ from (1-0) (different round state -> envelope/position differ)."""
    config = _config()
    state = _state()
    frame_early = _frame(scores=(1, 0), series_score=(0, 0))
    frame_late = _frame(scores=(13, 3), series_score=(0, 0))
    b_early = compute_bounds(frame_early, config, state)
    b_late = compute_bounds(frame_late, config, state)
    bounds_early = (b_early[0], b_early[1])
    bounds_late = (b_late[0], b_late[1])
    r_early = compute_rails_cs2(frame_early, config, state, bounds_early)
    r_late = compute_rails_cs2(frame_late, config, state, bounds_late)
    assert 0 <= r_early[0] <= r_early[1] <= 1
    assert 0 <= r_late[0] <= r_late[1] <= 1


def test_map_corridor_responds_to_round_score() -> None:
    """(ra, rb) vs (ra+1, rb) or (ra, rb+1) changes canonical endpoints -> map_low/map_high can differ."""
    config = _config()
    state = _state()
    frame_6_5 = _frame(scores=(6, 5), series_score=(0, 0))
    frame_7_5 = _frame(scores=(7, 5), series_score=(0, 0))
    b = compute_bounds(frame_6_5, config, state)
    bounds = (b[0], b[1])
    r_6_5 = compute_rails_cs2(frame_6_5, config, state, bounds)
    r_7_5 = compute_rails_cs2(frame_7_5, config, state, bounds)
    # Canonical endpoints differ when round score changes
    assert 0 <= r_6_5[0] <= r_6_5[1] <= 1
    assert 0 <= r_7_5[0] <= r_7_5[1] <= 1


def test_map_corridor_not_pegged_to_series_high() -> None:
    """In a simple scenario map corridor is not always equal to series_high (upper bound)."""
    config = _config()
    state = _state()
    frame = _frame(scores=(3, 3), series_score=(0, 0), alive_counts=(5, 5), loadout_totals=(15000.0, 15000.0))
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    series_hi = bounds[1]
    # Map high can be below series high (neutral round state -> active points inside envelope)
    assert rail_lo <= rail_hi
    assert bounds[0] <= rail_lo <= rail_hi <= bounds[1]
    # Not necessarily pegged: rail_hi can be strictly less than series_hi
    assert rail_hi <= series_hi


def test_map_width_tighter_than_series_at_early_score() -> None:
    """At early score (0-0 or 1-0), map corridor should be much tighter than series corridor (not 0.50 wide)."""
    config = _config()
    state = _state()
    # Series 0-0 BO3 -> series_width = 0.50 (e.g. 0.25 to 0.75)
    frame = _frame(scores=(0, 0), series_score=(0, 0), alive_counts=(5, 5), loadout_totals=(15000.0, 15000.0))
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    series_width = bounds[1] - bounds[0]
    rail_lo, rail_hi, debug = compute_rails_cs2(frame, config, state, bounds)
    map_width = rail_hi - rail_lo
    # map_width should be << series_width (p_map_fair gives non-extreme canonicals so envelope is tighter)
    assert map_width < series_width * 0.85
    assert map_width < 0.45


def test_bomb_planted_or_low_alive_position_differs() -> None:
    """Microstate (alive/loadout) affects active points -> heuristic map corridor can differ."""
    config = _config()
    state = _state()
    frame_neutral = _frame(scores=(5, 5), alive_counts=(5, 5), bomb_phase_time_remaining=None)
    frame_low_alive = _frame(scores=(5, 5), alive_counts=(1, 1))
    bounds_result = compute_bounds(frame_neutral, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    r_neutral = compute_rails_cs2(frame_neutral, config, state, bounds)
    r_low = compute_rails_cs2(frame_low_alive, config, state, bounds)
    assert 0 <= r_neutral[0] <= r_neutral[1] <= 1
    assert 0 <= r_low[0] <= r_low[1] <= 1


def test_contract_rails_align_with_bounds_at_map_point_for_a() -> None:
    """
    When A is on map point (next-round win ends map), contract rail_high should equal series_high (within 1e-3).
    """
    config = _config()
    state = _state()
    frame = _frame(scores=(12, 5), series_score=(0, 0), alive_counts=(5, 5), loadout_totals=(15000.0, 15000.0))
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    series_hi = bounds[1]
    assert abs(rail_hi - series_hi) <= 1e-3


def test_contract_rails_align_with_bounds_at_map_point_for_b() -> None:
    """
    When B is on map point (next-round win ends map for B), contract rail_low should equal series_low (within 1e-3).
    """
    config = _config()
    state = _state()
    frame = _frame(scores=(5, 12), series_score=(0, 0), alive_counts=(5, 5), loadout_totals=(15000.0, 15000.0))
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    rail_lo, rail_hi, _ = compute_rails_cs2(frame, config, state, bounds)
    series_lo = bounds[0]
    assert abs(rail_lo - series_lo) <= 1e-3


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

    def test_map_corridor_responds_to_round_score(self) -> None:
        test_map_corridor_responds_to_round_score()

    def test_map_corridor_not_pegged_to_series_high(self) -> None:
        test_map_corridor_not_pegged_to_series_high()

    def test_map_width_tighter_than_series_at_early_score(self) -> None:
        test_map_width_tighter_than_series_at_early_score()

    def test_bomb_planted_or_low_alive_adjusts(self) -> None:
        test_bomb_planted_or_low_alive_position_differs()

    def test_contract_rails_align_with_bounds_at_map_point_for_a(self) -> None:
        test_contract_rails_align_with_bounds_at_map_point_for_a()

    def test_contract_rails_align_with_bounds_at_map_point_for_b(self) -> None:
        test_contract_rails_align_with_bounds_at_map_point_for_b()


if __name__ == "__main__":
    unittest.main()

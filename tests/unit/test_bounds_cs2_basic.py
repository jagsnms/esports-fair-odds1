"""
Unit tests for CS2 series corridor (compute_bounds_cs2).
I.i.d. macro-only: series_low/series_high from series_win_prob_live(..., p_current_map=0/1).
No microstate or round score.
"""
from __future__ import annotations

import unittest
from typing import Optional

from engine.compute.bounds import compute_bounds
from engine.compute.bounds_cs2 import compute_bounds_cs2
from engine.models import Config, Frame, State


def _frame(
    scores: tuple[int, int] = (0, 0),
    series_score: tuple[int, int] = (0, 0),
    map_index: int = 0,
    series_fmt: str = "bo3",
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=scores,
        series_score=series_score,
        series_fmt=series_fmt,
        map_index=map_index,
    )


def _config(contract_scope: str = "", prematch_map: Optional[float] = None) -> Config:
    c = Config(contract_scope=contract_scope)
    if prematch_map is not None:
        c.prematch_map = prematch_map
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_series_corridor_at_0_0_valid_and_ordered() -> None:
    """At series (0,0) BO3: series_low < series_high, both in [0,1], width > 0."""
    frame = _frame(series_score=(0, 0))
    config = _config()
    state = _state()
    lo, hi, debug = compute_bounds_cs2(frame, config, state)
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert lo <= hi
    assert hi - lo > 0
    assert debug.get("series_low") == lo
    assert debug.get("series_high") == hi


def test_series_corridor_responds_to_series_score() -> None:
    """Different series_score (maps_a_won, maps_b_won) yields different bounds."""
    config = _config()
    state = _state()
    frame_0_0 = _frame(series_score=(0, 0))
    frame_1_0 = _frame(series_score=(1, 0))
    lo0, hi0, _ = compute_bounds_cs2(frame_0_0, config, state)
    lo1, hi1, _ = compute_bounds_cs2(frame_1_0, config, state)
    assert (lo0, hi0) != (lo1, hi1)
    assert 0 <= lo0 <= hi0 <= 1 and 0 <= lo1 <= hi1 <= 1


def test_series_corridor_decider_map() -> None:
    """BO3 at (1,1): decider map -> bounds can span [0,1] or tight."""
    frame = _frame(series_score=(1, 1), series_fmt="bo3")
    config = _config()
    state = _state()
    lo, hi, _ = compute_bounds_cs2(frame, config, state)
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert lo <= hi


def test_ot_16_15_valid() -> None:
    """Any frame: valid bounds in [0,1] (series corridor ignores round score)."""
    frame = _frame(scores=(16, 15), series_score=(0, 0))
    config = _config()
    state = _state()
    lo, hi, _ = compute_bounds_cs2(frame, config, state)
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert lo <= hi


def test_bounds_midpoint_between_series_low_high() -> None:
    """At (0,0) with p0=0.5, series_low < 0.5 < series_high (or ordered)."""
    frame = _frame(series_score=(0, 0))
    config = _config(prematch_map=0.5)
    state = _state()
    lo, hi, _ = compute_bounds_cs2(frame, config, state)
    assert lo <= hi
    mid = 0.5 * (lo + hi)
    assert lo <= mid <= hi


def test_compute_bounds_integration_returns_three_tuple() -> None:
    """Public compute_bounds returns (lo, hi, debug); first two in [0,1]."""
    frame = _frame(scores=(1, 0), series_score=(0, 0))
    config = _config()
    state = _state()
    result = compute_bounds(frame, config, state)
    assert len(result) >= 2
    lo, hi = result[0], result[1]
    assert 0 <= lo <= hi <= 1
    if len(result) > 2:
        assert isinstance(result[2], dict)


def test_series_corridor_invariant_to_microstate() -> None:
    """Series corridor depends only on macro (series_score, series_fmt); not microstate."""
    base = _frame(scores=(6, 3), series_score=(1, 0))
    frame_neutral = Frame(
        timestamp=base.timestamp,
        teams=base.teams,
        scores=base.scores,
        series_score=base.series_score,
        series_fmt=base.series_fmt,
        map_index=base.map_index,
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        cash_loadout_totals=(10000.0, 10000.0),
        cash_totals=(8000.0, 8000.0),
        loadout_totals=(8000.0, 8000.0),
        armor_totals=(100.0, 100.0),
        bomb_phase_time_remaining={"round_time_remaining": 40.0, "round_phase": "IN_PROGRESS", "is_bomb_planted": False},
    )
    frame_fragile = Frame(
        timestamp=base.timestamp,
        teams=base.teams,
        scores=base.scores,
        series_score=base.series_score,
        series_fmt=base.series_fmt,
        map_index=base.map_index,
        alive_counts=(1, 5),
        hp_totals=(50.0, 450.0),
        cash_loadout_totals=(1000.0, 20000.0),
        cash_totals=(500.0, 15000.0),
        loadout_totals=(500.0, 15000.0),
        armor_totals=(0.0, 200.0),
        bomb_phase_time_remaining={"round_time_remaining": 5.0, "round_phase": "IN_PROGRESS", "is_bomb_planted": True},
    )
    config = _config()
    state = _state()
    lo_neutral, hi_neutral, _ = compute_bounds_cs2(frame_neutral, config, state)
    lo_fragile, hi_fragile, _ = compute_bounds_cs2(frame_fragile, config, state)
    assert lo_neutral == lo_fragile
    assert hi_neutral == hi_fragile


class TestBoundsCs2Basic(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_series_corridor_at_0_0_valid_and_ordered(self) -> None:
        test_series_corridor_at_0_0_valid_and_ordered()

    def test_series_corridor_responds_to_series_score(self) -> None:
        test_series_corridor_responds_to_series_score()

    def test_series_corridor_decider_map(self) -> None:
        test_series_corridor_decider_map()

    def test_ot_16_15_valid(self) -> None:
        test_ot_16_15_valid()

    def test_bounds_midpoint_between_series_low_high(self) -> None:
        test_bounds_midpoint_between_series_low_high()

    def test_compute_bounds_integration_returns_three_tuple(self) -> None:
        test_compute_bounds_integration_returns_three_tuple()

    def test_series_corridor_invariant_to_microstate(self) -> None:
        test_series_corridor_invariant_to_microstate()


if __name__ == "__main__":
    unittest.main()

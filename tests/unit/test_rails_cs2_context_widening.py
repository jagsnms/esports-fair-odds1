"""
Regression tests for contextual widening of map corridor (rails_cs2).
Same score/series corridor: neutral context vs fragile context => map width fragile > neutral.
"""
from __future__ import annotations

import unittest

from engine.compute.bounds import compute_bounds
from engine.compute.rails_cs2 import compute_rails_cs2
from engine.models import Config, Frame, State


def _frame(
    scores: tuple[int, int] = (5, 5),
    series_score: tuple[int, int] = (0, 0),
    alive_counts: tuple[int, int] = (5, 5),
    hp_totals: tuple[float, float] = (500.0, 500.0),
    loadout_totals: tuple[float, float] | None = (10000.0, 10000.0),
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=scores,
        series_score=series_score,
        series_fmt="bo3",
        map_index=0,
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        loadout_totals=loadout_totals,
    )


def _config(context_widening_enabled: bool = False) -> Config:
    c = Config()
    c.context_widening_enabled = context_widening_enabled
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_early_round_width_capped() -> None:
    """With context_widening_enabled: early-round map_width_after_cap <= cap and >= pre-widen width."""
    config = _config(context_widening_enabled=True)
    state = _state()
    scores = (2, 1)
    series = (0, 0)
    frame = _frame(
        scores=scores,
        series_score=series,
        alive_counts=(5, 5),
        hp_totals=(400.0, 400.0),
        loadout_totals=(500.0, 500.0),
    )
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    map_width_before = debug.get("map_width_before")
    map_width_after_cap = debug.get("map_width_after_cap")
    width_cap_used = debug.get("width_cap_used")
    assert map_width_before is not None and map_width_before > 0.0
    assert map_width_after_cap is not None and width_cap_used is not None
    assert map_width_after_cap >= map_width_before
    effective_cap = max(width_cap_used, map_width_before)
    assert map_width_after_cap <= effective_cap


def test_late_round_cap_larger_than_early() -> None:
    """With context_widening_enabled: late-phase width cap >= early-phase cap."""
    config = _config(context_widening_enabled=True)
    state = _state()
    frame_early = _frame(scores=(2, 1), series_score=(0, 0))
    b_early = compute_bounds(frame_early, config, state)
    bounds_early = (b_early[0], b_early[1])
    _, _, debug_early = compute_rails_cs2(frame_early, config, state, bounds_early)
    cap_early = debug_early.get("width_cap_used")
    frame_late = _frame(scores=(12, 11), series_score=(1, 1))
    b_late = compute_bounds(frame_late, config, state)
    bounds_late = (b_late[0], b_late[1])
    _, _, debug_late = compute_rails_cs2(frame_late, config, state, bounds_late)
    cap_late = debug_late.get("width_cap_used")
    assert cap_early is not None and cap_late is not None
    assert cap_late >= cap_early


def test_debug_contains_context_risk_and_widths() -> None:
    """With context_widening_enabled, debug contains context_risk, uncertainty_multiplier, width keys."""
    config = _config(context_widening_enabled=True)
    state = _state()
    frame = _frame(scores=(5, 5), loadout_totals=(8000.0, 8000.0))
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    assert "context_risk" in debug
    assert "context_risk_components" in debug
    assert "uncertainty_multiplier" in debug
    assert "map_width_before" in debug
    assert "map_width_after_widen" in debug
    assert "map_width_after_cap" in debug
    assert "width_cap_used" in debug


class TestRailsCs2ContextWidening(unittest.TestCase):
    def test_early_round_width_capped(self) -> None:
        test_early_round_width_capped()

    def test_late_round_cap_larger_than_early(self) -> None:
        test_late_round_cap_larger_than_early()

    def test_debug_contains_context_risk_and_widths(self) -> None:
        test_debug_contains_context_risk_and_widths()


if __name__ == "__main__":
    unittest.main()

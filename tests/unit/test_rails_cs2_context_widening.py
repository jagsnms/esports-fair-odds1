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


def _config() -> Config:
    return Config()


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_context_widening_fragile_wider_than_neutral() -> None:
    """Same score and series corridor; neutral context vs fragile => map width fragile > neutral."""
    config = _config()
    state = _state()
    # Early score so bounds are wide and widening is not clamped to bound width
    scores = (2, 1)
    series = (0, 0)
    frame_neutral = _frame(
        scores=scores,
        series_score=series,
        alive_counts=(5, 5),
        hp_totals=(400.0, 400.0),
        loadout_totals=(12000.0, 11000.0),  # balanced, decent total
    )
    frame_fragile_low = _frame(
        scores=scores,
        series_score=series,
        alive_counts=(5, 5),
        hp_totals=(400.0, 400.0),
        loadout_totals=(500.0, 500.0),  # very low total -> high fragility
    )
    frame_fragile_asym = _frame(
        scores=scores,
        series_score=series,
        alive_counts=(5, 5),
        hp_totals=(400.0, 400.0),
        loadout_totals=(20000.0, 1000.0),  # large asymmetry
    )
    frame_fragile_missing = _frame(
        scores=scores,
        series_score=series,
        alive_counts=(0, 0),
        hp_totals=(0.0, 0.0),
        loadout_totals=None,  # missing microstate -> missingness_risk
    )

    bounds = compute_bounds(frame_neutral, config, state)
    _, _, debug_n = compute_rails_cs2(frame_neutral, config, state, bounds)
    _, _, debug_fl = compute_rails_cs2(frame_fragile_low, config, state, bounds)
    _, _, debug_fa = compute_rails_cs2(frame_fragile_asym, config, state, bounds)
    _, _, debug_fm = compute_rails_cs2(frame_fragile_missing, config, state, bounds)

    width_neutral = debug_n["map_width_after"]
    width_fragile_low = debug_fl["map_width_after"]
    width_fragile_asym = debug_fa["map_width_after"]
    width_fragile_missing = debug_fm["map_width_after"]

    assert width_fragile_low > width_neutral
    assert width_fragile_asym > width_neutral
    assert width_fragile_missing > width_neutral


def test_debug_contains_context_risk_and_widths() -> None:
    """compute_rails_cs2 debug dict contains context_risk, uncertainty_multiplier, map_width_before/after."""
    config = _config()
    state = _state()
    frame = _frame(scores=(5, 5), loadout_totals=(8000.0, 8000.0))
    bounds = compute_bounds(frame, config, state)
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    assert "context_risk" in debug
    assert "context_risk_components" in debug
    assert "uncertainty_multiplier" in debug
    assert "map_width_before" in debug
    assert "map_width_after" in debug
    assert debug["map_width_after"] >= debug["map_width_before"]


class TestRailsCs2ContextWidening(unittest.TestCase):
    def test_context_widening_fragile_wider_than_neutral(self) -> None:
        test_context_widening_fragile_wider_than_neutral()

    def test_debug_contains_context_risk_and_widths(self) -> None:
        test_debug_contains_context_risk_and_widths()


if __name__ == "__main__":
    unittest.main()

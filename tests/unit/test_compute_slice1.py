"""
Unit tests for compute slice 1: bounds, rails, resolve_p_hat.
"""
from __future__ import annotations

import unittest
from typing import Optional

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.resolve import resolve_p_hat
from engine.models import Config, Frame, State


def _frame(scores: tuple[int, int], series_score: tuple[int, int] = (0, 0)) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=scores,
        series_score=series_score,
        series_fmt="bo3",
    )


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def _config(contract_scope: str = "", prematch_map: Optional[float] = None) -> Config:
    c = Config(contract_scope=contract_scope)
    if prematch_map is not None:
        c.prematch_map = prematch_map
    return c


def test_bounds_in_01_and_low_le_high():
    """Bounds in [0,1] and bound_low <= bound_high."""
    frame = _frame((5, 5))
    config = _config()
    state = _state()
    lo, hi = compute_bounds(frame, config, state)
    assert 0 <= lo <= 1
    assert 0 <= hi <= 1
    assert lo <= hi


def test_rails_in_01_and_inside_bounds():
    """Rails in [0,1] and rail_low >= bound_low, rail_high <= bound_high (or equal for pass-through)."""
    frame = _frame((5, 5))
    config = _config()
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rails = compute_rails(frame, config, state, bounds)
    rlo, rhi = rails
    blo, bhi = bounds
    assert 0 <= rlo <= 1
    assert 0 <= rhi <= 1
    assert rlo >= blo - 1e-9
    assert rhi <= bhi + 1e-9


def test_p_hat_in_rails():
    """p_hat from resolve_p_hat lies within rails."""
    frame = _frame((5, 5))
    config = _config(prematch_map=0.6)
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rails = compute_rails(frame, config, state, bounds)
    p = resolve_p_hat(frame, config, state, rails)
    rlo, rhi = rails
    assert rlo <= p <= rhi


def test_early_round_wider_bounds_than_late():
    """Early round (1-0) has wider bounds than late round (12-3)."""
    config = _config()
    state = _state()
    frame_early = _frame((1, 0))
    frame_late = _frame((12, 3))
    lo_early, hi_early = compute_bounds(frame_early, config, state)
    lo_late, hi_late = compute_bounds(frame_late, config, state)
    width_early = hi_early - lo_early
    width_late = hi_late - lo_late
    assert width_early > width_late


def test_late_round_bounds_tight():
    """Late round gives relatively tight bounds (narrower than early)."""
    frame = _frame((13, 3))  # 16 rounds played; CS2 corridor or simple fallback
    config = _config()
    state = _state()
    lo, hi = compute_bounds(frame, config, state)
    assert hi - lo < 0.45


def test_resolve_clamps_to_rails():
    """prematch_map outside rails is clamped into rails."""
    frame = _frame((10, 2))  # late round, tight bounds around 0.5
    config = _config(prematch_map=0.9)  # 0.9 likely outside tight band
    state = _state()
    bounds = compute_bounds(frame, config, state)
    rails = compute_rails(frame, config, state, bounds)
    p = resolve_p_hat(frame, config, state, rails)
    rlo, rhi = rails
    assert rlo <= p <= rhi


class TestComputeSlice1(unittest.TestCase):
    """Run the same tests via unittest for environments without pytest."""

    def test_bounds_in_01_and_low_le_high(self):
        test_bounds_in_01_and_low_le_high()

    def test_rails_in_01_and_inside_bounds(self):
        test_rails_in_01_and_inside_bounds()

    def test_p_hat_in_rails(self):
        test_p_hat_in_rails()

    def test_early_round_wider_bounds_than_late(self):
        test_early_round_wider_bounds_than_late()

    def test_late_round_bounds_tight(self):
        test_late_round_bounds_tight()

    def test_resolve_clamps_to_rails(self):
        test_resolve_clamps_to_rails()


if __name__ == "__main__":
    unittest.main()

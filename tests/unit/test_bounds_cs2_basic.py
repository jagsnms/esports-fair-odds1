"""
Unit tests for CS2 map-state bounds (compute_bounds_cs2).
Scenarios: early 1-0 (wide), late 12-11 (narrow), 12-3 (center shift), OT 16-15.
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


def test_early_1_0_bounds_wide() -> None:
    """Early 1-0: bounds wide (width > 0.4 ok)."""
    frame = _frame(scores=(1, 0))
    config = _config()
    state = _state()
    lo, hi = compute_bounds_cs2(frame, config, state)
    width = hi - lo
    assert width > 0.4
    assert 0 <= lo <= 1 and 0 <= hi <= 1
    assert lo <= hi


def test_late_12_11_bounds_narrow() -> None:
    """Late 12-11: bounds narrow (width < early width)."""
    config = _config()
    state = _state()
    frame_early = _frame(scores=(1, 0))
    frame_late = _frame(scores=(12, 11))
    lo_e, hi_e = compute_bounds_cs2(frame_early, config, state)
    lo_l, hi_l = compute_bounds_cs2(frame_late, config, state)
    width_early = hi_e - lo_e
    width_late = hi_l - lo_l
    assert width_late < width_early
    assert 0 <= lo_l <= hi_l <= 1


def test_large_diff_12_3_center_shift_toward_leader() -> None:
    """Large diff 12-3: center (midpoint) should shift toward leader A (midpoint > 0.5)."""
    frame = _frame(scores=(12, 3))
    config = _config()
    state = _state()
    lo, hi = compute_bounds_cs2(frame, config, state)
    mid = 0.5 * (lo + hi)
    assert mid > 0.5
    assert 0 <= lo <= hi <= 1


def test_ot_16_15_valid_narrow() -> None:
    """OT case 16-15: valid bounds in [0,1], narrow-ish."""
    frame = _frame(scores=(16, 15))
    config = _config()
    state = _state()
    lo, hi = compute_bounds_cs2(frame, config, state)
    assert 0 <= lo <= 1
    assert 0 <= hi <= 1
    assert lo <= hi
    width = hi - lo
    assert width < 0.5


def test_bounds_contain_center() -> None:
    """Bounds always contain baseline+shift (center)."""
    frame = _frame(scores=(6, 5))
    config = _config(prematch_map=0.52)
    state = _state()
    lo, hi = compute_bounds_cs2(frame, config, state)
    mid = 0.5 * (lo + hi)
    assert lo <= mid <= hi


def test_compute_bounds_integration() -> None:
    """Public compute_bounds (map scope) returns same shape; early wide."""
    frame = _frame(scores=(1, 0))
    config = _config()
    state = _state()
    lo, hi = compute_bounds(frame, config, state)
    assert 0 <= lo <= hi <= 1
    assert hi - lo > 0.3


class TestBoundsCs2Basic(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_early_1_0_bounds_wide(self) -> None:
        test_early_1_0_bounds_wide()

    def test_late_12_11_bounds_narrow(self) -> None:
        test_late_12_11_bounds_narrow()

    def test_large_diff_12_3_center_shift_toward_leader(self) -> None:
        test_large_diff_12_3_center_shift_toward_leader()

    def test_ot_16_15_valid_narrow(self) -> None:
        test_ot_16_15_valid_narrow()

    def test_bounds_contain_center(self) -> None:
        test_bounds_contain_center()

    def test_compute_bounds_integration(self) -> None:
        test_compute_bounds_integration()


if __name__ == "__main__":
    unittest.main()

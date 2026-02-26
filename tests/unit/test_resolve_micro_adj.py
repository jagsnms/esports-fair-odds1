"""
Unit tests for resolve_p_hat with microstate adjustment (micro_adj_cs2).
"""
from __future__ import annotations

import unittest
from typing import Any, Optional

from engine.compute.micro_adj_cs2 import micro_adjustment_cs2
from engine.compute.resolve import resolve_p_hat
from engine.models import Config, Frame, State


def _frame(
    alive_counts: tuple[int, int] = (0, 0),
    hp_totals: tuple[float, float] = (0.0, 0.0),
    cash_loadout_totals: tuple[float, float] = (0.0, 0.0),
    bomb_phase_time_remaining: Any = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=cash_loadout_totals,
        bomb_phase_time_remaining=bomb_phase_time_remaining,
    )


def _config(
    prematch_map: Optional[float] = None,
    midround_enabled: bool = False,
) -> Config:
    c = Config()
    if prematch_map is not None:
        c.prematch_map = prematch_map
    c.midround_enabled = midround_enabled
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_micro_fields_missing_adj_zero() -> None:
    """When micro fields missing or zeroed -> adj == 0."""
    frame = _frame()  # all defaults (0,0), (0,0), (0,0), None
    adj = micro_adjustment_cs2(frame)
    assert adj == 0.0


def test_strong_a_advantage_adj_positive() -> None:
    """Strong A advantage (alive 5-2, hp 450-150, econ 12000-6000) -> adj > 0."""
    frame = _frame(
        alive_counts=(5, 2),
        hp_totals=(450.0, 150.0),
        cash_loadout_totals=(12000.0, 6000.0),
    )
    adj = micro_adjustment_cs2(frame)
    assert adj > 0
    assert adj <= 0.08


def test_strong_a_disadvantage_adj_negative() -> None:
    """Strong A disadvantage -> adj < 0."""
    frame = _frame(
        alive_counts=(2, 5),
        hp_totals=(150.0, 450.0),
        cash_loadout_totals=(6000.0, 12000.0),
    )
    adj = micro_adjustment_cs2(frame)
    assert adj < 0
    assert adj >= -0.08


def test_resolve_clamps_into_rails() -> None:
    """resolve_p_hat clamps into rails even if base+adj is outside."""
    # Base 0.9 + adj 0.08 = 0.98; rails (0.3, 0.7) -> result should be 0.7
    frame = _frame(alive_counts=(5, 0), hp_totals=(500.0, 0.0), cash_loadout_totals=(20000.0, 0.0))
    config = _config(prematch_map=0.9)
    state = _state()
    rails = (0.3, 0.7)
    p, _ = resolve_p_hat(frame, config, state, rails)
    assert p >= 0.3
    assert p <= 0.7
    assert p == 0.7  # base+adj would be > 0.7, clamped to rail_high


def test_resolve_clamps_to_01() -> None:
    """resolve_p_hat result is in [0, 1]."""
    frame = _frame(alive_counts=(5, 0), hp_totals=(500.0, 0.0))
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.0, 1.0)
    p, _ = resolve_p_hat(frame, config, state, rails)
    assert 0 <= p <= 1


def test_midround_disabled_unchanged_behavior() -> None:
    """midround_enabled=False => p_hat equals p_hat_old and p_hat_final (no blend)."""
    frame = _frame(alive_counts=(4, 2), hp_totals=(350.0, 250.0))
    config = _config(prematch_map=0.5, midround_enabled=False)
    state = _state()
    rails = (0.2, 0.8)
    p, dbg = resolve_p_hat(frame, config, state, rails)
    assert dbg["midround_enabled"] is False
    assert dbg["midround_weight"] == 0.0
    assert dbg["p_hat_old"] == dbg["p_hat_final"]
    assert p == dbg["p_hat_old"]
    assert p == dbg["p_hat_final"]


def test_midround_enabled_stays_within_rails() -> None:
    """midround_enabled=True => p_hat_final within rails."""
    frame = _frame(alive_counts=(5, 1), hp_totals=(400.0, 100.0))
    config = _config(prematch_map=0.5, midround_enabled=True)
    state = _state()
    rails = (0.25, 0.75)
    p, dbg = resolve_p_hat(frame, config, state, rails)
    assert dbg["midround_enabled"] is True
    assert dbg["midround_weight"] == 0.25
    rlo, rhi = rails
    assert rlo <= p <= rhi
    assert p == dbg["p_hat_final"]


def test_midround_monotonic_higher_q_intra_raises_p_hat() -> None:
    """With midround_enabled=True, higher q_intra => higher p_hat_final when rails allow."""
    state = _state()
    rails = (0.2, 0.8)  # wide enough that blend can move
    config_low = _config(prematch_map=0.5, midround_enabled=True)
    # Frame with A slightly favored -> q_intra ~0.5–0.6
    frame_low = _frame(alive_counts=(3, 3), hp_totals=(300.0, 300.0))
    p_low, dbg_low = resolve_p_hat(frame_low, config_low, state, rails)
    # Frame with A more favored -> higher q_intra
    frame_high = _frame(alive_counts=(5, 2), hp_totals=(450.0, 150.0))
    p_high, dbg_high = resolve_p_hat(frame_high, config_low, state, rails)
    q_low = dbg_low["q_intra"]["q_intra_round_win_a"]
    q_high = dbg_high["q_intra"]["q_intra_round_win_a"]
    assert q_high > q_low
    assert p_high >= p_low


class TestResolveMicroAdj(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_micro_fields_missing_adj_zero(self) -> None:
        test_micro_fields_missing_adj_zero()

    def test_strong_a_advantage_adj_positive(self) -> None:
        test_strong_a_advantage_adj_positive()

    def test_strong_a_disadvantage_adj_negative(self) -> None:
        test_strong_a_disadvantage_adj_negative()

    def test_resolve_clamps_into_rails(self) -> None:
        test_resolve_clamps_into_rails()

    def test_resolve_clamps_to_01(self) -> None:
        test_resolve_clamps_to_01()

    def test_midround_disabled_unchanged_behavior(self) -> None:
        test_midround_disabled_unchanged_behavior()

    def test_midround_enabled_stays_within_rails(self) -> None:
        test_midround_enabled_stays_within_rails()

    def test_midround_monotonic_higher_q_intra_raises_p_hat(self) -> None:
        test_midround_monotonic_higher_q_intra_raises_p_hat()


if __name__ == "__main__":
    unittest.main()

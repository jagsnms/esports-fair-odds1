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
    loadout_totals: tuple[float, float] | None = None,
    cash_loadout_totals: tuple[float, float] = (0.0, 0.0),  # legacy; should not affect CS2 compute
    bomb_phase_time_remaining: Any = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=cash_loadout_totals,
        loadout_totals=loadout_totals,
        bomb_phase_time_remaining=bomb_phase_time_remaining,
    )


def _config(prematch_map: Optional[float] = None) -> Config:
    c = Config()
    if prematch_map is not None:
        c.prematch_map = prematch_map
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
        loadout_totals=(12000.0, 6000.0),
    )
    adj = micro_adjustment_cs2(frame)
    assert adj > 0
    assert adj <= 0.08


def test_strong_a_disadvantage_adj_negative() -> None:
    """Strong A disadvantage -> adj < 0."""
    frame = _frame(
        alive_counts=(2, 5),
        hp_totals=(150.0, 450.0),
        loadout_totals=(6000.0, 12000.0),
    )
    adj = micro_adjustment_cs2(frame)
    assert adj < 0
    assert adj >= -0.08


def test_resolve_clamps_into_rails() -> None:
    """resolve_p_hat clamps into rails even if base+adj is outside."""
    # Base 0.9 + adj 0.08 = 0.98; rails (0.3, 0.7) -> result should be 0.7 (V2 skipped for non-IN_PROGRESS)
    frame = _frame(alive_counts=(5, 0), hp_totals=(500.0, 0.0), loadout_totals=(20000.0, 0.0))
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


def test_midround_buy_time_frozen_midpoint() -> None:
    """When round_phase is BUY_TIME, p_hat_final = rail midpoint (no loadout pegging)."""
    frame = _frame(
        alive_counts=(4, 2),
        hp_totals=(350.0, 250.0),
        bomb_phase_time_remaining={"round_phase": "BUY_TIME"},
    )
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.2, 0.8)
    rail_low, rail_high = rails
    p, dbg = resolve_p_hat(frame, config, state, rails)
    assert dbg["midround_weight"] == 0.0
    expected_mid = 0.5 * (rail_low + rail_high)
    assert dbg["p_hat_final"] == expected_mid
    assert p == expected_mid
    mv2 = dbg.get("midround_v2") or {}
    assert mv2.get("skipped") is True
    assert mv2.get("reason") == "buy_time_frozen_midpoint"
    assert mv2.get("round_phase") == "BUY_TIME"


def test_midround_freezetime_frozen_midpoint() -> None:
    """FREEZETIME: same as BUY_TIME, p_hat = rail midpoint."""
    frame = _frame(
        alive_counts=(5, 5),
        hp_totals=(400.0, 400.0),
        bomb_phase_time_remaining={"round_phase": "FREEZETIME"},
    )
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.2, 0.8)
    rail_low, rail_high = rails
    p, dbg = resolve_p_hat(frame, config, state, rails)
    expected_mid = 0.5 * (rail_low + rail_high)
    assert p == expected_mid
    mv2 = dbg.get("midround_v2") or {}
    assert mv2.get("skipped") is True
    assert mv2.get("reason") == "buy_time_frozen_midpoint"
    assert mv2.get("round_phase") == "FREEZETIME"


def test_midround_in_progress_stays_within_rails() -> None:
    """When round_phase is IN_PROGRESS, V2 is applied and p_hat_final within rails."""
    frame = _frame(
        alive_counts=(5, 1),
        hp_totals=(400.0, 100.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
    )
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.25, 0.75)
    p, dbg = resolve_p_hat(frame, config, state, rails)
    assert dbg["midround_weight"] == 0.25
    mv2 = dbg.get("midround_v2") or {}
    assert mv2.get("skipped") is not True
    rlo, rhi = rails
    assert rlo <= p <= rhi
    assert p == dbg["p_hat_final"]


def test_midround_monotonic_higher_q_intra_raises_p_hat() -> None:
    """With IN_PROGRESS (V2 applied), more A advantage => higher p_hat_final when rails allow."""
    state = _state()
    rails = (0.2, 0.8)
    config = _config(prematch_map=0.5)
    bomb = {"round_phase": "IN_PROGRESS"}
    frame_low = _frame(
        alive_counts=(3, 3),
        hp_totals=(300.0, 300.0),
        loadout_totals=(5000.0, 5000.0),
        bomb_phase_time_remaining=bomb,
    )
    p_low, dbg_low = resolve_p_hat(frame_low, config, state, rails)
    frame_high = _frame(
        alive_counts=(5, 2),
        hp_totals=(450.0, 150.0),
        loadout_totals=(12000.0, 6000.0),
        bomb_phase_time_remaining=bomb,
    )
    p_high, dbg_high = resolve_p_hat(frame_high, config, state, rails)
    v2_low = dbg_low.get("midround_v2") or {}
    v2_high = dbg_high.get("midround_v2") or {}
    q_low = v2_low.get("q_intra", dbg_low["q_intra"].get("q_intra_round_win_a", 0.5))
    q_high = v2_high.get("q_intra", dbg_high["q_intra"].get("q_intra_round_win_a", 0.5))
    assert q_high > q_low
    assert p_high >= p_low


def test_midround_v2_p_hat_equals_p_mid_clamped_when_rails_wide() -> None:
    """IN_PROGRESS with wide rails => p_hat_final equals midround_v2 p_mid_clamped."""
    state = _state()
    rails = (0.01, 0.99)  # wide so clamp to rails does not change p_mid_clamped
    config = _config(prematch_map=0.5)
    frame = _frame(
        alive_counts=(4, 2),
        hp_totals=(400.0, 200.0),
        loadout_totals=(10000.0, 5000.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
    )
    p, dbg = resolve_p_hat(frame, config, state, rails)
    assert dbg.get("midround_v2") is not None
    assert dbg["midround_v2"].get("skipped") is not True  # V2 was applied
    p_mid_clamped = dbg["midround_v2"]["p_mid_clamped"]
    assert p == p_mid_clamped


REQUIRED_EXPLAIN_KEYS = (
    "phase",
    "p_base_map",
    "p_base_series",
    "midround_weight",
    "q_intra_total",
    "q_terms",
    "micro_adj",
    "rails",
    "final",
)


def test_resolve_debug_contains_explain_with_required_keys() -> None:
    """HistoryPoint is built from resolve debug; debug must contain explain with required keys."""
    frame = _frame(
        alive_counts=(4, 2),
        hp_totals=(300.0, 200.0),
        loadout_totals=(8000.0, 4000.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
    )
    config = _config(prematch_map=0.55)
    state = _state()
    rails = (0.2, 0.8)
    _, dbg = resolve_p_hat(frame, config, state, rails)
    assert "explain" in dbg, "resolve_p_hat debug_dict must include explain"
    explain = dbg["explain"]
    for key in REQUIRED_EXPLAIN_KEYS:
        assert key in explain, f"explain must contain {key!r}"
    assert "alive_adj" in explain["micro_adj"]
    assert "hp_adj" in explain["micro_adj"]
    assert "econ_adj" in explain["micro_adj"]
    assert "rail_low" in explain["rails"]
    assert "rail_high" in explain["rails"]
    assert "corridor_width" in explain["rails"]
    assert "p_hat_final" in explain["final"]
    assert "clamp_reason" in explain["final"]


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

    def test_midround_buy_time_frozen_midpoint(self) -> None:
        test_midround_buy_time_frozen_midpoint()

    def test_midround_freezetime_frozen_midpoint(self) -> None:
        test_midround_freezetime_frozen_midpoint()

    def test_midround_in_progress_stays_within_rails(self) -> None:
        test_midround_in_progress_stays_within_rails()

    def test_midround_monotonic_higher_q_intra_raises_p_hat(self) -> None:
        test_midround_monotonic_higher_q_intra_raises_p_hat()

    def test_midround_v2_p_hat_equals_p_mid_clamped_when_rails_wide(self) -> None:
        test_midround_v2_p_hat_equals_p_mid_clamped_when_rails_wide()

    def test_resolve_debug_contains_explain_with_required_keys(self) -> None:
        test_resolve_debug_contains_explain_with_required_keys()


if __name__ == "__main__":
    unittest.main()

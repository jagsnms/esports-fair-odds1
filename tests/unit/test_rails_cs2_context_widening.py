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


def _config(
    context_widening_enabled: bool = False,
    *,
    context_widen_beta: float | None = None,
    uncertainty_mult_min: float | None = None,
    uncertainty_mult_max: float | None = None,
    context_risk_weight_leverage: float | None = None,
    context_risk_weight_fragility: float | None = None,
    context_risk_weight_missingness: float | None = None,
) -> Config:
    c = Config()
    c.context_widening_enabled = context_widening_enabled
    if context_widen_beta is not None:
        c.context_widen_beta = context_widen_beta
    if uncertainty_mult_min is not None:
        c.uncertainty_mult_min = uncertainty_mult_min
    if uncertainty_mult_max is not None:
        c.uncertainty_mult_max = uncertainty_mult_max
    if context_risk_weight_leverage is not None:
        c.context_risk_weight_leverage = context_risk_weight_leverage
    if context_risk_weight_fragility is not None:
        c.context_risk_weight_fragility = context_risk_weight_fragility
    if context_risk_weight_missingness is not None:
        c.context_risk_weight_missingness = context_risk_weight_missingness
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


def test_context_widen_beta_knob_increases_widening() -> None:
    """Higher context_widen_beta should not produce less widening for same frame/risk."""
    state = _state()
    frame = _frame(scores=(8, 8), loadout_totals=(9000.0, 1200.0))
    base_cfg = _config(context_widening_enabled=True, context_widen_beta=0.0)
    hi_cfg = _config(context_widening_enabled=True, context_widen_beta=1.0)
    b = compute_bounds(frame, base_cfg, state)
    bounds = (b[0], b[1])
    _, _, debug_base = compute_rails_cs2(frame, base_cfg, state, bounds)
    _, _, debug_hi = compute_rails_cs2(frame, hi_cfg, state, bounds)
    w_base = debug_base.get("map_width_after_widen")
    w_hi = debug_hi.get("map_width_after_widen")
    assert w_base is not None and w_hi is not None
    assert w_hi >= w_base


def test_context_risk_weight_knobs_affect_mix() -> None:
    """Weights should control the context_risk blend; missingness-only uses missingness_risk."""
    config = _config(
        context_widening_enabled=True,
        context_risk_weight_leverage=0.0,
        context_risk_weight_fragility=0.0,
        context_risk_weight_missingness=1.0,
    )
    state = _state()
    frame = _frame(scores=(11, 11), alive_counts=(5, 5), loadout_totals=None)
    b = compute_bounds(frame, config, state)
    bounds = (b[0], b[1])
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    risk = debug.get("context_risk")
    comps = debug.get("context_risk_components") or {}
    missingness = comps.get("missingness_risk")
    assert risk is not None and missingness is not None
    assert abs(float(risk) - float(missingness)) <= 1e-9
    weights = comps.get("weights") or {}
    assert weights.get("missingness") == 1.0


class TestRailsCs2ContextWidening(unittest.TestCase):
    def test_early_round_width_capped(self) -> None:
        test_early_round_width_capped()

    def test_late_round_cap_larger_than_early(self) -> None:
        test_late_round_cap_larger_than_early()

    def test_debug_contains_context_risk_and_widths(self) -> None:
        test_debug_contains_context_risk_and_widths()

    def test_context_widen_beta_knob_increases_widening(self) -> None:
        test_context_widen_beta_knob_increases_widening()

    def test_context_risk_weight_knobs_affect_mix(self) -> None:
        test_context_risk_weight_knobs_affect_mix()


if __name__ == "__main__":
    unittest.main()

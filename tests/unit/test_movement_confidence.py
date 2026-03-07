"""
Unit tests for time-varying movement confidence (Bible Ch 4 / Ch 6 Step 8).

Validates:
- compute_movement_confidence curve shape (monotonic, bounded, correct values)
- Integration with resolve_p_hat: early-round slow movement, late-round fast convergence
- Backward compatibility: when time_progress is unknown (0.5), confidence ≈ 0.25
"""
from __future__ import annotations

from typing import Any

from engine.compute.resolve import (
    CONF_EXPONENT,
    CONF_MAX,
    CONF_MIN,
    compute_movement_confidence,
    resolve_p_hat,
)
from engine.models import Config, Frame, State


def _frame(
    alive_counts: tuple[int, int] = (5, 3),
    hp_totals: tuple[float, float] = (400.0, 250.0),
    loadout_totals: tuple[float, float] = (10000.0, 6000.0),
    bomb_phase_time_remaining: Any = None,
    round_time_remaining_s: float | None = None,
    a_side: str | None = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=(0.0, 0.0),
        loadout_totals=loadout_totals,
        bomb_phase_time_remaining=bomb_phase_time_remaining,
        round_time_remaining_s=round_time_remaining_s,
        players_a=[],
        players_b=[],
        a_side=a_side,
    )


def _config(prematch_map: float = 0.5) -> Config:
    c = Config()
    c.prematch_map = prematch_map
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


# --- Pure function tests ---


def test_confidence_at_round_start_is_low():
    """Ch 4: early round → slow movement."""
    c = compute_movement_confidence(0.0)
    assert c == CONF_MIN
    assert c < 0.15


def test_confidence_at_round_end_is_high():
    """Ch 4: late round → rapid convergence."""
    c = compute_movement_confidence(1.0)
    assert c == CONF_MAX
    assert c > 0.5


def test_confidence_at_midpoint_matches_legacy():
    """Backward compat: time_progress=0.5 → 0.25 (the old fixed constant)."""
    c = compute_movement_confidence(0.5)
    expected = CONF_MIN + (CONF_MAX - CONF_MIN) * (0.5 ** CONF_EXPONENT)
    assert abs(c - expected) < 1e-12
    assert abs(c - 0.25) < 1e-12


def test_confidence_monotonically_increasing():
    """Bible Ch 4: confidence must increase as round progresses."""
    steps = [i / 20.0 for i in range(21)]
    values = [compute_movement_confidence(tp) for tp in steps]
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1], (
            f"confidence must be non-decreasing: tp={steps[i]}, "
            f"c={values[i]} < c_prev={values[i - 1]}"
        )


def test_confidence_bounded_01():
    """Confidence stays within [CONF_MIN, CONF_MAX] for all valid inputs."""
    for tp in [-0.1, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        c = compute_movement_confidence(tp)
        assert CONF_MIN <= c <= CONF_MAX, f"tp={tp}, c={c}"


def test_confidence_clamps_input():
    """Out-of-range time_progress is clamped to [0, 1]."""
    assert compute_movement_confidence(-1.0) == compute_movement_confidence(0.0)
    assert compute_movement_confidence(2.0) == compute_movement_confidence(1.0)


def test_confidence_acceleration():
    """Gain from 0.5→1.0 is larger than gain from 0.0→0.5 (power curve accelerates)."""
    c_0 = compute_movement_confidence(0.0)
    c_mid = compute_movement_confidence(0.5)
    c_end = compute_movement_confidence(1.0)
    gain_first_half = c_mid - c_0
    gain_second_half = c_end - c_mid
    assert gain_second_half > gain_first_half


# --- Integration with resolve_p_hat ---


def test_resolve_early_round_low_confidence():
    """Early round (100s remaining out of 120s) → low confidence → small p_hat movement."""
    frame = _frame(
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
        round_time_remaining_s=100.0,
    )
    _, dbg = resolve_p_hat(frame, _config(), _state(), (0.2, 0.8))
    weight = dbg["midround_weight"]
    assert weight < 0.15, f"early round should have low weight, got {weight}"


def test_resolve_late_round_high_confidence():
    """Late round (10s remaining out of 120s) → high confidence → large p_hat movement."""
    frame = _frame(
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
        round_time_remaining_s=10.0,
    )
    _, dbg = resolve_p_hat(frame, _config(), _state(), (0.2, 0.8))
    weight = dbg["midround_weight"]
    assert weight > 0.45, f"late round should have high weight, got {weight}"


def test_resolve_unknown_time_defaults_to_legacy():
    """When round_time_remaining_s is None, time_progress defaults to 0.5 → weight ≈ 0.25."""
    frame = _frame(
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
        round_time_remaining_s=None,
    )
    _, dbg = resolve_p_hat(frame, _config(), _state(), (0.2, 0.8))
    weight = dbg["midround_weight"]
    assert abs(weight - 0.25) < 1e-9, f"unknown time should give legacy weight, got {weight}"


def test_resolve_late_round_moves_p_hat_more_than_early():
    """Same microstate at different times: late-round p_hat is further from base than early-round."""
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.1, 0.9)
    bomb = {"round_phase": "IN_PROGRESS"}

    frame_early = _frame(
        bomb_phase_time_remaining=bomb,
        round_time_remaining_s=100.0,
    )
    p_early, _ = resolve_p_hat(frame_early, config, state, rails)

    frame_late = _frame(
        bomb_phase_time_remaining=bomb,
        round_time_remaining_s=10.0,
    )
    p_late, _ = resolve_p_hat(frame_late, config, state, rails)

    base = 0.5
    move_early = abs(p_early - base)
    move_late = abs(p_late - base)
    assert move_late > move_early, (
        f"late-round should move p_hat more: early={move_early}, late={move_late}"
    )


def test_resolve_p_hat_still_clamped_to_rails():
    """Even with high confidence, p_hat stays within rails."""
    frame = _frame(
        alive_counts=(5, 0),
        hp_totals=(500.0, 0.0),
        loadout_totals=(25000.0, 0.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS"},
        round_time_remaining_s=1.0,
    )
    config = _config(prematch_map=0.5)
    state = _state()
    rails = (0.3, 0.7)
    p, _ = resolve_p_hat(frame, config, state, rails)
    assert 0.3 <= p <= 0.7

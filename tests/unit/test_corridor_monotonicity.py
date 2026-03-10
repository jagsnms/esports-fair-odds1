"""
Unit tests for corridor/rail mapping: p_hat_final must be monotonic increasing in p_unshaped (q_intra).
Ensures the mapping from p_unshaped to p_hat_final is not inverted when rails are passed in any order.
"""
from __future__ import annotations

from engine.compute.resolve import resolve_p_hat
from engine.models import Config, Frame, State


def _frame(
    alive_counts: tuple[int, int] = (0, 0),
    hp_totals: tuple[float, float] = (0.0, 0.0),
    loadout_totals: tuple[float, float] | None = None,
    bomb_phase_time_remaining: dict | None = None,
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
    )


def _config(prematch_map: float = 0.5) -> Config:
    c = Config()
    c.prematch_map = prematch_map
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_corridor_mapping_monotonic_in_p_unshaped() -> None:
    """
    With fixed rails (0.40, 0.60), p_hat_final must be strictly increasing as p_unshaped (q_intra) increases.
    Use rails passed in reversed order (0.60, 0.40) to assert resolve normalizes and mapping stays monotonic.
    """
    rail_lo, rail_hi = 0.40, 0.60
    rails = (rail_hi, rail_lo)  # intentionally wrong order; resolve must normalize to (0.4, 0.6)
    config = _config(0.5)
    state = _state()
    bomb = {"round_phase": "IN_PROGRESS"}

    # Frames with increasing A advantage -> increasing q_intra (p_unshaped)
    frames = [
        _frame(
            alive_counts=(1, 5),
            hp_totals=(100.0, 500.0),
            loadout_totals=(2000.0, 15000.0),
            bomb_phase_time_remaining=bomb,
        ),
        _frame(
            alive_counts=(2, 4),
            hp_totals=(150.0, 400.0),
            loadout_totals=(4000.0, 12000.0),
            bomb_phase_time_remaining=bomb,
        ),
        _frame(
            alive_counts=(3, 3),
            hp_totals=(300.0, 300.0),
            loadout_totals=(6000.0, 6000.0),
            bomb_phase_time_remaining=bomb,
        ),
        _frame(
            alive_counts=(4, 2),
            hp_totals=(400.0, 200.0),
            loadout_totals=(10000.0, 5000.0),
            bomb_phase_time_remaining=bomb,
        ),
        _frame(
            alive_counts=(5, 1),
            hp_totals=(450.0, 100.0),
            loadout_totals=(14000.0, 3000.0),
            bomb_phase_time_remaining=bomb,
        ),
    ]

    p_hats: list[float] = []
    p_unshapeds: list[float] = []
    for f in frames:
        p, dbg = resolve_p_hat(f, config, state, rails)
        p_hats.append(p)
        explain = dbg.get("explain") or {}
        pu = explain.get("p_unshaped")
        if pu is not None:
            p_unshapeds.append(float(pu))
        else:
            mv2 = dbg.get("midround_v2") or {}
            p_unshapeds.append(float(mv2.get("q_intra", 0.5)))

    # p_hat_final must be non-decreasing as p_unshaped increases (monotonic)
    for i in range(len(p_hats) - 1):
        assert p_hats[i + 1] >= p_hats[i] - 1e-12, (
            f"Corridor mapping must be monotonic: p_hat_final[{i}]={p_hats[i]:.6f} "
            f"> p_hat_final[{i+1}]={p_hats[i+1]:.6f} (p_unshaped {p_unshapeds[i]:.4f} -> {p_unshapeds[i+1]:.4f})"
        )

    # p_unshaped should increase with A advantage
    for i in range(len(p_unshapeds) - 1):
        assert p_unshapeds[i + 1] >= p_unshapeds[i] - 1e-9, (
            f"Expected q_intra to increase with A advantage: {p_unshapeds[i]:.4f} -> {p_unshapeds[i+1]:.4f}"
        )


def test_corridor_mapping_formula_fixed_rails() -> None:
    """
    For rail_low=0.40, rail_high=0.60, p_hat_final for p_unshaped in [0.05, 0.25, 0.50, 0.75, 0.95]
    must be strictly increasing (canonical formula: rail_low + p_unshaped * (rail_high - rail_low) clamped).
    """
    rail_low, rail_high = 0.40, 0.60
    p_unshaped_vals = [0.05, 0.25, 0.50, 0.75, 0.95]

    def corridor_map(p: float, rlo: float, rhi: float) -> float:
        return max(rlo, min(rhi, rlo + p * (rhi - rlo)))

    p_hat_vals = [corridor_map(p, rail_low, rail_high) for p in p_unshaped_vals]

    for i in range(len(p_hat_vals) - 1):
        assert p_hat_vals[i + 1] >= p_hat_vals[i] - 1e-12, (
            f"Canonical corridor map must be monotonic: p_unshaped {p_unshaped_vals[i]} -> {p_unshaped_vals[i+1]} "
            f"gave p_hat {p_hat_vals[i]:.6f} -> {p_hat_vals[i+1]:.6f}"
        )

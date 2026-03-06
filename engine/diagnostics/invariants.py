"""
Invariant checks and diagnostic helpers for the pipeline.

Used to validate Derived/State/HistoryPoint before publish or replay.
"""
from __future__ import annotations

from typing import Any

from engine.models import Frame, State, HistoryPoint


def check_prob_range(p: float, low: float, high: float) -> list[str]:
    """
    Ensure p and [low, high] are in [0, 1] and low <= p <= high.
    Returns list of violation messages (empty if ok).
    """
    violations: list[str] = []
    if not (0 <= p <= 1):
        violations.append(f"p_hat out of range: {p}")
    if not (0 <= low <= 1):
        violations.append(f"bound_low out of range: {low}")
    if not (0 <= high <= 1):
        violations.append(f"bound_high out of range: {high}")
    if low > high:
        violations.append(f"bound_low > bound_high: {low} > {high}")
    if p < low or p > high:
        violations.append(f"p_hat outside [bound_low, bound_high]: {p} not in [{low}, {high}]")
    return violations


def check_monotonic_time(prev_time: float, point: HistoryPoint) -> list[str]:
    """
    Ensure point.time >= prev_time (for append-only history).
    Returns list of violation messages (empty if ok).
    """
    if point.time < prev_time:
        return [f"History time went backwards: {point.time} < {prev_time}"]
    return []


def check_team_identity_consistency(state: State, frame: Frame | None) -> list[str]:
    """
    Placeholder: ensure team identity in state is consistent with frame when lock is set.
    Returns list of violation messages (empty if ok).
    """
    # Stub: no-op until identity locking logic is implemented
    return []


def compute_corridor_invariants(
    *,
    series_low: float,
    series_high: float,
    map_low: float,
    map_high: float,
    p_hat: float,
    testing_mode: bool = True,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """
    Compute corridor invariants and classify violations.

    Structural violations:
    - series_map_order: corridor ordering is broken.

    Behavioral diagnostics:
    - p_hat_outside_map: p_hat escaped map corridor.

    In testing_mode, both structural and behavioral violations are reported in
    invariant_violations. Outside testing mode, invariant_violations contains
    structural violations only.
    """
    order_ok = (series_low - eps) <= map_low <= map_high <= (series_high + eps)
    p_in_map_ok = (map_low - eps) <= p_hat <= (map_high + eps)

    structural_violations: list[str] = []
    behavioral_violations: list[str] = []

    if not order_ok:
        structural_violations.append("series_map_order")
    if not p_in_map_ok:
        behavioral_violations.append("p_hat_outside_map")

    if testing_mode:
        primary_violations = structural_violations + behavioral_violations
    else:
        primary_violations = structural_violations[:]

    return {
        "invariant_mode": "testing" if testing_mode else "production",
        "invariant_series_map_order_ok": order_ok,
        "invariant_p_hat_in_map_ok": p_in_map_ok,
        "invariant_structural_violations": structural_violations,
        "invariant_behavioral_violations": behavioral_violations,
        "invariant_structural_ok": len(structural_violations) == 0,
        "invariant_behavioral_ok": len(behavioral_violations) == 0,
        "invariant_violations": primary_violations,
    }


def compute_phat_contract_diagnostics(
    *,
    q_intra_total: float | None,
    rail_low: float,
    rail_high: float,
    p_hat_prev: float,
    p_hat_final: float,
    movement_confidence: float,
    phase: str | None,
    round_time_remaining_s: float | None,
    is_bomb_planted: bool | None,
    alive_counts: tuple[int, int] | None,
    hp_totals: tuple[float, float] | None,
    loadout_totals: tuple[float, float] | None,
    round_phase: str | None,
    round_number: int | None,
    testing_mode: bool,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """
    Compute Bible-contract diagnostics for target/movement without changing runtime output.

    Structural diagnostics:
    - q_out_of_bounds
    - rail_order_invalid

    Behavioral diagnostics (testing mode only):
    - movement_contract_gap
      p_hat_final differs from expected movement step:
      p_expected = p_hat_prev + confidence * (target - p_hat_prev)
    """
    structural_violations: list[str] = []
    behavioral_violations: list[str] = []

    q_ok = isinstance(q_intra_total, (int, float)) and 0.0 <= float(q_intra_total) <= 1.0
    rails_ok = rail_low <= rail_high + eps
    if not q_ok:
        structural_violations.append("q_out_of_bounds")
    if not rails_ok:
        structural_violations.append("rail_order_invalid")

    q_used = float(q_intra_total) if q_ok else None
    target_p_hat = None
    if q_used is not None:
        target_p_hat = rail_low + q_used * (rail_high - rail_low)

    confidence = max(0.0, min(1.0, float(movement_confidence)))
    expected_after_movement = None
    if target_p_hat is not None:
        expected_after_movement = p_hat_prev + confidence * (target_p_hat - p_hat_prev)

    movement_gap = None
    if expected_after_movement is not None:
        movement_gap = abs(float(p_hat_final) - float(expected_after_movement))
        # Test-mode visibility for current semantic drift from coupling+movement contract.
        if testing_mode and (phase or "").strip().upper() == "IN_PROGRESS" and movement_gap > 0.02:
            behavioral_violations.append("movement_contract_gap")

    return {
        "contract_testing_mode": bool(testing_mode),
        "phase": phase,
        "round_time_remaining_s": round_time_remaining_s,
        "is_bomb_planted": is_bomb_planted,
        "alive_counts": alive_counts,
        "hp_totals": hp_totals,
        "loadout_totals": loadout_totals,
        "round_phase": round_phase,
        "round_number": round_number,
        "q_intra_total": q_used,
        "rail_low": float(rail_low),
        "rail_high": float(rail_high),
        "target_p_hat": target_p_hat,
        "p_hat_prev": float(p_hat_prev),
        "p_hat_final": float(p_hat_final),
        "movement_confidence": confidence,
        "expected_p_hat_after_movement": expected_after_movement,
        "movement_gap_abs": movement_gap,
        "structural_violations": structural_violations,
        "behavioral_violations": behavioral_violations,
        "structural_ok": len(structural_violations) == 0,
        "behavioral_ok": len(behavioral_violations) == 0,
    }

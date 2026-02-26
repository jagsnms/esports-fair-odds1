"""
Placeholder invariant checks for the pipeline.

Used to validate Derived/State/HistoryPoint before publish or replay.
"""
from __future__ import annotations

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

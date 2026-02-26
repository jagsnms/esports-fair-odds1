"""
Compute certainty bounds [bound_low, bound_high] from frame/state/config.
Map scope: CS2 map-state corridor (bounds_cs2); fallback simple method on exception.
"""
from __future__ import annotations

import math

from engine.models import Config, Frame, State

# MR12: first to 13, max 24 rounds (used by fallback)
WIN_TARGET = 13
MAX_ROUNDS_MAP = 24


def _compute_bounds_simple(frame: Frame, config: Config, state: State) -> tuple[float, float]:
    """Fallback: center 0.5, half-width shrinks with rounds (wide early, narrow late)."""
    ra = max(0, getattr(frame, "scores", (0, 0))[0])
    rb = max(0, getattr(frame, "scores", (0, 0))[1])
    rounds_played = ra + rb
    total_rounds = max(1, MAX_ROUNDS_MAP)
    frac_remaining = max(0.0, (total_rounds - rounds_played) / total_rounds)
    half_width = 0.25 * math.sqrt(frac_remaining)
    half_width = max(0.01, min(0.49, half_width))
    center = 0.5
    bound_low = max(0.01, center - half_width)
    bound_high = min(0.99, center + half_width)
    if bound_high <= bound_low:
        bound_high = min(0.99, bound_low + 0.02)
    return (bound_low, bound_high)


def compute_bounds(frame: Frame, config: Config, state: State) -> tuple[float, float]:
    """
    Return (bound_low, bound_high) in [0, 1].
    Map scope: compute_bounds_cs2 (score + series corridor); fallback to simple on exception.
    Non-map scope: (0.01, 0.99).
    """
    scope = getattr(config, "contract_scope", None) or ""
    is_map = scope == "" or (
        isinstance(scope, str) and "map" in scope.lower() and "series" not in scope.lower()
    )
    if not is_map:
        return (0.01, 0.99)
    try:
        from engine.compute.bounds_cs2 import compute_bounds_cs2
        return compute_bounds_cs2(frame, config, state)
    except Exception:
        return _compute_bounds_simple(frame, config, state)

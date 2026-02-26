"""
Compute certainty bounds [bound_low, bound_high] from frame/state/config.
Series/map scope: CS2 series corridor (bounds_cs2, i.i.d macro-only); fallback simple on exception.
Returns (bound_low, bound_high, debug_dict); debug_dict may be empty.
"""
from __future__ import annotations

import math
from typing import Any

from engine.models import Config, Frame, State

# MR12: first to 13, max 24 rounds (used by fallback)
WIN_TARGET = 13
MAX_ROUNDS_MAP = 24


def _compute_bounds_simple(frame: Frame, config: Config, state: State) -> tuple[float, float, dict[str, Any]]:
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
    return (bound_low, bound_high, {})


def compute_bounds(frame: Frame, config: Config, state: State) -> tuple[float, float, dict[str, Any]]:
    """
    Return (bound_low, bound_high, bounds_debug) in [0, 1].
    Series/map scope: compute_bounds_cs2 (i.i.d series corridor); fallback to simple on exception.
    Non-series/map scope: (0.01, 0.99), {}.
    """
    scope = getattr(config, "contract_scope", None) or ""
    scope_lower = scope.lower() if isinstance(scope, str) else ""
    use_series_corridor = scope == "" or "map" in scope_lower or "series" in scope_lower
    if not use_series_corridor:
        return (0.01, 0.99, {})
    try:
        from engine.compute.bounds_cs2 import compute_bounds_cs2
        lo, hi, debug = compute_bounds_cs2(frame, config, state)
        return (lo, hi, debug)
    except Exception:
        lo, hi, _ = _compute_bounds_simple(frame, config, state)
        return (lo, hi, {})

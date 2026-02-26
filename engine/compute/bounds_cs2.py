"""
CS2 series corridor (outer bounds): i.i.d. macro-only.
Uses series_win_prob_live with p_current_map=0/1; no microstate or round score.
Deterministic; stdlib only.
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.series_iid import series_win_prob_live


def _n_maps_from_series_fmt(series_fmt: str) -> int:
    """Best-of: bo3 -> 3, bo5 -> 5."""
    try:
        n = int(str(series_fmt or "bo3").replace("bo", "").strip() or "3")
    except (ValueError, TypeError):
        n = 3
    return 3 if n not in (3, 5) else n


def compute_bounds_cs2(frame: Frame, config: Config, state: State) -> tuple[float, float, dict[str, Any]]:
    """
    Series corridor (outer): series_low and series_high from i.i.d. series model.
    - series_low = series_win_prob_live(n_maps, maps_a_won, maps_b_won, p_current_map=0.0, p_future_map=p0_map)
    - series_high = series_win_prob_live(..., p_current_map=1.0, p_future_map=p0_map)
    Does NOT use microstate or intra-map round score. Only series_score, series_fmt, config.prematch_map.
    Returns (series_low, series_high, debug_dict).
    """
    series = getattr(frame, "series_score", (0, 0))
    maps_a_won = int(series[0]) if len(series) > 0 and series[0] is not None else 0
    maps_b_won = int(series[1]) if len(series) > 1 and series[1] is not None else 0
    series_fmt = getattr(frame, "series_fmt", "bo3") or "bo3"
    n_maps = _n_maps_from_series_fmt(series_fmt)
    p0_map = getattr(config, "prematch_map", None)
    if p0_map is not None and isinstance(p0_map, (int, float)):
        p0_map = max(1e-6, min(1.0 - 1e-6, float(p0_map)))
    else:
        p0_map = 0.5

    series_low = series_win_prob_live(n_maps, maps_a_won, maps_b_won, 0.0, p0_map)
    series_high = series_win_prob_live(n_maps, maps_a_won, maps_b_won, 1.0, p0_map)
    # Ensure lo <= hi
    if series_low > series_high:
        series_low, series_high = series_high, series_low
    series_low = max(0.0, min(1.0, series_low))
    series_high = max(0.0, min(1.0, series_high))

    debug: dict[str, Any] = {
        "series_corridor_n_maps": n_maps,
        "series_corridor_maps_a_won": maps_a_won,
        "series_corridor_maps_b_won": maps_b_won,
        "series_corridor_p0_map": p0_map,
        "series_low": series_low,
        "series_high": series_high,
    }
    return (series_low, series_high, debug)

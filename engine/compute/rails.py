"""
Compute rails [rail_low, rail_high] from frame/state/config and bounds.
Delegates to CS2 round-state envelope logic (rails_cs2); fallback pass-through if needed.
"""
from __future__ import annotations

from engine.models import Config, Frame, State

from engine.compute.rails_cs2 import compute_rails_cs2


def compute_rails(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """Return (rail_low, rail_high). Uses compute_rails_cs2 (round-state envelope)."""
    try:
        return compute_rails_cs2(frame, config, state, bounds)
    except Exception:
        lo, hi = bounds
        return (max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi)))

"""
Resolve p_hat from frame/config/state and rails. No score hack; base clamped into rails.
"""
from __future__ import annotations

from engine.models import Config, Frame, State


def resolve_p_hat(
    frame: Frame,
    config: Config,
    state: State,
    rails: tuple[float, float],
) -> float:
    """Base = prematch_map if set else 0.5; return base clamped into [rail_low, rail_high]."""
    rail_low, rail_high = rails
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        base = max(0.0, min(1.0, float(prematch)))
    else:
        base = 0.5
    return max(rail_low, min(rail_high, base))

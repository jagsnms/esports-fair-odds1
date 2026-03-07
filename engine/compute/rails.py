"""
Compute rails [rail_low, rail_high] (map corridor) from frame/state/config and bounds.
Delegates to CS2 round-state envelope + contextual widening (rails_cs2); fallback pass-through if needed.
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.rails_cs2 import compute_rails_cs2


def compute_rails(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
    *,
    source: str | None = None,
    replay_kind: str | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Return (rail_low, rail_high, rails_debug). Uses compute_rails_cs2 (envelope + contextual widening)."""
    try:
        return compute_rails_cs2(frame, config, state, bounds, source=source, replay_kind=replay_kind)
    except Exception:
        lo, hi = bounds
        return (max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi)), {})

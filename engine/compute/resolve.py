"""
Resolve p_hat from frame/config/state and rails. Base + microstate adjustment, clamped into rails.
Optional midround blend toward q_intra when config.midround_enabled is True.
Returns (p_hat, debug_dict).
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.micro_adj_cs2 import micro_adjustment_cs2
from engine.compute.q_intra_cs2 import compute_q_intra_cs2

# Max weight for blending toward q_intra when midround_enabled (conservative).
MIDROUND_W_MAX = 0.25


def resolve_p_hat(
    frame: Frame,
    config: Config,
    state: State,
    rails: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    """Base = prematch_map if set else 0.5; add micro_adjustment_cs2(frame); clamp into rails and [0,1].
    If config.midround_enabled: blend result slightly toward q_intra (w=0.25), then clamp to [0,1] and rails.
    Returns (p_hat, debug_dict).
    """
    rail_low, rail_high = rails
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        base = max(0.0, min(1.0, float(prematch)))
    else:
        base = 0.5
    adj = micro_adjustment_cs2(frame)
    p_pre_clamp01 = base + adj
    p_post_clamp01 = max(0.0, min(1.0, p_pre_clamp01))
    p_hat_old = max(rail_low, min(rail_high, p_post_clamp01))

    q_intra, q_intra_debug = compute_q_intra_cs2(frame, config=config)
    midround_enabled = getattr(config, "midround_enabled", False)

    if not midround_enabled:
        p_hat_blend_pre_rails = p_hat_old
        p_hat_final = p_hat_old
        w = 0.0
    else:
        w = MIDROUND_W_MAX
        p_hat_blend = (1.0 - w) * p_hat_old + w * q_intra
        p_hat_blend_pre_rails = max(0.0, min(1.0, p_hat_blend))
        p_hat_final = max(rail_low, min(rail_high, p_hat_blend_pre_rails))

    debug_dict: dict[str, Any] = {
        "p_hat_base": base,
        "micro_adj": adj,
        "p_hat_pre_clamp01": p_pre_clamp01,
        "p_hat_post_clamp01": p_post_clamp01,
        "rail_low": rail_low,
        "rail_high": rail_high,
        "q_intra": q_intra_debug,
        "midround_enabled": midround_enabled,
        "midround_weight": w,
        "p_hat_old": p_hat_old,
        "p_hat_blend_pre_rails": p_hat_blend_pre_rails,
        "p_hat_final": p_hat_final,
    }
    if "reason" in q_intra_debug:
        debug_dict["reason"] = q_intra_debug["reason"]

    return p_hat_final, debug_dict

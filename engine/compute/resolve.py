"""
Resolve p_hat from frame/config/state and rails. Base + microstate adjustment, clamped into rails.
When config.midround_enabled is True, uses midround V2 oracle (frozen rails, p_mid_clamped).
Returns (p_hat, debug_dict).
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.micro_adj_cs2 import micro_adjustment_cs2
from engine.compute.midround_v2_cs2 import (
    apply_cs2_midround_adjustment_v2_mixture,
    compute_cs2_midround_features,
)
from engine.compute.q_intra_cs2 import compute_q_intra_cs2


def resolve_p_hat(
    frame: Frame,
    config: Config,
    state: State,
    rails: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    """Base = prematch_map if set else 0.5; add micro_adjustment_cs2(frame); clamp into rails and [0,1].
    If config.midround_enabled: use midround V2 oracle (frozen_a=rail_high, frozen_b=rail_low) during
    IN_PROGRESS rounds (or when round_phase unknown); p_hat_final = clamp_to_rails(p_mid_clamped).
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

    midround_v2_result: dict[str, Any] | None = None
    phase_unknown = True

    if not midround_enabled:
        p_hat_final = p_hat_old
        midround_weight = 0.0
    else:
        midround_weight = 0.25
        features = compute_cs2_midround_features(frame, config=config)
        round_phase = features.get("round_phase")
        phase_unknown = round_phase is None
        is_live_round_phase = phase_unknown or (
            str(round_phase).lower() not in ("ended", "freezetime", "warmup")
            if round_phase is not None
            else True
        )
        if not is_live_round_phase:
            p_hat_final = p_hat_old
            midround_v2_result = {"skipped": True, "reason": "round_not_in_progress", "round_phase": round_phase}
        else:
            result = apply_cs2_midround_adjustment_v2_mixture(
                frozen_a=rail_high,
                frozen_b=rail_low,
                features=features,
            )
            midround_v2_result = result
            p_mid_clamped = result["p_mid_clamped"]
            p_hat_final = max(rail_low, min(rail_high, p_mid_clamped))

    debug_dict: dict[str, Any] = {
        "p_hat_base": base,
        "micro_adj": adj,
        "p_hat_pre_clamp01": p_pre_clamp01,
        "p_hat_post_clamp01": p_post_clamp01,
        "rail_low": rail_low,
        "rail_high": rail_high,
        "q_intra": q_intra_debug,
        "midround_enabled": midround_enabled,
        "midround_weight": midround_weight,
        "p_hat_old": p_hat_old,
        "p_hat_final": p_hat_final,
        "midround_v2": midround_v2_result,
        "phase_unknown": phase_unknown,
    }
    if "reason" in q_intra_debug:
        debug_dict["reason"] = q_intra_debug["reason"]

    return p_hat_final, debug_dict

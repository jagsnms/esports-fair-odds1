"""
Resolve p_hat from frame/config/state and rails. Base + microstate adjustment, clamped into rails.
Midround V2 is always computed; applied only when round_phase is IN_PROGRESS (phase-gated).
Returns (p_hat, debug_dict).
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.micro_adj_cs2 import micro_adjustment_cs2, micro_adjustment_cs2_breakdown
from engine.compute.midround_v2_cs2 import (
    apply_cs2_midround_adjustment_v2_mixture,
    compute_cs2_midround_features,
)
from engine.compute.q_intra_cs2 import compute_q_intra_cs2

# Only apply V2 mixture when round is in progress (phase-gated)
IN_PROGRESS_PHASES = frozenset({"IN_PROGRESS"})
# BUY_TIME / FREEZETIME: freeze p_hat at map rail midpoint (no loadout pegging)
BUY_TIME_PHASES = frozenset({"BUY_TIME", "FREEZETIME"})


def _build_explain(
    *,
    phase: str | None,
    p_base_map: float,
    p_base_series: float | None,
    midround_weight: float,
    midround_v2_result: dict[str, Any],
    q_intra_debug: dict[str, Any],
    micro_adj_breakdown: dict[str, float],
    rail_low: float,
    rail_high: float,
    p_hat_final: float,
) -> dict[str, Any]:
    """Build per-tick explain dict for HistoryPoint (logging / calibration / ML)."""
    q_intra_total = midround_v2_result.get("q_intra")
    if q_intra_total is None:
        q_intra_total = q_intra_debug.get("q_intra_round_win_a")
    q_terms: dict[str, Any] = {}
    for k in ("term_alive", "term_hp", "term_loadout", "term_bomb", "term_cash"):
        if k in midround_v2_result:
            q_terms[k] = midround_v2_result[k]
    corridor_width = rail_high - rail_low if rail_high >= rail_low else 0.0
    clamp_reason: str | None = None
    if rail_low >= rail_high:
        clamp_reason = "rails_collapsed"
    elif p_hat_final <= rail_low:
        clamp_reason = "rail_low"
    elif p_hat_final >= rail_high:
        clamp_reason = "rail_high"
    return {
        "phase": phase,
        "p_base_map": p_base_map,
        "p_base_series": p_base_series,
        "midround_weight": midround_weight,
        "q_intra_total": q_intra_total,
        "q_terms": q_terms,
        "micro_adj": micro_adj_breakdown,
        "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": corridor_width},
        "final": {"p_hat_final": p_hat_final, "clamp_reason": clamp_reason},
    }


def resolve_p_hat(
    frame: Frame,
    config: Config,
    state: State,
    rails: tuple[float, float],
) -> tuple[float, dict[str, Any]]:
    """Base = prematch_map if set else 0.5; add micro_adjustment_cs2(frame); clamp into rails and [0,1].
    Midround V2: always compute features; apply mixture only when round_phase in IN_PROGRESS.
    Else p_hat_final = p_hat_old and debug midround_v2 has skipped=True, reason=phase_not_in_progress.
    Returns (p_hat, debug_dict).
    """
    rail_low, rail_high = rails
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        base = max(0.0, min(1.0, float(prematch)))
    else:
        base = 0.5
    adj = micro_adjustment_cs2(frame)
    micro_adj_breakdown = micro_adjustment_cs2_breakdown(frame)
    p_pre_clamp01 = base + adj
    p_post_clamp01 = max(0.0, min(1.0, p_pre_clamp01))
    p_hat_old = max(rail_low, min(rail_high, p_post_clamp01))

    q_intra, q_intra_debug = compute_q_intra_cs2(frame, config=config)
    features = compute_cs2_midround_features(frame, config=config)
    round_phase = features.get("round_phase")
    phase_unknown = round_phase is None
    phase_upper = str(round_phase).strip().upper() if round_phase is not None else None
    apply_v2 = phase_upper in IN_PROGRESS_PHASES

    midround_v2_result: dict[str, Any]
    if phase_upper in BUY_TIME_PHASES:
        p_hat_final = 0.5 * (rail_low + rail_high)
        midround_weight = 0.0
        midround_v2_result = {
            "skipped": True,
            "reason": "buy_time_frozen_midpoint",
            "round_phase": round_phase,
        }
        explain = _build_explain(
            phase=phase_upper,
            p_base_map=base,
            p_base_series=None,
            midround_weight=midround_weight,
            midround_v2_result=midround_v2_result,
            q_intra_debug=q_intra_debug,
            micro_adj_breakdown=micro_adj_breakdown,
            rail_low=rail_low,
            rail_high=rail_high,
            p_hat_final=p_hat_final,
        )
        debug_dict = {
            "p_hat_base": base,
            "micro_adj": adj,
            "p_hat_pre_clamp01": p_pre_clamp01,
            "p_hat_post_clamp01": p_post_clamp01,
            "rail_low": rail_low,
            "rail_high": rail_high,
            "q_intra": q_intra_debug,
            "midround_weight": midround_weight,
            "p_hat_old": p_hat_old,
            "p_hat_final": p_hat_final,
            "midround_v2": midround_v2_result,
            "phase_unknown": phase_unknown,
            "explain": explain,
        }
        if "reason" in q_intra_debug:
            debug_dict["reason"] = q_intra_debug["reason"]
        return p_hat_final, debug_dict

    if not apply_v2:
        p_hat_final = p_hat_old
        midround_weight = 0.0
        midround_v2_result = {
            "skipped": True,
            "reason": "phase_not_in_progress",
            "round_phase": round_phase,
        }
    else:
        midround_weight = 0.25
        result = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=rail_high,
            frozen_b=rail_low,
            features=features,
        )
        midround_v2_result = result
        p_mid_clamped = result["p_mid_clamped"]
        p_hat_final = max(rail_low, min(rail_high, p_mid_clamped))

    explain = _build_explain(
        phase=phase_upper,
        p_base_map=base,
        p_base_series=None,
        midround_weight=midround_weight,
        midround_v2_result=midround_v2_result,
        q_intra_debug=q_intra_debug,
        micro_adj_breakdown=micro_adj_breakdown,
        rail_low=rail_low,
        rail_high=rail_high,
        p_hat_final=p_hat_final,
    )
    debug_dict = {
        "p_hat_base": base,
        "micro_adj": adj,
        "p_hat_pre_clamp01": p_pre_clamp01,
        "p_hat_post_clamp01": p_post_clamp01,
        "rail_low": rail_low,
        "rail_high": rail_high,
        "q_intra": q_intra_debug,
        "midround_weight": midround_weight,
        "p_hat_old": p_hat_old,
        "p_hat_final": p_hat_final,
        "midround_v2": midround_v2_result,
        "phase_unknown": phase_unknown,
        "explain": explain,
    }
    if "reason" in q_intra_debug:
        debug_dict["reason"] = q_intra_debug["reason"]

    return p_hat_final, debug_dict

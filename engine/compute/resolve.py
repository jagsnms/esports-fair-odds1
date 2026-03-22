"""
Resolve p_hat from frame/config/state and rails. Base + microstate adjustment, clamped into rails.
Midround V2 is always computed; applied only when round_phase is IN_PROGRESS (phase-gated).
Returns (p_hat, debug_dict).
"""
from __future__ import annotations

from typing import Any

from engine.diagnostics.invariants import compute_phat_contract_diagnostics
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
    p_hat_prev_source: str,
    midround_v2_result: dict[str, Any],
    q_intra_debug: dict[str, Any],
    micro_adj_breakdown: dict[str, float],
    rail_low: float,
    rail_high: float,
    p_hat_final: float,
    contract_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build per-tick explain dict for HistoryPoint (logging / calibration / ML).
    When midround_v2_result has raw_score_pre_urgency, adds score-space diagnostics for
    history_score_points.jsonl: score_raw, term_contribs, base_intercept, p_unshaped.
    """
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
    out: dict[str, Any] = {
        "phase": phase,
        "p_base_map": p_base_map,
        "p_base_series": p_base_series,
        "midround_weight": midround_weight,
        "p_hat_prev_source": p_hat_prev_source,
        "q_intra_total": q_intra_total,
        "q_terms": q_terms,
        "micro_adj": micro_adj_breakdown,
        "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": corridor_width},
        "final": {"p_hat_final": p_hat_final, "clamp_reason": clamp_reason},
    }
    if isinstance(contract_diag, dict):
        for key in (
            "round_phase",
            "alive_counts",
            "hp_totals",
            "loadout_totals",
            "target_p_hat",
            "p_hat_prev",
            "movement_confidence",
            "expected_p_hat_after_movement",
            "movement_gap_abs",
        ):
            if key in contract_diag:
                out[key] = contract_diag.get(key)
    # Score-space diagnostics (pre-asymptote): raw score and additive term contribs for history_score_points.jsonl
    raw_pre = midround_v2_result.get("raw_score_pre_urgency")
    if raw_pre is not None:
        term_contribs = {
            k: midround_v2_result.get(k, 0.0)
            for k in ("term_alive", "term_hp", "term_loadout", "term_bomb", "term_cash")
        }
        out["score_raw"] = float(raw_pre)
        out["term_contribs"] = {k: float(v) for k, v in term_contribs.items()}
        out["base_intercept"] = 0.0
        q_intra = midround_v2_result.get("q_intra")
        out["p_unshaped"] = float(q_intra) if q_intra is not None else None
        # Pre-weight raw signals and coefficients for learning true weights (score_diag_v2)
        term_raw = midround_v2_result.get("term_raw")
        term_coef = midround_v2_result.get("term_coef")
        if isinstance(term_raw, dict) and isinstance(term_coef, dict):
            out["term_raw"] = {k: float(v) for k, v in term_raw.items() if isinstance(v, (int, float))}
            out["term_coef"] = {k: float(v) for k, v in term_coef.items() if isinstance(v, (int, float))}
        wp = midround_v2_result.get("weight_profile")
        if isinstance(wp, str):
            out["weight_profile"] = wp
    return out


def _contract_diag(
    *,
    frame: Frame,
    config: Config,
    phase: str | None,
    q_intra_debug: dict[str, Any],
    midround_v2_result: dict[str, Any],
    rail_low: float,
    rail_high: float,
    p_hat_old: float,
    p_hat_final: float,
    movement_confidence: float,
) -> dict[str, Any]:
    q_intra_total = midround_v2_result.get("q_intra")
    if q_intra_total is None:
        q_intra_total = q_intra_debug.get("q_intra_round_win_a")
    testing_mode = bool(getattr(config, "contract_testing_mode", False))
    round_time_remaining_s = getattr(frame, "round_time_remaining_s", None)
    if not isinstance(round_time_remaining_s, (int, float)):
        round_time_remaining_s = None
    bomb_phase = getattr(frame, "bomb_phase_time_remaining", None)
    is_bomb_planted = None
    round_phase = None
    round_number = None
    if isinstance(bomb_phase, dict) and "is_bomb_planted" in bomb_phase:
        bp = bomb_phase.get("is_bomb_planted")
        is_bomb_planted = bool(bp) if bp is not None else None
    if isinstance(bomb_phase, dict):
        if isinstance(bomb_phase.get("round_phase"), str):
            round_phase = bomb_phase.get("round_phase")
        rn = bomb_phase.get("round_number")
        if isinstance(rn, int):
            round_number = rn
    alive_counts = getattr(frame, "alive_counts", None)
    if (
        not isinstance(alive_counts, (tuple, list))
        or len(alive_counts) != 2
        or not all(isinstance(v, int) for v in alive_counts)
    ):
        alive_counts = None
    else:
        alive_counts = (int(alive_counts[0]), int(alive_counts[1]))
    hp_totals = getattr(frame, "hp_totals", None)
    if (
        not isinstance(hp_totals, (tuple, list))
        or len(hp_totals) != 2
        or not all(isinstance(v, (int, float)) for v in hp_totals)
    ):
        hp_totals = None
    else:
        hp_totals = (float(hp_totals[0]), float(hp_totals[1]))
    loadout_totals = getattr(frame, "loadout_totals", None)
    if (
        not isinstance(loadout_totals, (tuple, list))
        or len(loadout_totals) != 2
        or not all(isinstance(v, (int, float)) for v in loadout_totals)
    ):
        loadout_totals = None
    else:
        loadout_totals = (float(loadout_totals[0]), float(loadout_totals[1]))
    return compute_phat_contract_diagnostics(
        q_intra_total=q_intra_total,
        rail_low=rail_low,
        rail_high=rail_high,
        p_hat_prev=p_hat_old,
        p_hat_final=p_hat_final,
        movement_confidence=movement_confidence,
        phase=phase,
        round_time_remaining_s=float(round_time_remaining_s) if round_time_remaining_s is not None else None,
        is_bomb_planted=is_bomb_planted,
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        loadout_totals=loadout_totals,
        round_phase=round_phase,
        round_number=round_number,
        testing_mode=testing_mode,
        timer_contract={
            "timer_contract_version": midround_v2_result.get("timer_contract_version"),
            "timer_state": midround_v2_result.get("timer_state"),
            "timer_source_class": midround_v2_result.get("timer_source_class"),
            "timer_remaining_s": midround_v2_result.get("timer_remaining_s"),
            "timer_valid": midround_v2_result.get("timer_valid"),
            "a_side_used": midround_v2_result.get("a_side_used"),
            "timer_direction_expected": midround_v2_result.get("timer_direction_expected"),
            "timer_direction_applied": midround_v2_result.get("timer_direction_applied"),
            "timer_direction_term": midround_v2_result.get("timer_direction_term"),
            "timer_direction_reason_code": midround_v2_result.get("timer_direction_reason_code"),
            "defuse_time_s": midround_v2_result.get("defuse_time_s"),
            "defuse_time_source": midround_v2_result.get("defuse_time_source"),
            "hard_boundary_active": midround_v2_result.get("hard_boundary_active"),
            "hard_boundary_reason_code": midround_v2_result.get("hard_boundary_reason_code"),
        },
    )


def resolve_p_hat(
    frame: Frame,
    config: Config,
    state: State,
    rails: tuple[float, float],
    *,
    p_hat_prev: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve current PHAT inside the ordered rail corridor.

    `p_hat_prev` is the true carried-forward prior PHAT state when available.
    If absent, resolve falls back to the legacy reconstructed anchor
    `prematch_map + micro_adjustment_cs2(frame)` clamped into rails.

    Midround V2: always compute features; apply mixture only when round_phase in IN_PROGRESS.
    Else p_hat_final preserves the carried-forward anchor and debug midround_v2 reports the hold reason.
    Returns (p_hat, debug_dict).
    """
    # Ensure corridor is ordered so mixture is monotonic: p_hat increases with q_intra (p_unshaped).
    # If rails were (hi, lo), mixture would invert; normalize so rail_low <= rail_high.
    _r0, _r1 = float(rails[0]), float(rails[1])
    rail_low = min(_r0, _r1)
    rail_high = max(_r0, _r1)
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        base = max(0.0, min(1.0, float(prematch)))
    else:
        base = 0.5
    adj = micro_adjustment_cs2(frame)
    micro_adj_breakdown = micro_adjustment_cs2_breakdown(frame)
    p_pre_clamp01 = base + adj
    p_post_clamp01 = max(0.0, min(1.0, p_pre_clamp01))
    reconstructed_p_hat = max(rail_low, min(rail_high, p_post_clamp01))
    if isinstance(p_hat_prev, (int, float)):
        p_hat_old = max(rail_low, min(rail_high, float(p_hat_prev)))
        p_hat_prev_source = "carried_forward"
    else:
        p_hat_old = reconstructed_p_hat
        p_hat_prev_source = "reconstructed_fallback"

    q_intra, q_intra_debug = compute_q_intra_cs2(frame, config=config)
    features = compute_cs2_midround_features(frame, config=config)
    round_phase = features.get("round_phase")
    phase_unknown = round_phase is None
    phase_upper = str(round_phase).strip().upper() if round_phase is not None else None
    apply_v2 = phase_upper in IN_PROGRESS_PHASES

    midround_v2_result: dict[str, Any]
    if phase_upper in BUY_TIME_PHASES:
        p_hat_final = p_hat_old
        midround_weight = 0.0
        midround_v2_result = {
            "skipped": True,
            "reason": "buy_time_hold_previous_phat",
            "round_phase": round_phase,
        }
        contract_diag = _contract_diag(
            frame=frame,
            config=config,
            phase=phase_upper,
            q_intra_debug=q_intra_debug,
            midround_v2_result=midround_v2_result,
            rail_low=rail_low,
            rail_high=rail_high,
            p_hat_old=p_hat_old,
            p_hat_final=p_hat_final,
            movement_confidence=midround_weight,
        )
        explain = _build_explain(
            phase=phase_upper,
            p_base_map=base,
            p_base_series=None,
            midround_weight=midround_weight,
            p_hat_prev_source=p_hat_prev_source,
            midround_v2_result=midround_v2_result,
            q_intra_debug=q_intra_debug,
            micro_adj_breakdown=micro_adj_breakdown,
            rail_low=rail_low,
            rail_high=rail_high,
            p_hat_final=p_hat_final,
            contract_diag=contract_diag,
        )
        debug_dict = {
            "p_hat_base": base,
            "micro_adj": adj,
            "p_hat_pre_clamp01": p_pre_clamp01,
            "p_hat_post_clamp01": p_post_clamp01,
            "p_hat_reconstructed": reconstructed_p_hat,
            "p_hat_prev_source": p_hat_prev_source,
            "rail_low": rail_low,
            "rail_high": rail_high,
            "q_intra": q_intra_debug,
            "midround_weight": midround_weight,
            "p_hat_old": p_hat_old,
            "p_hat_final": p_hat_final,
            "midround_v2": midround_v2_result,
            "phase_unknown": phase_unknown,
            "explain": explain,
            "contract_diagnostics": contract_diag,
        }
        if "reason" in q_intra_debug:
            debug_dict["reason"] = q_intra_debug["reason"]
        return p_hat_final, debug_dict

    if not apply_v2:
        p_hat_final = p_hat_old
        midround_weight = 0.0
        midround_v2_result = {
            "skipped": True,
            "reason": "phase_hold_previous_phat",
            "round_phase": round_phase,
        }
    else:
        midround_weight = 0.25
        result = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=rail_high,
            frozen_b=rail_low,
            features=features,
            config=config,
            frame=frame,
        )
        midround_v2_result = result
        # Stage 2: movement toward target (Bible Ch 6 Step 8), then clamp to rails.
        target_p_hat = result["p_mid"]
        p_hat_final = p_hat_old + midround_weight * (target_p_hat - p_hat_old)
        p_hat_final = max(rail_low, min(rail_high, p_hat_final))

    contract_diag = _contract_diag(
        frame=frame,
        config=config,
        phase=phase_upper,
        q_intra_debug=q_intra_debug,
        midround_v2_result=midround_v2_result,
        rail_low=rail_low,
        rail_high=rail_high,
        p_hat_old=p_hat_old,
        p_hat_final=p_hat_final,
        movement_confidence=midround_weight,
    )
    explain = _build_explain(
        phase=phase_upper,
        p_base_map=base,
        p_base_series=None,
        midround_weight=midround_weight,
        p_hat_prev_source=p_hat_prev_source,
        midround_v2_result=midround_v2_result,
        q_intra_debug=q_intra_debug,
        micro_adj_breakdown=micro_adj_breakdown,
        rail_low=rail_low,
        rail_high=rail_high,
        p_hat_final=p_hat_final,
        contract_diag=contract_diag,
    )
    debug_dict = {
        "p_hat_base": base,
        "micro_adj": adj,
        "p_hat_pre_clamp01": p_pre_clamp01,
        "p_hat_post_clamp01": p_post_clamp01,
        "p_hat_reconstructed": reconstructed_p_hat,
        "p_hat_prev_source": p_hat_prev_source,
        "rail_low": rail_low,
        "rail_high": rail_high,
        "q_intra": q_intra_debug,
        "midround_weight": midround_weight,
        "p_hat_old": p_hat_old,
        "p_hat_final": p_hat_final,
        "midround_v2": midround_v2_result,
        "phase_unknown": phase_unknown,
        "explain": explain,
        "contract_diagnostics": contract_diag,
    }
    if "reason" in q_intra_debug:
        debug_dict["reason"] = q_intra_debug["reason"]

    return p_hat_final, debug_dict

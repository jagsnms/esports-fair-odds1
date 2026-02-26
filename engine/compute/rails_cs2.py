"""
CS2 map corridor: two counterfactual next-round endpoints -> envelope -> active points.
Ported from app35_ml.py geometry. Contextual widening gated by config.context_widening_enabled (default OFF).
No numpy; clamps rails inside series corridor and [0,1].
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

from engine.compute.map_corridor_cs2 import (
    compute_cs2_branch_endpoint_envelope_debug,
    compute_cs2_branch_endpoint_position_debug,
)
from engine.compute.map_fair_cs2 import p_map_fair
from engine.compute.series_iid import series_win_prob_live

# MR12: first to 13; overtime 16, 19, ...
WIN_TARGET_REG = 13
OT_BLOCK = 3
MAX_ROUNDS_MAP = 24

# Contextual widening (only when context_widening_enabled=True)
CONTEXT_WIDEN_BETA = 0.25
UNCERTAINTY_MULT_MAX = 1.35
UNCERTAINTY_MULT_MIN = 1.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _cs2_win_target(rounds_a: int, rounds_b: int) -> int:
    """Current win target: 13 regulation; 16/19/... in OT (MR3 blocks)."""
    ra, rb = int(rounds_a), int(rounds_b)
    if min(ra, rb) < 12:
        return WIN_TARGET_REG
    sets = max(0, (min(ra, rb) - 12) // OT_BLOCK)
    return WIN_TARGET_REG + OT_BLOCK * (sets + 1)


def _n_maps_from_series_fmt(series_fmt: str) -> int:
    try:
        n = int(str(series_fmt or "bo3").replace("bo", "").strip() or "3")
    except (ValueError, TypeError):
        n = 3
    return 3 if n not in (3, 5) else n


def _compute_context_risk(frame: Frame) -> tuple[float, dict[str, Any]]:
    """Context risk [0,1] from leverage/fragility/missingness. Used only when context_widening_enabled."""
    components: dict[str, Any] = {}
    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    score_diff = abs(ra - rb)
    closeness = 1.0 - min(score_diff / 8.0, 1.0) if score_diff <= 8 else 0.0
    score_sum = ra + rb
    lateness = min(score_sum / float(MAX_ROUNDS_MAP), 1.0) if MAX_ROUNDS_MAP else 0.0
    leader = max(ra, rb)
    match_point_ish = 1.0 if leader >= 11 else (0.5 if leader >= 9 else 0.0)
    leverage_risk = _clip01(0.35 * closeness + 0.35 * lateness + 0.3 * match_point_ish)
    components["leverage_risk"] = leverage_risk
    loadout = getattr(frame, "loadout_totals", None)
    if isinstance(loadout, (tuple, list)) and len(loadout) >= 2:
        load_a = float(loadout[0]) if loadout[0] is not None else 0.0
        load_b = float(loadout[1]) if loadout[1] is not None else 0.0
        load_sum = load_a + load_b
        load_max = max(load_a, load_b)
        load_min = min(load_a, load_b)
        ratio = load_min / (load_max + 1e-6)
        low_total = 0.5 if load_sum < 5000 else (0.2 if load_sum < 10000 else 0.0)
        asymmetry = 0.5 if ratio < 0.3 else (0.2 if ratio < 0.5 else 0.0)
        fragility_risk = _clip01(low_total + asymmetry)
    else:
        fragility_risk = 0.5
        components["fragility_note"] = "loadout_totals_missing"
    components["fragility_risk"] = fragility_risk
    alive = getattr(frame, "alive_counts", None)
    hp = getattr(frame, "hp_totals", (0.0, 0.0))
    has_alive = isinstance(alive, (tuple, list)) and len(alive) >= 2 and alive[0] is not None and alive[1] is not None
    has_loadout = isinstance(loadout, (tuple, list)) and len(loadout) >= 2
    missingness_risk = 0.3 if not (has_alive and has_loadout) else 0.0
    components["missingness_risk"] = missingness_risk
    context_risk = _clip01(0.4 * leverage_risk + 0.4 * fragility_risk + 0.2 * missingness_risk)
    return context_risk, components


def _map_width_cap(scores: tuple[int, int]) -> tuple[float, dict[str, Any]]:
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    score_sum = ra + rb
    diff = abs(ra - rb)
    closeness = 1.0 - min(diff / 8.0, 1.0)
    if score_sum <= 6:
        base_cap, phase = 0.18, "early"
    elif score_sum <= 16:
        base_cap, phase = 0.26, "mid"
    else:
        base_cap, phase = 0.36, "late"
    extra = 0.04 * closeness if phase != "early" else 0.02 * closeness
    cap = base_cap + extra
    cap = min(cap, 0.22 if phase == "early" else (0.30 if phase == "mid" else 0.40))
    return cap, {"score_sum": score_sum, "closeness": closeness, "phase": phase}


def compute_rails_cs2(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """
    Map corridor: canonical_if_a / canonical_if_b from series_win_prob_live; envelope around each;
    active points from Frame microstate; map_low/high = min/max(active); clamp inside series.
    When context_widening_enabled: apply context_risk widening + width cap. Else: no widening.
    Returns (map_low, map_high, debug_dict).
    """
    series_lo, series_hi = _clip(bounds[0], 0.0, 1.0), _clip(bounds[1], 0.0, 1.0)
    if series_hi < series_lo:
        series_hi = min(1.0, series_lo + 0.02)
    series_width = series_hi - series_lo

    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    series = getattr(frame, "series_score", (0, 0))
    ma = int(series[0]) if len(series) > 0 and series[0] is not None else 0
    mb = int(series[1]) if len(series) > 1 and series[1] is not None else 0
    series_fmt = getattr(frame, "series_fmt", "bo3") or "bo3"
    bo = _n_maps_from_series_fmt(series_fmt)
    p0 = getattr(config, "prematch_map", None)
    if p0 is not None and isinstance(p0, (int, float)):
        p0 = max(1e-6, min(1.0 - 1e-6, float(p0)))
    else:
        p0 = 0.5
    wt = _cs2_win_target(ra, rb)

    # Map-level fair probability for counterfactual states; then series_win_prob_live
    p_map_if_a_val: float | None = None
    p_map_if_b_val: float | None = None
    if (rb + 1) >= wt:
        canonical_if_b = series_win_prob_live(bo, ma, mb + 1, p0, p0)
    else:
        p_map_if_b_val = p_map_fair(ra, rb + 1, config=config, state=state, frame=frame)
        canonical_if_b = series_win_prob_live(bo, ma, mb, p_map_if_b_val, p0)
    if (ra + 1) >= wt:
        canonical_if_a = series_win_prob_live(bo, ma + 1, mb, p0, p0)
    else:
        p_map_if_a_val = p_map_fair(ra + 1, rb, config=config, state=state, frame=frame)
        canonical_if_a = series_win_prob_live(bo, ma, mb, p_map_if_a_val, p0)
    canonical_if_a = _clip01(canonical_if_a)
    canonical_if_b = _clip01(canonical_if_b)

    # Envelope around each branch
    env = compute_cs2_branch_endpoint_envelope_debug(
        canonical_if_a,
        canonical_if_b,
        ra,
        rb,
        bo,
        ma,
        mb,
        wt,
    )
    debug: dict[str, Any] = {
        "canonical_if_a_round": canonical_if_a,
        "canonical_if_b_round": canonical_if_b,
        "p_map_if_a": p_map_if_a_val,
        "p_map_if_b": p_map_if_b_val,
        "base_span": env.get("endpoint_base_span_abs"),
        "k": env.get("endpoint_env_fraction_k"),
    }
    if not env.get("env_valid"):
        # Fallback: use canonical endpoints as rails, clamped to series
        rail_lo = _clip(min(canonical_if_a, canonical_if_b), series_lo, series_hi)
        rail_hi = _clip(max(canonical_if_a, canonical_if_b), series_lo, series_hi)
        rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
        min_map_width = 0.01
        if rail_hi - rail_lo < min_map_width:
            mid = 0.5 * (rail_lo + rail_hi)
            half = min_map_width / 2.0
            rail_lo = _clip(mid - half, series_lo, series_hi)
            rail_hi = _clip(mid + half, series_lo, series_hi)
            rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
        if rail_hi < rail_lo:
            rail_hi = min(1.0, rail_lo + min_map_width)
        debug["map_corridor_fallback"] = "invalid_envelope"
        return (rail_lo, rail_hi, debug)

    a_lo = env["endpoint_a_env_low"]
    a_hi = env["endpoint_a_env_high"]
    b_lo = env["endpoint_b_env_low"]
    b_hi = env["endpoint_b_env_high"]
    debug["envelope"] = {k: v for k, v in env.items() if k != "env_valid"}

    # Microstate for position
    alive = getattr(frame, "alive_counts", (5, 5))
    alive_a = float(alive[0]) if isinstance(alive, (tuple, list)) and len(alive) > 0 and alive[0] is not None else 5.0
    alive_b = float(alive[1]) if isinstance(alive, (tuple, list)) and len(alive) > 1 and alive[1] is not None else 5.0
    loadout = getattr(frame, "loadout_totals", None)
    if isinstance(loadout, (tuple, list)) and len(loadout) >= 2:
        load_a = float(loadout[0]) if loadout[0] is not None else 0.0
        load_b = float(loadout[1]) if loadout[1] is not None else 0.0
    else:
        load_a = load_b = 0.0
    armor = getattr(frame, "armor_totals", None)
    if isinstance(armor, (tuple, list)) and len(armor) >= 2:
        arm_a = float(armor[0]) if armor[0] is not None else 0.0
        arm_b = float(armor[1]) if armor[1] is not None else 0.0
    else:
        arm_a = arm_b = 0.0

    pos = compute_cs2_branch_endpoint_position_debug(
        a_lo, a_hi, b_lo, b_hi,
        alive_a, alive_b,
        load_a, load_b,
        arm_a, arm_b,
        branch_temp=0.40,
    )
    debug["quality_pos"] = {
        "d_alive": pos.get("endpoint_pos_d_alive"),
        "d_loadout": pos.get("endpoint_pos_d_loadout"),
        "d_armor": pos.get("endpoint_pos_d_armor"),
        "a_quality_pos": pos.get("endpoint_pos_a_quality_pos"),
        "b_quality_pos": pos.get("endpoint_pos_b_quality_pos"),
    }
    debug["active_points"] = {
        "a_active": pos.get("endpoint_pos_a_active_dbg"),
        "b_active": pos.get("endpoint_pos_b_active_dbg"),
    }

    if pos.get("pos_valid") and pos.get("endpoint_pos_a_active_dbg") is not None and pos.get("endpoint_pos_b_active_dbg") is not None:
        active_a = pos["endpoint_pos_a_active_dbg"]
        active_b = pos["endpoint_pos_b_active_dbg"]
        map_low = min(active_a, active_b)
        map_high = max(active_a, active_b)
    else:
        map_low = min(canonical_if_a, canonical_if_b)
        map_high = max(canonical_if_a, canonical_if_b)

    rail_lo = _clip(map_low, series_lo, series_hi)
    rail_hi = _clip(map_high, series_lo, series_hi)
    rail_lo = _clip01(rail_lo)
    rail_hi = _clip01(rail_hi)
    # Enforce minimum map corridor width to avoid degenerate collapse
    min_map_width = 0.01
    if rail_hi - rail_lo < min_map_width:
        mid = 0.5 * (rail_lo + rail_hi)
        half = min_map_width / 2.0
        rail_lo = _clip(mid - half, series_lo, series_hi)
        rail_hi = _clip(mid + half, series_lo, series_hi)
        rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
    if rail_hi < rail_lo:
        rail_hi = min(1.0, rail_lo + min_map_width)

    # Optional: context widening + width cap (gated)
    if getattr(config, "context_widening_enabled", False):
        context_risk, risk_components = _compute_context_risk(frame)
        uncertainty_mult = UNCERTAINTY_MULT_MIN + context_risk * (UNCERTAINTY_MULT_MAX - UNCERTAINTY_MULT_MIN)
        uncertainty_mult = max(UNCERTAINTY_MULT_MIN, min(UNCERTAINTY_MULT_MAX, uncertainty_mult))
        map_width_before = rail_hi - rail_lo
        current_halfwidth = map_width_before / 2.0
        widened_halfwidth = current_halfwidth * (1.0 + CONTEXT_WIDEN_BETA * context_risk)
        mid = (rail_lo + rail_hi) / 2.0
        rail_lo = mid - widened_halfwidth
        rail_hi = mid + widened_halfwidth
        rail_lo = _clip(rail_lo, series_lo, series_hi)
        rail_hi = _clip(rail_hi, series_lo, series_hi)
        rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
        if rail_hi < rail_lo:
            rail_hi = min(1.0, rail_lo + 0.01)
        map_width_after_widen = rail_hi - rail_lo
        width_cap_val, cap_meta = _map_width_cap(scores)
        width_cap_used = min(series_width, width_cap_val)
        map_width_after_cap = max(map_width_before, min(map_width_after_widen, width_cap_used))
        if map_width_after_cap <= 0.0:
            map_width_after_cap = map_width_before
        half_capped = 0.5 * map_width_after_cap
        rail_lo = mid - half_capped
        rail_hi = mid + half_capped
        rail_lo = _clip(rail_lo, series_lo, series_hi)
        rail_hi = _clip(rail_hi, series_lo, series_hi)
        rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
        if rail_hi < rail_lo:
            rail_hi = min(1.0, rail_lo + 0.01)
        debug["context_risk"] = context_risk
        debug["context_risk_components"] = risk_components
        debug["uncertainty_multiplier"] = uncertainty_mult
        debug["map_width_before"] = map_width_before
        debug["map_width_after_widen"] = map_width_after_widen
        debug["map_width_after_cap"] = rail_hi - rail_lo
        debug["width_cap_used"] = width_cap_used
        debug["width_cap_meta"] = cap_meta
    else:
        debug["context_widening_enabled"] = False

    return (rail_lo, rail_hi, debug)

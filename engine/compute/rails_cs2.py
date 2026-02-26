"""
CS2 round-state rails/envelope (map corridor): port of app35_ml round-state envelope logic.
Uses Frame scores, series_score, map_index; optional alive/hp/loadout for intraround nudge.
Contextual widening: context_risk (leverage/fragility/missingness) widens map corridor to avoid false certainty.
No numpy; always clamps rails into bounds and [0,1].
"""
from __future__ import annotations

from typing import Any

from engine.models import Config, Frame, State

# MR12: first to 13; overtime 16, 19, ...
WIN_TARGET_REG = 13
OT_BLOCK = 3
MAX_ROUNDS_MAP = 24

# Envelope fraction k: app _compute_cs2_branch_endpoint_envelope_debug
ENV_K_LO = 0.08
ENV_K_HI = 0.22
ENV_K_BASE = 0.08
ENV_K_SCALE = 0.14

# Optional intraround: bomb planted / low alive -> slight extra narrowing (cap)
INTRA_NARROW_CAP = 0.04

# Contextual widening: beta in widened_halfwidth = current_halfwidth * (1 + beta * context_risk)
CONTEXT_WIDEN_BETA = 1.0
# uncertainty_multiplier = 1 + context_risk -> in [1.0, 2.0]
UNCERTAINTY_MULT_MAX = 2.0
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


def _envelope_fraction_k(
    rounds_a: int,
    rounds_b: int,
    maps_a_won: int,
    maps_b_won: int,
    best_of: int,
    win_target: int,
) -> float:
    """
    Port of app _compute_cs2_branch_endpoint_envelope_debug leverage -> k.
    L_round from leader/wt and score diff; L_series from map index; k = 0.08 + 0.14 * (0.75*L_round + 0.25*L_series).
    """
    ra, rb = max(0, int(rounds_a)), max(0, int(rounds_b))
    wt = max(1, int(win_target))
    ma, mb = max(0, int(maps_a_won)), max(0, int(maps_b_won))
    bo = max(1, int(best_of))
    if bo not in (3, 5):
        bo = 3

    # 1) Round leverage L_round in [0,1]
    leader = max(ra, rb)
    P = leader / float(wt) if wt else 0.0
    d = abs(ra - rb)
    C = 1.0 - min(d / 8.0, 1.0)
    L_round = _clip01(0.6 * P + 0.4 * C)

    # 2) Series leverage L_series from map index
    map_idx = ma + mb + 1
    if bo == 3:
        series_leverage_map = {1: 0.45, 2: 0.65, 3: 1.00}
    else:
        series_leverage_map = {1: 0.30, 2: 0.45, 3: 0.65, 4: 0.85, 5: 1.00}
    L_series = series_leverage_map.get(map_idx, 0.60)

    # 3) k
    k_raw = ENV_K_BASE + ENV_K_SCALE * (0.75 * L_round + 0.25 * L_series)
    return _clip(k_raw, ENV_K_LO, ENV_K_HI)


def _intraround_narrowing(frame: Frame) -> float:
    """
    Optional: small extra narrowing when bomb planted or very low alive counts.
    Returns 0.0 to INTRA_NARROW_CAP. Defensive if fields missing.
    """
    out = 0.0
    bomb = getattr(frame, "bomb_phase_time_remaining", None)
    if isinstance(bomb, dict) and bomb.get("is_bomb_planted"):
        out += 0.02
    alive = getattr(frame, "alive_counts", (5, 5))
    if isinstance(alive, (tuple, list)) and len(alive) >= 2:
        a, b = int(alive[0]) if alive[0] is not None else 5, int(alive[1]) if alive[1] is not None else 5
        total_alive = a + b
        if total_alive <= 2:
            out += 0.02
        elif total_alive <= 4:
            out += 0.01
    return _clip(out, 0.0, INTRA_NARROW_CAP)


def _compute_context_risk(frame: Frame) -> tuple[float, dict[str, Any]]:
    """
    Compute context_risk in [0, 1] from leverage, fragility, missingness.
    Returns (context_risk, components_dict).
    """
    components: dict[str, Any] = {}
    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    score_diff = abs(ra - rb)
    score_sum = ra + rb
    # Leverage: closeness (tight score), lateness (rounds played), match-point-ish (leader near 12/13)
    closeness = 1.0 - min(score_diff / 8.0, 1.0) if score_diff <= 8 else 0.0
    lateness = min(score_sum / float(MAX_ROUNDS_MAP), 1.0) if MAX_ROUNDS_MAP else 0.0
    leader = max(ra, rb)
    match_point_ish = 1.0 if leader >= 11 else (0.5 if leader >= 9 else 0.0)
    leverage_risk = _clip01(0.35 * closeness + 0.35 * lateness + 0.3 * match_point_ish)
    components["leverage_risk"] = leverage_risk
    components["leverage_closeness"] = closeness
    components["leverage_lateness"] = lateness
    components["leverage_match_point_ish"] = match_point_ish

    # Fragility: loadout asymmetry or very low total loadout on one side
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
        fragility_risk = 0.5  # treat missing loadout as fragile
        components["fragility_note"] = "loadout_totals_missing"
    components["fragility_risk"] = fragility_risk

    # Missingness: key microstate missing -> increase risk
    alive = getattr(frame, "alive_counts", None)
    hp = getattr(frame, "hp_totals", (0.0, 0.0))
    has_alive = isinstance(alive, (tuple, list)) and len(alive) >= 2 and alive[0] is not None and alive[1] is not None
    has_hp = isinstance(hp, (tuple, list)) and len(hp) >= 2 and (hp[0] != 0 or hp[1] != 0 or True)
    has_loadout = isinstance(loadout, (tuple, list)) and len(loadout) >= 2
    key_microstate_ok = has_alive and (has_hp or has_loadout)
    missingness_risk = 0.3 if not key_microstate_ok else 0.0
    components["missingness_risk"] = missingness_risk
    components["inputs_present"] = {"alive": has_alive, "hp": has_hp, "loadout": has_loadout}

    context_risk = _clip01(0.4 * leverage_risk + 0.4 * fragility_risk + 0.2 * missingness_risk)
    return context_risk, components


def compute_rails_cs2(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
) -> tuple[float, float, dict[str, Any]]:
    """
    Compute (rail_lo, rail_hi, debug_dict) for map corridor.
    - Base rails from round-state envelope (same as before).
    - Contextual widening: context_risk widens halfwidth by (1 + beta*context_risk), clamped inside bounds.
    - Debug: context_risk, components, uncertainty_multiplier, map_width_before, map_width_after.
    """
    bound_lo, bound_hi = bounds
    bound_lo = _clip(bound_lo, 0.0, 1.0)
    bound_hi = _clip(bound_hi, 0.0, 1.0)
    if bound_hi < bound_lo:
        bound_hi = min(1.0, bound_lo + 0.02)
    bound_width = bound_hi - bound_lo

    # Center
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        center = _clip01(float(prematch))
    else:
        center = 0.5

    # Scores and series from frame
    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    series = getattr(frame, "series_score", (0, 0))
    ma = int(series[0]) if len(series) > 0 and series[0] is not None else 0
    mb = int(series[1]) if len(series) > 1 and series[1] is not None else 0
    map_index = getattr(frame, "map_index", 0)
    series_fmt = getattr(frame, "series_fmt", "bo3") or "bo3"
    try:
        bo = int(series_fmt.replace("bo", "")) if isinstance(series_fmt, str) else 3
    except (ValueError, TypeError):
        bo = 3
    bo = 3 if bo not in (3, 5) else bo

    wt = _cs2_win_target(ra, rb)
    k = _envelope_fraction_k(ra, rb, ma, mb, bo, wt)
    intra = _intraround_narrowing(frame)
    effective_k = _clip(k + intra, 0.0, ENV_K_HI + INTRA_NARROW_CAP)

    # Base rail width (before contextual widening)
    rail_half_width = 0.5 * bound_width * (1.0 - effective_k)
    rail_half_width = max(0.005, min(0.5 * bound_width, rail_half_width))

    rail_lo = center - rail_half_width
    rail_hi = center + rail_half_width
    rail_lo = _clip(rail_lo, bound_lo, bound_hi)
    rail_hi = _clip(rail_hi, bound_lo, bound_hi)
    rail_lo = _clip01(rail_lo)
    rail_hi = _clip01(rail_hi)
    if rail_hi < rail_lo:
        rail_hi = min(1.0, rail_lo + 0.01)

    map_width_before = rail_hi - rail_lo

    # Contextual widening
    context_risk, risk_components = _compute_context_risk(frame)
    uncertainty_multiplier = UNCERTAINTY_MULT_MIN + context_risk * (UNCERTAINTY_MULT_MAX - UNCERTAINTY_MULT_MIN)
    uncertainty_multiplier = max(UNCERTAINTY_MULT_MIN, min(UNCERTAINTY_MULT_MAX, uncertainty_multiplier))

    current_halfwidth = (rail_hi - rail_lo) / 2.0
    widened_halfwidth = current_halfwidth * (1.0 + CONTEXT_WIDEN_BETA * context_risk)
    mid = (rail_lo + rail_hi) / 2.0
    rail_lo = mid - widened_halfwidth
    rail_hi = mid + widened_halfwidth
    rail_lo = _clip(rail_lo, bound_lo, bound_hi)
    rail_hi = _clip(rail_hi, bound_lo, bound_hi)
    rail_lo = _clip01(rail_lo)
    rail_hi = _clip01(rail_hi)
    if rail_hi < rail_lo:
        rail_hi = min(1.0, rail_lo + 0.01)

    map_width_after = rail_hi - rail_lo

    debug: dict[str, Any] = {
        "context_risk": context_risk,
        "context_risk_components": risk_components,
        "uncertainty_multiplier": uncertainty_multiplier,
        "map_width_before": map_width_before,
        "map_width_after": map_width_after,
    }
    return (rail_lo, rail_hi, debug)

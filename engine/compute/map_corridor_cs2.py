"""
CS2 map corridor: branch endpoint envelope and position (stdlib only).
Ported from app35_ml.py: _compute_cs2_branch_endpoint_envelope_debug,
_compute_cs2_branch_endpoint_position_debug. Same structure and weights.
"""
from __future__ import annotations

import math
from typing import Any


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid_stable(x: float, temp: float = 1.0) -> float:
    """Numerically stable sigmoid; output in (0, 1). q = 1 / (1 + exp(-x/temp))."""
    if temp is None or temp <= 0:
        temp = 1.0
    z = float(x) / float(temp)
    if z >= 20.0:
        return 1.0
    if z <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def compute_cs2_branch_endpoint_envelope_debug(
    endpoint_a_base: float,
    endpoint_b_base: float,
    rounds_a: int,
    rounds_b: int,
    best_of: int,
    maps_a_won: int,
    maps_b_won: int,
    win_target: int = 13,
) -> dict[str, Any]:
    """Pure helper: per-branch endpoint envelope from base endpoints, round score, series state.
    Returns debug dict with env_valid=False and None fields on invalid inputs.
    """
    invalid: dict[str, Any] = {
        "env_valid": False,
        "endpoint_a_base": None,
        "endpoint_b_base": None,
        "endpoint_base_span_abs": None,
        "endpoint_env_fraction_k": None,
        "endpoint_env_round_leverage": None,
        "endpoint_env_series_leverage": None,
        "endpoint_env_half_width": None,
        "endpoint_a_env_low": None,
        "endpoint_a_env_high": None,
        "endpoint_b_env_low": None,
        "endpoint_b_env_high": None,
        "endpoint_a_env_center": None,
        "endpoint_b_env_center": None,
        "endpoint_env_map_index": None,
        "endpoint_env_best_of": None,
        "endpoint_env_win_target": None,
    }
    try:
        ea = float(endpoint_a_base)
        eb = float(endpoint_b_base)
        ra = int(rounds_a)
        rb = int(rounds_b)
        bo = int(best_of)
        ma = int(maps_a_won)
        mb = int(maps_b_won)
        wt = int(win_target)
    except (TypeError, ValueError):
        return invalid
    if wt <= 0 or ra < 0 or rb < 0 or ma < 0 or mb < 0 or bo not in (3, 5):
        return invalid

    # 1) Round leverage L_round in [0,1]
    leader = max(ra, rb)
    P = leader / float(wt)
    d = abs(ra - rb)
    C = 1.0 - min(d / 8.0, 1.0)
    L_round = _clip01(0.6 * P + 0.4 * C)

    # 2) Series leverage L_series in [0,1]
    map_idx = ma + mb + 1
    if bo == 3:
        series_leverage_map = {1: 0.45, 2: 0.65, 3: 1.00}
    elif bo == 5:
        series_leverage_map = {1: 0.30, 2: 0.45, 3: 0.65, 4: 0.85, 5: 1.00}
    else:
        series_leverage_map = {}
    L_series = series_leverage_map.get(map_idx, 0.60)

    # 3) Envelope fraction k
    k_raw = 0.08 + 0.14 * (0.75 * L_round + 0.25 * L_series)
    k = max(0.08, min(0.22, float(k_raw)))

    # 4) Base span and envelope half-width
    base_span = abs(ea - eb)
    env_half_width = base_span * k

    # 5) Branch-oriented envelope bounds
    a_env_low = _clip01(ea - env_half_width)
    a_env_high = _clip01(ea + env_half_width)
    b_env_low = _clip01(eb - env_half_width)
    b_env_high = _clip01(eb + env_half_width)

    # 6) Centers
    a_env_center = 0.5 * (a_env_low + a_env_high)
    b_env_center = 0.5 * (b_env_low + b_env_high)

    return {
        "env_valid": True,
        "endpoint_a_base": ea,
        "endpoint_b_base": eb,
        "endpoint_base_span_abs": base_span,
        "endpoint_env_fraction_k": k,
        "endpoint_env_round_leverage": L_round,
        "endpoint_env_series_leverage": L_series,
        "endpoint_env_half_width": env_half_width,
        "endpoint_a_env_low": a_env_low,
        "endpoint_a_env_high": a_env_high,
        "endpoint_b_env_low": b_env_low,
        "endpoint_b_env_high": b_env_high,
        "endpoint_a_env_center": a_env_center,
        "endpoint_b_env_center": b_env_center,
        "endpoint_env_map_index": map_idx,
        "endpoint_env_best_of": bo,
        "endpoint_env_win_target": wt,
    }


def compute_cs2_branch_endpoint_position_debug(
    a_env_low: float,
    a_env_high: float,
    b_env_low: float,
    b_env_high: float,
    team_a_alive_count: float,
    team_b_alive_count: float,
    team_a_alive_loadout_total: float,
    team_b_alive_loadout_total: float,
    team_a_armor_alive_total: float,
    team_b_armor_alive_total: float,
    branch_temp: float = 0.40,
) -> dict[str, Any]:
    """Pure helper: branch-quality position inside endpoint envelopes from intraround microstate.
    Returns pos_valid=False and None fields on invalid input.
    """
    EPS = 1e-9
    invalid: dict[str, Any] = {
        "pos_valid": False,
        "branch_temp_used": None,
        "endpoint_pos_d_alive": None,
        "endpoint_pos_d_loadout": None,
        "endpoint_pos_d_armor": None,
        "endpoint_pos_score_a": None,
        "endpoint_pos_score_b": None,
        "endpoint_pos_a_quality_pos": None,
        "endpoint_pos_b_quality_pos": None,
        "endpoint_pos_a_active_dbg": None,
        "endpoint_pos_b_active_dbg": None,
        "endpoint_pos_a_active_within_env": None,
        "endpoint_pos_b_active_within_env": None,
        "endpoint_pos_loadout_sum": None,
        "endpoint_pos_armor_sum": None,
    }
    try:
        a_lo = float(a_env_low)
        a_hi = float(a_env_high)
        b_lo = float(b_env_low)
        b_hi = float(b_env_high)
        n_a = float(team_a_alive_count)
        n_b = float(team_b_alive_count)
        load_a = float(team_a_alive_loadout_total)
        load_b = float(team_b_alive_loadout_total)
        arm_a = float(team_a_armor_alive_total)
        arm_b = float(team_b_armor_alive_total)
        temp = float(branch_temp)
    except (TypeError, ValueError):
        return invalid
    if not (
        -1e30 < a_lo < 1e30 and -1e30 < a_hi < 1e30 and -1e30 < b_lo < 1e30 and -1e30 < b_hi < 1e30
        and -1e30 < n_a < 1e30 and -1e30 < n_b < 1e30
        and -1e30 < load_a < 1e30 and -1e30 < load_b < 1e30
        and -1e30 < arm_a < 1e30 and -1e30 < arm_b < 1e30
        and -1e30 < temp < 1e30
    ):
        return invalid
    temp = max(float(temp), 1e-6)

    # Normalized microstate diffs
    d_alive = (n_a - n_b) / 5.0
    d_alive = max(-1.0, min(1.0, float(d_alive)))

    loadout_sum = load_a + load_b
    if loadout_sum > EPS:
        d_loadout = (load_a - load_b) / loadout_sum
    else:
        d_loadout = 0.0
    d_loadout = max(-1.0, min(1.0, float(d_loadout)))

    armor_sum = arm_a + arm_b
    if armor_sum > EPS:
        d_armor = (arm_a - arm_b) / armor_sum
    else:
        d_armor = 0.0
    d_armor = max(-1.0, min(1.0, float(d_armor)))

    # Branch-quality score (50/30/20)
    s_A = 0.50 * d_alive + 0.30 * d_loadout + 0.20 * d_armor
    s_B = -s_A
    s_A = max(-5.0, min(5.0, float(s_A)))
    s_B = max(-5.0, min(5.0, float(s_B)))

    # Map to [0,1] via sigmoid
    a_quality_pos = _sigmoid_stable(s_A, temp)
    b_quality_pos = _sigmoid_stable(s_B, temp)

    # Active endpoints inside envelope (linear interpolation)
    a_active_dbg = a_lo + (a_hi - a_lo) * a_quality_pos
    b_active_dbg = b_lo + (b_hi - b_lo) * b_quality_pos

    a_active_within_env = bool((a_lo - 1e-9) <= a_active_dbg <= (a_hi + 1e-9))
    b_active_within_env = bool((b_lo - 1e-9) <= b_active_dbg <= (b_hi + 1e-9))

    return {
        "pos_valid": True,
        "branch_temp_used": temp,
        "endpoint_pos_d_alive": d_alive,
        "endpoint_pos_d_loadout": d_loadout,
        "endpoint_pos_d_armor": d_armor,
        "endpoint_pos_score_a": s_A,
        "endpoint_pos_score_b": s_B,
        "endpoint_pos_a_quality_pos": a_quality_pos,
        "endpoint_pos_b_quality_pos": b_quality_pos,
        "endpoint_pos_a_active_dbg": a_active_dbg,
        "endpoint_pos_b_active_dbg": b_active_dbg,
        "endpoint_pos_a_active_within_env": a_active_within_env,
        "endpoint_pos_b_active_within_env": b_active_within_env,
        "endpoint_pos_loadout_sum": loadout_sum,
        "endpoint_pos_armor_sum": armor_sum,
    }

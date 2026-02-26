"""
CS2 round-state rails/envelope: port of app35_ml round-state envelope logic.
Uses Frame scores, series_score, map_index; optional alive/hp/cash/bomb for intraround nudge.
No numpy; always clamps rails into bounds and [0,1].
"""
from __future__ import annotations

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


def compute_rails_cs2(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
) -> tuple[float, float]:
    """
    Compute (rail_lo, rail_hi) from round-state envelope logic.
    - Center from config.prematch_map or 0.5.
    - Envelope fraction k from rounds/series (same math as app); higher k -> narrower rails.
    - Rail width = bound width * (1 - k) with optional intraround narrowing.
    - Always clamped into bounds and [0, 1]. Defensive if optional frame fields missing.
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

    # Rail width: shrink by envelope (late/decisive -> narrower)
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
    return (rail_lo, rail_hi)

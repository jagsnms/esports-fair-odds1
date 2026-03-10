"""
CS2 map-level fair probability (minimal port from app35 estimate_inplay_prob + soft_lock).
Score-driven + optional leverage; smooth, ~0.5 at 0-0; clamped to [0.02, 0.98] to avoid degeneracy.
Stdlib only (no numpy).
"""
from __future__ import annotations

import math
from typing import Any, Optional

from engine.models import Config, Frame, State

# Win target: MR12 first to 13; OT 16, 19, ...
WIN_TARGET_REG = 13
OT_BLOCK = 3

# Clamp output to avoid 0/1 extremes early
P_MAP_CLAMP_LO = 0.02
P_MAP_CLAMP_HI = 0.98

# Score sensitivity (stage-scaled): early leads swing less
BETA_SCORE = 0.22
# Soft lock: blend toward score-state when within ~3 rounds of winning
LOCK_START_OFFSET = 3
LOCK_RAMP = 3
BETA_LOCK = 0.90


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _cs2_win_target(rounds_a: int, rounds_b: int) -> int:
    ra, rb = int(rounds_a), int(rounds_b)
    if min(ra, rb) < 12:
        return WIN_TARGET_REG
    sets = max(0, (min(ra, rb) - 12) // OT_BLOCK)
    return WIN_TARGET_REG + OT_BLOCK * (sets + 1)


def _estimate_inplay_prob(
    p0: float,
    rounds_a: int,
    rounds_b: int,
    win_target: int,
    *,
    beta_score: float = BETA_SCORE,
) -> float:
    """
    Minimal score-driven map probability. Stage scaling: early leads swing less.
    No econ/pistol/side for minimal version.
    """
    ra = int(rounds_a)
    rb = int(rounds_b)
    score_diff = ra - rb
    rp = max(0, ra + rb)
    # Stage scaling: 0.65 + 0.35 * min(1, rp/12) so early round diff matters less
    stage_scale = 0.65 + 0.35 * min(1.0, rp / 12.0)
    x = _logit(p0) + (beta_score * stage_scale) * score_diff
    # Late-game nonlinearity (match-point nudge)
    lead_sign = 1 if score_diff > 0 else (-1 if score_diff < 0 else 0)
    if lead_sign != 0:
        leading_rounds = ra if lead_sign > 0 else rb
        start_at = win_target - LOCK_START_OFFSET
        closeness = (float(leading_rounds) - float(start_at)) / float(max(1, LOCK_RAMP))
        closeness = max(0.0, min(1.0, closeness))
        x += BETA_LOCK * (closeness ** 2) * float(lead_sign)
    return _sigmoid(x)


def _soft_lock_map_prob(p_map: float, rounds_a: int, rounds_b: int, win_target: int) -> float:
    """
    Blend p_map toward score-state probability when within ~3 rounds of winning,
    so map->series doesn't cliff at map end.
    """
    p_map = max(1e-6, min(1.0 - 1e-6, float(p_map)))
    ra, rb = int(rounds_a), int(rounds_b)
    lead = max(ra, rb)
    closeness = (float(lead) - float(win_target - LOCK_START_OFFSET)) / float(max(1, LOCK_RAMP))
    closeness = max(0.0, min(1.0, closeness))
    if closeness <= 0.0:
        return p_map
    from engine.compute.series_iid import series_prob_needed
    wa = max(0, win_target - ra)
    wb = max(0, win_target - rb)
    if wa <= 0:
        p_state = 1.0
    elif wb <= 0:
        p_state = 0.0
    else:
        p_state = series_prob_needed(wa, wb, 0.5)
    alpha = closeness ** 2
    return (1.0 - alpha) * p_map + alpha * float(p_state)


def p_map_fair(
    rounds_a: int,
    rounds_b: int,
    *,
    config: Optional[Config] = None,
    state: Optional[State] = None,
    frame: Optional[Frame] = None,
) -> float:
    """
    Map-level fair probability that Team A wins the current map.
    Smooth, centered ~0.5 at 0-0; shifts modestly with score. Clamped to [0.02, 0.98].
    """
    ra = int(rounds_a)
    rb = int(rounds_b)
    p0 = 0.5
    if config is not None:
        prematch = getattr(config, "prematch_map", None)
        if prematch is not None and isinstance(prematch, (int, float)):
            p0 = max(1e-6, min(1.0 - 1e-6, float(prematch)))
    wt = _cs2_win_target(ra, rb)
    p_raw = _estimate_inplay_prob(p0, ra, rb, wt)
    p_soft = _soft_lock_map_prob(p_raw, ra, rb, wt)
    return max(P_MAP_CLAMP_LO, min(P_MAP_CLAMP_HI, float(p_soft)))

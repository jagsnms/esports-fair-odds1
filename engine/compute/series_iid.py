"""
I.I.D. series win probability (stdlib math only).
Ported from app35_ml.py: series_prob_needed, series_win_prob_live.
Uses math.comb and clamps; no numpy.
"""
from __future__ import annotations

import math


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def series_prob_needed(wins_needed: int, losses_allowed_plus1: int, p: float) -> float:
    """Probability to reach 'wins_needed' wins before reaching 'losses_allowed_plus1' losses
    in independent Bernoulli trials with win prob p.
    losses_allowed_plus1 = losses_needed (max losses before losing series) + 1.
    For best-of-(2t-1): wins_needed=t, losses_allowed_plus1=t.
    """
    p = _clip01(min(1.0 - 1e-9, max(1e-9, p)))
    w = max(0, int(wins_needed))
    l_lim = max(0, int(losses_allowed_plus1))
    if w == 0:
        return 1.0
    if l_lim == 0:
        return 0.0
    q = 1.0 - p
    out = 0.0
    for k in range(0, l_lim):
        out += math.comb(w + k - 1, k) * (p**w) * (q**k)
    return _clip01(out)


def series_win_prob_live(
    n_maps: int,
    maps_a_won: int,
    maps_b_won: int,
    p_current_map: float,
    p_future_map: float,
) -> float:
    """
    Live series win probability for Team A.
    maps_a_won/maps_b_won are maps already won BEFORE the current map resolves.
    p_current_map is the live probability Team A wins the CURRENT map.
    p_future_map is used for any remaining future maps after the current one.
    Resolves directly to 1.0/0.0 on the deciding map branch.
    """
    bo_n = max(1, int(n_maps))
    target = bo_n // 2 + 1
    mwA = max(0, int(maps_a_won))
    mwB = max(0, int(maps_b_won))
    pf = float(p_future_map) if p_future_map is not None else float(p_current_map)
    pc = _clip01(min(1.0 - 1e-6, max(1e-6, float(p_current_map))))
    pf = _clip01(min(1.0 - 1e-6, max(1e-6, pf)))

    if mwA >= target:
        return 1.0
    if mwB >= target:
        return 0.0

    if mwA + 1 >= target:
        p_if_win_current = 1.0
    else:
        p_if_win_current = series_prob_needed(target - (mwA + 1), target - mwB, pf)

    if mwB + 1 >= target:
        p_if_lose_current = 0.0
    else:
        p_if_lose_current = series_prob_needed(target - mwA, target - (mwB + 1), pf)

    return float(pc * p_if_win_current + (1.0 - pc) * p_if_lose_current)


def derive_p_map_from_p_series(best_of: int, prematch_series: float) -> float:
    """
    Inverse: given series win prob (prematch, 0-0), find p_map such that
    series_win_prob_live(best_of, 0, 0, p_map, p_map) == prematch_series.
    Uses bisection on p_map in [0.02, 0.98]. Returns p_map in that range.
    """
    target = _clip01(float(prematch_series))
    bo = max(1, int(best_of))
    if bo not in (3, 5):
        bo = 3
    lo, hi = 0.02, 0.98
    for _ in range(60):
        mid = (lo + hi) * 0.5
        p_series = series_win_prob_live(bo, 0, 0, mid, mid)
        if abs(p_series - target) < 1e-9:
            return mid
        if p_series < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5

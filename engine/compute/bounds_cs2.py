"""
CS2 map-state corridor (bounds): score + series context, parity-style.
Uses frame.scores, series_score, series_fmt, map_index; same win target as rails_cs2.
Deterministic; no external deps.
"""
from __future__ import annotations

from engine.models import Config, Frame, State

from engine.compute.rails_cs2 import _cs2_win_target

# Bounds width: early ~0.35 halfwidth, late ~0.10
HALFWIDTH_EARLY = 0.35
HALFWIDTH_LATE = 0.10
# Score shift: clamp diff/wt to [-0.35, 0.35], then scale by S
SHIFT_CLIP = 0.35
# Shift strength S in [S_LO, 1.0]; increases with late-roundness and series leverage
S_LO = 0.4
# Series leverage tightens width: halfwidth *= (1 - TIGHTEN * L_series)
SERIES_TIGHTEN = 0.15


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _series_leverage(maps_a_won: int, maps_b_won: int, best_of: int) -> float:
    """L_series in [0,1] from map index; later maps higher."""
    ma, mb = max(0, maps_a_won), max(0, maps_b_won)
    bo = 3 if best_of not in (3, 5) else best_of
    map_idx = ma + mb + 1
    if bo == 3:
        m = {1: 0.45, 2: 0.65, 3: 1.00}
    else:
        m = {1: 0.30, 2: 0.45, 3: 0.65, 4: 0.85, 5: 1.00}
    return m.get(map_idx, 0.60)


def compute_bounds_cs2(frame: Frame, config: Config, state: State) -> tuple[float, float]:
    """
    Map-state corridor: baseline + score-based shift, width shrinks with round progress and series.
    - Baseline = prematch_map if set else 0.5.
    - Shift = clamp((ra-rb)/wt, -0.35, 0.35) * S; S increases with late-roundness and series leverage.
    - Halfwidth from ~0.35 early to ~0.10 late; tightened by series leverage.
    - Bounds always contain center; final lo, hi in [0,1], lo <= hi.
    """
    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    series = getattr(frame, "series_score", (0, 0))
    ma = int(series[0]) if len(series) > 0 and series[0] is not None else 0
    mb = int(series[1]) if len(series) > 1 and series[1] is not None else 0
    series_fmt = getattr(frame, "series_fmt", "bo3") or "bo3"
    try:
        bo = int(series_fmt.replace("bo", "")) if isinstance(series_fmt, str) else 3
    except (ValueError, TypeError):
        bo = 3
    bo = 3 if bo not in (3, 5) else bo

    wt = _cs2_win_target(ra, rb)
    wt_f = max(1.0, float(wt))

    # Baseline
    prematch = getattr(config, "prematch_map", None)
    if prematch is not None and isinstance(prematch, (int, float)):
        baseline = _clip01(float(prematch))
    else:
        baseline = 0.5

    # Score-based shift: diff/wt clamped, then scaled by S
    diff = ra - rb
    raw_shift = diff / wt_f
    raw_shift = _clip(raw_shift, -SHIFT_CLIP, SHIFT_CLIP)
    # S (shift strength): late-roundness + series leverage
    rounds_played = ra + rb
    leader = max(ra, rb)
    late_round = leader / wt_f if wt_f else 0.0
    late_round = _clip01(late_round)
    L_series = _series_leverage(ma, mb, bo)
    S = S_LO + (1.0 - S_LO) * (0.6 * late_round + 0.4 * L_series)
    S = _clip(S, S_LO, 1.0)
    shift = raw_shift * S
    center = _clip01(baseline + shift)

    # Halfwidth: early wide, late narrow; series tightens
    progress = rounds_played / (2.0 * wt_f) if wt_f else 0.0
    progress = _clip01(progress)
    halfwidth = HALFWIDTH_EARLY - (HALFWIDTH_EARLY - HALFWIDTH_LATE) * progress
    halfwidth *= 1.0 - SERIES_TIGHTEN * L_series
    halfwidth = max(0.02, min(0.49, halfwidth))

    lo = center - halfwidth
    hi = center + halfwidth
    lo = _clip01(lo)
    hi = _clip01(hi)
    # Ensure bounds contain center
    if center < lo:
        lo = max(0.0, center - 0.01)
    if center > hi:
        hi = min(1.0, center + 0.01)
    if lo > hi:
        hi = min(1.0, lo + 0.02)
    lo = _clip01(lo)
    hi = _clip01(hi)
    return (lo, hi)

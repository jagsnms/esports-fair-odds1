"""
CS2 intra-round advantage signal (debug-only v1).

Computes q = P(team A wins round) from microstate: alive counts, HP totals,
optional bomb/time as confidence/narrowing only. No econ in v1; no bomb direction bias.
"""
from __future__ import annotations

import math
from typing import Any

from engine.models import Frame

# Conservative weights (documented in debug). Not tuned aggressively.
W_ALIVE = 0.25  # per-player alive delta contribution to raw_score
W_HP_SCALE = 400.0  # hp_delta / W_HP_SCALE * W_HP_MAX -> contribution capped at W_HP_MAX
W_HP_MAX = 0.6  # max contribution from HP delta
SIGMOID_SCALE = 2.0  # q = 1 / (1 + exp(-SIGMOID_SCALE * raw_score))
# Bomb: narrowing only (pull raw toward 0 when bomb planted)
BOMB_NARROW_FACTOR = 0.92  # raw_score *= BOMB_NARROW_FACTOR when bomb planted

# Time: plausible round time remaining (seconds)
T_REMAINING_MIN = 0.0
T_REMAINING_MAX = 120.0


def _sigmoid(x: float) -> float:
    """Map real line to (0, 1)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    t = math.exp(x)
    return t / (1.0 + t)


def _get_alive_delta(frame: Frame) -> tuple[bool, float]:
    """Return (present, alive_delta) with alive_delta = A - B."""
    alive = getattr(frame, "alive_counts", None)
    if not isinstance(alive, (tuple, list)) or len(alive) < 2:
        return False, 0.0
    a = alive[0] if alive[0] is not None else 0
    b = alive[1] if alive[1] is not None else 0
    try:
        return True, float(int(a) - int(b))
    except (TypeError, ValueError):
        return False, 0.0


def _get_hp_delta(frame: Frame) -> tuple[bool, float]:
    """Return (present, hp_delta) with hp_delta = A - B."""
    hp = getattr(frame, "hp_totals", None)
    if not isinstance(hp, (tuple, list)) or len(hp) < 2:
        return False, 0.0
    ha = hp[0] if hp[0] is not None else 0.0
    hb = hp[1] if hp[1] is not None else 0.0
    try:
        return True, float(ha) - float(hb)
    except (TypeError, ValueError):
        return False, 0.0


def _get_bomb_planted(frame: Frame) -> bool:
    """True if bomb is planted (from bomb_phase_time_remaining dict)."""
    bomb_phase = getattr(frame, "bomb_phase_time_remaining", None)
    if not isinstance(bomb_phase, dict):
        return False
    return bool(bomb_phase.get("is_bomb_planted"))


def _get_time_remaining(frame: Frame) -> tuple[bool, float | None]:
    """Return (used, t_remaining). Used only if numeric and in plausible range."""
    bomb_phase = getattr(frame, "bomb_phase_time_remaining", None)
    if not isinstance(bomb_phase, dict):
        return False, None
    t = bomb_phase.get("round_time_remaining")
    if t is None:
        return False, None
    try:
        val = float(t)
    except (TypeError, ValueError):
        return False, None
    if T_REMAINING_MIN <= val <= T_REMAINING_MAX:
        return True, val
    return False, None


def compute_q_intra_cs2(frame: Frame, *, config: Any = None) -> tuple[float, dict]:
    """
    CS2 intra-round win probability for team A (debug-only v1).

    Uses alive_counts, hp_totals; bomb_planted as narrowing only; time_remaining
    for gating/debug. No econ; no bomb direction bias.

    Returns:
        (q, debug_dict) with q in [0, 1] and debug containing required keys.
    """
    alive_present, alive_delta = _get_alive_delta(frame)
    hp_present, hp_delta = _get_hp_delta(frame)
    bomb_planted = _get_bomb_planted(frame)
    time_used, t_remaining = _get_time_remaining(frame)

    inputs_present = {
        "alive": alive_present,
        "hp": hp_present,
        "bomb": bomb_planted,
        "time": time_used,
    }

    # Required microstate: at least one of alive or hp usable
    if not (alive_present or hp_present):
        debug = {
            "q_intra_round_win_a": 0.5,
            "raw_score": 0.0,
            "alive_delta": 0.0,
            "hp_delta": 0.0,
            "bomb_term_used": False,
            "time_term_used": False,
            "t_remaining": None,
            "used_econ": False,
            "used_bomb_direction": False,
            "inputs_present": inputs_present,
            "reason": "missing_microstate",
            "weights": {"w_alive": W_ALIVE, "w_hp_scale": W_HP_SCALE, "w_hp_max": W_HP_MAX, "sigmoid_scale": SIGMOID_SCALE},
        }
        return 0.5, debug

    # Raw score from alive + hp only (no econ, no bomb direction)
    raw_score = 0.0
    if alive_present:
        raw_score += W_ALIVE * alive_delta
    if hp_present:
        hp_contrib = (hp_delta / W_HP_SCALE) * W_HP_MAX if W_HP_SCALE else 0.0
        hp_contrib = max(-W_HP_MAX, min(W_HP_MAX, hp_contrib))
        raw_score += hp_contrib

    # Bomb: narrowing only (no direction)
    bomb_term_used = bomb_planted
    if bomb_term_used:
        raw_score *= BOMB_NARROW_FACTOR

    q = _sigmoid(SIGMOID_SCALE * raw_score)

    debug = {
        "q_intra_round_win_a": q,
        "raw_score": raw_score,
        "alive_delta": alive_delta,
        "hp_delta": hp_delta,
        "bomb_term_used": bomb_term_used,
        "time_term_used": time_used,
        "t_remaining": t_remaining if time_used else None,
        "used_econ": False,
        "used_bomb_direction": False,
        "inputs_present": inputs_present,
        "weights": {"w_alive": W_ALIVE, "w_hp_scale": W_HP_SCALE, "w_hp_max": W_HP_MAX, "sigmoid_scale": SIGMOID_SCALE},
    }
    return q, debug

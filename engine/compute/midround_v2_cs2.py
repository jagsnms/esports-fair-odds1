"""
CS2 midround V2 mixture oracle: features from Frame, mixture p_mid = frozen_b + q*(frozen_a - frozen_b).
Ported from app35 _compute_cs2_midround_features and _apply_cs2_midround_adjustment_v2_mixture.
"""
from __future__ import annotations

import math
from typing import Any

from engine.models import Frame

# Oracle constants (app35 parity)
MAX_ROUND_TIME_S = 120.0
MIXTURE_TEMP = 0.8
ALIVE_WEIGHT = 0.035  # per-player alive diff
HP_WEIGHT = 0.010     # per-100 HP
BOMB_WEIGHT = 0.060
LOADOUT_WEIGHT = 0.012  # per-1000
ARMOR_WEIGHT = 0.008    # per-100 (optional)
URGENCY_FLOOR = 0.15
URGENCY_SCALE = 0.85
EPS = 1e-6


def _sigmoid(x: float, temp: float = 1.0) -> float:
    """q = 1 / (1 + exp(-x/temp)). Numerically stable."""
    if temp is None or temp <= 0:
        temp = 1.0
    z = float(x) / float(temp)
    if z >= 20.0:
        return 1.0
    if z <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _normalize_side(side: Any) -> str | None:
    if side is None:
        return None
    s = str(side).strip().upper()
    return s if s in ("T", "CT") else None


def compute_cs2_midround_features(frame: Frame, *, config: Any = None) -> dict[str, Any]:
    """
    Extract intraround features from Frame defensively.
    Returns dict with alive_diff, hp_*, loadout_*, armor_*, bomb_planted, time_*, a_side,
    inputs_present, and reliability (stub).
    """
    alive = getattr(frame, "alive_counts", None)
    hp = getattr(frame, "hp_totals", None)
    loadout = getattr(frame, "loadout_totals", None)
    armor = getattr(frame, "armor_totals", None)
    bomb_phase = getattr(frame, "bomb_phase_time_remaining", None)
    a_side = _normalize_side(getattr(frame, "a_side", None))

    # Parse bomb_phase (dict with round_time_remaining, round_phase, is_bomb_planted)
    bomb_planted = None
    round_time_remaining_s = None
    round_phase = None
    if isinstance(bomb_phase, dict):
        bp = bomb_phase.get("is_bomb_planted")
        bomb_planted = bool(bp) if bp is not None else None
        t = bomb_phase.get("round_time_remaining")
        if t is not None:
            try:
                round_time_remaining_s = float(t)
                if round_time_remaining_s > 200:  # ms?
                    round_time_remaining_s = round_time_remaining_s / 1000.0
            except (TypeError, ValueError):
                pass
        rp = bomb_phase.get("round_phase")
        round_phase = str(rp) if rp is not None else None

    # Alive
    alive_a = alive_b = None
    if isinstance(alive, (tuple, list)) and len(alive) >= 2:
        try:
            alive_a = int(alive[0]) if alive[0] is not None else None
            alive_b = int(alive[1]) if alive[1] is not None else None
        except (TypeError, ValueError):
            pass
    alive_delta = (alive_a - alive_b) if (alive_a is not None and alive_b is not None) else 0

    # HP
    hp_a = hp_b = 0.0
    if isinstance(hp, (tuple, list)) and len(hp) >= 2:
        try:
            hp_a = float(hp[0]) if hp[0] is not None else 0.0
            hp_b = float(hp[1]) if hp[1] is not None else 0.0
        except (TypeError, ValueError):
            pass
    hp_delta = hp_a - hp_b

    # Loadout (alive-only equipment value)
    load_a = load_b = 0.0
    if isinstance(loadout, (tuple, list)) and len(loadout) >= 2:
        try:
            load_a = float(loadout[0]) if loadout[0] is not None else 0.0
            load_b = float(loadout[1]) if loadout[1] is not None else 0.0
        except (TypeError, ValueError):
            pass
    loadout_delta = load_a - load_b

    # Armor (optional)
    armor_a = armor_b = 0.0
    has_armor = False
    if isinstance(armor, (tuple, list)) and len(armor) >= 2:
        try:
            armor_a = float(armor[0]) if armor[0] is not None else 0.0
            armor_b = float(armor[1]) if armor[1] is not None else 0.0
            has_armor = True
        except (TypeError, ValueError):
            pass
    armor_delta = armor_a - armor_b

    # Time progress [0, 1]: 0 = round start, 1 = late
    max_round_s = MAX_ROUND_TIME_S
    if config is not None and hasattr(config, "max_round_time_s") and getattr(config, "max_round_time_s") is not None:
        try:
            max_round_s = float(getattr(config, "max_round_time_s"))
        except (TypeError, ValueError):
            pass
    time_progress = 0.5
    if round_time_remaining_s is not None and round_time_remaining_s >= 0 and max_round_s > 0:
        time_progress = max(0.0, min(1.0, 1.0 - (round_time_remaining_s / max_round_s)))

    inputs_present = {
        "alive": alive_a is not None and alive_b is not None,
        "hp": True,  # we always have floats
        "loadout": loadout is not None and isinstance(loadout, (tuple, list)) and len(loadout) >= 2,
        "armor": has_armor,
        "bomb": bomb_planted is not None,
        "time": round_time_remaining_s is not None,
        "a_side": a_side is not None,
    }

    reliability = {
        "has_players": True,
        "has_clock": round_time_remaining_s is not None,
    }

    return {
        "alive_diff": alive_delta,
        "alive_a": alive_a,
        "alive_b": alive_b,
        "hp_diff_alive": hp_delta,
        "hp_a_total": hp_a,
        "hp_b_total": hp_b,
        "loadout_diff_alive": loadout_delta,
        "load_a_total": load_a,
        "load_b_total": load_b,
        "armor_diff_alive": armor_delta if has_armor else None,
        "bomb_planted": 1 if bomb_planted else 0,
        "bomb_planted_bool": bomb_planted,
        "time_remaining_s": round_time_remaining_s,
        "time_progress": time_progress,
        "round_phase": round_phase,
        "a_side": a_side,
        "inputs_present": inputs_present,
        "reliability": reliability,
        "feature_ok": inputs_present["alive"] or inputs_present["loadout"] or (hp_a + hp_b > 0),
    }


def apply_cs2_midround_adjustment_v2_mixture(
    *,
    frozen_a: float,
    frozen_b: float,
    features: dict[str, Any],
) -> dict[str, Any]:
    """
    V2 mixture: q_intra from alive/hp/loadout/bomb (and optional armor); urgency from time.
    p_mid = frozen_b + q_intra*(frozen_a - frozen_b); clamp between endpoints.
    If missing key microstate (no alive/hp/loadout), return q_intra=0.5 and p_mid_clamped = midpoint.
    """
    out: dict[str, Any] = {}
    alive_delta = float(features.get("alive_diff", 0))
    hp_delta = float(features.get("hp_diff_alive", 0))
    hp_a = float(features.get("hp_a_total", 0))
    hp_b = float(features.get("hp_b_total", 0))
    load_delta = float(features.get("loadout_diff_alive", 0))
    load_a = float(features.get("load_a_total", 0))
    load_b = float(features.get("load_b_total", 0))
    armor_delta = features.get("armor_diff_alive")
    if armor_delta is not None:
        armor_delta = float(armor_delta)
    bomb = features.get("bomb_planted", 0)
    if isinstance(bomb, bool):
        bomb = 1 if bomb else 0
    bomb = int(bomb) if bomb else 0
    a_side = (features.get("a_side") or "").strip().upper() or None
    if a_side and a_side not in ("T", "CT"):
        a_side = None
    time_progress = max(0.0, min(1.0, float(features.get("time_progress", 0.5))))
    inputs_present = features.get("inputs_present") or {}

    has_alive = inputs_present.get("alive", False)
    has_loadout = inputs_present.get("loadout", False)
    hp_sum = hp_a + hp_b
    has_hp = hp_sum > 0 or (hp_a != 0 or hp_b != 0)
    key_microstate_ok = has_alive or has_hp or has_loadout

    if not key_microstate_ok:
        mid = (frozen_a + frozen_b) / 2.0
        return {
            "q_intra": 0.5,
            "raw_score": 0.0,
            "urgency": 0.5,
            "time_progress": 0.5,
            "p_mid": mid,
            "p_mid_clamped": mid,
            "alive_delta": 0.0,
            "hp_delta": 0.0,
            "loadout_delta": 0.0,
            "armor_delta": None,
            "score_alive": 0.0,
            "score_hp": 0.0,
            "score_loadout": 0.0,
            "score_armor": 0.0,
            "score_bomb": 0.0,
            "used_time": False,
            "used_loadout": False,
            "used_bomb_direction": False,
            "used_armor": False,
            "reason": "missing_microstate",
            "temp": MIXTURE_TEMP,
        }

    # Score components (oracle formula)
    score_alive = (alive_delta / 5.0) * ALIVE_WEIGHT if has_alive else 0.0
    if hp_sum > 0:
        score_hp = (hp_delta / hp_sum) * 100.0 * HP_WEIGHT
    else:
        score_hp = (hp_delta / 100.0) * HP_WEIGHT
    load_sum = load_a + load_b
    if load_sum > 0:
        score_loadout = (load_delta / load_sum) * 1000.0 * LOADOUT_WEIGHT if has_loadout else 0.0
    else:
        score_loadout = (load_delta / 1000.0) * LOADOUT_WEIGHT if has_loadout else 0.0
    score_armor = 0.0
    if armor_delta is not None:
        score_armor = (armor_delta / 100.0) * ARMOR_WEIGHT
    if bomb:
        if a_side == "T":
            score_bomb = BOMB_WEIGHT
        elif a_side == "CT":
            score_bomb = -BOMB_WEIGHT
        else:
            score_bomb = 0.0
    else:
        score_bomb = 0.0

    urgency = URGENCY_FLOOR + URGENCY_SCALE * time_progress
    raw_score = (score_alive + score_hp + score_loadout + score_armor + score_bomb) * urgency
    q_intra = _sigmoid(raw_score, MIXTURE_TEMP)

    fa = float(frozen_a)
    fb = float(frozen_b)
    p_mid = fb + q_intra * (fa - fb)
    lo = min(fa, fb)
    hi = max(fa, fb)
    if hi <= lo:
        p_mid_clamped = (fa + fb) / 2.0
    else:
        lo_ep = lo + EPS
        hi_ep = hi - EPS
        p_mid_clamped = max(lo_ep, min(hi_ep, p_mid))

    out["q_intra"] = q_intra
    out["raw_score"] = raw_score
    out["urgency"] = urgency
    out["time_progress"] = time_progress
    out["p_mid"] = p_mid
    out["p_mid_clamped"] = p_mid_clamped
    out["alive_delta"] = alive_delta
    out["hp_delta"] = hp_delta
    out["loadout_delta"] = load_delta
    out["armor_delta"] = armor_delta
    out["score_alive"] = score_alive
    out["score_hp"] = score_hp
    out["score_loadout"] = score_loadout
    out["score_armor"] = score_armor
    out["score_bomb"] = score_bomb
    out["used_time"] = features.get("time_remaining_s") is not None
    out["used_loadout"] = has_loadout
    out["used_bomb_direction"] = bool(bomb and a_side in ("T", "CT"))
    out["used_armor"] = armor_delta is not None
    out["temp"] = MIXTURE_TEMP
    return out

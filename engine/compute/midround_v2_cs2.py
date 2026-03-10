"""
CS2 midround V2 mixture oracle: features from Frame, mixture p_mid = frozen_b + q*(frozen_a - frozen_b).
Ported from app35 _compute_cs2_midround_features and _apply_cs2_midround_adjustment_v2_mixture.
"""
from __future__ import annotations

import math
import os
from typing import Any

from engine.models import Frame

# Oracle constants (app35 parity)
MAX_ROUND_TIME_S = 120.0
MIXTURE_TEMP = 0.8
# Default weights (current hardcoded profile); see WEIGHTS_CURRENT / WEIGHTS_LEARNED_V1
ALIVE_WEIGHT = 0.035  # per-player alive diff
HP_FRAC_WEIGHT = 0.04  # HP team-vs-team: term_hp = weight * (hp_frac_a - 0.5)
BOMB_WEIGHT = 0.060
LOADOUT_WEIGHT = 0.012  # per-1000
ARMOR_WEIGHT = 0.008    # per-100 (optional)
URGENCY_FLOOR = 0.15
URGENCY_SCALE = 0.85
EPS = 1e-6
TIMER_CONTRACT_VERSION = "timer_contract.v1"

TIMER_STATE_PRE_PLANT = "PRE_PLANT"
TIMER_STATE_POST_PLANT = "POST_PLANT"
TIMER_STATE_UNKNOWN = "UNKNOWN"

TIMER_SOURCE_CANONICAL_REPLAY_RAW = "canonical_replay_raw"
TIMER_SOURCE_BO3_LIVE_NORMALIZED = "bo3_live_normalized"
TIMER_SOURCE_GRID_REDUCED = "grid_reduced"
TIMER_SOURCE_UNKNOWN = "unknown"

TIMER_DIRECTION_FAVOR_CT = "FAVOR_CT"
TIMER_DIRECTION_FAVOR_T = "FAVOR_T"
TIMER_DIRECTION_NONE = "NONE"

PREPLANT_CT_FAVOR_APPLIED = "PREPLANT_CT_FAVOR_APPLIED"
POSTPLANT_T_FAVOR_APPLIED = "POSTPLANT_T_FAVOR_APPLIED"
TIMER_DIRECTION_SKIPPED_TIMER_MISSING = "TIMER_DIRECTION_SKIPPED_TIMER_MISSING"
TIMER_DIRECTION_SKIPPED_TIMER_INVALID = "TIMER_DIRECTION_SKIPPED_TIMER_INVALID"
TIMER_DIRECTION_SKIPPED_PLANT_STATE_UNKNOWN = "TIMER_DIRECTION_SKIPPED_PLANT_STATE_UNKNOWN"
TIMER_DIRECTION_SKIPPED_A_SIDE_UNKNOWN = "TIMER_DIRECTION_SKIPPED_A_SIDE_UNKNOWN"
TIMER_DIRECTION_SKIPPED_UNSUPPORTED_SOURCE = "TIMER_DIRECTION_SKIPPED_UNSUPPORTED_SOURCE"

HARD_BOUNDARY_ACTIVE_CT_IMPOSSIBLE = "HARD_BOUNDARY_ACTIVE_CT_IMPOSSIBLE"
HARD_BOUNDARY_NOT_ACTIVE_ABOVE_THRESHOLD = "HARD_BOUNDARY_NOT_ACTIVE_ABOVE_THRESHOLD"
HARD_BOUNDARY_SKIPPED_NOT_POSTPLANT = "HARD_BOUNDARY_SKIPPED_NOT_POSTPLANT"
HARD_BOUNDARY_SKIPPED_TIMER_MISSING = "HARD_BOUNDARY_SKIPPED_TIMER_MISSING"
HARD_BOUNDARY_SKIPPED_TIMER_INVALID = "HARD_BOUNDARY_SKIPPED_TIMER_INVALID"
HARD_BOUNDARY_SKIPPED_UNSUPPORTED_SOURCE = "HARD_BOUNDARY_SKIPPED_UNSUPPORTED_SOURCE"

DEFUSE_SOURCE_KIT_5S = "KIT_5S"
DEFUSE_SOURCE_NO_KIT_10S = "NO_KIT_10S"
DEFUSE_SOURCE_UNKNOWN_FLOOR_5S = "UNKNOWN_FLOOR_5S"
DEFUSE_SOURCE_UNAVAILABLE = "UNAVAILABLE"

# Timer directional pressure strength inside q-score space.
TIMER_DIRECTION_WEIGHT = 0.06

# Weight profiles for A/B testing (current, learned_v1, learned_v2 converged calibration)
WEIGHTS_CURRENT = {
    "alive": 0.035,
    "hp": 0.04,
    "loadout": 0.012,
    "bomb": 0.06,
    "cash": 0.0,
}
WEIGHTS_LEARNED_V1 = {
    "alive": 0.06,
    "hp": 0.07,
    "loadout": 0.003,
    "bomb": 0.10,
    "cash": 0.0,
}
WEIGHTS_LEARNED_V2 = {
    "alive": 0.08,
    "hp": 0.12,
    "loadout": 0.002,
    "bomb": 0.10,
    "cash": 0.0,
}
# Fitted suggested_coef from midround_fit_weights, sign-corrected so Team-A-positive term_raw yields positive score (q_intra > 0.5)
WEIGHTS_LEARNED_FIT = {
    "alive": 6.80,
    "hp": 11.41,
    "loadout": 0.00392,
    "bomb": 3.11,
    "cash": 0.0,
}

_ALLOWED_PROFILES = ("current", "learned_v1", "learned_v2", "learned_fit")


def _get_weight_profile(config: Any = None) -> str:
    """Resolve active profile: config.midround_v2_weight_profile else env MIDROUND_V2_WEIGHT_PROFILE else 'current'."""
    if config is not None and hasattr(config, "midround_v2_weight_profile"):
        p = getattr(config, "midround_v2_weight_profile", None)
        if isinstance(p, str) and p.strip():
            p = p.strip().lower()
            if p in _ALLOWED_PROFILES:
                return p
    raw = os.environ.get("MIDROUND_V2_WEIGHT_PROFILE", "current")
    if isinstance(raw, str) and raw.strip().lower() in _ALLOWED_PROFILES:
        return raw.strip().lower()
    return "current"


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


def _classify_timer_source(config: Any = None) -> str:
    source = str(getattr(config, "source", "") or "").strip().upper()
    if source == "REPLAY":
        return TIMER_SOURCE_CANONICAL_REPLAY_RAW
    if source == "BO3":
        return TIMER_SOURCE_BO3_LIVE_NORMALIZED
    if source == "GRID":
        return TIMER_SOURCE_GRID_REDUCED
    return TIMER_SOURCE_UNKNOWN


def _coerce_timer_remaining_seconds(value: Any) -> tuple[float | None, bool, bool, bool]:
    """
    Return (seconds, valid, missing, invalid).
    Valid timer is finite and non-negative, in canonical seconds.
    """
    if value is None:
        return None, False, True, False
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None, False, False, True
    if not math.isfinite(t) or t < 0.0:
        return None, False, False, True
    return t, True, False, False


def _timer_state_from_bomb_planted(bomb_planted: Any) -> str:
    if isinstance(bomb_planted, bool):
        return TIMER_STATE_POST_PLANT if bomb_planted else TIMER_STATE_PRE_PLANT
    return TIMER_STATE_UNKNOWN


def _player_field(player: Any, key: str) -> Any:
    if isinstance(player, dict):
        return player.get(key)
    return getattr(player, key, None)


def _resolve_ct_defuse_time(frame: Frame | None, a_side: str | None) -> tuple[float, str, bool]:
    """
    Resolve CT effective defuse threshold under frozen Stage 0 contract.
    Returns (defuse_time_s, defuse_time_source, confident_signal).
    """
    if frame is None or a_side not in ("T", "CT"):
        return 5.0, DEFUSE_SOURCE_UNKNOWN_FLOOR_5S, False
    ct_players = getattr(frame, "players_b", []) if a_side == "T" else getattr(frame, "players_a", [])
    if not isinstance(ct_players, list) or not ct_players:
        return 5.0, DEFUSE_SOURCE_UNKNOWN_FLOOR_5S, False

    alive_players: list[Any] = []
    for p in ct_players:
        alive = _player_field(p, "alive")
        if alive is True:
            alive_players.append(p)
    considered = alive_players if alive_players else ct_players

    known_kits: list[bool] = []
    has_unknown = False
    for p in considered:
        hk = _player_field(p, "has_kit")
        if isinstance(hk, bool):
            known_kits.append(hk)
        else:
            has_unknown = True

    if any(known_kits):
        return 5.0, DEFUSE_SOURCE_KIT_5S, True
    if known_kits and not has_unknown and all(v is False for v in known_kits):
        return 10.0, DEFUSE_SOURCE_NO_KIT_10S, True
    return 5.0, DEFUSE_SOURCE_UNKNOWN_FLOOR_5S, False


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

    # Canonical round time: use Frame.round_time_remaining_s (normalized at ingest)
    round_time_remaining_s = getattr(frame, "round_time_remaining_s", None)
    bomb_planted = None
    round_phase = None
    if isinstance(bomb_phase, dict):
        bp = bomb_phase.get("is_bomb_planted")
        bomb_planted = bool(bp) if bp is not None else None
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

    # HP (explicit team totals + derived)
    hp_a = hp_b = 0.0
    if isinstance(hp, (tuple, list)) and len(hp) >= 2:
        try:
            hp_a = float(hp[0]) if hp[0] is not None else 0.0
            hp_b = float(hp[1]) if hp[1] is not None else 0.0
        except (TypeError, ValueError):
            pass
    hp_delta = hp_a - hp_b
    hp_sum = hp_a + hp_b
    hp_frac_a = hp_a / max(hp_sum, 1.0)
    hp_asym = (hp_a - hp_b) / max(max(hp_a, hp_b), 1.0) if (hp_a != 0 or hp_b != 0) else None
    hp_ratio = (min(hp_a, hp_b) / max(hp_a, hp_b)) if max(hp_a, hp_b) > 0 else None

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
        "hp_a": hp_a,
        "hp_b": hp_b,
        "hp_a_total": hp_a,
        "hp_b_total": hp_b,
        "hp_sum": hp_sum,
        "hp_frac_a": hp_frac_a,
        "hp_asym": hp_asym,
        "hp_ratio": hp_ratio,
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


def _compute_timer_contract(
    *,
    features: dict[str, Any],
    frame: Frame | None,
    config: Any,
    max_round_s: float,
) -> dict[str, Any]:
    source_class = _classify_timer_source(config)
    a_side = _normalize_side(features.get("a_side"))
    bomb_planted_bool = features.get("bomb_planted_bool")
    timer_state = _timer_state_from_bomb_planted(bomb_planted_bool)
    timer_remaining_s, timer_valid, timer_missing, timer_invalid = _coerce_timer_remaining_seconds(
        features.get("time_remaining_s")
    )

    if timer_state == TIMER_STATE_PRE_PLANT:
        timer_direction_expected = TIMER_DIRECTION_FAVOR_CT
    elif timer_state == TIMER_STATE_POST_PLANT:
        timer_direction_expected = TIMER_DIRECTION_FAVOR_T
    else:
        timer_direction_expected = TIMER_DIRECTION_NONE

    defuse_time_s: float | None = None
    defuse_time_source = DEFUSE_SOURCE_UNAVAILABLE
    defuse_confident = False
    if timer_state == TIMER_STATE_POST_PLANT:
        defuse_time_s, defuse_time_source, defuse_confident = _resolve_ct_defuse_time(frame, a_side)

    unsupported_source = source_class == TIMER_SOURCE_UNKNOWN
    if (
        not unsupported_source
        and source_class == TIMER_SOURCE_GRID_REDUCED
        and timer_state == TIMER_STATE_POST_PLANT
        and not defuse_confident
    ):
        unsupported_source = True

    timer_direction_applied = False
    timer_direction_term = 0.0
    if timer_state == TIMER_STATE_UNKNOWN:
        timer_direction_reason_code = TIMER_DIRECTION_SKIPPED_PLANT_STATE_UNKNOWN
    elif timer_missing:
        timer_direction_reason_code = TIMER_DIRECTION_SKIPPED_TIMER_MISSING
    elif timer_invalid:
        timer_direction_reason_code = TIMER_DIRECTION_SKIPPED_TIMER_INVALID
    elif unsupported_source:
        timer_direction_reason_code = TIMER_DIRECTION_SKIPPED_UNSUPPORTED_SOURCE
    elif a_side not in ("T", "CT"):
        timer_direction_reason_code = TIMER_DIRECTION_SKIPPED_A_SIDE_UNKNOWN
    else:
        timer_direction_applied = True
        horizon = max(max_round_s, 1.0)
        timer_pressure = max(0.0, min(1.0, 1.0 - (float(timer_remaining_s) / horizon)))
        if timer_direction_expected == TIMER_DIRECTION_FAVOR_CT:
            sign = 1.0 if a_side == "CT" else -1.0
            timer_direction_reason_code = PREPLANT_CT_FAVOR_APPLIED
        else:
            sign = 1.0 if a_side == "T" else -1.0
            timer_direction_reason_code = POSTPLANT_T_FAVOR_APPLIED
        timer_direction_term = sign * TIMER_DIRECTION_WEIGHT * timer_pressure

    hard_boundary_active = False
    hard_boundary_q_a_override: float | None = None
    if timer_state != TIMER_STATE_POST_PLANT:
        hard_boundary_reason_code = HARD_BOUNDARY_SKIPPED_NOT_POSTPLANT
        defuse_time_s = None
        defuse_time_source = DEFUSE_SOURCE_UNAVAILABLE
    elif timer_missing:
        hard_boundary_reason_code = HARD_BOUNDARY_SKIPPED_TIMER_MISSING
        defuse_time_s = None
        defuse_time_source = DEFUSE_SOURCE_UNAVAILABLE
    elif timer_invalid:
        hard_boundary_reason_code = HARD_BOUNDARY_SKIPPED_TIMER_INVALID
        defuse_time_s = None
        defuse_time_source = DEFUSE_SOURCE_UNAVAILABLE
    elif unsupported_source or a_side not in ("T", "CT"):
        hard_boundary_reason_code = HARD_BOUNDARY_SKIPPED_UNSUPPORTED_SOURCE
        defuse_time_s = None
        defuse_time_source = DEFUSE_SOURCE_UNAVAILABLE
    elif float(timer_remaining_s) < float(defuse_time_s):
        hard_boundary_active = True
        hard_boundary_reason_code = HARD_BOUNDARY_ACTIVE_CT_IMPOSSIBLE
        hard_boundary_q_a_override = 0.0 if a_side == "CT" else 1.0
    else:
        hard_boundary_reason_code = HARD_BOUNDARY_NOT_ACTIVE_ABOVE_THRESHOLD

    return {
        "timer_contract_version": TIMER_CONTRACT_VERSION,
        "timer_state": timer_state,
        "timer_source_class": source_class,
        "timer_remaining_s": timer_remaining_s,
        "timer_valid": timer_valid,
        "a_side_used": a_side or "UNKNOWN",
        "timer_direction_expected": timer_direction_expected,
        "timer_direction_applied": timer_direction_applied,
        "timer_direction_term": timer_direction_term,
        "timer_direction_reason_code": timer_direction_reason_code,
        "defuse_time_s": defuse_time_s,
        "defuse_time_source": defuse_time_source,
        "hard_boundary_active": hard_boundary_active,
        "hard_boundary_reason_code": hard_boundary_reason_code,
        "hard_boundary_q_a_override": hard_boundary_q_a_override,
    }


def apply_cs2_midround_adjustment_v2_mixture(
    *,
    frozen_a: float,
    frozen_b: float,
    features: dict[str, Any],
    config: Any = None,
    frame: Frame | None = None,
) -> dict[str, Any]:
    """
    V2 mixture: q_intra from alive/hp/loadout/bomb (and optional armor); urgency from time.
    p_mid = frozen_b + q_intra*(frozen_a - frozen_b); clamp between endpoints.
    If missing key microstate (no alive/hp/loadout), return q_intra=0.5 and p_mid_clamped = midpoint.
    Weights come from config.midround_v2_weight_profile or env MIDROUND_V2_WEIGHT_PROFILE ("current" | "learned_v1" | "learned_v2" | "learned_fit").
    """
    profile = _get_weight_profile(config)
    if profile == "learned_v1":
        active = WEIGHTS_LEARNED_V1
    elif profile == "learned_v2":
        active = WEIGHTS_LEARNED_V2
    elif profile == "learned_fit":
        active = WEIGHTS_LEARNED_FIT
    else:
        active = WEIGHTS_CURRENT
    out: dict[str, Any] = {}
    alive_delta = float(features.get("alive_diff", 0))
    hp_delta = float(features.get("hp_diff_alive", 0))
    hp_a = float(features.get("hp_a", features.get("hp_a_total", 0)))
    hp_b = float(features.get("hp_b", features.get("hp_b_total", 0)))
    hp_frac_a = features.get("hp_frac_a")
    if hp_frac_a is None:
        hp_sum_inner = hp_a + hp_b
        hp_frac_a = hp_a / max(hp_sum_inner, 1.0)
    else:
        hp_frac_a = max(0.0, min(1.0, float(hp_frac_a)))
    hp_asym = features.get("hp_asym")
    if hp_asym is None:
        hp_asym = (hp_a - hp_b) / max(max(hp_a, hp_b), 1.0) if (hp_a != 0 or hp_b != 0) else None
    else:
        hp_asym = float(hp_asym)
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

    # Timer contract semantics (direction + post-plant hard boundary) are q-path only.
    max_round_s = MAX_ROUND_TIME_S
    if config is not None and hasattr(config, "max_round_time_s") and getattr(config, "max_round_time_s") is not None:
        try:
            max_round_s = float(getattr(config, "max_round_time_s"))
        except (TypeError, ValueError):
            pass
    timer_contract = _compute_timer_contract(
        features=features,
        frame=frame,
        config=config,
        max_round_s=max_round_s,
    )
    timer_direction_term = float(timer_contract.get("timer_direction_term", 0.0) or 0.0)

    # Score components (oracle formula; weights from active profile)
    w_alive = active["alive"]
    w_hp = active["hp"]
    w_loadout = active["loadout"]
    w_bomb = active["bomb"]
    score_alive = (alive_delta / 5.0) * w_alive if key_microstate_ok and has_alive else 0.0
    term_hp = w_hp * (hp_frac_a - 0.5) if key_microstate_ok else 0.0
    load_sum = load_a + load_b
    if key_microstate_ok and load_sum > 0:
        score_loadout = (load_delta / load_sum) * 1000.0 * w_loadout if has_loadout else 0.0
    else:
        score_loadout = (load_delta / 1000.0) * w_loadout if (key_microstate_ok and has_loadout) else 0.0
    score_armor = 0.0
    if key_microstate_ok and armor_delta is not None:
        score_armor = (armor_delta / 100.0) * ARMOR_WEIGHT
    if key_microstate_ok and bomb:
        if a_side == "T":
            score_bomb = w_bomb
        elif a_side == "CT":
            score_bomb = -w_bomb
        else:
            score_bomb = 0.0
    else:
        score_bomb = 0.0

    # Term breakdown (bounded HP fraction as first-class driver)
    term_alive = score_alive
    term_loadout = score_loadout
    term_bomb = score_bomb
    term_cash = 0.0  # no cash term in current formula
    raw_score_pre_urgency = score_alive + term_hp + score_loadout + score_armor + score_bomb + timer_direction_term
    urgency = URGENCY_FLOOR + URGENCY_SCALE * time_progress
    raw_score_post_urgency = raw_score_pre_urgency * urgency
    raw_score = raw_score_post_urgency
    q_intra = _sigmoid(raw_score, MIXTURE_TEMP)
    hard_boundary_override = timer_contract.get("hard_boundary_q_a_override")
    if hard_boundary_override is not None:
        q_intra = float(hard_boundary_override)

    # Pre-weight raw signals and coefficients for score_diag (learn true weights)
    alive_raw = (alive_delta / 5.0) if (key_microstate_ok and has_alive) else 0.0
    hp_raw = (hp_frac_a - 0.5) if (key_microstate_ok and hp_frac_a is not None) else 0.0
    if key_microstate_ok and load_sum > 0:
        loadout_raw = (load_delta / load_sum) * 1000.0 if has_loadout else 0.0
    else:
        loadout_raw = (load_delta / 1000.0) if (key_microstate_ok and has_loadout) else 0.0
    if key_microstate_ok and bomb and a_side == "T":
        bomb_raw = 1.0
    elif key_microstate_ok and bomb and a_side == "CT":
        bomb_raw = -1.0
    else:
        bomb_raw = 0.0
    cash_raw = 0.0
    term_raw = {
        "alive": alive_raw,
        "hp": hp_raw,
        "loadout": loadout_raw,
        "bomb": bomb_raw,
        "cash": cash_raw,
    }
    term_coef = {
        "alive": active["alive"],
        "hp": active["hp"],
        "loadout": active["loadout"],
        "bomb": active["bomb"],
        "cash": active["cash"],
    }

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
    out["raw_score_pre_urgency"] = raw_score_pre_urgency
    out["raw_score_post_urgency"] = raw_score_post_urgency
    out["urgency"] = urgency
    out["time_progress"] = time_progress
    out["p_mid"] = p_mid
    out["p_mid_clamped"] = p_mid_clamped
    out["alive_delta"] = alive_delta
    out["hp_delta"] = hp_delta
    out["hp_a"] = hp_a
    out["hp_b"] = hp_b
    out["hp_frac_a"] = hp_frac_a
    out["hp_asym"] = None if (hp_a == 0 and hp_b == 0) else hp_asym
    out["term_alive"] = term_alive
    out["term_hp"] = term_hp
    out["term_loadout"] = term_loadout
    out["term_bomb"] = term_bomb
    out["term_cash"] = term_cash
    out["timer_direction_term"] = timer_direction_term
    out["loadout_delta"] = load_delta
    out["armor_delta"] = armor_delta
    out["score_alive"] = score_alive
    out["score_hp"] = term_hp
    out["score_loadout"] = score_loadout
    out["score_armor"] = score_armor
    out["score_bomb"] = score_bomb
    out["used_time"] = bool(timer_contract.get("timer_valid"))
    out["used_loadout"] = has_loadout
    out["used_bomb_direction"] = bool(bomb and a_side in ("T", "CT"))
    out["used_armor"] = armor_delta is not None
    out["temp"] = MIXTURE_TEMP
    out["term_raw"] = term_raw
    out["term_coef"] = term_coef
    out["weight_profile"] = profile
    out["timer_contract_version"] = timer_contract.get("timer_contract_version")
    out["timer_state"] = timer_contract.get("timer_state")
    out["timer_source_class"] = timer_contract.get("timer_source_class")
    out["timer_remaining_s"] = timer_contract.get("timer_remaining_s")
    out["timer_valid"] = timer_contract.get("timer_valid")
    out["a_side_used"] = timer_contract.get("a_side_used")
    out["timer_direction_expected"] = timer_contract.get("timer_direction_expected")
    out["timer_direction_applied"] = timer_contract.get("timer_direction_applied")
    out["timer_direction_reason_code"] = timer_contract.get("timer_direction_reason_code")
    out["defuse_time_s"] = timer_contract.get("defuse_time_s")
    out["defuse_time_source"] = timer_contract.get("defuse_time_source")
    out["hard_boundary_active"] = timer_contract.get("hard_boundary_active")
    out["hard_boundary_reason_code"] = timer_contract.get("hard_boundary_reason_code")
    if not key_microstate_ok:
        out["reason"] = "missing_microstate"
    return out

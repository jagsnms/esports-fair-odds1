"""
Raw frame inputs and fragility terms for observability (derived.debug).
No behavior changes; safe None handling.
"""
from __future__ import annotations

from typing import Any

from engine.models import Frame

# Phase names that indicate pause (timeout, intermission, etc.)
PHASE_PAUSE_SET = frozenset({
    "TIMEOUT", "TECH_TIMEOUT", "PAUSED", "HALFTIME", "INTERMISSION",
    "POSTGAME", "MAP_END", "WARMUP", "FREEZETIME",
})


def build_raw_debug(frame: Frame) -> dict[str, Any]:
    """
    Build a read-only dict of raw frame inputs and inputs_present booleans.
    Missing or None values are omitted or indicated via inputs_present.
    """
    out: dict[str, Any] = {}
    scores = getattr(frame, "scores", (0, 0))
    out["scores"] = list(scores) if scores is not None else None
    out["scores_present"] = scores is not None and len(scores) >= 2

    series_score = getattr(frame, "series_score", (0, 0))
    out["series_score"] = list(series_score) if series_score is not None else None
    out["series_score_present"] = series_score is not None and len(series_score) >= 2

    out["map_index"] = getattr(frame, "map_index", 0)
    out["map_index_present"] = True

    map_name = getattr(frame, "map_name", "")
    out["map_name"] = map_name if map_name else None
    out["map_name_present"] = bool(map_name)

    bomb = getattr(frame, "bomb_phase_time_remaining", None)
    if isinstance(bomb, dict):
        round_phase = bomb.get("round_phase") or bomb.get("phase")
        out["round_phase"] = round_phase
        out["round_phase_present"] = round_phase is not None
        out["is_bomb_planted"] = bomb.get("is_bomb_planted")
        out["is_bomb_planted_present"] = "is_bomb_planted" in bomb
    else:
        out["round_phase_present"] = False
        out["is_bomb_planted_present"] = False
    rtr_raw = getattr(frame, "round_time_remaining_raw", None)
    rtr_s = getattr(frame, "round_time_remaining_s", None)
    out["round_time_remaining_raw"] = rtr_raw
    out["round_time_remaining_s"] = rtr_s
    out["round_time_remaining_present"] = rtr_s is not None
    out["round_time_invalid"] = rtr_raw is not None and rtr_s is None

    a_side = getattr(frame, "a_side", None)
    out["a_side"] = a_side
    out["a_side_present"] = a_side is not None and bool(str(a_side).strip())

    alive = getattr(frame, "alive_counts", (0, 0))
    out["alive_counts"] = list(alive) if alive is not None else None
    out["alive_counts_present"] = alive is not None and len(alive) >= 2

    hp = getattr(frame, "hp_totals", (0.0, 0.0))
    out["hp_totals"] = list(hp) if hp is not None else None
    out["hp_totals_present"] = hp is not None and len(hp) >= 2

    cash = getattr(frame, "cash_totals", None)
    out["cash_totals"] = list(cash) if cash is not None else None
    out["cash_totals_present"] = cash is not None and len(cash) >= 2

    loadout = getattr(frame, "loadout_totals", None)
    out["loadout_totals"] = list(loadout) if loadout is not None else None
    out["loadout_totals_present"] = loadout is not None and len(loadout) >= 2

    wealth = getattr(frame, "wealth_totals", None)
    out["wealth_totals"] = list(wealth) if wealth is not None else None
    out["wealth_totals_present"] = wealth is not None and len(wealth) >= 2

    out["loadout_source"] = getattr(frame, "loadout_source", None)
    out["loadout_ev_count_a"] = getattr(frame, "loadout_ev_count_a", None)
    out["loadout_ev_count_b"] = getattr(frame, "loadout_ev_count_b", None)
    out["loadout_est_count_a"] = getattr(frame, "loadout_est_count_a", None)
    out["loadout_est_count_b"] = getattr(frame, "loadout_est_count_b", None)

    armor = getattr(frame, "armor_totals", None)
    out["armor_totals"] = list(armor) if armor is not None else None
    out["armor_totals_present"] = armor is not None and len(armor) >= 2

    return out


def compute_fragility_debug(frame: Frame) -> dict[str, Any]:
    """
    Compute fragility-related terms from frame with safe None handling.
    Returns cash_*, loadout_*, wealth_*, bank_vs_spend_*, loadout_source/counts,
    low_cash_flag, low_loadout_flag, missing_microstate_flag, clock_invalid_flag, phase_pause_flag.
    """
    out: dict[str, Any] = {}

    cash = getattr(frame, "cash_totals", None)
    if cash is not None and len(cash) >= 2:
        try:
            ca = float(cash[0]) if cash[0] is not None else 0.0
            cb = float(cash[1]) if cash[1] is not None else 0.0
        except (TypeError, ValueError):
            ca, cb = 0.0, 0.0
        out["cash_a"] = ca
        out["cash_b"] = cb
        out["cash_ratio"] = min(ca, cb) / max(ca, cb) if max(ca, cb) > 0 else None
        out["cash_asymmetry"] = (ca - cb) / max(ca, cb, 1.0)
        out["low_cash_flag"] = min(ca, cb) < 2000
    else:
        out["cash_a"] = None
        out["cash_b"] = None
        out["cash_ratio"] = None
        out["cash_asymmetry"] = None
        out["low_cash_flag"] = None

    loadout = getattr(frame, "loadout_totals", None)
    if loadout is not None and len(loadout) >= 2:
        try:
            la = float(loadout[0]) if loadout[0] is not None else 0.0
            lb = float(loadout[1]) if loadout[1] is not None else 0.0
        except (TypeError, ValueError):
            la, lb = 0.0, 0.0
        out["loadout_a"] = la
        out["loadout_b"] = lb
        out["loadout_sum"] = la + lb
        out["loadout_min"] = min(la, lb)
        out["loadout_max"] = max(la, lb)
        if max(la, lb) > 0:
            out["loadout_ratio"] = min(la, lb) / max(la, lb)
            out["econ_asymmetry"] = (la - lb) / max(la, lb)
            out["loadout_asymmetry"] = (la - lb) / max(la, lb)
        else:
            out["loadout_ratio"] = None
            out["econ_asymmetry"] = None
            out["loadout_asymmetry"] = None
        out["low_loadout_flag"] = min(la, lb) < 5000
    else:
        out["loadout_a"] = None
        out["loadout_b"] = None
        out["loadout_sum"] = None
        out["loadout_min"] = None
        out["loadout_max"] = None
        out["loadout_ratio"] = None
        out["econ_asymmetry"] = None
        out["loadout_asymmetry"] = None
        out["low_loadout_flag"] = None

    wealth = getattr(frame, "wealth_totals", None)
    if wealth is not None and len(wealth) >= 2:
        try:
            wa = float(wealth[0]) if wealth[0] is not None else 0.0
            wb = float(wealth[1]) if wealth[1] is not None else 0.0
        except (TypeError, ValueError):
            wa, wb = 0.0, 0.0
        out["wealth_a"] = wa
        out["wealth_b"] = wb
        if max(wa, wb) > 0:
            out["wealth_ratio"] = min(wa, wb) / max(wa, wb)
            out["wealth_asymmetry"] = (wa - wb) / max(wa, wb)
        else:
            out["wealth_ratio"] = None
            out["wealth_asymmetry"] = None
        loadout_a = out.get("loadout_a")
        loadout_b = out.get("loadout_b")
        out["bank_vs_spend_a"] = (loadout_a / max(wa, 1.0)) if isinstance(loadout_a, (int, float)) and wa >= 0 else None
        out["bank_vs_spend_b"] = (loadout_b / max(wb, 1.0)) if isinstance(loadout_b, (int, float)) and wb >= 0 else None
    else:
        out["wealth_a"] = None
        out["wealth_b"] = None
        out["wealth_ratio"] = None
        out["wealth_asymmetry"] = None
        out["bank_vs_spend_a"] = None
        out["bank_vs_spend_b"] = None

    out["loadout_source"] = getattr(frame, "loadout_source", None)
    out["loadout_ev_count_a"] = getattr(frame, "loadout_ev_count_a", None)
    out["loadout_ev_count_b"] = getattr(frame, "loadout_ev_count_b", None)
    out["loadout_est_count_a"] = getattr(frame, "loadout_est_count_a", None)
    out["loadout_est_count_b"] = getattr(frame, "loadout_est_count_b", None)

    alive = getattr(frame, "alive_counts", None)
    hp = getattr(frame, "hp_totals", (0.0, 0.0))
    loadout_present = loadout is not None and len(loadout) >= 2
    alive_present = alive is not None and len(alive) >= 2
    hp_present = hp is not None and len(hp) >= 2
    out["missing_microstate_flag"] = not (alive_present and hp_present and loadout_present)

    rtr_raw = getattr(frame, "round_time_remaining_raw", None)
    rtr_s = getattr(frame, "round_time_remaining_s", None)
    out["clock_invalid_flag"] = rtr_raw is not None and rtr_s is None

    bomb = getattr(frame, "bomb_phase_time_remaining", None)
    if isinstance(bomb, dict):
        phase = bomb.get("round_phase") or bomb.get("phase")
        if phase is not None:
            phase_upper = str(phase).strip().upper()
            out["phase_pause_flag"] = phase_upper in PHASE_PAUSE_SET
        else:
            out["phase_pause_flag"] = None
    else:
        out["phase_pause_flag"] = None

    return out

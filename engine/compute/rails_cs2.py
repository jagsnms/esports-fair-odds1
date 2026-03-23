"""
CS2 map corridor: two counterfactual next-round endpoints -> envelope -> active points.
Ported from app35_ml.py geometry. Contextual widening gated by config.context_widening_enabled (default OFF).
No numpy; clamps rails inside series corridor and [0,1].

This module now distinguishes:
- contract rails: strict next-round counterfactuals, rail_low = P(series | lose next round),
  rail_high = P(series | win next round), with only minimal safety post-processing.
- heuristic rails: legacy envelope/active/context-widened map corridor, preserved in debug for comparison.
"""
from __future__ import annotations

import logging
import math
from typing import Any

from engine.models import Config, Frame, State

from engine.compute.map_corridor_cs2 import (
    compute_cs2_branch_endpoint_envelope_debug,
    compute_cs2_branch_endpoint_position_debug,
)
from engine.compute.map_fair_cs2 import p_map_fair
from engine.compute.series_iid import series_win_prob_live

logger = logging.getLogger(__name__)

# MR12: first to 13; overtime 16, 19, ...
WIN_TARGET_REG = 13
OT_BLOCK = 3
MAX_ROUNDS_MAP = 24

# Contextual widening (only when context_widening_enabled=True)
CONTEXT_WIDEN_BETA = 0.25
UNCERTAINTY_MULT_MAX = 1.35
UNCERTAINTY_MULT_MIN = 1.0

# Contract rails: tiny minimum width epsilon to avoid degenerate collapse.
CONTRACT_MIN_WIDTH = 1e-4

# --- Rail input contract v1 (legacy fallback semantics). ---
RAIL_INPUT_V1_ALLOWED_FIELDS = (
    "bounds.low",
    "bounds.high",
    "frame.scores",
    "frame.series_score",
    "frame.series_fmt",
    "config.prematch_map",
)
RAIL_INPUT_V1_FORBIDDEN_FIELDS = (
    "frame.hp_totals",
    "frame.alive_counts",
    "frame.round_time_remaining_s",
    "frame.bomb_phase_time_remaining",
    "frame.loadout_totals",
    "frame.cash_totals",
    "frame.armor_totals",
)

# --- Rail input contract v2 (Stage 3 live-activation + asymmetric endpoints). ---
RAIL_INPUT_CONTRACT_VERSION = "v2-stage3"
RAIL_INPUT_V2_CONTRACT_VERSION = RAIL_INPUT_CONTRACT_VERSION
RAIL_INPUT_POLICY_FORCE_V1 = "force_v1"
RAIL_INPUT_POLICY_V2_STRICT = "v2_strict"
RAIL_INPUT_POLICY_STATES = (
    RAIL_INPUT_POLICY_FORCE_V1,
    RAIL_INPUT_POLICY_V2_STRICT,
)
# Default Stage 1 policy: strict activation, deterministic fallback.
RAIL_INPUT_V2_POLICY = RAIL_INPUT_POLICY_V2_STRICT
RAIL_INPUT_V2_REQUIRED_FIELDS = (
    "frame.alive_counts",
    "frame.cash_totals",
    "frame.loadout_totals",
    "frame.scores",
    "frame.series_score",
    "frame.series_fmt",
)
RAIL_INPUT_V2_OPTIONAL_FIELDS = (
    "frame.armor_totals",
    "frame.wealth_totals",
    "frame.loadout_source",
    "frame.loadout_ev_count_a",
    "frame.loadout_ev_count_b",
    "frame.loadout_est_count_a",
    "frame.loadout_est_count_b",
    "config.prematch_map",
)
RAIL_INPUT_V2_FORBIDDEN_FIELDS = (
    "frame.hp_totals",
    "frame.round_time_remaining_s",
    "frame.round_time_s",
    "frame.bomb_phase_time_remaining",
)
# Fallback reason codes (deterministic).
V2_FALLBACK_POLICY_FORCE_V1 = "POLICY_FORCE_V1"
V2_FALLBACK_REQUIRED_MISSING = "V2_REQUIRED_FIELDS_MISSING"
V2_FALLBACK_REQUIRED_INVALID = "V2_REQUIRED_FIELDS_INVALID"
V2_FALLBACK_POLICY_UNSUPPORTED = "V2_POLICY_UNSUPPORTED"
V2_ACTIVATED = "V2_STRICT_ACTIVATED"

# Semantic carryover signal scaling used only when v2_strict is activated.
V2_CARRYOVER_EDGE_SCALE = 0.08
V2_WINNER_EDGE_BIAS = 0.03


def _v2_required_field_valid(key: str, frame: Frame, config: Config) -> bool:
    """Return True if the required field is present and valid for v2 carryover semantics."""
    if not key.startswith("frame."):
        return False
    attr = key.split(".", 1)[1]
    val = getattr(frame, attr, None)
    if val is None:
        return False
    if attr in ("scores", "series_score"):
        if not isinstance(val, (tuple, list)) or len(val) < 2:
            return False
        try:
            a, b = int(val[0]) if val[0] is not None else None, int(val[1]) if val[1] is not None else None
            return a is not None and b is not None
        except (TypeError, ValueError):
            return False
    if attr == "series_fmt":
        return isinstance(val, str) and bool(str(val).strip())
    if attr in ("cash_totals", "loadout_totals", "armor_totals"):
        if not isinstance(val, (tuple, list)) or len(val) < 2:
            return False
        try:
            a = float(val[0]) if val[0] is not None else None
            b = float(val[1]) if val[1] is not None else None
            return a is not None and b is not None
        except (TypeError, ValueError):
            return False
    if attr == "alive_counts":
        if not isinstance(val, (tuple, list)) or len(val) < 2:
            return False
        try:
            a = int(val[0]) if val[0] is not None else None
            b = int(val[1]) if val[1] is not None else None
            return a is not None and b is not None
        except (TypeError, ValueError):
            return False
    return False


def _resolve_prematch_map_for_rails(config: Config) -> tuple[float, str]:
    """Return the prematch map value used by rail construction plus explicit provenance."""
    val = getattr(config, "prematch_map", None)
    if val is not None:
        try:
            f = float(val)
            if math.isfinite(f) and 1e-6 <= f <= 1.0 - 1e-6:
                return (f, "config_prematch_map")
        except (TypeError, ValueError):
            pass
    return (0.5, "neutral_fallback")


def _rail_input_v2_provenance(
    frame: Frame,
    config: Config,
    *,
    source: str | None = None,
    replay_kind: str | None = None,
) -> dict[str, Any]:
    """Build v2 policy/activation provenance for Stage 1 semantic-switch."""
    present_required: list[str] = []
    missing_required: list[str] = []
    invalid_required: list[str] = []
    for key in RAIL_INPUT_V2_REQUIRED_FIELDS:
        if key == "config.prematch_map":
            val = getattr(config, "prematch_map", None)
        else:
            attr = key.split(".", 1)[1]
            val = getattr(frame, attr, None)
        if val is None:
            missing_required.append(key)
            continue
        if _v2_required_field_valid(key, frame, config):
            present_required.append(key)
        else:
            invalid_required.append(key)

    present_optional: list[str] = []
    for key in RAIL_INPUT_V2_OPTIONAL_FIELDS:
        if key == "config.prematch_map":
            val = getattr(config, "prematch_map", None)
            if val is not None:
                present_optional.append(key)
            continue
        if key.startswith("frame."):
            attr = key.split(".", 1)[1]
            if getattr(frame, attr, None) is not None:
                present_optional.append(key)

    n_required = len(RAIL_INPUT_V2_REQUIRED_FIELDS)
    n_present = len(present_required)
    required_coverage_ratio = (n_present / n_required) if n_required else 0.0
    required_complete = len(missing_required) == 0 and len(invalid_required) == 0

    policy_state = str(getattr(config, "rail_input_contract_policy", RAIL_INPUT_V2_POLICY) or "").strip()
    if not policy_state:
        policy_state = RAIL_INPUT_V2_POLICY

    v2_activated = False
    if policy_state == RAIL_INPUT_POLICY_FORCE_V1:
        reason_code = V2_FALLBACK_POLICY_FORCE_V1
    elif policy_state == RAIL_INPUT_POLICY_V2_STRICT:
        if invalid_required:
            reason_code = V2_FALLBACK_REQUIRED_INVALID
        elif missing_required:
            reason_code = V2_FALLBACK_REQUIRED_MISSING
        else:
            reason_code = V2_ACTIVATED
            v2_activated = True
    else:
        reason_code = V2_FALLBACK_POLICY_UNSUPPORTED

    v1_fallback_used = not v2_activated
    active_semantics = "v2" if v2_activated else "v1"
    prematch_map_used, prematch_map_source = _resolve_prematch_map_for_rails(config)

    return {
        "rail_input_contract_version": RAIL_INPUT_V2_CONTRACT_VERSION,
        "rail_input_contract_policy": policy_state,
        "rail_input_policy_states_supported": list(RAIL_INPUT_POLICY_STATES),
        "rail_input_v2_required_fields": list(RAIL_INPUT_V2_REQUIRED_FIELDS),
        "rail_input_v2_optional_fields": list(RAIL_INPUT_V2_OPTIONAL_FIELDS),
        "rail_input_v2_forbidden_fields": list(RAIL_INPUT_V2_FORBIDDEN_FIELDS),
        "rail_input_v2_present_required_fields": present_required,
        "rail_input_v2_missing_required_fields": missing_required,
        "rail_input_v2_invalid_required_fields": invalid_required,
        "rail_input_v2_present_optional_fields": present_optional,
        "rail_input_v2_required_coverage_ratio": required_coverage_ratio,
        "rail_input_v2_required_complete": required_complete,
        "rail_input_v2_activated": v2_activated,
        "rail_input_v1_fallback_used": v1_fallback_used,
        "rail_input_v1_fallback_reason_code": reason_code,
        "rail_input_activation_reason_code": reason_code,
        "rail_input_active_endpoint_semantics": active_semantics,
        "rail_input_v2_prematch_map_used": prematch_map_used,
        "rail_input_v2_prematch_map_source": prematch_map_source,
        "rail_input_source": source,
        "rail_input_replay_kind": replay_kind,
    }


def _rail_input_provenance(
    frame: Frame,
    forbidden_fields: tuple[str, ...],
    allowed_fields: tuple[str, ...],
    allowed_consumed: tuple[str, ...],
) -> dict[str, Any]:
    """Build per-evaluation provenance: allowed/forbidden sets and consumed/ignored."""
    forbidden_ignored: list[str] = []
    for key in forbidden_fields:
        if not key.startswith("frame."):
            continue
        attr = key.split(".", 1)[1]
        val = getattr(frame, attr, None)
        if val is not None:
            forbidden_ignored.append(key)
    return {
        "rail_input_contract_version": RAIL_INPUT_CONTRACT_VERSION,
        "rail_input_allowed_fields": list(allowed_fields),
        "rail_input_forbidden_fields": list(forbidden_fields),
        "rail_input_allowed_consumed": list(allowed_consumed),
        "rail_input_forbidden_ignored": forbidden_ignored,
    }


def _safe_norm_delta(a: float, b: float) -> float:
    denom = abs(a) + abs(b)
    if denom <= 1e-6:
        return 0.0
    return _clip((a - b) / denom, -1.0, 1.0)


def _future_buy_fragility_score(
    *,
    cash_total: float,
    loadout_total: float,
    armor_total: float,
    wealth_total: float,
    alive_count: float,
) -> float:
    """Approximate next-round fragility from persistent resources only.

    This is intentionally coarse and deterministic: it is a semantic decomposition
    of future-buy resilience, not a tuned prediction model.
    """
    cash_pressure = 1.0 - _clip01(cash_total / 16000.0)
    wealth_pressure = 1.0 - _clip01(wealth_total / 28000.0)
    retained_quality = loadout_total + (8.0 * armor_total)
    retained_pressure = 1.0 - _clip01(retained_quality / 18000.0)
    survivor_pressure = 1.0 - _clip01(alive_count / 5.0)
    return _clip01(
        (0.35 * cash_pressure)
        + (0.30 * wealth_pressure)
        + (0.20 * retained_pressure)
        + (0.15 * survivor_pressure)
    )


def _resource_resilience_score(
    *,
    cash_total: float,
    loadout_total: float,
    armor_total: float,
    wealth_total: float,
    alive_count: float,
) -> float:
    """Coarse persistent next-round resilience score from boundary-stable resources only."""
    cash_quality = _clip01(cash_total / 16000.0)
    wealth_quality = _clip01(wealth_total / 28000.0)
    retained_quality = loadout_total + (8.0 * armor_total)
    retained_quality_score = _clip01(retained_quality / 18000.0)
    survivor_quality = _clip01(alive_count / 5.0)
    return _clip01(
        (0.35 * cash_quality)
        + (0.25 * wealth_quality)
        + (0.25 * retained_quality_score)
        + (0.15 * survivor_quality)
    )


def _branch_score_leverage(
    *,
    canonical_endpoint_v1: float,
    p0: float,
    outcome: str,
) -> float:
    """Normalize branch leverage from the v1 score/map state itself.

    This preserves score-leverage / comeback-burden asymmetry without hard-coding one case.
    """
    if outcome == "a_win":
        denom = max(1e-6, 1.0 - p0)
        return _clip01((canonical_endpoint_v1 - p0) / denom)
    denom = max(1e-6, p0)
    return _clip01((p0 - canonical_endpoint_v1) / denom)


def _compute_v2_carryover_edge(
    frame: Frame,
    *,
    canonical_if_a_v1: float,
    canonical_if_b_v1: float,
    p0: float,
) -> tuple[float, dict[str, float]]:
    """Compute deterministic carryover/asymmetry signals from boundary-persistent inputs only."""
    cash = getattr(frame, "cash_totals", (0.0, 0.0))
    loadout = getattr(frame, "loadout_totals", (0.0, 0.0))
    armor = getattr(frame, "armor_totals", None)
    alive = getattr(frame, "alive_counts", (0.0, 0.0))
    wealth = getattr(frame, "wealth_totals", None)
    cash_a, cash_b = float(cash[0]), float(cash[1])
    load_a, load_b = float(loadout[0]), float(loadout[1])
    if isinstance(armor, (tuple, list)) and len(armor) >= 2:
        arm_a = float(armor[0]) if armor[0] is not None else 0.0
        arm_b = float(armor[1]) if armor[1] is not None else 0.0
        armor_present = True
    else:
        arm_a = arm_b = 0.0
        armor_present = False
    alive_a = float(alive[0]) if alive[0] is not None else 0.0
    alive_b = float(alive[1]) if alive[1] is not None else 0.0
    if isinstance(wealth, (tuple, list)) and len(wealth) >= 2:
        wealth_a = float(wealth[0]) if wealth[0] is not None else (cash_a + load_a)
        wealth_b = float(wealth[1]) if wealth[1] is not None else (cash_b + load_b)
    else:
        wealth_a = cash_a + load_a
        wealth_b = cash_b + load_b
    d_cash = _safe_norm_delta(cash_a, cash_b)
    d_load = _safe_norm_delta(load_a, load_b)
    d_armor = _safe_norm_delta(arm_a, arm_b)
    d_alive = _safe_norm_delta(alive_a, alive_b)
    d_wealth = _safe_norm_delta(wealth_a, wealth_b)
    retained_quality_edge = _clip((0.60 * d_load) + (0.25 * d_armor) + (0.15 * d_alive), -1.0, 1.0)
    economy_edge = _clip((0.55 * d_cash) + (0.45 * d_wealth), -1.0, 1.0)
    fragility_a = _future_buy_fragility_score(
        cash_total=cash_a,
        loadout_total=load_a,
        armor_total=arm_a,
        wealth_total=wealth_a,
        alive_count=alive_a,
    )
    fragility_b = _future_buy_fragility_score(
        cash_total=cash_b,
        loadout_total=load_b,
        armor_total=arm_b,
        wealth_total=wealth_b,
        alive_count=alive_b,
    )
    future_buy_fragility_edge = _clip(fragility_b - fragility_a, -1.0, 1.0)
    resilience_a = _resource_resilience_score(
        cash_total=cash_a,
        loadout_total=load_a,
        armor_total=arm_a,
        wealth_total=wealth_a,
        alive_count=alive_a,
    )
    resilience_b = _resource_resilience_score(
        cash_total=cash_b,
        loadout_total=load_b,
        armor_total=arm_b,
        wealth_total=wealth_b,
        alive_count=alive_b,
    )
    score_leverage_if_a = _branch_score_leverage(
        canonical_endpoint_v1=canonical_if_a_v1,
        p0=p0,
        outcome="a_win",
    )
    score_leverage_if_b = _branch_score_leverage(
        canonical_endpoint_v1=canonical_if_b_v1,
        p0=p0,
        outcome="a_loss",
    )
    branch_edge_if_a_round = _clip01(
        (0.40 * resilience_a)
        + (0.25 * fragility_b)
        + (0.20 * score_leverage_if_a)
        + (0.15 * _clip01(0.5 * (1.0 + retained_quality_edge)))
    )
    branch_edge_if_b_round = _clip01(
        (0.40 * resilience_b)
        + (0.25 * fragility_a)
        + (0.20 * score_leverage_if_b)
        + (0.15 * _clip01(0.5 * (1.0 - retained_quality_edge)))
    )
    carryover_edge = _clip(branch_edge_if_a_round - branch_edge_if_b_round, -1.0, 1.0)
    return carryover_edge, {
        "carryover_edge": carryover_edge,
        "cash_delta_norm": d_cash,
        "loadout_delta_norm": d_load,
        "armor_delta_norm": d_armor,
        "alive_delta_norm": d_alive,
        "wealth_delta_norm": d_wealth,
        "retained_quality_edge": retained_quality_edge,
        "economy_edge": economy_edge,
        "future_buy_fragility_a": fragility_a,
        "future_buy_fragility_b": fragility_b,
        "future_buy_fragility_edge": future_buy_fragility_edge,
        "resource_resilience_a": resilience_a,
        "resource_resilience_b": resilience_b,
        "branch_score_leverage_if_a_round": score_leverage_if_a,
        "branch_score_leverage_if_b_round": score_leverage_if_b,
        "branch_edge_if_a_round": branch_edge_if_a_round,
        "branch_edge_if_b_round": branch_edge_if_b_round,
        "branch_endpoint_shift_if_a_round": (V2_CARRYOVER_EDGE_SCALE * branch_edge_if_a_round) + V2_WINNER_EDGE_BIAS,
        "branch_endpoint_shift_if_b_round": (V2_CARRYOVER_EDGE_SCALE * branch_edge_if_b_round) + V2_WINNER_EDGE_BIAS,
        "armor_totals_missing_assumed_zero": 0.0 if armor_present else 1.0,
    }


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


def _n_maps_from_series_fmt(series_fmt: str) -> int:
    try:
        n = int(str(series_fmt or "bo3").replace("bo", "").strip() or "3")
    except (ValueError, TypeError):
        n = 3
    return 3 if n not in (3, 5) else n


def _compute_context_risk(frame: Frame) -> tuple[float, dict[str, Any]]:
    """Context risk [0,1] from leverage/fragility/missingness. Used only when context_widening_enabled."""
    components: dict[str, Any] = {}
    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    score_diff = abs(ra - rb)
    closeness = 1.0 - min(score_diff / 8.0, 1.0) if score_diff <= 8 else 0.0
    score_sum = ra + rb
    lateness = min(score_sum / float(MAX_ROUNDS_MAP), 1.0) if MAX_ROUNDS_MAP else 0.0
    leader = max(ra, rb)
    match_point_ish = 1.0 if leader >= 11 else (0.5 if leader >= 9 else 0.0)
    leverage_risk = _clip01(0.35 * closeness + 0.35 * lateness + 0.3 * match_point_ish)
    components["leverage_risk"] = leverage_risk
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
        fragility_risk = 0.5
        components["fragility_note"] = "loadout_totals_missing"
    components["fragility_risk"] = fragility_risk
    alive = getattr(frame, "alive_counts", None)
    hp = getattr(frame, "hp_totals", (0.0, 0.0))
    has_alive = isinstance(alive, (tuple, list)) and len(alive) >= 2 and alive[0] is not None and alive[1] is not None
    has_loadout = isinstance(loadout, (tuple, list)) and len(loadout) >= 2
    missingness_risk = 0.3 if not (has_alive and has_loadout) else 0.0
    components["missingness_risk"] = missingness_risk
    context_risk = _clip01(0.4 * leverage_risk + 0.4 * fragility_risk + 0.2 * missingness_risk)
    return context_risk, components


def _map_width_cap(scores: tuple[int, int]) -> tuple[float, dict[str, Any]]:
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    score_sum = ra + rb
    diff = abs(ra - rb)
    closeness = 1.0 - min(diff / 8.0, 1.0)
    if score_sum <= 6:
        base_cap, phase = 0.18, "early"
    elif score_sum <= 16:
        base_cap, phase = 0.26, "mid"
    else:
        base_cap, phase = 0.36, "late"
    extra = 0.04 * closeness if phase != "early" else 0.02 * closeness
    cap = base_cap + extra
    cap = min(cap, 0.22 if phase == "early" else (0.30 if phase == "mid" else 0.40))
    return cap, {"score_sum": score_sum, "closeness": closeness, "phase": phase}


def compute_rails_cs2(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float],
    *,
    source: str | None = None,
    replay_kind: str | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """
    Map corridor from strict next-round counterfactuals.

    - Strict counterfactual (contract) rails:
        cf_low  = P(series | lose next round)
        cf_high = P(series | win  next round)
      rail_low / rail_high returned from this function are the contract rails after only safe post-processing.

    - Heuristic rails (legacy behavior):
      envelope around each canonical endpoint -> active points from Frame microstate -> optional contextual widening.
      These are preserved in debug as rails_heur_low / rails_heur_high (not used for p_hat clamp/invariants).
    """
    series_lo, series_hi = _clip(bounds[0], 0.0, 1.0), _clip(bounds[1], 0.0, 1.0)
    if series_hi < series_lo:
        series_hi = min(1.0, series_lo + 0.02)
    series_width = series_hi - series_lo

    scores = getattr(frame, "scores", (0, 0))
    ra = int(scores[0]) if len(scores) > 0 and scores[0] is not None else 0
    rb = int(scores[1]) if len(scores) > 1 and scores[1] is not None else 0
    series = getattr(frame, "series_score", (0, 0))
    ma = int(series[0]) if len(series) > 0 and series[0] is not None else 0
    mb = int(series[1]) if len(series) > 1 and series[1] is not None else 0
    series_fmt = getattr(frame, "series_fmt", "bo3") or "bo3"
    bo = _n_maps_from_series_fmt(series_fmt)
    p0, p0_source = _resolve_prematch_map_for_rails(config)
    wt = _cs2_win_target(ra, rb)

    # Map-level fair probability for counterfactual states under v1 baseline.
    p_map_if_a_v1: float | None = None
    p_map_if_b_v1: float | None = None
    if (rb + 1) >= wt:
        canonical_if_b_v1 = series_win_prob_live(bo, ma, mb + 1, p0, p0)
    else:
        p_map_if_b_v1 = p_map_fair(ra, rb + 1, config=config, state=state, frame=frame)
        canonical_if_b_v1 = series_win_prob_live(bo, ma, mb, p_map_if_b_v1, p0)
    if (ra + 1) >= wt:
        canonical_if_a_v1 = series_win_prob_live(bo, ma + 1, mb, p0, p0)
    else:
        p_map_if_a_v1 = p_map_fair(ra + 1, rb, config=config, state=state, frame=frame)
        canonical_if_a_v1 = series_win_prob_live(bo, ma, mb, p_map_if_a_v1, p0)
    canonical_if_a_v1 = _clip01(canonical_if_a_v1)
    canonical_if_b_v1 = _clip01(canonical_if_b_v1)

    v2_provenance = _rail_input_v2_provenance(frame, config, source=source, replay_kind=replay_kind)
    use_v2 = bool(v2_provenance.get("rail_input_v2_activated"))
    policy_state = str(v2_provenance.get("rail_input_contract_policy", RAIL_INPUT_V2_POLICY))

    # Active endpoint semantics: v1 fallback or v2 strict (when all required fields valid).
    p_map_if_a_val = p_map_if_a_v1
    p_map_if_b_val = p_map_if_b_v1
    canonical_if_a = canonical_if_a_v1
    canonical_if_b = canonical_if_b_v1
    carryover_debug: dict[str, Any] = {}
    if use_v2:
        carryover_edge, carryover_debug = _compute_v2_carryover_edge(
            frame,
            canonical_if_a_v1=canonical_if_a_v1,
            canonical_if_b_v1=canonical_if_b_v1,
            p0=p0,
        )
        if p_map_if_a_v1 is not None:
            p_map_if_a_val = _clip(
                p_map_if_a_v1 + (V2_CARRYOVER_EDGE_SCALE * carryover_debug["branch_edge_if_a_round"]) + V2_WINNER_EDGE_BIAS,
                1e-6,
                1.0 - 1e-6,
            )
            canonical_if_a = _clip01(series_win_prob_live(bo, ma, mb, p_map_if_a_val, p0))
        if p_map_if_b_v1 is not None:
            p_map_if_b_val = _clip(
                p_map_if_b_v1 - (V2_CARRYOVER_EDGE_SCALE * carryover_debug["branch_edge_if_b_round"]) - V2_WINNER_EDGE_BIAS,
                1e-6,
                1.0 - 1e-6,
            )
            canonical_if_b = _clip01(series_win_prob_live(bo, ma, mb, p_map_if_b_val, p0))

    # Strict counterfactual endpoints before any envelope/active/context transforms.
    cf_low = min(canonical_if_a, canonical_if_b)
    cf_high = max(canonical_if_a, canonical_if_b)
    cf_low = _clip(cf_low, series_lo, series_hi)
    cf_high = _clip(cf_high, series_lo, series_hi)
    cf_low = _clip01(cf_low)
    cf_high = _clip01(cf_high)

    # Provenance: expose active semantic input contract.
    if use_v2:
        allowed_fields = (
            "bounds.low",
            "bounds.high",
            "frame.alive_counts",
            "frame.cash_totals",
            "frame.loadout_totals",
            "frame.armor_totals",
            "frame.wealth_totals",
            "frame.scores",
            "frame.series_score",
            "frame.series_fmt",
            "config.prematch_map",
        )
        forbidden_fields = RAIL_INPUT_V2_FORBIDDEN_FIELDS
    else:
        allowed_fields = RAIL_INPUT_V1_ALLOWED_FIELDS
        forbidden_fields = RAIL_INPUT_V1_FORBIDDEN_FIELDS
    provenance = _rail_input_provenance(frame, forbidden_fields, allowed_fields, allowed_fields)

    # Envelope around each branch (heuristic rails).
    env = compute_cs2_branch_endpoint_envelope_debug(
        canonical_if_a,
        canonical_if_b,
        ra,
        rb,
        bo,
        ma,
        mb,
        wt,
    )
    debug: dict[str, Any] = {
        "canonical_if_a_round": canonical_if_a,
        "canonical_if_b_round": canonical_if_b,
        "canonical_if_a_round_v1": canonical_if_a_v1,
        "canonical_if_b_round_v1": canonical_if_b_v1,
        "p_map_if_a": p_map_if_a_val,
        "p_map_if_b": p_map_if_b_val,
        "p_map_if_a_v1": p_map_if_a_v1,
        "p_map_if_b_v1": p_map_if_b_v1,
        "rails_cf_low": cf_low,
        "rails_cf_high": cf_high,
        "rails_cf_map_score": (ra, rb),
        "rails_cf_series_score": (ma, mb),
        "rails_cf_series_fmt": series_fmt,
        "rail_input_semantic_policy_state": policy_state,
        "rail_input_semantic_v2_activated": use_v2,
        "rail_input_semantic_prematch_map_source": p0_source,
        "base_span": env.get("endpoint_base_span_abs"),
        "k": env.get("endpoint_env_fraction_k"),
    }
    for k, v in carryover_debug.items():
        debug[f"rail_input_v2_{k}"] = v
    for k, v in provenance.items():
        debug[k] = v
    for k, v in v2_provenance.items():
        debug[k] = v
    env_valid = bool(env.get("env_valid"))

    # Heuristic rails (legacy): envelope -> active points -> optional contextual widening.
    # If envelope is invalid, fall back to canonical endpoints as the heuristic rails.
    if not env_valid:
        debug["map_corridor_fallback"] = "invalid_envelope"
        map_low_heur = min(canonical_if_a, canonical_if_b)
        map_high_heur = max(canonical_if_a, canonical_if_b)
    else:
        a_lo = env["endpoint_a_env_low"]
        a_hi = env["endpoint_a_env_high"]
        b_lo = env["endpoint_b_env_low"]
        b_hi = env["endpoint_b_env_high"]
        debug["envelope"] = {k: v for k, v in env.items() if k != "env_valid"}

        # Microstate for position
        alive = getattr(frame, "alive_counts", (5, 5))
        alive_a = (
            float(alive[0])
            if isinstance(alive, (tuple, list)) and len(alive) > 0 and alive[0] is not None
            else 5.0
        )
        alive_b = (
            float(alive[1])
            if isinstance(alive, (tuple, list)) and len(alive) > 1 and alive[1] is not None
            else 5.0
        )
        loadout = getattr(frame, "loadout_totals", None)
        if isinstance(loadout, (tuple, list)) and len(loadout) >= 2:
            load_a = float(loadout[0]) if loadout[0] is not None else 0.0
            load_b = float(loadout[1]) if loadout[1] is not None else 0.0
        else:
            load_a = load_b = 0.0
        armor = getattr(frame, "armor_totals", None)
        if isinstance(armor, (tuple, list)) and len(armor) >= 2:
            arm_a = float(armor[0]) if armor[0] is not None else 0.0
            arm_b = float(armor[1]) if armor[1] is not None else 0.0
        else:
            arm_a = arm_b = 0.0

        pos = compute_cs2_branch_endpoint_position_debug(
            a_lo,
            a_hi,
            b_lo,
            b_hi,
            alive_a,
            alive_b,
            load_a,
            load_b,
            arm_a,
            arm_b,
            branch_temp=0.40,
        )
        debug["quality_pos"] = {
            "d_alive": pos.get("endpoint_pos_d_alive"),
            "d_loadout": pos.get("endpoint_pos_d_loadout"),
            "d_armor": pos.get("endpoint_pos_d_armor"),
            "a_quality_pos": pos.get("endpoint_pos_a_quality_pos"),
            "b_quality_pos": pos.get("endpoint_pos_b_quality_pos"),
        }
        debug["active_points"] = {
            "a_active": pos.get("endpoint_pos_a_active_dbg"),
            "b_active": pos.get("endpoint_pos_b_active_dbg"),
        }

        if (
            pos.get("pos_valid")
            and pos.get("endpoint_pos_a_active_dbg") is not None
            and pos.get("endpoint_pos_b_active_dbg") is not None
        ):
            active_a = pos["endpoint_pos_a_active_dbg"]
            active_b = pos["endpoint_pos_b_active_dbg"]
            map_low_heur = min(active_a, active_b)
            map_high_heur = max(active_a, active_b)
        else:
            map_low_heur = min(canonical_if_a, canonical_if_b)
            map_high_heur = max(canonical_if_a, canonical_if_b)

    # Clamp heuristic rails inside series corridor and [0,1], with legacy min-width + optional contextual widening.
    rail_lo_heur = _clip(map_low_heur, series_lo, series_hi)
    rail_hi_heur = _clip(map_high_heur, series_lo, series_hi)
    rail_lo_heur = _clip01(rail_lo_heur)
    rail_hi_heur = _clip01(rail_hi_heur)
    # Enforce minimum map corridor width (legacy behavior).
    min_map_width = 0.01
    if rail_hi_heur - rail_lo_heur < min_map_width:
        mid_h = 0.5 * (rail_lo_heur + rail_hi_heur)
        half_h = min_map_width / 2.0
        rail_lo_heur = _clip(mid_h - half_h, series_lo, series_hi)
        rail_hi_heur = _clip(mid_h + half_h, series_lo, series_hi)
        rail_lo_heur, rail_hi_heur = _clip01(rail_lo_heur), _clip01(rail_hi_heur)
    if rail_hi_heur < rail_lo_heur:
        rail_hi_heur = min(1.0, rail_lo_heur + min_map_width)

    # Optional: context widening + width cap (gated) – applied only to heuristic rails.
    if getattr(config, "context_widening_enabled", False):
        context_risk, risk_components = _compute_context_risk(frame)
        uncertainty_mult = UNCERTAINTY_MULT_MIN + context_risk * (UNCERTAINTY_MULT_MAX - UNCERTAINTY_MULT_MIN)
        uncertainty_mult = max(UNCERTAINTY_MULT_MIN, min(UNCERTAINTY_MULT_MAX, uncertainty_mult))
        map_width_before = rail_hi_heur - rail_lo_heur
        current_halfwidth = map_width_before / 2.0
        widened_halfwidth = current_halfwidth * (1.0 + CONTEXT_WIDEN_BETA * context_risk)
        mid_h = (rail_lo_heur + rail_hi_heur) / 2.0
        rail_lo_heur = mid_h - widened_halfwidth
        rail_hi_heur = mid_h + widened_halfwidth
        rail_lo_heur = _clip(rail_lo_heur, series_lo, series_hi)
        rail_hi_heur = _clip(rail_hi_heur, series_lo, series_hi)
        rail_lo_heur, rail_hi_heur = _clip01(rail_lo_heur), _clip01(rail_hi_heur)
        if rail_hi_heur < rail_lo_heur:
            rail_hi_heur = min(1.0, rail_lo_heur + 0.01)
        map_width_after_widen = rail_hi_heur - rail_lo_heur
        width_cap_val, cap_meta = _map_width_cap(scores)
        width_cap_used = min(series_width, width_cap_val)
        map_width_after_cap = max(map_width_before, min(map_width_after_widen, width_cap_used))
        if map_width_after_cap <= 0.0:
            map_width_after_cap = map_width_before
        half_capped = 0.5 * map_width_after_cap
        rail_lo_heur = mid_h - half_capped
        rail_hi_heur = mid_h + half_capped
        rail_lo_heur = _clip(rail_lo_heur, series_lo, series_hi)
        rail_hi_heur = _clip(rail_hi_heur, series_lo, series_hi)
        rail_lo_heur, rail_hi_heur = _clip01(rail_lo_heur), _clip01(rail_hi_heur)
        if rail_hi_heur < rail_lo_heur:
            rail_hi_heur = min(1.0, rail_lo_heur + 0.01)
        debug["context_risk"] = context_risk
        debug["context_risk_components"] = risk_components
        debug["uncertainty_multiplier"] = uncertainty_mult
        debug["map_width_before"] = map_width_before
        debug["map_width_after_widen"] = map_width_after_widen
        debug["map_width_after_cap"] = rail_hi_heur - rail_lo_heur
        debug["width_cap_used"] = width_cap_used
        debug["width_cap_meta"] = cap_meta
    else:
        debug["context_widening_enabled"] = False

    # Preserve heuristic rails and deltas vs contract rails in debug.
    debug["rails_heur_low"] = rail_lo_heur
    debug["rails_heur_high"] = rail_hi_heur
    debug["rails_delta_low"] = rail_lo_heur - cf_low
    debug["rails_delta_high"] = rail_hi_heur - cf_high

    # Contract rails: use strict counterfactual endpoints with minimal safety post-processing.
    rail_lo = cf_low
    rail_hi = cf_high
    rail_lo = _clip(rail_lo, series_lo, series_hi)
    rail_hi = _clip(rail_hi, series_lo, series_hi)
    rail_lo = _clip01(rail_lo)
    rail_hi = _clip01(rail_hi)

    # Tiny minimum width epsilon to avoid degenerate collapse.
    applied_eps = False
    if rail_hi - rail_lo < CONTRACT_MIN_WIDTH:
        applied_eps = True
        mid_c = 0.5 * (rail_lo + rail_hi)
        half_c = CONTRACT_MIN_WIDTH / 2.0
        rail_lo = _clip(mid_c - half_c, series_lo, series_hi)
        rail_hi = _clip(mid_c + half_c, series_lo, series_hi)
        rail_lo, rail_hi = _clip01(rail_lo), _clip01(rail_hi)
        if rail_hi < rail_lo:
            rail_hi = min(1.0, rail_lo + CONTRACT_MIN_WIDTH)
    debug["contract_min_width_eps"] = CONTRACT_MIN_WIDTH
    debug["contract_min_width_applied"] = applied_eps

    # Map-point assertions (debug-only): at map point, contract rails should touch series bounds.
    a_on_map_point = (ra + 1) >= wt
    b_on_map_point = (rb + 1) >= wt
    debug["map_point_a_on_point"] = a_on_map_point
    debug["map_point_b_on_point"] = b_on_map_point
    if a_on_map_point:
        delta_hi = rail_hi - series_hi
        debug["map_point_a_delta_hi"] = delta_hi
        if abs(delta_hi) > 1e-3:
            logger.warning(
                "contract rails map-point mismatch for A",
                extra={
                    "scores": (ra, rb),
                    "series_score": (ma, mb),
                    "series_bounds": (series_lo, series_hi),
                    "rail_hi": rail_hi,
                    "delta_hi": delta_hi,
                    "bo": bo,
                    "win_target": wt,
                },
            )
    if b_on_map_point:
        delta_lo = rail_lo - series_lo
        debug["map_point_b_delta_lo"] = delta_lo
        if abs(delta_lo) > 1e-3:
            logger.warning(
                "contract rails map-point mismatch for B",
                extra={
                    "scores": (ra, rb),
                    "series_score": (ma, mb),
                    "series_bounds": (series_lo, series_hi),
                    "rail_lo": rail_lo,
                    "delta_lo": delta_lo,
                    "bo": bo,
                    "win_target": wt,
                },
            )

    debug["rail_low_contract"] = rail_lo
    debug["rail_high_contract"] = rail_hi

    return (rail_lo, rail_hi, debug)

"""
Carryover scenario suite: deterministic tests for Stage 1 rail semantic switch.
Proves policy gate, activation/fallback, carryover directionality, structural safety,
forbidden-transient invariance, and mirror consistency.
"""
from __future__ import annotations

from typing import Any

from engine.compute.bounds import compute_bounds
from engine.compute.rails_cs2 import (
    RAIL_INPUT_POLICY_FORCE_V1,
    RAIL_INPUT_POLICY_V2_STRICT,
    V2_ACTIVATED,
    V2_FALLBACK_POLICY_FORCE_V1,
    V2_FALLBACK_REQUIRED_INVALID,
    V2_FALLBACK_REQUIRED_MISSING,
    compute_rails_cs2,
)
from engine.models import Config, Frame, State


# --- Scenario builder (deterministic, reusable) ---

def _frame(
    scores: tuple[int, int] = (0, 0),
    series_score: tuple[int, int] = (0, 0),
    series_fmt: str = "bo3",
    cash_totals: tuple[float, float] | None = None,
    loadout_totals: tuple[float, float] | None = None,
    armor_totals: tuple[float, float] | None = None,
    hp_totals: tuple[float, float] = (0.0, 0.0),
    alive_counts: tuple[int, int] = (5, 5),
    round_time_remaining_s: float | None = None,
    bomb_phase_time_remaining: Any = None,
) -> Frame:
    return Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=scores,
        series_score=series_score,
        series_fmt=series_fmt,
        map_index=0,
        alive_counts=alive_counts,
        hp_totals=hp_totals,
        cash_loadout_totals=(0.0, 0.0),
        loadout_totals=loadout_totals,
        cash_totals=cash_totals,
        armor_totals=armor_totals,
        bomb_phase_time_remaining=bomb_phase_time_remaining,
        round_time_remaining_s=round_time_remaining_s,
    )


def _config(
    prematch_map: float | None = 0.5,
    rail_input_contract_policy: str | None = RAIL_INPUT_POLICY_V2_STRICT,
) -> Config:
    c = Config()
    if prematch_map is not None:
        c.prematch_map = prematch_map
    if rail_input_contract_policy is not None:
        c.rail_input_contract_policy = rail_input_contract_policy
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def run_scenario(
    frame: Frame,
    config: Config,
    state: State,
    bounds: tuple[float, float] | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Run compute_rails_cs2; if bounds not provided, compute from frame/config/state."""
    if bounds is None:
        lo, hi, _ = compute_bounds(frame, config, state)
        bounds = (lo, hi)
    return compute_rails_cs2(frame, config, state, bounds)


def _mid(rail_lo: float, rail_hi: float) -> float:
    """Midpoint of rails (relational ordering proxy)."""
    return (rail_lo + rail_hi) / 2.0


# --- Neutral and directional carryover presets (fixed score/series for comparability) ---

_SCORE = (6, 6)
_SERIES = (1, 1)
_FMT = "bo3"

NEUTRAL_CARRYOVER = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (2000.0, 2000.0),
    "loadout_totals": (6000.0, 6000.0),
    "armor_totals": (300.0, 300.0),
}

TEAM_A_MODEST = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (3000.0, 2000.0),
    "loadout_totals": (7000.0, 5000.0),
    "armor_totals": (350.0, 250.0),
}

TEAM_A_LARGE = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (5000.0, 1500.0),
    "loadout_totals": (9000.0, 4500.0),
    "armor_totals": (450.0, 250.0),
}

TEAM_B_MODEST = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (2000.0, 3000.0),
    "loadout_totals": (5000.0, 7000.0),
    "armor_totals": (250.0, 350.0),
}

# Economy-heavy A advantage (cash skew), loadout/armor-heavy A advantage (loadout+armor skew)
TEAM_A_ECONOMY_HEAVY = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (5500.0, 1000.0),
    "loadout_totals": (6000.0, 6000.0),
    "armor_totals": (300.0, 300.0),
}

TEAM_A_LOADOUT_ARMOR_HEAVY = {
    "scores": _SCORE,
    "series_score": _SERIES,
    "series_fmt": _FMT,
    "cash_totals": (2000.0, 2000.0),
    "loadout_totals": (9500.0, 3500.0),
    "armor_totals": (500.0, 150.0),
}


# --- S1: v2_strict activates with full valid required fields ---

def test_s1_v2_strict_activates_with_full_valid_required_fields() -> None:
    """S1: Full valid required fields under v2_strict -> V2_STRICT_ACTIVATED, semantics v2."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    frame = _frame(**TEAM_A_MODEST)
    _, _, debug = run_scenario(frame, config, state)
    assert debug["rail_input_v1_fallback_reason_code"] == V2_ACTIVATED
    assert debug["rail_input_v2_activated"] is True
    assert debug["rail_input_active_endpoint_semantics"] == "v2"
    assert debug["rail_input_v2_required_complete"] is True


# --- S2: missing required field fallback ---

def test_s2_missing_required_field_fallback() -> None:
    """S2: Missing required field -> V2_REQUIRED_FIELDS_MISSING, v1 fallback."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    frame = _frame(scores=(2, 1), series_score=(0, 0), cash_totals=None, loadout_totals=None, armor_totals=None)
    _, _, debug = run_scenario(frame, config, state)
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_MISSING
    assert debug["rail_input_v2_activated"] is False
    assert len(debug["rail_input_v2_missing_required_fields"]) >= 1


# --- S3: invalid required field fallback ---

def test_s3_invalid_required_field_fallback() -> None:
    """S3: Invalid required field (e.g. empty series_fmt) -> V2_REQUIRED_FIELDS_INVALID."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    frame = _frame(
        scores=(1, 0),
        series_score=(0, 0),
        series_fmt="",
        cash_totals=(1000.0, 1000.0),
        loadout_totals=(5000.0, 5000.0),
        armor_totals=(200.0, 200.0),
    )
    _, _, debug = run_scenario(frame, config, state)
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_INVALID
    assert debug["rail_input_v2_activated"] is False
    assert "frame.series_fmt" in debug["rail_input_v2_invalid_required_fields"]


# --- S4: force_v1 blocks activation ---

def test_s4_force_v1_blocks_activation() -> None:
    """S4: force_v1 policy -> POLICY_FORCE_V1 even when all required fields valid."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_FORCE_V1)
    state = _state()
    frame = _frame(**TEAM_A_MODEST)
    _, _, debug = run_scenario(frame, config, state)
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_POLICY_FORCE_V1
    assert debug["rail_input_v2_activated"] is False
    assert debug["rail_input_active_endpoint_semantics"] == "v1"


# --- S5: Team A modest carryover vs neutral ---

def test_s5_team_a_modest_vs_neutral() -> None:
    """S5: M(Team A modest) > M(neutral) with fixed score/series."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    r_lo_n, r_hi_n, d_n = run_scenario(_frame(**NEUTRAL_CARRYOVER), config, state, bounds)
    r_lo_a, r_hi_a, d_a = run_scenario(_frame(**TEAM_A_MODEST), config, state, bounds)
    assert d_n["rail_input_v2_activated"] is True and d_a["rail_input_v2_activated"] is True
    M_n = _mid(r_lo_n, r_hi_n)
    M_a = _mid(r_lo_a, r_hi_a)
    assert M_a > M_n, "Team A modest carryover should shift midpoint above neutral"


# --- S6: Team A large edge vs modest edge ---

def test_s6_team_a_large_vs_modest() -> None:
    """S6: M(Team A large) > M(Team A modest) > M(neutral)."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    r_lo_n, r_hi_n, _ = run_scenario(_frame(**NEUTRAL_CARRYOVER), config, state, bounds)
    r_lo_mod, r_hi_mod, _ = run_scenario(_frame(**TEAM_A_MODEST), config, state, bounds)
    r_lo_large, r_hi_large, _ = run_scenario(_frame(**TEAM_A_LARGE), config, state, bounds)
    M_n = _mid(r_lo_n, r_hi_n)
    M_mod = _mid(r_lo_mod, r_hi_mod)
    M_large = _mid(r_lo_large, r_hi_large)
    assert M_mod > M_n
    assert M_large > M_mod


# --- S7: economy-heavy vs loadout/armor-heavy (both Team A favorable vs neutral only) ---

def test_s7_economy_and_loadout_armor_both_team_a_favorable_vs_neutral() -> None:
    """S7: Both economy-heavy and loadout/armor-heavy A advantage move midpoint above neutral.
    Do NOT assert one exceeds the other (formula does not guarantee that)."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    r_lo_n, r_hi_n, _ = run_scenario(_frame(**NEUTRAL_CARRYOVER), config, state, bounds)
    r_lo_eco, r_hi_eco, d_eco = run_scenario(_frame(**TEAM_A_ECONOMY_HEAVY), config, state, bounds)
    r_lo_la, r_hi_la, d_la = run_scenario(_frame(**TEAM_A_LOADOUT_ARMOR_HEAVY), config, state, bounds)
    assert d_eco["rail_input_v2_activated"] is True and d_la["rail_input_v2_activated"] is True
    M_n = _mid(r_lo_n, r_hi_n)
    M_eco = _mid(r_lo_eco, r_hi_eco)
    M_la = _mid(r_lo_la, r_hi_la)
    assert M_eco > M_n, "Economy-heavy A advantage should be above neutral"
    assert M_la > M_n, "Loadout/armor-heavy A advantage should be above neutral"


# --- Directional ordering: M_A_large > M_A_modest > M_neutral > M_B_modest ---

def test_directional_ordering_m_a_large_modest_neutral_m_b_modest() -> None:
    """Full chain: M(Team A large) > M(Team A modest) > M(neutral) > M(Team B modest)."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    r_lo_n, r_hi_n, _ = run_scenario(_frame(**NEUTRAL_CARRYOVER), config, state, bounds)
    r_lo_b, r_hi_b, _ = run_scenario(_frame(**TEAM_B_MODEST), config, state, bounds)
    r_lo_mod, r_hi_mod, _ = run_scenario(_frame(**TEAM_A_MODEST), config, state, bounds)
    r_lo_large, r_hi_large, _ = run_scenario(_frame(**TEAM_A_LARGE), config, state, bounds)
    M_b = _mid(r_lo_b, r_hi_b)
    M_n = _mid(r_lo_n, r_hi_n)
    M_mod = _mid(r_lo_mod, r_hi_mod)
    M_large = _mid(r_lo_large, r_hi_large)
    assert M_b < M_n < M_mod < M_large


# --- S8: mirrored Team B version of S5 ---

def test_s8_team_b_modest_vs_neutral() -> None:
    """S8: M(Team B modest) < M(neutral) (mirror of S5)."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    r_lo_n, r_hi_n, d_n = run_scenario(_frame(**NEUTRAL_CARRYOVER), config, state, bounds)
    r_lo_b, r_hi_b, d_b = run_scenario(_frame(**TEAM_B_MODEST), config, state, bounds)
    assert d_n["rail_input_v2_activated"] is True and d_b["rail_input_v2_activated"] is True
    M_n = _mid(r_lo_n, r_hi_n)
    M_b = _mid(r_lo_b, r_hi_b)
    assert M_b < M_n, "Team B modest carryover should shift midpoint below neutral"


# --- S9: late-map pressure with carryover differences ---

def test_s9_late_map_pressure_structural_safety() -> None:
    """S9: Late-map score (e.g. 10-9) with neutral vs A carryover: rails valid, ordered, in [0,1]."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    late_neutral = {"scores": (10, 9), "series_score": (1, 0), "series_fmt": "bo3",
                    "cash_totals": (2000.0, 2000.0), "loadout_totals": (6000.0, 6000.0), "armor_totals": (300.0, 300.0)}
    late_a = {"scores": (10, 9), "series_score": (1, 0), "series_fmt": "bo3",
              "cash_totals": (4000.0, 1500.0), "loadout_totals": (8000.0, 5000.0), "armor_totals": (400.0, 200.0)}
    for kwargs in (late_neutral, late_a):
        r_lo, r_hi, debug = run_scenario(_frame(**kwargs), config, state, bounds)
        assert 0 <= r_lo <= 1 and 0 <= r_hi <= 1
        assert r_lo <= r_hi
        assert debug.get("rail_low_contract") is not None and debug.get("rail_high_contract") is not None


# --- S10: map-point structural pair under v2 activation ---

def test_s10_map_point_structural_pair_under_v2() -> None:
    """S10: At map point (e.g. 12-11) with v2 active: structural safeguards, rail ordering, in-bounds."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    frame = _frame(
        scores=(12, 11),
        series_score=(0, 0),
        series_fmt="bo3",
        cash_totals=(3500.0, 3000.0),
        loadout_totals=(8000.0, 7000.0),
        armor_totals=(400.0, 350.0),
    )
    bounds_result = compute_bounds(frame, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    r_lo, r_hi, debug = run_scenario(frame, config, state, bounds)
    assert debug["rail_input_v2_activated"] is True
    assert 0 <= r_lo <= 1 and 0 <= r_hi <= 1
    assert r_lo <= r_hi
    assert r_hi - r_lo >= 1e-5, "contract min-width or non-degenerate width"
    assert bounds[0] <= r_lo <= bounds[1] and bounds[0] <= r_hi <= bounds[1]


# --- S11: forbidden transient invariance (strict) ---

def test_s11_forbidden_transient_invariance_strict() -> None:
    """S11: With fixed allowed inputs (scores, series, cash, loadout, armor), perturb only
    still-forbidden transients (hp, round_time, bomb_phase); rails must stay identical."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    base = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        cash_totals=(3000.0, 2000.0),
        loadout_totals=(8000.0, 6000.0),
        armor_totals=(400.0, 300.0),
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        round_time_remaining_s=None,
        bomb_phase_time_remaining=None,
    )
    bounds_result = compute_bounds(base, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    r_lo_base, r_hi_base, _ = run_scenario(base, config, state, bounds)

    perturbed = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        cash_totals=(3000.0, 2000.0),
        loadout_totals=(8000.0, 6000.0),
        armor_totals=(400.0, 300.0),
        alive_counts=(5, 5),
        hp_totals=(100.0, 200.0),
        round_time_remaining_s=45.0,
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS", "round_time_remaining": 45.0},
    )
    r_lo_p, r_hi_p, _ = run_scenario(perturbed, config, state, bounds)
    assert r_lo_base == r_lo_p, "rails must be invariant to forbidden transient perturbation"
    assert r_hi_base == r_hi_p, "rails must be invariant to forbidden transient perturbation"


def test_s12_branch_asymmetry_from_resource_fragility() -> None:
    """Resource fragility/resilience should create non-mirror endpoint shifts, not one shared rail move."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    frame = _frame(
        scores=(6, 6),
        series_score=(1, 1),
        series_fmt="bo3",
        alive_counts=(5, 2),
        cash_totals=(5000.0, 1000.0),
        loadout_totals=(9000.0, 3500.0),
        armor_totals=(400.0, 100.0),
    )
    _, _, debug = run_scenario(frame, config, state, bounds)
    assert debug["rail_input_v2_activated"] is True
    shift_if_a = debug["p_map_if_a"] - debug["p_map_if_a_v1"]
    shift_if_b = debug["p_map_if_b_v1"] - debug["p_map_if_b"]
    assert debug["rail_input_v2_branch_edge_if_a_round"] > debug["rail_input_v2_branch_edge_if_b_round"]
    assert shift_if_a > shift_if_b


def test_s13_branch_asymmetry_from_score_leverage_and_comeback_burden() -> None:
    """Lopsided score states should create branch leverage asymmetry even with neutral resources."""
    config = _config(rail_input_contract_policy=RAIL_INPUT_POLICY_V2_STRICT)
    state = _state()
    bounds = (0.0, 1.0)
    frame = _frame(
        scores=(10, 2),
        series_score=(1, 0),
        series_fmt="bo3",
        alive_counts=(5, 5),
        cash_totals=(2500.0, 2500.0),
        loadout_totals=(7000.0, 7000.0),
        armor_totals=(300.0, 300.0),
    )
    _, _, debug = run_scenario(frame, config, state, bounds)
    assert debug["rail_input_v2_activated"] is True
    assert debug["rail_input_v2_branch_score_leverage_if_a_round"] > debug["rail_input_v2_branch_score_leverage_if_b_round"]
    shift_if_a = debug["p_map_if_a"] - debug["p_map_if_a_v1"]
    shift_if_b = debug["p_map_if_b_v1"] - debug["p_map_if_b"]
    assert shift_if_a > shift_if_b

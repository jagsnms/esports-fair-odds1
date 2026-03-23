"""
Stage 1 rail semantic-switch contract tests.
Policy is explicit (force_v1 | v2_strict), no partial-v2 mode.
"""
from __future__ import annotations

from engine.compute.bounds import compute_bounds
from engine.compute.rails import compute_rails
from engine.compute.rails_cs2 import (
    RAIL_INPUT_V2_CONTRACT_VERSION,
    RAIL_INPUT_POLICY_FORCE_V1,
    RAIL_INPUT_POLICY_V2_STRICT,
    RAIL_INPUT_V2_POLICY,
    RAIL_INPUT_V2_REQUIRED_FIELDS,
    V2_ACTIVATED,
    V2_FALLBACK_POLICY_FORCE_V1,
    V2_FALLBACK_POLICY_UNSUPPORTED,
    V2_FALLBACK_REQUIRED_MISSING,
    V2_FALLBACK_REQUIRED_INVALID,
    compute_rails_cs2,
)
from engine.models import Config, Frame, State


def _frame(
    scores: tuple[int, int] = (0, 0),
    series_score: tuple[int, int] = (0, 0),
    series_fmt: str = "bo3",
    alive_counts: tuple[int, int] = (5, 5),
    hp_totals: tuple[float, float] = (0.0, 0.0),
    loadout_totals: tuple[float, float] | None = None,
    bomb_phase_time_remaining=None,
    round_time_remaining_s: float | None = None,
    cash_totals: tuple[float, float] | None = None,
    armor_totals: tuple[float, float] | None = None,
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
        bomb_phase_time_remaining=bomb_phase_time_remaining,
        round_time_remaining_s=round_time_remaining_s,
        cash_totals=cash_totals,
        armor_totals=armor_totals,
    )


def _config(prematch_map: float | None = None) -> Config:
    c = Config()
    if prematch_map is not None:
        c.prematch_map = prematch_map
    return c


def _state() -> State:
    return State(config=Config(), last_frame=None, map_index=0, last_total_rounds=0)


def test_provenance_keys_emitted() -> None:
    """Every contract-rail evaluation emits v2 policy/activation provenance keys."""
    frame = _frame(scores=(3, 2), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds = (0.0, 1.0)
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    assert debug.get("rail_input_contract_version") == RAIL_INPUT_V2_CONTRACT_VERSION
    assert debug.get("rail_input_contract_policy") == RAIL_INPUT_V2_POLICY
    assert "rail_input_allowed_fields" in debug
    assert "rail_input_forbidden_fields" in debug
    assert "rail_input_allowed_consumed" in debug
    assert "rail_input_forbidden_ignored" in debug
    assert "rail_input_policy_states_supported" in debug
    assert debug.get("rail_input_v2_activated") is False
    assert debug.get("rail_input_active_endpoint_semantics") == "v1"
    assert "rail_input_v2_required_fields" in debug
    assert "rail_input_v2_missing_required_fields" in debug
    assert "rail_input_v2_required_coverage_ratio" in debug
    assert "rail_input_v1_fallback_reason_code" in debug
    assert "bounds.low" in debug["rail_input_allowed_consumed"]
    assert "bounds.high" in debug["rail_input_allowed_consumed"]


def test_forbidden_perturbation_invariance() -> None:
    """Perturbing still-forbidden transient inputs does not change contract rails under v2_strict activation."""
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    base = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        loadout_totals=(8000.0, 6000.0),
        bomb_phase_time_remaining=None,
        round_time_remaining_s=100.0,
        cash_totals=(3000.0, 2000.0),
        armor_totals=(400.0, 300.0),
    )
    bounds_result = compute_bounds(base, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    r_lo_base, r_hi_base, _ = compute_rails_cs2(base, config, state, bounds)

    # Perturb only forbidden transient inputs.
    perturbed = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        alive_counts=(5, 5),
        hp_totals=(100.0, 200.0),
        loadout_totals=(8000.0, 6000.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS", "round_time_remaining": 45.0},
        round_time_remaining_s=45.0,
        cash_totals=(3000.0, 2000.0),
        armor_totals=(400.0, 300.0),
    )
    r_lo_p, r_hi_p, _ = compute_rails_cs2(perturbed, config, state, bounds)
    assert r_lo_base == r_lo_p, "contract rails must be invariant to forbidden input perturbation"
    assert r_hi_base == r_hi_p, "contract rails must be invariant to forbidden input perturbation"


def test_survivor_carryover_sensitivity_when_v2_active() -> None:
    """Alive-count survivor advantage is now part of strict v2 endpoint semantics."""
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    bounds = (0.0, 1.0)
    neutral = _frame(
        scores=(6, 6),
        series_score=(1, 1),
        series_fmt="bo3",
        alive_counts=(3, 3),
        cash_totals=(2500.0, 2500.0),
        loadout_totals=(7000.0, 7000.0),
        armor_totals=(300.0, 300.0),
    )
    advantaged = _frame(
        scores=(6, 6),
        series_score=(1, 1),
        series_fmt="bo3",
        alive_counts=(5, 2),
        cash_totals=(2500.0, 2500.0),
        loadout_totals=(7000.0, 7000.0),
        armor_totals=(300.0, 300.0),
    )
    lo_neutral, hi_neutral, dbg_neutral = compute_rails_cs2(neutral, config, state, bounds)
    lo_adv, hi_adv, dbg_adv = compute_rails_cs2(advantaged, config, state, bounds)
    assert dbg_neutral["rail_input_v2_activated"] is True
    assert dbg_adv["rail_input_v2_activated"] is True
    assert dbg_adv["rail_input_v2_alive_delta_norm"] > dbg_neutral["rail_input_v2_alive_delta_norm"]
    assert ((lo_adv + hi_adv) / 2.0) > ((lo_neutral + hi_neutral) / 2.0)


def test_carryover_sensitivity_when_v2_active() -> None:
    """With fixed score/series, changing required carryover vectors changes contract endpoints in v2_strict."""
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    bounds = (0.0, 1.0)
    base = _frame(
        scores=(6, 6),
        series_score=(1, 1),
        series_fmt="bo3",
        cash_totals=(2000.0, 2000.0),
        loadout_totals=(6000.0, 6000.0),
        armor_totals=(300.0, 300.0),
    )
    advantaged = _frame(
        scores=(6, 6),
        series_score=(1, 1),
        series_fmt="bo3",
        cash_totals=(5000.0, 1500.0),
        loadout_totals=(9000.0, 4500.0),
        armor_totals=(450.0, 250.0),
    )
    lo_base, hi_base, dbg_base = compute_rails_cs2(base, config, state, bounds)
    lo_adv, hi_adv, dbg_adv = compute_rails_cs2(advantaged, config, state, bounds)
    assert dbg_base["rail_input_v2_activated"] is True
    assert dbg_adv["rail_input_v2_activated"] is True
    assert (lo_base != lo_adv) or (hi_base != hi_adv)


def test_bo3_vs_grid_contract_parity() -> None:
    """Same v2 policy classification and rails for BO3/GRID/REPLAY when inputs are identical."""
    config_bo3 = _config()
    config_bo3.source = "BO3"
    config_bo3.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    config_grid = _config()
    config_grid.source = "GRID"
    config_grid.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    config_replay = _config()
    config_replay.source = "REPLAY"
    config_replay.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    frame = _frame(
        scores=(6, 5),
        series_score=(1, 0),
        series_fmt="bo3",
        cash_totals=(3000.0, 2500.0),
        loadout_totals=(8000.0, 7000.0),
        armor_totals=(400.0, 350.0),
    )
    bounds_result = compute_bounds(frame, config_bo3, state)
    bounds = (bounds_result[0], bounds_result[1])
    _, _, debug_bo3 = compute_rails_cs2(frame, config_bo3, state, bounds)
    _, _, debug_grid = compute_rails_cs2(frame, config_grid, state, bounds)
    _, _, debug_replay = compute_rails_cs2(frame, config_replay, state, bounds)
    assert debug_bo3["rail_input_contract_version"] == debug_grid["rail_input_contract_version"] == RAIL_INPUT_V2_CONTRACT_VERSION
    assert debug_bo3["rail_input_contract_version"] == debug_replay["rail_input_contract_version"]
    assert debug_bo3["rail_input_contract_policy"] == debug_grid["rail_input_contract_policy"] == debug_replay["rail_input_contract_policy"]
    assert debug_bo3["rail_input_v1_fallback_reason_code"] == debug_grid["rail_input_v1_fallback_reason_code"]
    assert debug_bo3["rail_input_v1_fallback_reason_code"] == debug_replay["rail_input_v1_fallback_reason_code"]
    assert debug_bo3["rail_input_v2_required_coverage_ratio"] == debug_grid["rail_input_v2_required_coverage_ratio"]
    assert debug_bo3["rail_input_v2_required_coverage_ratio"] == debug_replay["rail_input_v2_required_coverage_ratio"]
    assert debug_bo3["rail_low_contract"] == debug_grid["rail_low_contract"]
    assert debug_bo3["rail_high_contract"] == debug_grid["rail_high_contract"]
    assert debug_bo3["rail_low_contract"] == debug_replay["rail_low_contract"]
    assert debug_bo3["rail_high_contract"] == debug_replay["rail_high_contract"]


def test_force_v1_policy_forces_fallback_even_when_required_complete() -> None:
    """force_v1 keeps v1 semantics even if all required fields are valid."""
    frame = _frame(scores=(1, 0), series_score=(0, 0), cash_totals=(1000.0, 1000.0), loadout_totals=(5000.0, 5000.0), armor_totals=(200.0, 200.0))
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_FORCE_V1
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_used"] is True
    assert debug["rail_input_v2_activated"] is False
    assert debug["rail_input_active_endpoint_semantics"] == "v1"
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_POLICY_FORCE_V1


def test_v2_fallback_reason_missing_when_required_absent() -> None:
    """When v2 required fields are missing, fallback reason is V2_REQUIRED_FIELDS_MISSING."""
    # Frame without cash_totals, loadout_totals, armor_totals (None)
    frame = _frame(scores=(2, 1), series_score=(0, 0), cash_totals=None, loadout_totals=None, armor_totals=None)
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_MISSING
    assert debug["rail_input_v2_activated"] is False
    assert "frame.cash_totals" in debug["rail_input_v2_missing_required_fields"] or "frame.loadout_totals" in debug["rail_input_v2_missing_required_fields"] or "frame.armor_totals" in debug["rail_input_v2_missing_required_fields"]


def test_v2_strict_activates_when_required_complete() -> None:
    """When all required fields are valid and policy is v2_strict, v2 activates."""
    frame = _frame(
        scores=(3, 2),
        series_score=(0, 0),
        series_fmt="bo3",
        cash_totals=(2000.0, 1500.0),
        loadout_totals=(8000.0, 6000.0),
        armor_totals=(300.0, 250.0),
    )
    config = _config(prematch_map=0.52)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_reason_code"] == V2_ACTIVATED
    assert debug["rail_input_v2_required_complete"] is True
    assert debug["rail_input_v2_activated"] is True
    assert debug["rail_input_active_endpoint_semantics"] == "v2"
    assert debug["rail_input_v2_required_coverage_ratio"] == 1.0
    assert "frame.alive_counts" in debug["rail_input_v2_required_fields"]
    assert "frame.alive_counts" in debug["rail_input_allowed_consumed"]


def test_v2_fallback_reason_invalid_when_required_bad_type() -> None:
    """When a v2 required field is present but invalid, fallback reason is V2_REQUIRED_FIELDS_INVALID."""
    # prematch_map outside [0,1] or wrong type can be invalid; use a frame with invalid series_fmt (empty)
    frame = _frame(scores=(1, 0), series_score=(0, 0), series_fmt="", cash_totals=(0.0, 0.0), loadout_totals=(0.0, 0.0), armor_totals=(0.0, 0.0))
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = RAIL_INPUT_POLICY_V2_STRICT
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    # series_fmt "" is invalid (must be non-empty str), so we get invalid_required
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_INVALID
    assert "frame.series_fmt" in debug["rail_input_v2_invalid_required_fields"]


def test_v2_policy_unsupported_reason_code() -> None:
    """Unsupported policy values deterministically fall back with V2_POLICY_UNSUPPORTED."""
    frame = _frame(
        scores=(3, 2),
        series_score=(0, 0),
        cash_totals=(1500.0, 1500.0),
        loadout_totals=(5000.0, 5000.0),
        armor_totals=(300.0, 300.0),
    )
    config = _config(prematch_map=0.5)
    config.rail_input_contract_policy = "unknown_policy_value"
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v2_activated"] is False
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_POLICY_UNSUPPORTED


def test_rail_input_source_replay_kind_in_provenance() -> None:
    """When source and replay_kind are passed to compute_rails_cs2 they appear in debug provenance."""
    frame = _frame(scores=(0, 0), series_score=(0, 0))
    config = _config()
    state = _state()
    _, _, debug = compute_rails_cs2(
        frame, config, state, (0.0, 1.0), source="REPLAY", replay_kind="raw"
    )
    assert debug.get("rail_input_source") == "REPLAY"
    assert debug.get("rail_input_replay_kind") == "raw"


def test_compute_rails_forwards_source_replay_kind() -> None:
    """compute_rails (rails.py) forwards source/replay_kind to compute_rails_cs2 and they appear in debug."""
    frame = _frame(scores=(0, 0), series_score=(0, 0))
    config = _config()
    state = _state()
    _, _, debug = compute_rails(
        frame, config, state, (0.0, 1.0), source="GRID", replay_kind="live"
    )
    assert debug.get("rail_input_source") == "GRID"
    assert debug.get("rail_input_replay_kind") == "live"

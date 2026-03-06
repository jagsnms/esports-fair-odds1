"""
Rail input contract (v1 + v2 carryover observability) and provenance.
Stage 1: contract definition + observability only; no endpoint redesign.
v2 Stage 1: observe v2 required/optional; always use v1 endpoints (fallback).
"""
from __future__ import annotations

from engine.compute.bounds import compute_bounds
from engine.compute.rails_cs2 import (
    RAIL_INPUT_ALLOWED_FIELDS,
    RAIL_INPUT_CONTRACT_VERSION,
    RAIL_INPUT_FORBIDDEN_FIELDS,
    RAIL_INPUT_V2_CONTRACT_VERSION,
    RAIL_INPUT_V2_POLICY,
    RAIL_INPUT_V2_REQUIRED_FIELDS,
    V2_FALLBACK_REQUIRED_MISSING,
    V2_FALLBACK_REQUIRED_INVALID,
    V2_FALLBACK_STAGE1_LOCKED,
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
    """Every contract-rail evaluation emits v1 provenance keys and v2 observability keys."""
    frame = _frame(scores=(3, 2), series_score=(0, 0))
    config = _config()
    state = _state()
    bounds = (0.0, 1.0)
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    # v2 contract version and policy (Stage 1)
    assert debug.get("rail_input_contract_version") == RAIL_INPUT_V2_CONTRACT_VERSION
    assert debug.get("rail_input_contract_policy") == RAIL_INPUT_V2_POLICY
    # v1 keys retained for backward compatibility
    assert "rail_input_allowed_fields" in debug
    assert "rail_input_forbidden_fields" in debug
    assert "rail_input_allowed_consumed" in debug
    assert "rail_input_forbidden_ignored" in debug
    assert debug["rail_input_allowed_fields"] == list(RAIL_INPUT_ALLOWED_FIELDS)
    assert debug["rail_input_forbidden_fields"] == list(RAIL_INPUT_FORBIDDEN_FIELDS)
    assert set(debug["rail_input_allowed_consumed"]) == set(RAIL_INPUT_ALLOWED_FIELDS)
    # v2 observability
    assert debug.get("rail_input_v1_fallback_used") is True
    assert debug.get("rail_input_active_endpoint_semantics") == "v1"
    assert "rail_input_v2_required_fields" in debug
    assert "rail_input_v2_missing_required_fields" in debug
    assert "rail_input_v2_required_coverage_ratio" in debug
    assert "rail_input_v1_fallback_reason_code" in debug


def test_forbidden_perturbation_invariance() -> None:
    """Perturbing forbidden inputs while holding allowed inputs fixed does not change rail_low/rail_high."""
    config = _config(prematch_map=0.5)
    state = _state()
    base = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        alive_counts=(5, 5),
        hp_totals=(500.0, 500.0),
        loadout_totals=None,
        bomb_phase_time_remaining=None,
        round_time_remaining_s=None,
    )
    bounds_result = compute_bounds(base, config, state)
    bounds = (bounds_result[0], bounds_result[1])
    r_lo_base, r_hi_base, _ = compute_rails_cs2(base, config, state, bounds)

    # Perturb only forbidden: different alive, hp, loadout, bomb, round_time, cash, armor
    perturbed = _frame(
        scores=(5, 4),
        series_score=(1, 0),
        series_fmt="bo3",
        alive_counts=(2, 3),
        hp_totals=(100.0, 200.0),
        loadout_totals=(8000.0, 6000.0),
        bomb_phase_time_remaining={"round_phase": "IN_PROGRESS", "round_time_remaining": 45.0},
        round_time_remaining_s=45.0,
        cash_totals=(2000.0, 1500.0),
        armor_totals=(400.0, 300.0),
    )
    r_lo_p, r_hi_p, _ = compute_rails_cs2(perturbed, config, state, bounds)
    assert r_lo_base == r_lo_p, "contract rails must be invariant to forbidden input perturbation"
    assert r_hi_base == r_hi_p, "contract rails must be invariant to forbidden input perturbation"


def test_allowed_consumed_reflects_contract() -> None:
    """rail_input_allowed_consumed lists exactly the inputs used for contract rails."""
    frame = _frame(scores=(1, 0), series_score=(0, 0))
    config = _config(prematch_map=0.48)
    state = _state()
    bounds = (0.2, 0.8)
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    consumed = debug["rail_input_allowed_consumed"]
    assert "bounds.low" in consumed
    assert "bounds.high" in consumed
    assert "frame.scores" in consumed
    assert "frame.series_score" in consumed
    assert "frame.series_fmt" in consumed
    assert "config.prematch_map" in consumed
    assert len(consumed) == 6


def test_forbidden_ignored_when_present() -> None:
    """When frame has forbidden attributes set, they appear in rail_input_forbidden_ignored."""
    frame = _frame(
        scores=(2, 1),
        hp_totals=(400.0, 300.0),
        loadout_totals=(5000.0, 5000.0),
        bomb_phase_time_remaining={"round_phase": "live"},
        round_time_remaining_s=90.0,
    )
    config = _config()
    state = _state()
    bounds = (0.0, 1.0)
    _, _, debug = compute_rails_cs2(frame, config, state, bounds)
    ignored = debug["rail_input_forbidden_ignored"]
    assert "frame.hp_totals" in ignored
    assert "frame.alive_counts" in ignored  # default (5,5) is not None
    assert "frame.loadout_totals" in ignored
    assert "frame.bomb_phase_time_remaining" in ignored
    assert "frame.round_time_remaining_s" in ignored


def test_allowed_input_change_changes_rails() -> None:
    """Changing an allowed input (e.g. scores) is reflected in rails and in consumed provenance."""
    config = _config()
    state = _state()
    bounds = (0.0, 1.0)
    f1 = _frame(scores=(3, 3), series_score=(0, 0))
    f2 = _frame(scores=(4, 2), series_score=(0, 0))
    _, _, d1 = compute_rails_cs2(f1, config, state, bounds)
    _, _, d2 = compute_rails_cs2(f2, config, state, bounds)
    # Different map score -> canonical endpoints can differ
    assert d1["rails_cf_map_score"] != d2["rails_cf_map_score"]
    assert d1["rail_input_allowed_consumed"] == d2["rail_input_allowed_consumed"]


def test_bo3_vs_grid_contract_parity() -> None:
    """Same contract version and v2 classification for BO3 and GRID when inputs are identical."""
    config_bo3 = _config()
    config_bo3.source = "BO3"
    config_grid = _config()
    config_grid.source = "GRID"
    state = _state()
    frame = _frame(scores=(6, 5), series_score=(1, 0), series_fmt="bo3")
    bounds_result = compute_bounds(frame, config_bo3, state)
    bounds = (bounds_result[0], bounds_result[1])
    _, _, debug_bo3 = compute_rails_cs2(frame, config_bo3, state, bounds)
    _, _, debug_grid = compute_rails_cs2(frame, config_grid, state, bounds)
    assert debug_bo3["rail_input_contract_version"] == debug_grid["rail_input_contract_version"] == RAIL_INPUT_V2_CONTRACT_VERSION
    assert debug_bo3["rail_input_allowed_fields"] == debug_grid["rail_input_allowed_fields"]
    assert debug_bo3["rail_input_forbidden_fields"] == debug_grid["rail_input_forbidden_fields"]
    assert debug_bo3["rail_input_allowed_consumed"] == debug_grid["rail_input_allowed_consumed"]
    assert debug_bo3["rail_input_forbidden_ignored"] == debug_grid["rail_input_forbidden_ignored"]
    assert debug_bo3["rail_input_v1_fallback_reason_code"] == debug_grid["rail_input_v1_fallback_reason_code"]
    assert debug_bo3["rail_input_v2_required_coverage_ratio"] == debug_grid["rail_input_v2_required_coverage_ratio"]
    assert debug_bo3["rail_low_contract"] == debug_grid["rail_low_contract"]
    assert debug_bo3["rail_high_contract"] == debug_grid["rail_high_contract"]


def test_v2_fallback_always_used() -> None:
    """Stage 1: v1 fallback always used; active endpoint semantics remain v1."""
    frame = _frame(scores=(1, 0), series_score=(0, 0), cash_totals=(1000.0, 1000.0), loadout_totals=(5000.0, 5000.0), armor_totals=(200.0, 200.0))
    config = _config(prematch_map=0.5)
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_used"] is True
    assert debug["rail_input_active_endpoint_semantics"] == "v1"


def test_v2_fallback_reason_missing_when_required_absent() -> None:
    """When v2 required fields are missing, fallback reason is V2_REQUIRED_FIELDS_MISSING."""
    # Frame without cash_totals, loadout_totals, armor_totals (None)
    frame = _frame(scores=(2, 1), series_score=(0, 0), cash_totals=None, loadout_totals=None, armor_totals=None)
    config = _config(prematch_map=0.5)
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_MISSING
    assert "frame.cash_totals" in debug["rail_input_v2_missing_required_fields"] or "frame.loadout_totals" in debug["rail_input_v2_missing_required_fields"] or "frame.armor_totals" in debug["rail_input_v2_missing_required_fields"]


def test_v2_fallback_reason_stage1_locked_when_required_complete() -> None:
    """When all v2 required fields present and valid, fallback reason is STAGE1_LOCKED_NO_SEMANTIC_SWITCH."""
    frame = _frame(
        scores=(3, 2),
        series_score=(0, 0),
        series_fmt="bo3",
        cash_totals=(2000.0, 1500.0),
        loadout_totals=(8000.0, 6000.0),
        armor_totals=(300.0, 250.0),
    )
    config = _config(prematch_map=0.52)
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_STAGE1_LOCKED
    assert debug["rail_input_v2_required_complete"] is True
    assert debug["rail_input_v2_required_coverage_ratio"] == 1.0


def test_v2_fallback_reason_invalid_when_required_bad_type() -> None:
    """When a v2 required field is present but invalid, fallback reason is V2_REQUIRED_FIELDS_INVALID."""
    # prematch_map outside [0,1] or wrong type can be invalid; use a frame with invalid series_fmt (empty)
    frame = _frame(scores=(1, 0), series_score=(0, 0), series_fmt="", cash_totals=(0.0, 0.0), loadout_totals=(0.0, 0.0), armor_totals=(0.0, 0.0))
    config = _config(prematch_map=0.5)
    state = _state()
    _, _, debug = compute_rails_cs2(frame, config, state, (0.0, 1.0))
    # series_fmt "" is invalid (must be non-empty str), so we get invalid_required
    assert debug["rail_input_v1_fallback_reason_code"] == V2_FALLBACK_REQUIRED_INVALID
    assert "frame.series_fmt" in debug["rail_input_v2_invalid_required_fields"]

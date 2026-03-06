"""Unit tests for Bible contract diagnostics classification."""

from __future__ import annotations

from engine.diagnostics.invariants import compute_phat_contract_diagnostics


def test_contract_diagnostics_structural_q_out_of_bounds() -> None:
    diag = compute_phat_contract_diagnostics(
        q_intra_total=1.2,
        rail_low=0.2,
        rail_high=0.8,
        p_hat_prev=0.5,
        p_hat_final=0.5,
        movement_confidence=0.25,
        phase="IN_PROGRESS",
        round_time_remaining_s=88.0,
        bomb_time_remaining_s=None,
        is_bomb_planted=False,
        alive_counts=(5, 4),
        hp_totals=(400.0, 320.0),
        loadout_totals=(12000.0, 9000.0),
        round_phase="IN_PROGRESS",
        round_number=12,
        testing_mode=True,
    )
    assert "q_out_of_bounds" in diag["structural_violations"]
    assert diag["target_p_hat"] is None


def test_contract_diagnostics_structural_rail_order_invalid() -> None:
    diag = compute_phat_contract_diagnostics(
        q_intra_total=0.5,
        rail_low=0.8,
        rail_high=0.2,
        p_hat_prev=0.5,
        p_hat_final=0.5,
        movement_confidence=0.25,
        phase="IN_PROGRESS",
        round_time_remaining_s=42.0,
        bomb_time_remaining_s=20.0,
        is_bomb_planted=True,
        alive_counts=(2, 5),
        hp_totals=(180.0, 420.0),
        loadout_totals=(5000.0, 11000.0),
        round_phase="IN_PROGRESS",
        round_number=21,
        testing_mode=True,
    )
    assert "rail_order_invalid" in diag["structural_violations"]


def test_contract_diagnostics_behavioral_gap_testing_only() -> None:
    common = dict(
        q_intra_total=0.9,
        rail_low=0.2,
        rail_high=0.8,
        p_hat_prev=0.2,
        p_hat_final=0.8,
        movement_confidence=0.25,
        phase="IN_PROGRESS",
        round_time_remaining_s=15.0,
        bomb_time_remaining_s=12.0,
        is_bomb_planted=True,
        alive_counts=(3, 1),
        hp_totals=(250.0, 100.0),
        loadout_totals=(8000.0, 3000.0),
        round_phase="IN_PROGRESS",
        round_number=28,
    )
    diag_testing = compute_phat_contract_diagnostics(testing_mode=True, **common)
    diag_production = compute_phat_contract_diagnostics(testing_mode=False, **common)
    assert "movement_contract_gap" in diag_testing["behavioral_violations"]
    assert diag_production["behavioral_violations"] == []


def test_contract_diagnostics_emits_core_state_fields() -> None:
    diag = compute_phat_contract_diagnostics(
        q_intra_total=0.6,
        rail_low=0.3,
        rail_high=0.7,
        p_hat_prev=0.4,
        p_hat_final=0.5,
        movement_confidence=0.25,
        phase="IN_PROGRESS",
        round_time_remaining_s=30.0,
        bomb_time_remaining_s=18.0,
        is_bomb_planted=False,
        alive_counts=(4, 3),
        hp_totals=(300.0, 220.0),
        loadout_totals=(9000.0, 7000.0),
        round_phase="IN_PROGRESS",
        round_number=10,
        testing_mode=True,
    )
    assert diag["rail_low"] == 0.3
    assert diag["rail_high"] == 0.7
    assert diag["p_hat_prev"] == 0.4
    assert diag["p_hat_final"] == 0.5
    assert diag["round_time_remaining_s"] == 30.0
    assert diag["bomb_time_remaining_s"] == 18.0
    assert diag["is_bomb_planted"] is False
    assert diag["alive_counts"] == (4, 3)
    assert diag["hp_totals"] == (300.0, 220.0)
    assert diag["loadout_totals"] == (9000.0, 7000.0)
    assert diag["round_phase"] == "IN_PROGRESS"
    assert diag["round_number"] == 10

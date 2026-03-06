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
    )
    diag_testing = compute_phat_contract_diagnostics(testing_mode=True, **common)
    diag_production = compute_phat_contract_diagnostics(testing_mode=False, **common)
    assert "movement_contract_gap" in diag_testing["behavioral_violations"]
    assert diag_production["behavioral_violations"] == []

"""Unit tests for corridor invariant classification and reporting."""

from __future__ import annotations

from engine.diagnostics.invariants import compute_corridor_invariants


def test_corridor_invariants_all_ok() -> None:
    inv = compute_corridor_invariants(
        series_low=0.2,
        series_high=0.8,
        map_low=0.3,
        map_high=0.7,
        p_hat=0.5,
    )
    assert inv["invariant_series_map_order_ok"] is True
    assert inv["invariant_p_hat_in_map_ok"] is True
    assert inv["invariant_structural_violations"] == []
    assert inv["invariant_behavioral_violations"] == []
    assert inv["invariant_violations"] == []


def test_corridor_invariants_order_violation_only() -> None:
    inv = compute_corridor_invariants(
        series_low=0.2,
        series_high=0.8,
        map_low=0.1,
        map_high=0.7,
        p_hat=0.5,
    )
    assert inv["invariant_series_map_order_ok"] is False
    assert inv["invariant_p_hat_in_map_ok"] is True
    assert inv["invariant_structural_violations"] == ["series_map_order"]
    assert inv["invariant_behavioral_violations"] == []
    assert inv["invariant_violations"] == ["series_map_order"]


def test_corridor_invariants_p_hat_outside_map() -> None:
    inv = compute_corridor_invariants(
        series_low=0.2,
        series_high=0.8,
        map_low=0.3,
        map_high=0.6,
        p_hat=0.75,
    )
    assert inv["invariant_series_map_order_ok"] is True
    assert inv["invariant_p_hat_in_map_ok"] is False
    assert inv["invariant_structural_violations"] == []
    assert inv["invariant_behavioral_violations"] == ["p_hat_outside_map"]
    assert inv["invariant_violations"] == ["p_hat_outside_map"]


def test_corridor_invariants_production_mode_reports_structural_only() -> None:
    inv = compute_corridor_invariants(
        series_low=0.2,
        series_high=0.8,
        map_low=0.3,
        map_high=0.6,
        p_hat=0.75,
        testing_mode=False,
    )
    assert inv["invariant_mode"] == "production"
    assert inv["invariant_structural_violations"] == []
    assert inv["invariant_behavioral_violations"] == ["p_hat_outside_map"]
    # In production mode, primary invariant list remains structural-only.
    assert inv["invariant_violations"] == []


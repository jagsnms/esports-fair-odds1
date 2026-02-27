"""Unit tests for non-fatal corridor invariant checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[2] / "backend" / "services" / "runner.py"
SPEC = importlib.util.spec_from_file_location("runner_module", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER_MODULE)
compute_corridor_invariants = RUNNER_MODULE.compute_corridor_invariants


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
    assert inv["invariant_violations"] == ["p_hat_outside_map"]


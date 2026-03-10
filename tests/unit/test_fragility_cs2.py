"""
Unit tests for raw/fragility debug: build_raw_debug, compute_fragility_debug.
"""
from __future__ import annotations

from engine.models import Frame
from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug


def test_fragility_loadout_ratio_and_low_flag_and_econ_asymmetry() -> None:
    """Frame with loadout_totals=(20000, 4000) yields loadout_ratio=0.2, low_loadout_flag True, econ_asymmetry positive."""
    frame = Frame(
        scores=(5, 3),
        series_score=(0, 0),
        loadout_totals=(20000.0, 4000.0),
        alive_counts=(5, 5),
        hp_totals=(100.0, 100.0),
    )
    fragility = compute_fragility_debug(frame)
    assert fragility["loadout_a"] == 20000.0
    assert fragility["loadout_b"] == 4000.0
    assert fragility["loadout_min"] == 4000.0
    assert fragility["loadout_max"] == 20000.0
    assert abs((fragility["loadout_ratio"] or 0) - 0.2) < 1e-9
    assert fragility["low_loadout_flag"] is True  # min 4000 < 5000
    # econ_asymmetry = (a - b) / max = (20000 - 4000) / 20000 = 0.8
    assert fragility["econ_asymmetry"] is not None
    assert (fragility["econ_asymmetry"] or 0) > 0
    assert abs((fragility["econ_asymmetry"] or 0) - 0.8) < 1e-9
    assert fragility["missing_microstate_flag"] is False


def test_fragility_missing_loadout_none_and_missing_microstate_flag() -> None:
    """Frame missing loadout -> fragility loadout fields None and missing_microstate_flag True."""
    frame = Frame(
        scores=(0, 0),
        series_score=(0, 0),
        loadout_totals=None,
        alive_counts=(5, 5),
        hp_totals=(0.0, 0.0),
    )
    fragility = compute_fragility_debug(frame)
    assert fragility["loadout_a"] is None
    assert fragility["loadout_b"] is None
    assert fragility["loadout_sum"] is None
    assert fragility["loadout_min"] is None
    assert fragility["loadout_max"] is None
    assert fragility["loadout_ratio"] is None
    assert fragility["econ_asymmetry"] is None
    assert fragility["low_loadout_flag"] is None
    assert fragility["missing_microstate_flag"] is True


def test_build_raw_debug_includes_present_flags() -> None:
    """build_raw_debug includes inputs_present booleans and raw values when present."""
    frame = Frame(
        scores=(3, 2),
        series_score=(1, 0),
        map_index=0,
        map_name="mirage",
        a_side="CT",
        alive_counts=(5, 4),
        hp_totals=(400.0, 350.0),
        loadout_totals=(10000.0, 8000.0),
    )
    raw = build_raw_debug(frame)
    assert raw["scores"] == [3, 2]
    assert raw["scores_present"] is True
    assert raw["series_score"] == [1, 0]
    assert raw["loadout_totals"] == [10000.0, 8000.0]
    assert raw["loadout_totals_present"] is True
    assert raw["map_name"] == "mirage"
    assert raw["a_side"] == "CT"

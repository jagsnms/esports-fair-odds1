"""
Unit tests for prematch series -> map derivation: derive_p_map_from_p_series.
"""
from __future__ import annotations

from engine.compute.series_iid import derive_p_map_from_p_series, series_win_prob_live


def test_derive_bo3_half_series_gives_half_map() -> None:
    """derive_p_map_from_p_series(3, 0.5) ~= 0.5 (symmetric BO3)."""
    p_map = derive_p_map_from_p_series(3, 0.5)
    assert abs(p_map - 0.5) < 0.01
    p_series = series_win_prob_live(3, 0, 0, p_map, p_map)
    assert abs(p_series - 0.5) < 1e-6


def test_derive_monotonic_higher_series_gives_higher_map() -> None:
    """0.6 series -> p_map > p_map(0.5); 0.4 series -> p_map < p_map(0.5)."""
    p_map_05 = derive_p_map_from_p_series(3, 0.5)
    p_map_06 = derive_p_map_from_p_series(3, 0.6)
    p_map_04 = derive_p_map_from_p_series(3, 0.4)
    assert p_map_06 > p_map_05
    assert p_map_04 < p_map_05
    assert abs(series_win_prob_live(3, 0, 0, p_map_06, p_map_06) - 0.6) < 1e-6
    assert abs(series_win_prob_live(3, 0, 0, p_map_04, p_map_04) - 0.4) < 1e-6

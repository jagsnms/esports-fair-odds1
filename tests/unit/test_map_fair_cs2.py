"""
Unit tests for CS2 map-level fair probability (p_map_fair).
"""
from __future__ import annotations

import unittest

from engine.compute.map_fair_cs2 import p_map_fair
from engine.models import Config


def test_p_map_fair_near_05_at_0_0() -> None:
    """At ra=0, rb=0, p_map_fair should be near 0.5 (e.g. between 0.45 and 0.55)."""
    p = p_map_fair(0, 0)
    assert 0.45 <= p <= 0.55
    assert 0.02 <= p <= 0.98


def test_p_map_fair_modest_shift_after_one_round() -> None:
    """After one round (1-0 vs 0-1), p_map_fair should shift modestly, not slam to extremes."""
    p_1_0 = p_map_fair(1, 0)
    p_0_1 = p_map_fair(0, 1)
    # Within +/-0.10 of 0.5 (modest shift)
    assert 0.40 <= p_1_0 <= 0.60
    assert 0.40 <= p_0_1 <= 0.60
    # 1-0 favors A, 0-1 favors B
    assert p_1_0 > p_0_1
    assert 0.02 <= p_1_0 <= 0.98
    assert 0.02 <= p_0_1 <= 0.98


def test_p_map_fair_clamped_no_extremes_early() -> None:
    """Early scores (e.g. 2-0, 0-2) should not output 0 or 1."""
    for ra, rb in [(2, 0), (0, 2), (3, 1)]:
        p = p_map_fair(ra, rb)
        assert p >= 0.02
        assert p <= 0.98


class TestMapFairCs2(unittest.TestCase):
    def test_p_map_fair_near_05_at_0_0(self) -> None:
        test_p_map_fair_near_05_at_0_0()

    def test_p_map_fair_modest_shift_after_one_round(self) -> None:
        test_p_map_fair_modest_shift_after_one_round()

    def test_p_map_fair_clamped_no_extremes_early(self) -> None:
        test_p_map_fair_clamped_no_extremes_early()


if __name__ == "__main__":
    unittest.main()

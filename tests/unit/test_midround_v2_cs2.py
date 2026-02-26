"""
Unit tests for CS2 midround V2 mixture oracle (midround_v2_cs2).
"""
from __future__ import annotations

import unittest

from engine.compute.midround_v2_cs2 import (
    apply_cs2_midround_adjustment_v2_mixture,
    compute_cs2_midround_features,
)
from engine.models import Frame


def _features(
    alive_diff: int = 0,
    hp_a: float = 0.0,
    hp_b: float = 0.0,
    load_a: float = 0.0,
    load_b: float = 0.0,
    bomb_planted: int = 0,
    a_side: str | None = None,
    time_progress: float = 0.5,
    armor_delta: float | None = None,
) -> dict:
    """Minimal features dict for mixture tests."""
    d = {
        "alive_diff": alive_diff,
        "hp_diff_alive": hp_a - hp_b,
        "hp_a_total": hp_a,
        "hp_b_total": hp_b,
        "loadout_diff_alive": load_a - load_b,
        "load_a_total": load_a,
        "load_b_total": load_b,
        "bomb_planted": bomb_planted,
        "a_side": a_side or "",
        "time_progress": time_progress,
        "inputs_present": {
            "alive": True,
            "hp": True,
            "loadout": True,
            "armor": armor_delta is not None,
            "bomb": True,
            "time": True,
            "a_side": a_side in ("T", "CT"),
        },
    }
    if armor_delta is not None:
        d["armor_diff_alive"] = armor_delta
    return d


def test_clamps_between_endpoints() -> None:
    """p_mid_clamped is always between frozen_a and frozen_b."""
    features = _features(alive_diff=2, hp_a=400, hp_b=200, load_a=10000, load_b=5000)
    for frozen_a, frozen_b in [(0.3, 0.7), (0.7, 0.3), (0.5, 0.5)]:
        result = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=frozen_a, frozen_b=frozen_b, features=features
        )
        lo = min(frozen_a, frozen_b)
        hi = max(frozen_a, frozen_b)
        assert lo <= result["p_mid_clamped"] <= hi


def test_bomb_direction_gated_by_side() -> None:
    """Bomb term applies direction only when a_side is T or CT; unknown side -> no bomb direction."""
    features_bomb_t = _features(bomb_planted=1, a_side="T", alive_diff=0, hp_a=500, hp_b=500, load_a=5000, load_b=5000)
    features_bomb_ct = _features(bomb_planted=1, a_side="CT", alive_diff=0, hp_a=500, hp_b=500, load_a=5000, load_b=5000)
    features_bomb_unknown = _features(bomb_planted=1, a_side=None, alive_diff=0, hp_a=500, hp_b=500, load_a=5000, load_b=5000)

    r_t = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=features_bomb_t)
    r_ct = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=features_bomb_ct)
    r_unknown = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=features_bomb_unknown)

    assert r_t["used_bomb_direction"] is True
    assert r_ct["used_bomb_direction"] is True
    assert r_unknown["used_bomb_direction"] is False
    assert r_t["q_intra"] > r_unknown["q_intra"]
    assert r_ct["q_intra"] < r_unknown["q_intra"]


def test_monotonic_alive_advantage_increases_q_and_p_mid() -> None:
    """Increasing alive advantage (A vs B) increases q_intra and p_mid_clamped when frozen_a > frozen_b."""
    frozen_a, frozen_b = 0.7, 0.3
    f_low = _features(alive_diff=0, hp_a=300, hp_b=300, load_a=5000, load_b=5000)
    f_high = _features(alive_diff=3, hp_a=400, hp_b=200, load_a=8000, load_b=4000)

    r_low = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_low)
    r_high = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_high)

    assert r_high["q_intra"] > r_low["q_intra"]
    assert r_high["p_mid_clamped"] > r_low["p_mid_clamped"]


def test_missing_microstate_midpoint() -> None:
    """When key microstate missing, q_intra=0.5 and p_mid_clamped = (frozen_a + frozen_b)/2."""
    features = {
        "alive_diff": 0,
        "hp_diff_alive": 0.0,
        "hp_a_total": 0.0,
        "hp_b_total": 0.0,
        "loadout_diff_alive": 0.0,
        "load_a_total": 0.0,
        "load_b_total": 0.0,
        "bomb_planted": 0,
        "time_progress": 0.5,
        "inputs_present": {
            "alive": False,
            "hp": False,
            "loadout": False,
            "armor": False,
            "bomb": False,
            "time": False,
            "a_side": False,
        },
    }
    result = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=features)
    assert result.get("reason") == "missing_microstate"
    assert result["q_intra"] == 0.5
    assert result["p_mid_clamped"] == 0.5
    assert result["p_mid"] == 0.5


def test_compute_features_from_frame() -> None:
    """compute_cs2_midround_features extracts from Frame and returns inputs_present + reliability."""
    frame = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(5, 4),
        alive_counts=(4, 2),
        hp_totals=(400.0, 200.0),
        loadout_totals=(12000.0, 6000.0),
        armor_totals=(100.0, 80.0),
        a_side="T",
        bomb_phase_time_remaining={ "round_time_remaining": 45.0, "is_bomb_planted": True, "round_phase": "live" },
    )
    features = compute_cs2_midround_features(frame)
    assert features["alive_diff"] == 2
    assert features["hp_diff_alive"] == 200.0
    assert features["loadout_diff_alive"] == 6000.0
    assert features["bomb_planted"] == 1
    assert features["a_side"] == "T"
    assert "inputs_present" in features
    assert "reliability" in features
    assert features["reliability"].get("has_clock") is True


class TestMidroundV2Cs2(unittest.TestCase):
    """Run the same tests via unittest."""

    def test_clamps_between_endpoints(self) -> None:
        test_clamps_between_endpoints()

    def test_bomb_direction_gated_by_side(self) -> None:
        test_bomb_direction_gated_by_side()

    def test_monotonic_alive_advantage_increases_q_and_p_mid(self) -> None:
        test_monotonic_alive_advantage_increases_q_and_p_mid()

    def test_missing_microstate_midpoint(self) -> None:
        test_missing_microstate_midpoint()

    def test_compute_features_from_frame(self) -> None:
        test_compute_features_from_frame()


if __name__ == "__main__":
    unittest.main()

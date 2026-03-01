"""
Unit tests for CS2 midround V2 mixture oracle (midround_v2_cs2).
"""
from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

from engine.compute.midround_v2_cs2 import (
    HP_FRAC_WEIGHT,
    WEIGHTS_LEARNED_FIT,
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
    hp_sum = hp_a + hp_b
    hp_frac_a = hp_a / max(hp_sum, 1.0)
    hp_asym = (hp_a - hp_b) / max(max(hp_a, hp_b), 1.0)
    hp_ratio = (min(hp_a, hp_b) / max(hp_a, hp_b)) if max(hp_a, hp_b) > 0 else None
    d = {
        "alive_diff": alive_diff,
        "hp_diff_alive": hp_a - hp_b,
        "hp_a": hp_a,
        "hp_b": hp_b,
        "hp_a_total": hp_a,
        "hp_b_total": hp_b,
        "hp_sum": hp_sum,
        "hp_frac_a": hp_frac_a,
        "hp_asym": hp_asym,
        "hp_ratio": hp_ratio,
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
        bomb_phase_time_remaining={"round_time_remaining": 45.0, "is_bomb_planted": True, "round_phase": "live"},
        round_time_remaining_s=45.0,
    )
    features = compute_cs2_midround_features(frame)
    assert features["alive_diff"] == 2
    assert features["hp_diff_alive"] == 200.0
    assert features["hp_a"] == 400.0
    assert features["hp_b"] == 200.0
    assert features["hp_sum"] == 600.0
    assert features["hp_frac_a"] == 400.0 / 600.0
    assert features["hp_ratio"] == 200.0 / 400.0
    assert features["loadout_diff_alive"] == 6000.0
    assert features["bomb_planted"] == 1
    assert features["a_side"] == "T"
    assert "inputs_present" in features
    assert "reliability" in features
    assert features["reliability"].get("has_clock") is True


def test_hp_advantage_increases_q_and_p_mid_when_frozen_a_gt_frozen_b() -> None:
    """With same alive counts and same loadout, increasing hp_a while decreasing hp_b
    increases q_intra and p_mid_clamped when frozen_a > frozen_b."""
    frozen_a, frozen_b = 0.7, 0.3
    # Same alive (0) and same loadout (5000 each); only HP changes
    f_low = _features(alive_diff=0, hp_a=200, hp_b=400, load_a=5000, load_b=5000)
    f_mid = _features(alive_diff=0, hp_a=300, hp_b=300, load_a=5000, load_b=5000)
    f_high = _features(alive_diff=0, hp_a=400, hp_b=200, load_a=5000, load_b=5000)

    r_low = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_low)
    r_mid = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_mid)
    r_high = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_high)

    assert r_low["q_intra"] < r_mid["q_intra"] < r_high["q_intra"]
    assert r_low["p_mid_clamped"] < r_mid["p_mid_clamped"] < r_high["p_mid_clamped"]
    assert r_high["term_hp"] > 0
    assert r_low["term_hp"] < 0


def test_hp_fraction_bounded() -> None:
    """hp_frac_a: (500,0)->1.0, (0,500)->0.0; term_hp and q_intra reflect bounds."""
    f_a_only = _features(alive_diff=0, hp_a=500, hp_b=0, load_a=5000, load_b=5000)
    f_b_only = _features(alive_diff=0, hp_a=0, hp_b=500, load_a=5000, load_b=5000)
    assert f_a_only["hp_frac_a"] == 1.0
    assert f_b_only["hp_frac_a"] == 0.0
    result_a = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=f_a_only)
    result_b = apply_cs2_midround_adjustment_v2_mixture(frozen_a=0.7, frozen_b=0.3, features=f_b_only)
    assert result_a["hp_frac_a"] == 1.0
    assert result_b["hp_frac_a"] == 0.0
    assert result_a["term_hp"] == 0.5 * HP_FRAC_WEIGHT  # HP_FRAC_WEIGHT * (1 - 0.5)
    assert result_b["term_hp"] == -0.5 * HP_FRAC_WEIGHT  # HP_FRAC_WEIGHT * (0 - 0.5)
    assert result_a["q_intra"] > 0.5
    assert result_b["q_intra"] < 0.5
    assert result_a["p_mid_clamped"] > result_b["p_mid_clamped"]


def test_same_alive_loadout_higher_hp_frac_a_raises_q_and_p_mid() -> None:
    """Same alive/loadout, higher hp_frac_a => higher q_intra and p_mid_clamped (frozen_a > frozen_b)."""
    frozen_a, frozen_b = 0.7, 0.3
    # Identical alive (2v2) and loadout (6k each); only HP fraction varies
    f_low = _features(alive_diff=0, hp_a=200, hp_b=400, load_a=6000, load_b=6000)
    f_high = _features(alive_diff=0, hp_a=400, hp_b=200, load_a=6000, load_b=6000)
    r_low = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_low)
    r_high = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f_high)
    assert r_high["hp_frac_a"] > r_low["hp_frac_a"]
    assert r_high["q_intra"] > r_low["q_intra"]
    assert r_high["p_mid_clamped"] > r_low["p_mid_clamped"]


def test_symmetric_2v2_near_equal_hp_q_intra_near_half() -> None:
    """Symmetric 2v2, near-equal HP => q_intra in [0.4, 0.6] when other terms neutral."""
    frozen_a, frozen_b = 0.6, 0.4
    f = _features(alive_diff=0, hp_a=250, hp_b=250, load_a=5000, load_b=5000, bomb_planted=0, a_side=None)
    result = apply_cs2_midround_adjustment_v2_mixture(frozen_a=frozen_a, frozen_b=frozen_b, features=f)
    assert 0.4 <= result["q_intra"] <= 0.6
    assert result["hp_frac_a"] == 0.5
    assert result["term_hp"] == 0.0


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

    def test_hp_advantage_increases_q_and_p_mid_when_frozen_a_gt_frozen_b(self) -> None:
        test_hp_advantage_increases_q_and_p_mid_when_frozen_a_gt_frozen_b()

    def test_hp_fraction_bounded(self) -> None:
        test_hp_fraction_bounded()

    def test_same_alive_loadout_higher_hp_frac_a_raises_q_and_p_mid(self) -> None:
        test_same_alive_loadout_higher_hp_frac_a_raises_q_and_p_mid()

    def test_symmetric_2v2_near_equal_hp_q_intra_near_half(self) -> None:
        test_symmetric_2v2_near_equal_hp_q_intra_near_half()

    def test_weight_profile_learned_v1_and_current(self) -> None:
        test_weight_profile_learned_v1_and_current()

    def test_learned_fit_team_a_advantage_gives_q_above_half(self) -> None:
        test_learned_fit_team_a_advantage_gives_q_above_half()


def test_learned_fit_team_a_advantage_gives_q_above_half() -> None:
    """Regression: learned_fit must be Team-A-positive; clear A advantage -> raw_score > 0 and q_intra > 0.5."""
    # Team A advantage: more alive, more HP share, more loadout
    features = _features(
        alive_diff=2,
        hp_a=400.0,
        hp_b=200.0,
        load_a=12000.0,
        load_b=4000.0,
        bomb_planted=0,
    )
    config = SimpleNamespace(midround_v2_weight_profile="learned_fit")
    result = apply_cs2_midround_adjustment_v2_mixture(
        frozen_a=0.7,
        frozen_b=0.3,
        features=features,
        config=config,
    )
    assert result.get("weight_profile") == "learned_fit"
    raw_score = result.get("raw_score_pre_urgency")
    q_intra = result.get("q_intra")
    assert raw_score is not None and raw_score > 0, (
        f"learned_fit with Team A advantage must give raw_score_pre_urgency > 0, got {raw_score}"
    )
    assert q_intra is not None and q_intra > 0.5, (
        f"learned_fit with Team A advantage must give q_intra (p_unshaped) > 0.5, got {q_intra}"
    )


def test_weight_profile_learned_v1_and_current() -> None:
    """With MIDROUND_V2_WEIGHT_PROFILE=learned_v1, term_coef uses learned_v1 weights; with current, uses current."""
    features = _features(alive_diff=1, hp_a=400, hp_b=300, load_a=5000, load_b=5000)
    saved = os.environ.get("MIDROUND_V2_WEIGHT_PROFILE")

    try:
        os.environ["MIDROUND_V2_WEIGHT_PROFILE"] = "learned_v1"
        result = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=0.7, frozen_b=0.3, features=features
        )
        assert result.get("weight_profile") == "learned_v1"
        tc = result.get("term_coef", {})
        assert tc.get("loadout") == 0.003
        assert tc.get("alive") == 0.06
        assert tc.get("hp") == 0.07
        assert tc.get("bomb") == 0.10
        assert tc.get("cash") == 0.0

        os.environ["MIDROUND_V2_WEIGHT_PROFILE"] = "learned_v2"
        result_v2 = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=0.7, frozen_b=0.3, features=features
        )
        assert result_v2.get("weight_profile") == "learned_v2"
        tc_v2 = result_v2.get("term_coef", {})
        assert tc_v2.get("alive") == 0.08
        assert tc_v2.get("hp") == 0.12
        assert tc_v2.get("loadout") == 0.002
        assert tc_v2.get("bomb") == 0.10
        assert tc_v2.get("cash") == 0.0

        os.environ["MIDROUND_V2_WEIGHT_PROFILE"] = "learned_fit"
        result_fit = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=0.7, frozen_b=0.3, features=features
        )
        assert result_fit.get("weight_profile") == "learned_fit"
        tc_fit = result_fit.get("term_coef", {})
        for key in ("alive", "hp", "loadout", "bomb", "cash"):
            assert tc_fit.get(key) == WEIGHTS_LEARNED_FIT[key], (
                f"learned_fit term_coef[{key}]={tc_fit.get(key)} != WEIGHTS_LEARNED_FIT[{key}]={WEIGHTS_LEARNED_FIT[key]}"
            )

        os.environ["MIDROUND_V2_WEIGHT_PROFILE"] = "current"
        result2 = apply_cs2_midround_adjustment_v2_mixture(
            frozen_a=0.7, frozen_b=0.3, features=features
        )
        assert result2.get("weight_profile") == "current"
        tc2 = result2.get("term_coef", {})
        assert tc2.get("loadout") == 0.012
        assert tc2.get("alive") == 0.035
        assert tc2.get("hp") == 0.04
        assert tc2.get("bomb") == 0.06
    finally:
        if saved is not None:
            os.environ["MIDROUND_V2_WEIGHT_PROFILE"] = saved
        else:
            os.environ.pop("MIDROUND_V2_WEIGHT_PROFILE", None)


if __name__ == "__main__":
    unittest.main()

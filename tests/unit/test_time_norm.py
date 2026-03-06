"""
Unit tests for canonical round time normalization (time_norm).
"""
from __future__ import annotations

import unittest

from engine.normalize.time_norm import normalize_round_time


def test_ms_111003_to_seconds() -> None:
    """round_time_remaining=111003 (ms) -> round_time_remaining_s ~= 111.003."""
    r = normalize_round_time(111003)
    assert r["raw"] == 111003
    assert r["seconds"] is not None
    assert abs(r["seconds"] - 111.003) < 0.001
    assert r["is_ms"] is True
    assert r["invalid_reason"] is None


def test_ms_35998_to_seconds() -> None:
    """round_time_remaining=35998 (ms) -> ~= 35.998."""
    r = normalize_round_time(35998)
    assert r["seconds"] is not None
    assert abs(r["seconds"] - 35.998) < 0.001
    assert r["is_ms"] is True
    assert r["invalid_reason"] is None


def test_negative_ms_out_of_range() -> None:
    """round_time_remaining=-130066 -> seconds None, invalid_reason set."""
    r = normalize_round_time(-130066)
    assert r["raw"] == -130066
    assert r["seconds"] is None
    assert r["invalid_reason"] == "out_of_range"
    assert r["is_ms"] is True


def test_huge_positive_ms_out_of_range() -> None:
    """Very large ms value -> seconds None, invalid_reason out_of_range."""
    r = normalize_round_time(300000)
    assert r["raw"] == 300000
    assert r["seconds"] is None
    assert r["invalid_reason"] == "out_of_range"
    assert r["is_ms"] is True


def test_small_seconds_stays_45() -> None:
    """Small seconds value (e.g. 45) stays 45 (not treated as ms)."""
    r = normalize_round_time(45)
    assert r["seconds"] == 45.0
    assert r["is_ms"] is False
    assert r["invalid_reason"] is None


def test_none_input() -> None:
    """None -> seconds None, no invalid_reason."""
    r = normalize_round_time(None)
    assert r["raw"] is None
    assert r["seconds"] is None
    assert r["invalid_reason"] is None


def test_frame_gets_normalized_time_from_bo3_snapshot() -> None:
    """bo3_snapshot_to_frame sets round_time_remaining_s and round_time_remaining_raw from snapshot."""
    from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
    raw = {
        "team_one": {"name": "A", "score": 0},
        "team_two": {"name": "B", "score": 0},
        "round_time_remaining": 111003,
        "bomb_time_remaining": 35500,
        "round_phase": "live",
    }
    frame = bo3_snapshot_to_frame(raw, team_a_is_team_one=True)
    assert frame.round_time_remaining_raw == 111003
    assert frame.round_time_remaining_s is not None
    assert abs(frame.round_time_remaining_s - 111.003) < 0.001
    assert frame.bomb_phase_time_remaining is not None
    assert abs(frame.bomb_phase_time_remaining.get("round_time_remaining", 0) - 111.003) < 0.001
    assert abs(frame.bomb_phase_time_remaining.get("bomb_time_remaining", 0) - 35.5) < 0.001


class TestTimeNorm(unittest.TestCase):
    def test_ms_111003_to_seconds(self) -> None:
        test_ms_111003_to_seconds()

    def test_ms_35998_to_seconds(self) -> None:
        test_ms_35998_to_seconds()

    def test_negative_ms_out_of_range(self) -> None:
        test_negative_ms_out_of_range()

    def test_small_seconds_stays_45(self) -> None:
        test_small_seconds_stays_45()

    def test_none_input(self) -> None:
        test_none_input()

    def test_frame_gets_normalized_time_from_bo3_snapshot(self) -> None:
        test_frame_gets_normalized_time_from_bo3_snapshot()

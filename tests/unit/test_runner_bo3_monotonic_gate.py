"""
Unit tests for BO3 tick monotonic gating (no FastAPI dependency).
Tests that out-of-order frames are rejected (ts_backwards, clock_rewind, alive_sig_rewind).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.services.bo3_freshness import (
    Bo3FreshnessGate,
    CLOCK_REWIND_EPS_S,
    coerce_ts_ms,
)


def test_coerce_ts_ms() -> None:
    assert coerce_ts_ms(1700000000) == 1700000000000  # seconds -> ms
    assert coerce_ts_ms(1700000000000) == 1700000000000  # already ms
    assert coerce_ts_ms(20.5) == 20500
    assert coerce_ts_ms("20.5") == 20500
    assert coerce_ts_ms(None) is None
    assert coerce_ts_ms("") is None
    assert coerce_ts_ms(-1) is None


def test_accept_new_then_old_rejects_old_ts_backwards() -> None:
    """Feed newer frame first, then older; second must be rejected (ts_backwards)."""
    gate = Bo3FreshnessGate()
    key = (1, 0, 5)  # game_number=1, map_index=0, round_number=5
    raw_new = {"game_number": 1, "round_number": 5, "updated_at": 2000}  # ts 2000 ms
    raw_old = {"game_number": 1, "round_number": 5, "updated_at": 1500}  # ts 1500 ms

    frame_new = SimpleNamespace(
        map_index=0,
        timestamp=2.0,
        round_time_remaining_s=20.0,
        alive_counts=(1, 1),
    )
    frame_old = SimpleNamespace(
        map_index=0,
        timestamp=1.5,
        round_time_remaining_s=35.0,
        alive_counts=(2, 3),
    )

    accept_new, reason_new, diag_new = gate.accept_frame(frame_new, raw_new)
    assert accept_new is True, "new frame should be accepted"
    assert reason_new is None
    assert diag_new.get("reason") is None

    accept_old, reason_old, diag_old = gate.accept_frame(frame_old, raw_old)
    assert accept_old is False, "older frame should be rejected"
    assert reason_old == "ts_backwards"
    assert diag_old.get("reason") == "ts_backwards"
    assert diag_old.get("ts_ms") == 1500000  # 1500 sec -> ms
    assert diag_old.get("last_ts_ms") == 2000000  # 2000 sec -> ms

    # Cache should still hold newer values
    entry = gate._cache.get(key)
    assert entry is not None
    assert entry["last_ts_ms"] == 2000000
    assert entry["last_round_time_remaining_s"] == 20.0


def test_accept_new_then_old_rejects_old_clock_rewind() -> None:
    """Same (game,map,round): accept ts 3000 with clock 15s; then ts 3500 with clock 40s is clock rewind."""
    gate = Bo3FreshnessGate()
    raw_new = {"game_number": 1, "round_number": 3, "updated_at": 3000}
    raw_late_rewind = {"game_number": 1, "round_number": 3, "updated_at": 3500}  # later ts but higher clock = rewind

    frame_new = SimpleNamespace(
        map_index=0,
        timestamp=3.0,
        round_time_remaining_s=15.0,
        alive_counts=(2, 2),
        scores=(8, 7),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )
    frame_late_rewind = SimpleNamespace(
        map_index=0,
        timestamp=3.5,
        round_time_remaining_s=40.0,
        alive_counts=(2, 2),
        scores=(8, 7),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )

    accept_new, _, _ = gate.accept_frame(frame_new, raw_new)
    assert accept_new is True

    accept_late, reason_late, diag_late = gate.accept_frame(frame_late_rewind, raw_late_rewind)
    assert accept_late is False
    assert reason_late == "clock_rewind"
    assert diag_late.get("reason") == "clock_rewind"
    assert diag_late.get("round_time_remaining_s") == 40.0
    assert diag_late.get("last_round_time_remaining_s") == 15.0
    assert diag_late.get("clock_rewind_meaningful_advancement") == []


def test_clock_rewind_with_alive_count_drop_is_accepted() -> None:
    gate = Bo3FreshnessGate()
    raw_first = {"game_number": 1, "round_number": 7, "updated_at": 3000}
    raw_second = {"game_number": 1, "round_number": 7, "updated_at": 3500}

    frame_first = SimpleNamespace(
        map_index=0,
        timestamp=3.0,
        round_time_remaining_s=15.0,
        alive_counts=(5, 5),
        scores=(6, 6),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )
    frame_second = SimpleNamespace(
        map_index=0,
        timestamp=3.5,
        round_time_remaining_s=40.0,
        alive_counts=(4, 5),
        scores=(6, 6),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )

    accept_first, _, _ = gate.accept_frame(frame_first, raw_first)
    assert accept_first is True

    accept_second, reason_second, diag_second = gate.accept_frame(frame_second, raw_second)
    assert accept_second is True
    assert reason_second is None
    assert diag_second.get("reason") is None
    assert diag_second.get("clock_rewind_meaningful_advancement") == ["alive_count_drop"]


def test_clock_rewind_with_score_progression_is_accepted() -> None:
    gate = Bo3FreshnessGate()
    raw_first = {"game_number": 1, "round_number": 16, "updated_at": 10000}
    raw_second = {"game_number": 1, "round_number": 16, "updated_at": 10500}

    frame_first = SimpleNamespace(
        map_index=0,
        timestamp=10.0,
        round_time_remaining_s=5.0,
        alive_counts=(1, 1),
        scores=(8, 7),
        bomb_phase_time_remaining={"round_phase": "round_end", "is_bomb_planted": False},
    )
    frame_second = SimpleNamespace(
        map_index=0,
        timestamp=10.5,
        round_time_remaining_s=20.0,
        alive_counts=(5, 5),
        scores=(9, 7),
        bomb_phase_time_remaining={"round_phase": "freeze", "is_bomb_planted": False},
    )

    accept_first, _, _ = gate.accept_frame(frame_first, raw_first)
    assert accept_first is True

    accept_second, reason_second, diag_second = gate.accept_frame(frame_second, raw_second)
    assert accept_second is True
    assert reason_second is None
    assert diag_second.get("clock_rewind_meaningful_advancement") == ["score_progression"]


def test_accept_old_then_new_accepts_both() -> None:
    """Feed old first then new; both accepted (monotonic order)."""
    gate = Bo3FreshnessGate()
    raw_old = {"game_number": 1, "round_number": 1, "updated_at": 1000}
    raw_new = {"game_number": 1, "round_number": 1, "updated_at": 2000}

    frame_old = SimpleNamespace(
        map_index=0,
        timestamp=1.0,
        round_time_remaining_s=80.0,
        alive_counts=(5, 5),
        scores=(0, 0),
        bomb_phase_time_remaining={"round_phase": "freeze", "is_bomb_planted": False},
    )
    frame_new = SimpleNamespace(
        map_index=0,
        timestamp=2.0,
        round_time_remaining_s=20.0,
        alive_counts=(2, 3),
        scores=(0, 0),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )

    accept_old, _, _ = gate.accept_frame(frame_old, raw_old)
    assert accept_old is True
    accept_new, _, _ = gate.accept_frame(frame_new, raw_new)
    assert accept_new is True

    key = (1, 0, 1)
    entry = gate._cache[key]
    assert entry["last_ts_ms"] == 2000000  # 2000 sec -> ms
    assert entry["last_round_time_remaining_s"] == 20.0
    assert entry["alive_sig_ts"].get("2v3") == 2000000


def test_alive_sig_rewind_rejected() -> None:
    """If we already saw 2v3 at ts 2000, then see 2v3 again at ts 1500 -> alive_sig_rewind."""
    gate = Bo3FreshnessGate()
    raw_first = {"game_number": 1, "round_number": 2, "updated_at": 2000}
    raw_repeat_older = {"game_number": 1, "round_number": 2, "updated_at": 1500}

    frame_first = SimpleNamespace(
        map_index=0,
        timestamp=2.0,
        round_time_remaining_s=30.0,
        alive_counts=(2, 3),
        scores=(0, 0),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )
    frame_repeat_older = SimpleNamespace(
        map_index=0,
        timestamp=1.5,
        round_time_remaining_s=60.0,
        alive_counts=(2, 3),
        scores=(0, 0),
        bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
    )

    accept_first, _, _ = gate.accept_frame(frame_first, raw_first)
    assert accept_first is True

    accept_repeat, reason, diag = gate.accept_frame(frame_repeat_older, raw_repeat_older)
    assert accept_repeat is False
    assert reason in ("ts_backwards", "alive_sig_rewind")
    assert diag.get("reason") in ("ts_backwards", "alive_sig_rewind")


def test_clear_cache_resets_state() -> None:
    gate = Bo3FreshnessGate()
    raw = {"game_number": 1, "round_number": 1, "updated_at": 1000}
    frame = SimpleNamespace(
        map_index=0,
        timestamp=1.0,
        round_time_remaining_s=90.0,
        alive_counts=(5, 5),
        scores=(0, 0),
        bomb_phase_time_remaining={"round_phase": "freeze", "is_bomb_planted": False},
    )
    gate.accept_frame(frame, raw)
    assert len(gate._cache) == 1
    gate.clear_cache()
    assert len(gate._cache) == 0
    accept, _, _ = gate.accept_frame(frame, raw)
    assert accept is True


def test_different_keys_independent() -> None:
    """Different (game, map, round) keys do not interfere."""
    gate = Bo3FreshnessGate()
    gate.accept_frame(
        SimpleNamespace(
            map_index=0,
            timestamp=1.0,
            round_time_remaining_s=80.0,
            alive_counts=(5, 5),
            scores=(0, 0),
            bomb_phase_time_remaining={"round_phase": "freeze", "is_bomb_planted": False},
        ),
        {"game_number": 1, "round_number": 1, "updated_at": 1000},
    )
    gate.accept_frame(
        SimpleNamespace(
            map_index=0,
            timestamp=2.0,
            round_time_remaining_s=70.0,
            alive_counts=(4, 5),
            scores=(0, 0),
            bomb_phase_time_remaining={"round_phase": "live", "is_bomb_planted": False},
        ),
        {"game_number": 1, "round_number": 2, "updated_at": 2000},
    )
    accept, _, _ = gate.accept_frame(
        SimpleNamespace(
            map_index=1,
            timestamp=3.0,
            round_time_remaining_s=90.0,
            alive_counts=(5, 5),
            scores=(0, 0),
            bomb_phase_time_remaining={"round_phase": "freeze", "is_bomb_planted": False},
        ),
        {"game_number": 2, "round_number": 1, "updated_at": 3000},
    )
    assert accept is True
    assert len(gate._cache) == 3

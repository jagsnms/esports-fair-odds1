"""
Unit tests for source-agnostic monotonic key and gate (no FastAPI dependency).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.telemetry.monotonic import (
    MonotonicKey,
    compute_monotonic_key_from_bo3_snapshot,
    should_accept,
)


# --- MonotonicKey ordering ---


def test_key_ordering_increasing_game_round_seq_accepts() -> None:
    """Increasing (game, round, seq) is ordered correctly; later key > earlier."""
    k1 = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    k2 = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=101.0)
    assert k1 < k2
    assert k2 > k1
    accept, reason = should_accept(k1, k2)
    assert accept is True
    assert reason is None


def test_key_ordering_round_advance_accepts() -> None:
    """Round advance: higher round_number is after lower."""
    k1 = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    k2 = MonotonicKey(game_number=1, round_number=2, seq_index=0, ts=200.0)
    assert k1 < k2
    accept, reason = should_accept(k1, k2)
    assert accept is True
    assert reason is None


def test_key_ordering_decreasing_rejects() -> None:
    """Decreasing key (regression) is rejected."""
    last = MonotonicKey(game_number=1, round_number=2, seq_index=0, ts=200.0)
    new = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    assert new < last
    accept, reason = should_accept(last, new)
    assert accept is False
    assert reason == "regression"


def test_key_ordering_same_key_rejects() -> None:
    """Same key is not strictly after; reject as regression."""
    k = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    accept, reason = should_accept(k, k)
    assert accept is False
    assert reason == "regression"


def test_missing_key_rejects_when_required() -> None:
    """None new_key with reject_missing_key=True yields reject."""
    accept, reason = should_accept(None, None, reject_missing_key=True)
    assert accept is False
    assert reason == "missing_key"

    accept, reason = should_accept(
        MonotonicKey(1, 1, 0, 100.0), None, reject_missing_key=True
    )
    assert accept is False
    assert reason == "missing_key"


def test_missing_key_accepts_when_not_required() -> None:
    """None new_key with reject_missing_key=False yields accept (backwards-compat)."""
    accept, reason = should_accept(None, None, reject_missing_key=False)
    assert accept is True
    assert reason is None


def test_first_key_accepts_when_last_is_none() -> None:
    """When last_accepted_key is None, any non-None new key is accepted."""
    new = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    accept, reason = should_accept(None, new)
    assert accept is True
    assert reason is None


def test_key_to_display() -> None:
    """to_display returns a string for diagnostics."""
    k = MonotonicKey(game_number=1, round_number=2, seq_index=0, ts=123.45)
    s = k.to_display()
    assert "1" in s and "2" in s and "123" in s or "123.45" in s


# --- compute_monotonic_key_from_bo3_snapshot ---


def test_compute_key_from_bo3_raw_and_frame() -> None:
    """Key is built from raw snapshot (game_number, round_number) and frame (map_index, timestamp)."""
    raw = {"game_number": 2, "round_number": 5, "updated_at": 1700000000}
    frame = SimpleNamespace(map_index=1, timestamp=1700000000.5)
    key = compute_monotonic_key_from_bo3_snapshot(frame, raw)
    assert key is not None
    assert key.game_number == 2
    assert key.round_number == 5
    assert key.seq_index == 1
    assert key.ts == 1700000000.5 or key.ts == 1700000000.0


def test_compute_key_from_bo3_raw_only() -> None:
    """Key uses raw snapshot when frame is None; map_index/seq become 0."""
    raw = {"game_number": 1, "round_number": 3, "updated_at": 1000}
    key = compute_monotonic_key_from_bo3_snapshot(None, raw)
    assert key is not None
    assert key.game_number == 1
    assert key.round_number == 3
    assert key.seq_index == 0
    assert key.ts == 1000.0 or key.ts == 1.0  # sec or ms interpretation


def test_compute_key_empty_raw_uses_defaults() -> None:
    """Empty raw yields game_number=1, round_number=0, seq_index from frame."""
    frame = SimpleNamespace(map_index=2, timestamp=50.0)
    key = compute_monotonic_key_from_bo3_snapshot(frame, {})
    assert key is not None
    assert key.game_number == 1
    assert key.round_number == 0
    assert key.seq_index == 2
    assert key.ts == 50.0

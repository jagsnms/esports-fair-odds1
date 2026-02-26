"""
Unit tests for breach detection: compute_breach_flags and breach event buffer.
"""
from __future__ import annotations

import asyncio

from backend.services.runner import compute_breach_flags
from backend.store.memory_store import MemoryStore


# --- compute_breach_flags ---


def test_no_breach_when_market_inside_corridors() -> None:
    """market_mid inside [map_low, map_high] and [series_low, series_high] -> no breach."""
    b_map_hi, b_map_lo, b_series_hi, b_series_lo, mag, btype = compute_breach_flags(
        0.5, 0.2, 0.8, 0.4, 0.6
    )
    assert b_map_hi is False
    assert b_map_lo is False
    assert b_series_hi is False
    assert b_series_lo is False
    assert mag is None
    assert btype is None


def test_breach_map_hi() -> None:
    """market_mid > map_high -> breach_map_hi True, breach_type map_hi, breach_mag positive."""
    b_map_hi, b_map_lo, b_series_hi, b_series_lo, mag, btype = compute_breach_flags(
        0.9, 0.0, 1.0, 0.3, 0.7
    )
    assert b_map_hi is True
    assert b_map_lo is False
    assert btype == "map_hi"
    assert mag is not None
    assert abs(mag - 0.2) < 1e-9


def test_breach_map_lo() -> None:
    """market_mid < map_low -> breach_map_lo True, breach_type map_lo."""
    b_map_hi, b_map_lo, b_series_hi, b_series_lo, mag, btype = compute_breach_flags(
        0.1, 0.0, 1.0, 0.3, 0.7
    )
    assert b_map_lo is True
    assert b_map_hi is False
    assert btype == "map_lo"
    assert mag is not None
    assert abs(mag - 0.2) < 1e-9


def test_breach_series_hi_and_map_hi_picks_max_mag() -> None:
    """When both series_hi and map_hi breach, primary is the one with larger magnitude."""
    # market_mid=0.95, series_high=0.8 -> series_hi mag=0.15; map_high=0.7 -> map_hi mag=0.25
    b_map_hi, b_map_lo, b_series_hi, b_series_lo, mag, btype = compute_breach_flags(
        0.95, 0.0, 0.8, 0.0, 0.7
    )
    assert b_series_hi is True
    assert b_map_hi is True
    assert btype == "map_hi"
    assert mag is not None
    assert abs(mag - 0.25) < 1e-9


def test_market_mid_none_no_breach() -> None:
    """market_mid None -> all flags False, mag and breach_type None."""
    b_map_hi, b_map_lo, b_series_hi, b_series_lo, mag, btype = compute_breach_flags(
        None, 0.2, 0.8, 0.4, 0.6
    )
    assert b_map_hi is False
    assert b_map_lo is False
    assert b_series_hi is False
    assert b_series_lo is False
    assert mag is None
    assert btype is None


# --- breach event buffer (MemoryStore) ---


def test_breach_buffer_append_and_get() -> None:
    """append_breach_event then get_breach_events returns the event."""
    async def run() -> None:
        store = MemoryStore(max_history=10, max_breach_events=50)
        evt = {
            "ts_epoch": 1000.0,
            "match_id": 123,
            "seg": 0,
            "scores": [5, 3],
            "series_score": [1, 0],
            "map_index": 0,
            "market_mid": 0.72,
            "p_hat": 0.55,
            "series_low": 0.2,
            "series_high": 0.8,
            "map_low": 0.4,
            "map_high": 0.6,
            "breach_type": "map_hi",
            "breach_mag": 0.12,
        }
        await store.append_breach_event(evt)
        events = await store.get_breach_events(limit=10)
        assert len(events) == 1
        assert events[0]["breach_type"] == "map_hi"
        assert events[0]["breach_mag"] == 0.12
        assert events[0]["ts_epoch"] == 1000.0
    asyncio.run(run())


def test_breach_buffer_maxlen() -> None:
    """Breach buffer respects maxlen (newest kept)."""
    async def run() -> None:
        store = MemoryStore(max_history=10, max_breach_events=3)
        for i in range(5):
            await store.append_breach_event({
                "ts_epoch": float(1000 + i),
                "match_id": 1,
                "seg": 0,
                "scores": [0, 0],
                "series_score": [0, 0],
                "map_index": 0,
                "market_mid": 0.5,
                "p_hat": 0.5,
                "series_low": 0.0,
                "series_high": 1.0,
                "map_low": 0.0,
                "map_high": 1.0,
                "breach_type": "map_hi",
                "breach_mag": float(i),
            })
        events = await store.get_breach_events(limit=10)
        assert len(events) == 3
        # Oldest in buffer should be ts_epoch 1002, 1003, 1004
        assert events[0]["ts_epoch"] == 1002.0
        assert events[-1]["ts_epoch"] == 1004.0
    asyncio.run(run())


def test_breach_buffer_limit_param() -> None:
    """get_breach_events(limit=N) returns at most N events."""
    async def run() -> None:
        store = MemoryStore(max_history=10, max_breach_events=20)
        for i in range(5):
            await store.append_breach_event({
                "ts_epoch": float(1000 + i),
                "match_id": 1,
                "seg": 0,
                "scores": [0, 0],
                "series_score": [0, 0],
                "map_index": 0,
                "market_mid": 0.5,
                "p_hat": 0.5,
                "series_low": 0.0,
                "series_high": 1.0,
                "map_low": 0.0,
                "map_high": 1.0,
                "breach_type": "map_hi",
                "breach_mag": 0.1,
            })
        events = await store.get_breach_events(limit=2)
        assert len(events) == 2
    asyncio.run(run())

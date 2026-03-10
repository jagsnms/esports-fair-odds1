"""
Unit tests for MarketDelayBuffer: get_delayed returns snapshot closest to (now - delay_sec).
"""
from __future__ import annotations

from unittest.mock import patch

from backend.services.market_buffer import MarketDelayBuffer


def test_empty_buffer_returns_none() -> None:
    """get_delayed on empty buffer returns None."""
    buf = MarketDelayBuffer(maxlen=10)
    assert buf.get_delayed(0) is None
    assert buf.get_delayed(120) is None


def test_single_entry_returned() -> None:
    """Single push then get_delayed returns that entry (any delay)."""
    buf = MarketDelayBuffer(maxlen=10)
    buf.push({"ts_epoch": 1000.0, "bid": 0.4, "ask": 0.5, "mid": 0.45, "ticker": "T"})
    with patch("backend.services.market_buffer.time") as mtime:
        mtime.time.return_value = 1000.0
        snap = buf.get_delayed(0)
    assert snap is not None
    assert snap["ts_epoch"] == 1000.0
    assert snap["mid"] == 0.45


def test_closest_at_or_before_target() -> None:
    """get_delayed(delay_sec) returns entry with ts_epoch closest to (now - delay_sec), preferring at or before."""
    buf = MarketDelayBuffer(maxlen=10)
    buf.push({"ts_epoch": 100.0, "bid": 0.3, "ask": 0.4, "mid": 0.35, "ticker": "T"})
    buf.push({"ts_epoch": 200.0, "bid": 0.35, "ask": 0.45, "mid": 0.40, "ticker": "T"})
    buf.push({"ts_epoch": 300.0, "bid": 0.4, "ask": 0.5, "mid": 0.45, "ticker": "T"})
    # now=350, delay_sec=100 -> target_ts=250. Best at or before 250 is 200.
    with patch("backend.services.market_buffer.time") as mtime:
        mtime.time.return_value = 350.0
        snap = buf.get_delayed(100)
    assert snap is not None
    assert snap["ts_epoch"] == 200.0
    assert snap["mid"] == 0.40


def test_fallback_closest_by_abs_when_none_before_target() -> None:
    """When no entry is at or before target_ts, return closest by absolute difference."""
    buf = MarketDelayBuffer(maxlen=10)
    buf.push({"ts_epoch": 500.0, "bid": 0.5, "ask": 0.6, "mid": 0.55, "ticker": "T"})
    # now=100, delay_sec=0 -> target_ts=100. No entry <= 100; closest is 500.
    with patch("backend.services.market_buffer.time") as mtime:
        mtime.time.return_value = 100.0
        snap = buf.get_delayed(0)
    assert snap is not None
    assert snap["ts_epoch"] == 500.0

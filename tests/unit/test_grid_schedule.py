"""
Unit tests for GRID per-series fetch schedule (interval + 429 backoff). No FastAPI.
"""
from __future__ import annotations

from engine.ingest.grid_constants import (
    GRID_MIN_FETCH_INTERVAL_S,
    GRID_RATE_LIMIT_BACKOFF_S,
)
from engine.ingest.grid_schedule import (
    after_rate_limit,
    after_success,
    is_rate_limit_response,
    next_fetch_allowed,
    next_fetch_in_s,
)


def test_fetch_allowed_at_t0() -> None:
    """Fetch is allowed when no previous schedule (t=0)."""
    next_ts: dict[str, float] = {}
    assert next_fetch_allowed("s1", next_ts, 0.0) is True
    assert next_fetch_in_s("s1", next_ts, 0.0) == 0.0


def test_blocked_until_interval() -> None:
    """After success, fetch is blocked until interval has passed."""
    next_ts: dict[str, float] = {}
    now = 100.0
    after_success("s1", next_ts, now, interval_s=GRID_MIN_FETCH_INTERVAL_S)
    assert next_fetch_allowed("s1", next_ts, now) is False
    assert next_fetch_allowed("s1", next_ts, now + 11.9) is False
    assert next_fetch_allowed("s1", next_ts, now + 12.0) is True
    assert next_fetch_in_s("s1", next_ts, now) == 12.0
    assert next_fetch_in_s("s1", next_ts, now + 6.0) == 6.0
    assert next_fetch_in_s("s1", next_ts, now + 12.0) == 0.0


def test_429_triggers_backoff() -> None:
    """429 response triggers backoff duration."""
    next_ts: dict[str, float] = {}
    now = 200.0
    after_rate_limit("s1", next_ts, now, backoff_s=GRID_RATE_LIMIT_BACKOFF_S)
    assert next_fetch_allowed("s1", next_ts, now) is False
    assert next_fetch_allowed("s1", next_ts, now + 59.9) is False
    assert next_fetch_allowed("s1", next_ts, now + 60.0) is True
    assert next_fetch_in_s("s1", next_ts, now) == 60.0


def test_default_interval_and_backoff() -> None:
    """Defaults are 12s interval and 60s backoff."""
    assert GRID_MIN_FETCH_INTERVAL_S == 12.0
    assert GRID_RATE_LIMIT_BACKOFF_S == 60.0


def test_is_rate_limit_response() -> None:
    """429 detected from error message."""
    assert is_rate_limit_response({"errors": [{"message": "HTTP 429"}]}) is True
    assert is_rate_limit_response({"errors": [{"message": "HTTP 429 Too Many Requests"}]}) is True
    assert is_rate_limit_response({"errors": [{"message": "HTTP 500"}]}) is False
    assert is_rate_limit_response({"data": {}}) is False
    assert is_rate_limit_response({}) is False

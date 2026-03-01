"""
GRID per-series fetch schedule: min interval and 429 backoff. Pure helpers for tests and runner.
No FastAPI deps.
"""
from __future__ import annotations

from typing import MutableMapping

from engine.ingest.grid_constants import (
    GRID_MIN_FETCH_INTERVAL_S,
    GRID_RATE_LIMIT_BACKOFF_S,
)


def next_fetch_allowed(series_id: str, next_fetch_ts: MutableMapping[str, float], now: float) -> bool:
    """True if a fetch is allowed for this series (now >= next_fetch_ts or not yet scheduled)."""
    next_ts = next_fetch_ts.get(series_id, 0.0)
    return now >= next_ts


def next_fetch_in_s(series_id: str, next_fetch_ts: MutableMapping[str, float], now: float) -> float:
    """Seconds until next fetch allowed; 0 if ready now."""
    next_ts = next_fetch_ts.get(series_id, 0.0)
    return max(0.0, next_ts - now)


def after_success(
    series_id: str,
    next_fetch_ts: MutableMapping[str, float],
    now: float,
    interval_s: float = GRID_MIN_FETCH_INTERVAL_S,
) -> None:
    """Set next allowed fetch time after a successful fetch."""
    next_fetch_ts[series_id] = now + interval_s


def after_rate_limit(
    series_id: str,
    next_fetch_ts: MutableMapping[str, float],
    now: float,
    backoff_s: float = GRID_RATE_LIMIT_BACKOFF_S,
) -> None:
    """Set next allowed fetch time after a 429 rate limit."""
    next_fetch_ts[series_id] = now + backoff_s


def is_rate_limit_response(payload: dict) -> bool:
    """True if payload indicates HTTP 429 (from legacy client error message)."""
    for e in payload.get("errors") or []:
        msg = str((e or {}).get("message", ""))
        if "429" in msg:
            return True
    return False

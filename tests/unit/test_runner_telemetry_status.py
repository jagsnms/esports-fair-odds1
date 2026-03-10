"""
Unit tests for telemetry status derivation (derive_telemetry_status).
Tri-state: NO_DATA, TELEMETRY_LOST, FEED_ALIVE. Thresholds centralized in runner.
"""
from __future__ import annotations

import pytest

from backend.services.runner import (
    TELEMETRY_FEED_DEAD_S,
    TELEMETRY_STALLED_S,
    derive_telemetry_status,
)


def test_no_data_session_created_but_no_frame_ever() -> None:
    """Session created but no frame ever -> NO_DATA."""
    now = 1000.0
    out = derive_telemetry_status(
        now,
        last_update_ts=None,
        last_fetch_ts=None,
        last_good_ts=None,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["telemetry_status"] == "NO_DATA"
    assert out["age_s"] is None
    assert "no_update_yet" in (out["telemetry_reason"] or "")


def test_no_data_with_last_error() -> None:
    """NO_DATA with last_error still NO_DATA; reason includes error."""
    out = derive_telemetry_status(
        1000.0,
        last_update_ts=None,
        last_fetch_ts=None,
        last_good_ts=None,
        telemetry_ok=False,
        telemetry_reason=None,
        last_error="drv_invalid",
    )
    assert out["telemetry_status"] == "NO_DATA"
    assert "drv_invalid" in (out["telemetry_reason"] or "")


def test_telemetry_lost_last_good_stalled_beyond_threshold() -> None:
    """last_good exists but now stalled beyond TELEMETRY_STALLED_S -> TELEMETRY_LOST."""
    now = 1000.0
    last_good_ts = now - TELEMETRY_STALLED_S - 10  # 70s ago
    last_update_ts = now - 5  # recent update
    last_fetch_ts = now - 5
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=last_fetch_ts,
        last_good_ts=last_good_ts,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["telemetry_status"] == "TELEMETRY_LOST"
    assert out["good_age_s"] is not None
    assert out["good_age_s"] >= TELEMETRY_STALLED_S


def test_telemetry_lost_feed_stalled() -> None:
    """No fetch for FEED_DEAD_S -> TELEMETRY_LOST."""
    now = 1000.0
    last_update_ts = now - TELEMETRY_FEED_DEAD_S - 5
    last_fetch_ts = now - TELEMETRY_FEED_DEAD_S - 5
    last_good_ts = now - 10
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=last_fetch_ts,
        last_good_ts=last_good_ts,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["telemetry_status"] == "TELEMETRY_LOST"
    assert "feed_stalled" in (out["telemetry_reason"] or "")


def test_telemetry_lost_telemetry_not_ok() -> None:
    """telemetry_ok False -> TELEMETRY_LOST (feed alive but telem invalid)."""
    now = 1000.0
    last_update_ts = now - 5
    last_fetch_ts = now - 5
    last_good_ts = None  # never had good telem
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=last_fetch_ts,
        last_good_ts=last_good_ts,
        telemetry_ok=False,
        telemetry_reason="missing_microstate",
    )
    assert out["telemetry_status"] == "TELEMETRY_LOST"
    assert "missing_microstate" in (out["telemetry_reason"] or "")


def test_feed_alive_normal_updates() -> None:
    """Normal updates, recent good telemetry -> FEED_ALIVE."""
    now = 1000.0
    last_update_ts = now - 5
    last_fetch_ts = now - 5
    last_good_ts = now - 10
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=last_fetch_ts,
        last_good_ts=last_good_ts,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["telemetry_status"] == "FEED_ALIVE"
    assert out["age_s"] is not None
    assert out["good_age_s"] is not None
    assert out["good_age_s"] < TELEMETRY_STALLED_S


def test_feed_alive_no_last_fetch_ts_uses_last_update_ts() -> None:
    """When last_fetch_ts is None, fetch_age uses last_update_ts (e.g. GRID)."""
    now = 1000.0
    last_update_ts = now - 5
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=None,
        last_good_ts=now - 5,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["telemetry_status"] == "FEED_ALIVE"
    assert out["fetch_age_s"] == out["age_s"]


def test_derive_returns_ages() -> None:
    """Helper returns age_s, fetch_age_s, good_age_s as expected."""
    now = 1000.0
    last_update_ts = now - 12.3
    last_fetch_ts = now - 12.0
    last_good_ts = now - 8.0
    out = derive_telemetry_status(
        now,
        last_update_ts=last_update_ts,
        last_fetch_ts=last_fetch_ts,
        last_good_ts=last_good_ts,
        telemetry_ok=True,
        telemetry_reason=None,
    )
    assert out["age_s"] == 12.3
    assert out["fetch_age_s"] == 12.0
    assert out["good_age_s"] == 8.0

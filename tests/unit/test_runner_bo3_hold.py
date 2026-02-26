"""
Unit tests for BO3 runner: non-live status (PAUSED/STALE/EMPTY/INVALID_CLOCK) still appends and broadcasts a hold point.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from engine.models import Config, Derived, State

from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


async def test_bo3_non_live_status_appends_hold_point() -> None:
    """
    When BO3 snapshot status is not live (e.g. STALE), runner appends a HistoryPoint
    with last known derived values and broadcasts it (chart stays continuous).
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=123, poll_interval_s=5.0)

    # Seed store with current state/derived so "last known values" exist
    state = State(config=config, segment_id=1)
    derived = Derived(
        p_hat=0.55,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.4,
        rail_high=0.6,
        kappa=0.0,
        debug={},
    )
    await store.set_current(state, derived)

    broadcasts: list[dict] = []
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(side_effect=lambda msg: broadcasts.append(msg))

    runner = Runner(store=store, broadcaster=broadcaster)

    # Minimal snapshot so get_snapshot returns and bo3_snapshot_to_frame succeeds
    minimal_snap = {
        "team_one": {"name": "A", "score": 0},
        "team_two": {"name": "B", "score": 0},
        "created_at": "ts1",
    }

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=minimal_snap)),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("stale", "snapshot unchanged for 4 polls", "ts1", False),
        ),
    ):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    history = await store.get_history(limit=10)
    assert len(history) == 1, "non-live status should still append one hold point"
    point = history[0]
    assert point["p"] == 0.55
    assert point["lo"] == 0.2
    assert point["hi"] == 0.8
    assert point.get("seg") == 1

    point_broadcasts = [b for b in broadcasts if b.get("type") == "point"]
    assert len(point_broadcasts) == 1, "non-live status should broadcast one point"
    assert point_broadcasts[0]["point"]["p"] == 0.55


def test_bo3_non_live_status_appends_hold_point_sync() -> None:
    """Entry point for pytest: run async test."""
    asyncio.run(test_bo3_non_live_status_appends_hold_point())


async def test_invalid_clock_does_not_zero_derived() -> None:
    """
    When snapshot has invalid clock (round_time_remaining out of range), status is invalid_clock.
    Runner appends hold point with last derived values (no zeros); bo3_health is GOOD (or PAUSED)
    with bo3_health_reason=invalid_clock and time_term_used=False.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=123, poll_interval_s=5.0)
    state = State(config=config, segment_id=1)
    derived = Derived(
        p_hat=0.62,
        bound_low=0.15,
        bound_high=0.85,
        rail_low=0.35,
        rail_high=0.75,
        kappa=0.0,
        debug={},
    )
    await store.set_current(state, derived)

    broadcasts: list[dict] = []
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(side_effect=lambda msg: broadcasts.append(msg))
    runner = Runner(store=store, broadcaster=broadcaster)

    minimal_snap = {
        "team_one": {"name": "A", "score": 0},
        "team_two": {"name": "B", "score": 0},
        "created_at": "ts1",
        "round_time_remaining": 250.0,
    }

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=minimal_snap)),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("invalid_clock", "round_time_remaining 250.0s out of range [-5, 200]", "ts1", False),
        ),
    ):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    history = await store.get_history(limit=10)
    assert len(history) == 1
    point = history[0]
    assert point["p"] == 0.62
    assert point["lo"] == 0.15
    assert point["hi"] == 0.85
    current = await store.get_current()
    debug = (current.get("derived") or {}).get("debug") or {}
    assert debug.get("bo3_health") in ("GOOD", "PAUSED")
    assert debug.get("bo3_health_reason") == "invalid_clock"
    assert debug.get("time_term_used") is False


async def test_error_path_holds_last_derived_no_zeros() -> None:
    """
    When BO3 fetch raises (e.g. network error), runner sets current derived from last derived;
    p_hat and corridors are preserved (no zeros), only debug fields updated.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=123, poll_interval_s=5.0)
    state = State(config=config, segment_id=2)
    derived = Derived(
        p_hat=0.48,
        bound_low=0.1,
        bound_high=0.9,
        rail_low=0.25,
        rail_high=0.7,
        kappa=0.0,
        debug={"bo3_fetch_ok": True},
    )
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)

    with patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(side_effect=RuntimeError("network error"))):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    current = await store.get_current()
    d = current.get("derived") or {}
    assert d.get("p_hat") == 0.48
    assert d.get("bound_low") == 0.1
    assert d.get("bound_high") == 0.9
    assert d.get("rail_low") == 0.25
    assert d.get("rail_high") == 0.7
    debug = d.get("debug") or {}
    assert debug.get("bo3_health") == "ERROR"
    assert debug.get("bo3_feed_error") == "network error" or debug.get("bo3_buffer_last_error") == "network error"
    history = await store.get_history(limit=10)
    assert len(history) == 0


def test_invalid_clock_does_not_zero_derived_sync() -> None:
    asyncio.run(test_invalid_clock_does_not_zero_derived())


def test_error_path_holds_last_derived_no_zeros_sync() -> None:
    asyncio.run(test_error_path_holds_last_derived_no_zeros())


async def test_buffer_holds_on_fetch_failure_after_first_success() -> None:
    """
    When fetch fails but buffer already has a snapshot, consumer still appends a point (from buffer).
    Debug shows bo3_buffer_has_snapshot True and bo3_buffer_age_s present/increasing.
    Mock get_snapshot to succeed once then raise on all retries.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=123, poll_interval_s=5.0)
    state = State(config=config, segment_id=1)
    derived = Derived(
        p_hat=0.5,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.4,
        rail_high=0.6,
        kappa=0.0,
        debug={},
    )
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)

    minimal_snap = {
        "team_one": {"name": "A", "score": 0},
        "team_two": {"name": "B", "score": 0},
        "created_at": "ts1",
    }
    # First call succeeds, subsequent calls (retries on 2nd tick) raise
    get_snapshot_mock = AsyncMock(side_effect=[minimal_snap, RuntimeError("fail"), RuntimeError("fail"), RuntimeError("fail")])

    with (
        patch("engine.ingest.bo3_client.get_snapshot", get_snapshot_mock),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("stale", "unchanged", "ts1", False),
        ),
    ):
        did1 = await runner._tick_bo3(config)
        did2 = await runner._tick_bo3(config)

    assert did1 is True
    assert did2 is True
    history = await store.get_history(limit=10)
    assert len(history) >= 2, "consumer should append a point on both ticks (buffer used on 2nd)"
    current = await store.get_current()
    debug = (current.get("derived") or {}).get("debug") or {}
    assert debug.get("bo3_buffer_has_snapshot") is True
    age_s = debug.get("bo3_buffer_age_s")
    assert age_s is not None, "buffer_age_s should be set when buffer has snapshot"
    assert age_s >= 0, "buffer_age_s should be non-negative"


def test_buffer_holds_on_fetch_failure_after_first_success_sync() -> None:
    asyncio.run(test_buffer_holds_on_fetch_failure_after_first_success())

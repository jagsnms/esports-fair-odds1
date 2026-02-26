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

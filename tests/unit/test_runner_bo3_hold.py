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


async def test_broadcast_point_does_not_mutate_store_state() -> None:
    """
    _broadcast_point must not mutate the dict returned by store.get_current (players_a/players_b stay present).
    """
    store = MemoryStore(max_history=10)
    config = Config(source="BO3", match_id=1, poll_interval_s=5.0)
    # Seed state with a last_frame that has players_a/players_b
    from engine.models import Frame, PlayerRow

    frame = Frame(
        timestamp=0.0,
        teams=("A", "B"),
        scores=(0, 0),
        players_a=[PlayerRow(name="A1")],
        players_b=[PlayerRow(name="B1")],
    )
    state = State(config=config, last_frame=frame, segment_id=0)
    derived = Derived()
    await store.set_current(state, derived)

    broadcasts: list[dict] = []
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(side_effect=lambda msg: broadcasts.append(msg))
    runner = Runner(store=store, broadcaster=broadcaster)

    # Call _broadcast_point directly with a dummy point
    from engine.models import HistoryPoint
    point = HistoryPoint(time=0.0, p_hat=0.5, bound_low=0.1, bound_high=0.9, rail_low=0.1, rail_high=0.9, market_mid=None, segment_id=0)
    await runner._broadcast_point(point)

    # Store state should still have players_a/players_b in last_frame
    current_after = await store.get_current()
    last_frame_after = ((current_after.get("state") or {}).get("last_frame") or {})
    assert last_frame_after.get("players_a"), "players_a should still be present in store state"
    assert last_frame_after.get("players_b"), "players_b should still be present in store state"

    # Wire payloads: point messages contain only point; frame messages contain frame with players
    point_msgs = [m for m in broadcasts if m.get("type") == "point"]
    frame_msgs = [m for m in broadcasts if m.get("type") == "frame"]
    assert len(point_msgs) == 1
    assert "current" not in point_msgs[0]
    assert "frame" not in point_msgs[0]
    assert len(frame_msgs) == 1
    frame_payload = frame_msgs[0].get("frame") or {}
    assert frame_payload.get("players_a")
    assert frame_payload.get("players_b")


def test_broadcast_point_does_not_mutate_store_state_sync() -> None:
    asyncio.run(test_broadcast_point_does_not_mutate_store_state())


async def test_tick_replay_processes_one_payload_and_advances_index() -> None:
    """
    _tick_replay must process exactly one replay payload per tick through the full pipeline
    (normalize -> reduce -> bounds -> rails -> resolve -> append_point -> broadcast)
    and increment replay_index. No early return before payload processing.
    """
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/bo3_pulls.jsonl",
        match_id=None,
        poll_interval_s=5.0,
        replay_loop=True,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcasts: list[dict] = []
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(side_effect=lambda msg: broadcasts.append(msg))
    runner = Runner(store=store, broadcaster=broadcaster)

    minimal_payload = {
        "team_one": {"name": "Team A", "score": 1, "id": 1},
        "team_two": {"name": "Team B", "score": 0, "id": 2},
        "created_at": "2024-01-01T00:00:00Z",
        "round_phase": "IN_PROGRESS",
    }
    mock_entries = [{"match_id": 99, "payload": minimal_payload, "source": "BO3", "ok": True}]

    with patch(
        "engine.replay.bo3_jsonl.load_bo3_jsonl_entries",
        return_value=mock_entries,
    ):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    assert runner._replay_index == 1, "one tick must consume one payload and increment index"
    history = await store.get_history(limit=10)
    assert len(history) == 1, "replay tick must append one history point via pipeline"
    assert history[0].get("p") is not None
    point_msgs = [b for b in broadcasts if b.get("type") == "point"]
    frame_msgs = [b for b in broadcasts if b.get("type") == "frame"]
    assert len(point_msgs) == 1, "exactly one point per processed payload"
    assert len(frame_msgs) == 1, "exactly one frame per processed payload (no duplicates)"


def test_tick_replay_processes_one_payload_and_advances_index_sync() -> None:
    asyncio.run(test_tick_replay_processes_one_payload_and_advances_index())


async def test_tick_replay_end_of_replay_loop_resets_index() -> None:
    """When replay_loop is True and index >= len(payloads), runner resets index to 0 and bumps segment."""
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/bo3_pulls.jsonl",
        match_id=None,
        poll_interval_s=5.0,
        replay_loop=True,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)
    minimal_payload = {
        "team_one": {"name": "A", "score": 0, "id": 1},
        "team_two": {"name": "B", "score": 0, "id": 2},
        "created_at": "ts",
    }
    runner._replay_payloads = [minimal_payload]
    runner._replay_index = 1
    runner._replay_path = "logs/bo3_pulls.jsonl"
    runner._replay_match_id = None

    did_replay = await runner._tick_replay(config)

    assert did_replay is True
    assert runner._replay_index == 0, "end-of-replay with loop must reset index to 0"
    history = await store.get_history(limit=10)
    assert len(history) == 1, "boundary point appended at loop"


def test_tick_replay_end_of_replay_loop_resets_index_sync() -> None:
    asyncio.run(test_tick_replay_end_of_replay_loop_resets_index())


async def test_tick_replay_end_of_replay_no_loop_does_not_crash() -> None:
    """When replay_loop is False and index >= len(payloads), runner does not crash; keeps last state."""
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/bo3_pulls.jsonl",
        match_id=None,
        poll_interval_s=5.0,
        replay_loop=False,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)
    runner._replay_payloads = [{"team_one": {"id": 1}, "team_two": {"id": 2}}]
    runner._replay_index = 1
    runner._replay_path = "logs/bo3_pulls.jsonl"
    runner._replay_match_id = None

    did_replay = await runner._tick_replay(config)

    assert did_replay is True
    assert runner._replay_index == 1, "no loop: index stays at end, no reset"
    history = await store.get_history(limit=10)
    assert len(history) == 0, "no boundary point when loop is False"


def test_tick_replay_end_of_replay_no_loop_does_not_crash_sync() -> None:
    asyncio.run(test_tick_replay_end_of_replay_no_loop_does_not_crash())


async def test_tick_replay_emits_round_result_event_when_present() -> None:
    """Replay should emit round_result HistoryPoint events when payload indicates a finished round."""
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/bo3_pulls.jsonl",
        match_id=999,
        poll_interval_s=5.0,
        replay_loop=True,
        team_a_is_team_one=True,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcasts: list[dict] = []
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(side_effect=lambda msg: broadcasts.append(msg))
    runner = Runner(store=store, broadcaster=broadcaster)

    # This payload should trigger a round_result emit (FINISHED + round_number + winning_team_id).
    payload = {
        "team_one": {"name": "A", "id": 101, "score": 1},
        "team_two": {"name": "B", "id": 202, "score": 0},
        "round_phase": "FINISHED",
        "round_number": 1,
        "winning_team_id": 101,
        "created_at": "ts",
    }
    runner._replay_payloads = [payload]
    runner._replay_index = 0
    runner._replay_path = "logs/bo3_pulls.jsonl"
    runner._replay_match_id = 999

    did_replay = await runner._tick_replay(config)

    assert did_replay is True
    hist = await store.get_history(limit=10)
    # We should have at least one event point (and the main point) in history now.
    assert any((p.get("event") or {}).get("event_type") == "round_result" for p in hist), "round_result event should be emitted in replay"


def test_tick_replay_emits_round_result_event_when_present_sync() -> None:
    asyncio.run(test_tick_replay_emits_round_result_event_when_present())


async def test_round_result_fallback_map_index_from_last_seen_game_number() -> None:
    """
    When payload has round_phase=FINISHED, round_number, winning_team_id but NO game_number,
    round_result event uses _bo3_last_seen_game_number so map_index/game_number are set for calibration join.
    """
    from engine.models import State

    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    runner._bo3_last_seen_game_number = 2

    raw = {
        "team_one": {"id": 101, "score": 2},
        "team_two": {"id": 202, "score": 1},
        "round_phase": "FINISHED",
        "round_number": 5,
        "winning_team_id": 101,
    }
    new_state = State(config=config, segment_id=0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw=raw,
        config=config,
        new_state=new_state,
        t=1000.0,
        p_hat=0.5,
        bound_low=0.01,
        bound_high=0.99,
        rail_low=0.01,
        rail_high=0.99,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=999,
    )

    hist = await store.get_history(limit=10)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_points) == 1, "exactly one round_result emitted"
    point = round_points[0]
    assert point.get("map_index") == 1, "map_index = game_number - 1 from fallback (game_number=2)"
    assert point.get("round_number") == 5
    assert point.get("game_number") == 2
    ev = point.get("event") or {}
    assert ev.get("map_index") == 1
    assert ev.get("game_number") == 2


def test_round_result_fallback_map_index_from_last_seen_game_number_sync() -> None:
    asyncio.run(test_round_result_fallback_map_index_from_last_seen_game_number())

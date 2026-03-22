"""
Unit tests for BO3 runner: non-live status (PAUSED/STALE/EMPTY/INVALID_CLOCK) still appends and broadcasts a hold point.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from engine.models import Config, Derived, Frame, HistoryPoint, State

from backend.services.runner import Runner, _inter_map_break_phat_and_dbg
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
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert round_points, "round_result event should be emitted in replay"
    assert round_points[0].get("match_id") == 999


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
    assert point.get("match_id") == 999
    assert point.get("map_index") == 1, "map_index = game_number - 1 from fallback (game_number=2)"
    assert point.get("round_number") == 5
    assert point.get("game_number") == 2
    ev = point.get("event") or {}
    assert ev.get("map_index") == 1
    assert ev.get("game_number") == 2


def test_round_result_fallback_map_index_from_last_seen_game_number_sync() -> None:
    asyncio.run(test_round_result_fallback_map_index_from_last_seen_game_number())


def _audited_match_111722_game1_round12_fragment() -> list[dict]:
    """Deterministic extracted raw fragment for the audited round-12 -> round-13 case."""
    return [
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 12,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:42Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:43Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 3, "match_score": 0},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:44Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 3, "match_score": 0},
            "round_phase": "BUY_TIME",
            "round_number": 13,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:45Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 3, "match_score": 0},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:46Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 9, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 3, "match_score": 0},
            "round_phase": "BUY_TIME",
            "round_number": 13,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:22:47Z",
        },
    ]


def _audited_match_111722_game1_round12_leadin() -> list[dict]:
    """Minimal prior state from the same map that previously contaminated round 12 winner seeding."""
    return [
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 0, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 0, "match_score": 0},
            "round_phase": "BUY_TIME",
            "round_number": 1,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:01:45Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 1, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 0, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 2,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:03:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 2, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 0, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 3,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:04:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 2, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 1, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 4,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:05:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 3, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 1, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 5,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:06:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 4, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 1, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 6,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:07:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 4, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 7,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:08:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 5, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 8,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:09:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 6, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 9,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:10:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 7, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 10,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:11:00Z",
        },
        {
            "team_one": {"name": "Team Spirit", "id": 654, "score": 8, "match_score": 0},
            "team_two": {"name": "Team Liquid", "id": 790, "score": 2, "match_score": 0},
            "round_phase": "PAUSED",
            "round_number": 11,
            "game_number": 1,
            "winning_team_id": 0,
            "created_at": "2026-03-19T14:12:00Z",
        },
    ]


async def test_round_result_does_not_seed_new_round_from_stale_prior_round_score_baseline() -> None:
    """Audited round-12 case should not seed a new-round winner from prior-round score baseline."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=111722, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    for offset, raw in enumerate(_audited_match_111722_game1_round12_leadin()):
        await runner._maybe_emit_outcome_events_from_bo3_payload(
            raw=raw,
            config=config,
            new_state=new_state,
            t=1500.0 + offset,
            p_hat=0.5,
            bound_low=0.01,
            bound_high=0.99,
            rail_low=0.01,
            rail_high=0.99,
            market_mid=None,
            dbg={},
            team_a_is_team_one=True,
            match_id_used=111722,
        )

    first_round12_tick = _audited_match_111722_game1_round12_fragment()[0]
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw=first_round12_tick,
        config=config,
        new_state=new_state,
        t=1600.0,
        p_hat=0.5,
        bound_low=0.01,
        bound_high=0.99,
        rail_low=0.01,
        rail_high=0.99,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=111722,
    )

    assert runner._bo3_pending_round_result_identity is None
    assert runner._bo3_pending_round_result_winner_team_id is None
    assert runner._bo3_last_emitted_round_number == 11
    assert runner._bo3_last_emitted_round_winner_team_id == 654

    for offset, raw in enumerate(_audited_match_111722_game1_round12_fragment()[1:], start=1):
        await runner._maybe_emit_outcome_events_from_bo3_payload(
            raw=raw,
            config=config,
            new_state=new_state,
            t=1600.0 + offset,
            p_hat=0.5,
            bound_low=0.01,
            bound_high=0.99,
            rail_low=0.01,
            rail_high=0.99,
            market_mid=None,
            dbg={},
            team_a_is_team_one=True,
            match_id_used=111722,
        )

    hist = await store.get_history(limit=40)
    round12_points = [
        p for p in hist
        if (p.get("event") or {}).get("event_type") == "round_result"
        and p.get("game_number") == 1
        and p.get("round_number") == 12
    ]
    assert len(round12_points) == 1
    event = round12_points[0].get("event") or {}
    assert event.get("round_winner_team_id") == 790
    assert event.get("round_winner_is_team_a") is False


def test_round_result_does_not_seed_new_round_from_stale_prior_round_score_baseline_sync() -> None:
    asyncio.run(test_round_result_does_not_seed_new_round_from_stale_prior_round_score_baseline())


async def test_round_result_carries_inferred_winner_until_round_advances() -> None:
    """A same-round score delta should still emit once the round later advances."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 2, "match_score": 1},
            "team_two": {"id": 202, "score": 3, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 6,
            "game_number": 3,
            "winning_team_id": 0,
        },
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 3, "match_score": 1},
            "team_two": {"id": 202, "score": 3, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 6,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1001.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 3, "match_score": 1},
            "team_two": {"id": 202, "score": 3, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 7,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1002.0,
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

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_points) == 1
    event = round_points[0].get("event") or {}
    assert round_points[0].get("round_number") == 6
    assert round_points[0].get("game_number") == 3
    assert round_points[0].get("map_index") == 2
    assert event.get("round_winner_team_id") == 101
    assert event.get("round_winner_is_team_a") is True


def test_round_result_carries_inferred_winner_until_round_advances_sync() -> None:
    asyncio.run(test_round_result_carries_inferred_winner_until_round_advances())


async def test_round_result_carryover_does_not_duplicate_on_repeated_ticks() -> None:
    """Carryover should emit exactly once even if same-round and boundary ticks repeat."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    payloads = [
        {
            "team_one": {"id": 101, "score": 4, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 10,
            "game_number": 3,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 101, "score": 5, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 10,
            "game_number": 3,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 101, "score": 5, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 10,
            "game_number": 3,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 101, "score": 5, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 11,
            "game_number": 3,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 101, "score": 5, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 11,
            "game_number": 3,
            "winning_team_id": 0,
        },
    ]

    for offset, raw in enumerate(payloads):
        await runner._maybe_emit_outcome_events_from_bo3_payload(
            raw=raw,
            config=config,
            new_state=new_state,
            t=1100.0 + offset,
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

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_points) == 1
    assert (round_points[0].get("event") or {}).get("round_winner_team_id") == 101


def test_round_result_carryover_does_not_duplicate_on_repeated_ticks_sync() -> None:
    asyncio.run(test_round_result_carryover_does_not_duplicate_on_repeated_ticks())


async def test_explicit_round_winner_overrides_carried_inference() -> None:
    """A later explicit winning_team_id should supersede the carried score-delta inference."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1200.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 6, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1201.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 6, "match_score": 1},
            "round_phase": "FINISHED",
            "round_number": 12,
            "game_number": 3,
            "winning_team_id": 202,
        },
        config=config,
        new_state=new_state,
        t=1202.0,
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

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_points) == 1
    assert (round_points[0].get("event") or {}).get("round_winner_team_id") == 202
    assert (round_points[0].get("event") or {}).get("round_winner_is_team_a") is False


def test_explicit_round_winner_overrides_carried_inference_sync() -> None:
    asyncio.run(test_explicit_round_winner_overrides_carried_inference())


async def test_explicit_round_winner_carries_to_boundary_when_boundary_tick_has_zero() -> None:
    """A nonzero explicit winner seen before boundary should still emit when the boundary tick reports zero."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 5, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1250.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 6, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 12,
            "game_number": 3,
            "winning_team_id": 202,
        },
        config=config,
        new_state=new_state,
        t=1251.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 6, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 13,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1252.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 6, "match_score": 1},
            "team_two": {"id": 202, "score": 6, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 13,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1253.0,
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

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_points) == 1
    event = round_points[0].get("event") or {}
    assert round_points[0].get("round_number") == 12
    assert event.get("round_winner_team_id") == 202
    assert event.get("round_winner_is_team_a") is False


def test_explicit_round_winner_carries_to_boundary_when_boundary_tick_has_zero_sync() -> None:
    asyncio.run(test_explicit_round_winner_carries_to_boundary_when_boundary_tick_has_zero())


async def test_round_result_carryover_resets_on_map_change() -> None:
    """Carried inference must not leak into a different game/map context."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 8, "match_score": 1},
            "team_two": {"id": 202, "score": 8, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 17,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1300.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 9, "match_score": 1},
            "team_two": {"id": 202, "score": 8, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 17,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1301.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 0, "match_score": 1},
            "team_two": {"id": 202, "score": 0, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 1,
            "game_number": 4,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1302.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 0, "match_score": 1},
            "team_two": {"id": 202, "score": 0, "match_score": 1},
            "round_phase": "BUY_TIME",
            "round_number": 1,
            "game_number": 4,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1303.0,
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

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    assert round_points == []


def test_round_result_carryover_resets_on_map_change_sync() -> None:
    asyncio.run(test_round_result_carryover_resets_on_map_change())


async def test_segment_result_still_emits_on_credible_match_score_increment() -> None:
    """Touched path must preserve existing credible match_score increment segment_result behavior."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=999, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(
        config=config,
        segment_id=0,
        last_frame=Frame(timestamp=1401.0, teams=("A", "B"), scores=(13, 8)),
    )

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 12, "match_score": 1},
            "team_two": {"id": 202, "score": 8, "match_score": 1},
            "round_phase": "IN_PROGRESS",
            "round_number": 21,
            "game_number": 3,
            "winning_team_id": 0,
        },
        config=config,
        new_state=new_state,
        t=1400.0,
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
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 101, "score": 13, "match_score": 2},
            "team_two": {"id": 202, "score": 8, "match_score": 1},
            "round_phase": "FINISHED",
            "round_number": 21,
            "game_number": 3,
            "winning_team_id": 101,
        },
        config=config,
        new_state=new_state,
        t=1401.0,
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

    hist = await store.get_history(limit=20)
    segment_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "segment_result"]
    assert len(segment_points) == 1
    assert segment_points[0].get("match_id") == 999
    event = segment_points[0].get("event") or {}
    assert event.get("map_winner_team_id") == 101
    assert event.get("final_rounds_a") == 13
    assert event.get("final_rounds_b") == 8


def test_segment_result_still_emits_on_credible_match_score_increment_sync() -> None:
    asyncio.run(test_segment_result_still_emits_on_credible_match_score_increment())


async def test_audited_live_map_final_emits_segment_result_with_top_level_match_id() -> None:
    """
    Audited live case: segment_result emits on the credible increment and now carries the
    same top-level match_id used by round_result, so match-scoped history filtering finds it.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=114254, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.01, bound_high=0.99, rail_low=0.01, rail_high=0.99, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)

    fragment = [
        {
            "team_one": {"id": 10994, "score": 12, "match_score": 0},
            "team_two": {"id": 22190, "score": 8, "match_score": 0},
            "match_fixture": {"team_one_score": 0, "team_two_score": 0},
            "round_phase": "IN_PROGRESS",
            "round_number": 21,
            "game_number": 1,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 10994, "score": 12, "match_score": 0},
            "team_two": {"id": 22190, "score": 9, "match_score": 0},
            "match_fixture": {"team_one_score": 0, "team_two_score": 0},
            "round_phase": "IN_PROGRESS",
            "round_number": 22,
            "game_number": 1,
            "winning_team_id": 0,
        },
        {
            "team_one": {"id": 10994, "score": 13, "match_score": 1},
            "team_two": {"id": 22190, "score": 9, "match_score": 0},
            "match_fixture": {"team_one_score": 0, "team_two_score": 0},
            "round_phase": "FINISHED",
            "round_number": 22,
            "game_number": 1,
            "winning_team_id": 10994,
        },
    ]
    new_state = State(
        config=config,
        segment_id=0,
        last_frame=Frame(timestamp=1502.0, teams=("Spirit", "Liquid"), scores=(13, 9)),
    )

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw=fragment[0],
        config=config,
        new_state=new_state,
        t=1500.0,
        p_hat=0.5,
        bound_low=0.01,
        bound_high=0.99,
        rail_low=0.01,
        rail_high=0.99,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=114254,
    )
    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw=fragment[1],
        config=config,
        new_state=new_state,
        t=1501.0,
        p_hat=0.5,
        bound_low=0.01,
        bound_high=0.99,
        rail_low=0.01,
        rail_high=0.99,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=114254,
    )

    assert runner._bo3_last_seen_match_score_by_game == {1: (0, 0)}
    final_ms1 = int(fragment[2]["team_one"]["match_score"])
    final_ms2 = int(fragment[2]["team_two"]["match_score"])
    prev_ms1, prev_ms2 = runner._bo3_last_seen_match_score_by_game[1]
    assert (final_ms1 - prev_ms1, final_ms2 - prev_ms2) == (1, 0)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw=fragment[2],
        config=config,
        new_state=new_state,
        t=1502.0,
        p_hat=0.5,
        bound_low=0.01,
        bound_high=0.99,
        rail_low=0.01,
        rail_high=0.99,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=114254,
    )

    hist = await store.get_history(limit=20)
    round_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "round_result"]
    segment_points = [p for p in hist if (p.get("event") or {}).get("event_type") == "segment_result"]
    segment_points_for_match = [
        p for p in segment_points if p.get("match_id") == 114254
    ]

    assert len(round_points) == 2
    assert round_points[-1]["match_id"] == 114254
    assert (round_points[-1].get("event") or {}).get("round_winner_team_id") == 10994

    assert len(segment_points) == 1
    assert segment_points[0].get("match_id") == 114254
    assert len(segment_points_for_match) == 1
    segment_event = segment_points[0].get("event") or {}
    assert segment_event.get("map_winner_team_id") == 10994
    assert segment_event.get("final_rounds_a") == 13
    assert segment_event.get("final_rounds_b") == 9


def test_audited_live_map_final_emits_segment_result_with_top_level_match_id_sync() -> None:
    asyncio.run(test_audited_live_map_final_emits_segment_result_with_top_level_match_id())


async def test_bo3_invalid_driver_missing_microstate_does_not_append_history_point() -> None:
    """
    Live BO3 tick with missing microstate (no player_states) must NOT append a history point.
    Driver validity gate: drv_valid_microstate False => skip append/broadcast (legacy-style gating).
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=456, poll_interval_s=5.0)
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

    # Snapshot WITHOUT player_states => frame has loadout_totals=None => missing_microstate_flag True
    minimal_snap_no_players = {
        "team_one": {"name": "A", "score": 0, "id": 1},
        "team_two": {"name": "B", "score": 0, "id": 2},
        "created_at": "ts1",
        "round_phase": "IN_PROGRESS",
        "game_number": 1,
    }

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=minimal_snap_no_players)),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("live", "ok", "ts1", True),
        ),
    ):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    history = await store.get_history(limit=10)
    assert len(history) == 0, "live BO3 tick with missing microstate must not append any history point"


async def test_bo3_valid_driver_with_microstate_appends_history_point() -> None:
    """
    Live BO3 tick WITH player_states (valid microstate) must append one joinable history point.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=789, poll_interval_s=5.0)
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

    # Snapshot WITH player_states => frame has microstate => missing_microstate_flag False (if clock valid)
    snap_with_players = {
        "team_one": {
            "name": "A",
            "score": 0,
            "id": 1,
            "player_states": [
                {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                {"is_alive": True, "health": 100, "balance": 3500, "equipment_value": 2500},
            ],
        },
        "team_two": {
            "name": "B",
            "score": 0,
            "id": 2,
            "player_states": [
                {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                {"is_alive": True, "health": 80, "balance": 2000, "equipment_value": 1500},
            ],
        },
        "created_at": "ts1",
        "round_phase": "IN_PROGRESS",
        "game_number": 1,
        "round_time_remaining": 115.0,
    }

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=snap_with_players)),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("live", "ok", "ts1", True),
        ),
    ):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    history = await store.get_history(limit=10)
    assert len(history) == 1, "live BO3 tick with valid microstate must append one history point"
    point = history[0]
    assert point.get("match_id") == 789
    assert point.get("game_number") == 1
    assert point.get("map_index") == 0
    explain = point.get("explain") or {}
    assert explain.get("round_phase") == "IN_PROGRESS"
    assert explain.get("q_intra_total") is not None
    assert explain.get("alive_counts") == (2, 2)
    assert explain.get("hp_totals") == (200.0, 180.0)
    assert explain.get("loadout_totals") == (5500.0, 4500.0)
    assert "target_p_hat" in explain
    assert "p_hat_prev" in explain
    assert "movement_confidence" in explain
    assert "expected_p_hat_after_movement" in explain
    assert "movement_gap_abs" in explain


async def test_bo3_live_tick_carries_forward_true_previous_p_hat() -> None:
    """Live BO3 resolve path must use the carried-forward persisted PHAT state as p_hat_prev."""
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=790, poll_interval_s=5.0)
    state = State(config=config, segment_id=1)
    derived = Derived(
        p_hat=0.73,
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

    snap_with_players = {
        "team_one": {
            "name": "A",
            "score": 0,
            "id": 1,
            "player_states": [
                {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                {"is_alive": True, "health": 100, "balance": 3500, "equipment_value": 2500},
            ],
        },
        "team_two": {
            "name": "B",
            "score": 0,
            "id": 2,
            "player_states": [
                {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                {"is_alive": True, "health": 80, "balance": 2000, "equipment_value": 1500},
            ],
        },
        "created_at": "ts1",
        "round_phase": "IN_PROGRESS",
        "game_number": 1,
        "round_time_remaining": 115.0,
    }

    with (
        patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=snap_with_players)),
        patch(
            "backend.services.runner._bo3_snapshot_status",
            return_value=("live", "ok", "ts1", True),
        ),
    ):
        did_bo3 = await runner._tick_bo3(config)

    assert did_bo3 is True
    history = await store.get_history(limit=10)
    assert len(history) == 1
    point = history[0]
    explain = history[0].get("explain") or {}
    assert explain.get("p_hat_prev_source") == "carried_forward"
    expected_prev = max(point["rail_low"], min(point["rail_high"], 0.73))
    assert explain.get("p_hat_prev") == expected_prev


def test_bo3_live_tick_carries_forward_true_previous_p_hat_sync() -> None:
    asyncio.run(test_bo3_live_tick_carries_forward_true_previous_p_hat())


def test_inter_map_break_preserves_carried_forward_p_hat_when_available() -> None:
    """Inter-map helper keeps last PHAT continuity instead of silently recentring it."""
    p_hat, dbg = _inter_map_break_phat_and_dbg(0.2, 0.8, "map_transition", 0.71, {})
    expected = max(dbg["map_low"], min(dbg["map_high"], 0.71))
    assert p_hat == expected
    assert dbg["p_hat_old"] == 0.71
    assert dbg["p_hat_final"] == expected
    assert dbg["map_low"] <= p_hat <= dbg["map_high"]


def test_bo3_invalid_driver_missing_microstate_does_not_append_history_point_sync() -> None:
    asyncio.run(test_bo3_invalid_driver_missing_microstate_does_not_append_history_point())


def test_bo3_valid_driver_with_microstate_appends_history_point_sync() -> None:
    asyncio.run(test_bo3_valid_driver_with_microstate_appends_history_point())


async def test_persisted_history_rows_allow_round_boundary_reconstruction() -> None:
    """
    Persisted artifacts alone should let us reconstruct compute-before, event row, and compute-after
    around a round boundary using stable match/map/round identity plus time order.
    """
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=222333, poll_interval_s=5.0, team_a_is_team_one=True)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.64, bound_low=0.2, bound_high=0.8, rail_low=0.35, rail_high=0.75, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock()
    runner = Runner(store=store, broadcaster=broadcaster)
    new_state = State(config=config, segment_id=0)

    pre_point = HistoryPoint(
        time=2000.0,
        p_hat=0.64,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.35,
        rail_high=0.75,
        segment_id=0,
        match_id=222333,
        map_index=0,
        game_number=1,
        round_number=12,
        explain={
            "phase": "IN_PROGRESS",
            "round_phase": "IN_PROGRESS",
            "q_intra_total": 0.82,
            "alive_counts": (4, 1),
            "hp_totals": (260.0, 72.0),
            "loadout_totals": (14800.0, 4200.0),
            "target_p_hat": 0.72,
            "p_hat_prev": 0.61,
            "movement_confidence": 0.25,
            "expected_p_hat_after_movement": 0.6375,
            "movement_gap_abs": 0.0025,
            "final": {"p_hat_final": 0.64, "clamp_reason": None},
            "score_raw": 0.16,
            "term_contribs": {
                "term_alive": 0.08,
                "term_hp": 0.05,
                "term_loadout": 0.03,
                "term_bomb": 0.0,
                "term_cash": 0.0,
            },
            "base_intercept": 0.0,
            "p_unshaped": 0.82,
        },
    )
    await store.append_point(pre_point, new_state, derived)

    await runner._maybe_emit_outcome_events_from_bo3_payload(
        raw={
            "team_one": {"id": 1001, "score": 9, "match_score": 0},
            "team_two": {"id": 1002, "score": 3, "match_score": 0},
            "round_phase": "FINISHED",
            "round_number": 12,
            "game_number": 1,
            "winning_team_id": 1001,
        },
        config=config,
        new_state=new_state,
        t=2001.0,
        p_hat=0.64,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.35,
        rail_high=0.75,
        market_mid=None,
        dbg={},
        team_a_is_team_one=True,
        match_id_used=222333,
    )

    post_point = HistoryPoint(
        time=2002.0,
        p_hat=0.57,
        bound_low=0.2,
        bound_high=0.8,
        rail_low=0.35,
        rail_high=0.75,
        segment_id=0,
        match_id=222333,
        map_index=0,
        game_number=1,
        round_number=13,
        explain={
            "phase": "BUY_TIME",
            "round_phase": "BUY_TIME",
            "q_intra_total": 0.5,
            "alive_counts": (5, 5),
            "hp_totals": (500.0, 500.0),
            "loadout_totals": (16000.0, 16000.0),
            "target_p_hat": None,
            "p_hat_prev": 0.57,
            "movement_confidence": 0.0,
            "expected_p_hat_after_movement": None,
            "movement_gap_abs": None,
            "final": {"p_hat_final": 0.57, "clamp_reason": None},
        },
    )
    await store.append_point(post_point, new_state, Derived(p_hat=0.57, bound_low=0.2, bound_high=0.8, rail_low=0.35, rail_high=0.75, kappa=0.0))

    hist = await store.get_history(limit=10)
    rows_for_match = [p for p in hist if p.get("match_id") == 222333]
    round_result_rows = [p for p in rows_for_match if (p.get("event") or {}).get("event_type") == "round_result"]
    assert len(round_result_rows) == 1
    event_row = round_result_rows[0]
    event_idx = rows_for_match.index(event_row)
    assert event_idx > 0
    assert event_idx < len(rows_for_match) - 1

    pre_row = rows_for_match[event_idx - 1]
    post_row = rows_for_match[event_idx + 1]
    assert pre_row.get("round_number") == 12
    assert (pre_row.get("explain") or {}).get("round_phase") == "IN_PROGRESS"
    assert event_row.get("game_number") == 1
    assert event_row.get("map_index") == 0
    assert event_row.get("round_number") == 12
    assert (event_row.get("event") or {}).get("round_winner_team_id") == 1001
    assert post_row.get("round_number") == 13
    assert (post_row.get("explain") or {}).get("round_phase") == "BUY_TIME"


def test_persisted_history_rows_allow_round_boundary_reconstruction_sync() -> None:
    asyncio.run(test_persisted_history_rows_allow_round_boundary_reconstruction())


def test_bo3_provider_side_strings_persist_a_side_on_live_compute_and_score_rows(tmp_path: Path) -> None:
    async def _run() -> None:
        history_path = tmp_path / "history_points.jsonl"
        score_path = tmp_path / "history_score_points.jsonl"
        with (
            patch("backend.store.memory_store._HISTORY_RECORD_ENABLED", True),
            patch("backend.store.memory_store._HISTORY_RECORD_JSONL_PATH", str(history_path)),
            patch("backend.store.memory_store._HISTORY_SCORE_RECORD_ENABLED", True),
            patch("backend.store.memory_store._HISTORY_SCORE_RECORD_JSONL_PATH", str(score_path)),
        ):
            store = MemoryStore(max_history=100)
            config = Config(source="BO3", match_id=789, poll_interval_s=5.0, team_a_is_team_one=True)
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

            snap_with_provider_side_strings = {
                "team_one": {
                    "name": "A",
                    "score": 0,
                    "id": 1,
                    "side": "TERRORIST",
                    "player_states": [
                        {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                        {"is_alive": True, "health": 100, "balance": 3500, "equipment_value": 2500},
                    ],
                },
                "team_two": {
                    "name": "B",
                    "score": 0,
                    "id": 2,
                    "side": "COUNTER_TERRORIST",
                    "player_states": [
                        {"is_alive": True, "health": 100, "balance": 4000, "equipment_value": 3000},
                        {"is_alive": True, "health": 80, "balance": 2000, "equipment_value": 1500},
                    ],
                },
                "created_at": "ts1",
                "round_phase": "IN_PROGRESS",
                "game_number": 1,
                "round_number": 1,
                "round_time_remaining": 115.0,
            }

            with (
                patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(return_value=snap_with_provider_side_strings)),
                patch(
                    "backend.services.runner._bo3_snapshot_status",
                    return_value=("live", "ok", "ts1", True),
                ),
            ):
                did_bo3 = await runner._tick_bo3(config)

            assert did_bo3 is True
            history = await store.get_history(limit=10)
            assert len(history) == 1
            assert history[0].get("game_number") == 1
            assert history[0].get("round_number") == 1
            assert history[0].get("a_side") == "T"

            score_rows = [
                json.loads(line)
                for line in score_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            assert len(score_rows) == 1
            assert score_rows[0].get("game_number") == 1
            assert score_rows[0].get("round_number") == 1
            assert score_rows[0].get("a_side") == "T"

    asyncio.run(_run())

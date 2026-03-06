"""
Stage 3B: Replay raw vs point contract — assert on real runner-produced debug.
Raw replay: debug.replay_mode == "raw_contract"; point replay: debug.replay_mode == "point_passthrough".
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.models import Config, Derived, State
from backend.store.memory_store import MemoryStore
from backend.services.runner import Runner, _is_raw_bo3_snapshot, _is_point_like_payload


def test_replay_raw_vs_point_detection_helpers() -> None:
    """Helper detection: raw vs point payloads are distinguished (support for real runner tests)."""
    raw_payload = {
        "team_one": {"name": "A", "score": 1},
        "team_two": {"name": "B", "score": 0},
        "round_phase": "IN_PROGRESS",
    }
    point_like = {"t": 1000.0, "p": 0.55, "lo": 0.2, "hi": 0.8, "rail_low": 0.4, "rail_high": 0.6}
    assert _is_raw_bo3_snapshot(raw_payload) is True
    assert _is_point_like_payload(point_like) is True
    assert _is_raw_bo3_snapshot(point_like) is False
    assert _is_point_like_payload(raw_payload) is False


@pytest.mark.asyncio
async def test_real_runner_raw_replay_tags_contract_mode() -> None:
    """
    Run one raw BO3 snapshot through _tick_replay; assert runner-produced
    derived.debug has replay_mode == "raw_contract" (real runner output).
    """
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

    minimal_raw = {
        "team_one": {"name": "Team A", "score": 1, "id": 1},
        "team_two": {"name": "Team B", "score": 0, "id": 2},
        "created_at": "2024-01-01T00:00:00Z",
        "round_phase": "IN_PROGRESS",
    }
    mock_entries = [{"match_id": 99, "payload": minimal_raw, "source": "BO3", "ok": True}]

    with patch(
        "engine.replay.bo3_jsonl.load_bo3_jsonl_entries",
        return_value=mock_entries,
    ):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    cur = await store.get_current()
    assert isinstance(cur, dict)
    derived_obj = cur.get("derived")
    assert isinstance(derived_obj, dict)
    dbg = derived_obj.get("debug")
    assert isinstance(dbg, dict)
    assert dbg.get("replay_mode") == "raw_contract", "raw replay must tag runner output as raw_contract"
    assert "explain" in dbg


def test_real_runner_raw_replay_tags_contract_mode_sync() -> None:
    asyncio.run(test_real_runner_raw_replay_tags_contract_mode())


@pytest.mark.asyncio
async def test_real_runner_point_replay_tags_passthrough_mode() -> None:
    """
    Run one point-like payload through _tick_replay; assert runner-produced
    derived.debug has replay_mode == "point_passthrough" (real runner output).
    """
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/points.jsonl",
        match_id=None,
        poll_interval_s=5.0,
        replay_loop=False,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)

    point_payload = {
        "t": 1000.0,
        "p": 0.55,
        "lo": 0.2,
        "hi": 0.8,
        "rail_low": 0.4,
        "rail_high": 0.6,
        "seg": 0,
    }
    # One entry so iter_payloads yields (1, point_payload); runner sees point-like and sets format "point"
    mock_entries = [{"match_id": 1, "payload": point_payload, "source": "BO3", "ok": True}]

    with patch(
        "engine.replay.bo3_jsonl.load_bo3_jsonl_entries",
        return_value=mock_entries,
    ):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    cur = await store.get_current()
    assert isinstance(cur, dict)
    derived_obj = cur.get("derived")
    assert isinstance(derived_obj, dict)
    dbg = derived_obj.get("debug")
    assert isinstance(dbg, dict)
    assert dbg.get("replay_mode") == "point_passthrough", "point replay must tag runner output as point_passthrough"
    assert "explain" in dbg


def test_real_runner_point_replay_tags_passthrough_mode_sync() -> None:
    asyncio.run(test_real_runner_point_replay_tags_passthrough_mode())

"""
Stage 1 replay contract gate tests:
- raw replay remains canonical
- point-like replay rejects by default
- transition passthrough only when explicit + sunset-valid
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.models import Config, Derived, State
from backend.store.memory_store import MemoryStore
from backend.services.runner import (
    REPLAY_POINT_POLICY_DECISION_REJECT,
    REPLAY_POINT_POLICY_DECISION_TRANSITION_PASSTHROUGH,
    REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY,
    REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED,
    Runner,
    _is_point_like_payload,
    _is_raw_bo3_snapshot,
)


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
    contract_status = runner.get_replay_contract_status()
    assert contract_status["point_like_inputs_seen"] == 0
    assert contract_status["point_like_inputs_rejected"] == 0


def test_real_runner_raw_replay_tags_contract_mode_sync() -> None:
    asyncio.run(test_real_runner_raw_replay_tags_contract_mode())


@pytest.mark.asyncio
async def test_point_replay_default_policy_rejects_without_append_or_broadcast() -> None:
    """
    Stage 1 default behavior: reject point-like replay payloads for canonical replay.
    No append/broadcast should occur.
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
    history = await store.get_history()
    assert history == [], "default reject policy must not append point-like replay points"
    broadcaster.broadcast.assert_not_awaited()
    contract_status = runner.get_replay_contract_status()
    assert contract_status["point_like_inputs_seen"] == 1
    assert contract_status["point_like_inputs_rejected"] == 1
    assert contract_status["point_like_inputs_transition_passthrough"] == 0
    assert contract_status["point_like_reject_reason_counts"] == {
        REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY: 1
    }
    assert contract_status["last_point_like_policy_decision"] == REPLAY_POINT_POLICY_DECISION_REJECT
    assert contract_status["last_point_like_policy_reason"] == REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY


@pytest.mark.asyncio
async def test_point_replay_transition_mode_allows_passthrough_when_sunset_valid() -> None:
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/points.jsonl",
        replay_loop=False,
        replay_point_transition_enabled=True,
        replay_point_transition_sunset_epoch=time.time() + 60.0,
    )
    await store.set_current(
        State(config=config, segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)
    point_payload = {"t": 1000.0, "p": 0.55, "lo": 0.2, "hi": 0.8, "rail_low": 0.4, "rail_high": 0.6, "seg": 0}
    mock_entries = [{"match_id": 1, "payload": point_payload, "source": "BO3", "ok": True}]

    with patch("engine.replay.bo3_jsonl.load_bo3_jsonl_entries", return_value=mock_entries):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    history = await store.get_history()
    assert len(history) == 1
    broadcaster.broadcast.assert_awaited()
    cur = await store.get_current()
    dbg = (cur.get("derived") or {}).get("debug") if isinstance(cur, dict) else {}
    assert isinstance(dbg, dict)
    assert dbg.get("replay_mode") == "point_passthrough"
    assert dbg.get("replay_quarantine_status") == "transition_passthrough_allowed"
    assert dbg.get("replay_point_policy_decision") == REPLAY_POINT_POLICY_DECISION_TRANSITION_PASSTHROUGH
    assert dbg.get("replay_contract_policy") == "reject_point_like"
    contract_status = runner.get_replay_contract_status()
    assert contract_status["point_like_inputs_seen"] == 1
    assert contract_status["point_like_inputs_rejected"] == 0
    assert contract_status["point_like_inputs_transition_passthrough"] == 1
    assert contract_status["point_like_reject_reason_counts"] == {}


@pytest.mark.asyncio
async def test_point_replay_transition_mode_rejects_when_sunset_expired() -> None:
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/points.jsonl",
        replay_loop=False,
        replay_point_transition_enabled=True,
        replay_point_transition_sunset_epoch=time.time() - 5.0,
    )
    await store.set_current(
        State(config=config, segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)
    point_payload = {"t": 1000.0, "p": 0.55, "lo": 0.2, "hi": 0.8, "rail_low": 0.4, "rail_high": 0.6, "seg": 0}
    mock_entries = [{"match_id": 1, "payload": point_payload, "source": "BO3", "ok": True}]

    with patch("engine.replay.bo3_jsonl.load_bo3_jsonl_entries", return_value=mock_entries):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    assert await store.get_history() == []
    broadcaster.broadcast.assert_not_awaited()
    contract_status = runner.get_replay_contract_status()
    assert contract_status["point_like_inputs_seen"] == 1
    assert contract_status["point_like_inputs_rejected"] == 1
    assert contract_status["point_like_reject_reason_counts"] == {
        REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED: 1
    }

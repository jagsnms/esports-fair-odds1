from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.routes_replay import replay_load, replay_status
from backend.services.runner import REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY, Runner
from backend.store.memory_store import MemoryStore
from engine.models import Config, Derived, State


@pytest.mark.asyncio
async def test_replay_status_exposes_contract_policy_and_counters() -> None:
    store = MemoryStore(max_history=100)
    await store.set_current(
        State(config=Config(source="REPLAY", replay_path="logs/points.jsonl", replay_loop=False), segment_id=0),
        Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0),
    )
    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)

    point_payload = {"t": 1000.0, "p": 0.55, "lo": 0.2, "hi": 0.8, "rail_low": 0.4, "rail_high": 0.6, "seg": 0}
    mock_entries = [{"match_id": 1, "payload": point_payload, "source": "BO3", "ok": True}]
    config = await store.get_config()
    with patch("engine.replay.bo3_jsonl.load_bo3_jsonl_entries", return_value=mock_entries):
        await runner._tick_replay(config)

    with patch("backend.deps.get_runner", return_value=runner):
        status = await replay_status(store=store)

    assert status["replay_contract_policy"] == "reject_point_like"
    assert status["replay_point_transition_enabled"] is False
    assert "replay_contract_status" in status
    contract_status = status["replay_contract_status"]
    assert contract_status["point_like_inputs_seen"] == 1
    assert contract_status["point_like_inputs_rejected"] == 1
    assert contract_status["point_like_reject_reason_counts"] == {
        REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY: 1
    }


@pytest.mark.asyncio
async def test_replay_load_accepts_contract_gate_fields() -> None:
    store = MemoryStore(max_history=100)
    body = {
        "path": "logs/history_points.jsonl",
        "replay_contract_policy": "reject_point_like",
        "replay_point_transition_enabled": True,
        "replay_point_transition_sunset_epoch": 4102444800,  # year 2100
    }
    resp = await replay_load(body=body, store=store)
    assert isinstance(resp.get("replay_load_preflight"), dict)
    status = await replay_status(store=store)
    assert status["replay_contract_policy"] == "reject_point_like"
    assert status["replay_point_transition_enabled"] is True
    assert status["replay_point_transition_sunset_epoch"] == 4102444800.0

"""
Stage 3B: Inter-map-break parity — assert on real runner-produced output.
Parity: when inter_map_break is True, runner-produced derived.debug has canonical keys
(inter_map_break, inter_map_break_reason, p_hat_old, p_hat_final, series_low, series_high,
map_low, map_high, explain with phase inter_map_break).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engine.models import Config, Derived, State
from backend.store.memory_store import MemoryStore
from backend.services.runner import Runner


_REQUIRED_INTER_MAP_BREAK_KEYS = {
    "inter_map_break",
    "inter_map_break_reason",
    "p_hat_old",
    "p_hat_final",
    "series_low",
    "series_high",
    "map_low",
    "map_high",
    "explain",
}


@pytest.mark.asyncio
async def test_real_runner_inter_map_break_produces_canonical_debug() -> None:
    """
    Run replay with detect_inter_map_break forced True; assert stored derived.debug
    from runner has canonical inter_map_break shape (real runner output, not handcrafted).
    """
    store = MemoryStore(max_history=100)
    config = Config(
        source="REPLAY",
        replay_path="logs/bo3_pulls.jsonl",
        match_id=99,
        poll_interval_s=5.0,
        replay_loop=False,
    )
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.3, rail_high=0.7, kappa=0.0)
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)
    runner = Runner(store=store, broadcaster=broadcaster)

    minimal_raw_payload = {
        "team_one": {"name": "A", "score": 0, "id": 1},
        "team_two": {"name": "B", "score": 0, "id": 2},
        "created_at": "2024-01-01T00:00:00Z",
        "round_phase": "IN_PROGRESS",
    }
    runner._replay_payloads = [minimal_raw_payload]
    runner._replay_index = 0
    runner._replay_path = "logs/bo3_pulls.jsonl"
    runner._replay_match_id = 99
    runner._replay_format = "raw"

    with patch(
        "engine.diagnostics.inter_map_break.detect_inter_map_break",
        return_value=(True, "test_reason"),
    ):
        did_replay = await runner._tick_replay(config)

    assert did_replay is True
    cur = await store.get_current()
    assert isinstance(cur, dict)
    derived_obj = cur.get("derived")
    assert isinstance(derived_obj, dict)
    dbg = derived_obj.get("debug")
    assert isinstance(dbg, dict), "runner must produce derived.debug"

    for key in _REQUIRED_INTER_MAP_BREAK_KEYS:
        assert key in dbg, f"runner-produced debug must contain {key!r}"

    assert dbg["inter_map_break"] is True
    assert dbg["inter_map_break_reason"] == "test_reason"
    explain = dbg.get("explain")
    assert isinstance(explain, dict)
    assert explain.get("phase") == "inter_map_break"
    assert dbg.get("replay_mode") == "raw_contract"


def test_real_runner_inter_map_break_produces_canonical_debug_sync() -> None:
    asyncio.run(test_real_runner_inter_map_break_produces_canonical_debug())

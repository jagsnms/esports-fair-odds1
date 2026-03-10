"""
Unit tests for BO3 multi-session isolation: buffer and freshness gate are per-session.
No cross-session state leakage when ticking match A, then B, then A.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from engine.models import Config, Derived, State
from engine.telemetry import SessionKey, SourceKind

from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


def _minimal_snap(match_id: int, game_number: int = 1) -> dict:
    """Minimal valid BO3 snapshot with match-specific game_number for assertion."""
    return {
        "team_one": {"name": "A", "score": 0, "id": 1},
        "team_two": {"name": "B", "score": 0, "id": 2},
        "game_number": game_number,
        "round_number": 1,
        "created_at": "ts1",
        "updated_at": 1000,
    }


async def _run_isolation_test() -> None:
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=1, poll_interval_s=5.0)
    state = State(config=config, segment_id=1)
    derived = Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.4, rail_high=0.6, kappa=0.0, debug={})
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)

    async def mock_get_snapshot(mid: int):
        if mid == 1:
            return _minimal_snap(1, game_number=1)
        if mid == 2:
            return _minimal_snap(2, game_number=2)
        return _minimal_snap(mid, game_number=1)

    runner = Runner(store=store, broadcaster=broadcaster)
    with patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(side_effect=mock_get_snapshot)):
        await runner._tick_bo3_for_match(1, config, is_primary=True)
        await runner._tick_bo3_for_match(2, config, is_primary=False)
        await runner._tick_bo3_for_match(1, config, is_primary=True)

    key_a = SessionKey(source=SourceKind.BO3, id="1")
    key_b = SessionKey(source=SourceKind.BO3, id="2")
    assert key_a in runner._sessions, "Session A should exist"
    assert key_b in runner._sessions, "Session B should exist"

    session_a = runner._sessions[key_a]
    session_b = runner._sessions[key_b]

    # A's buffer must still be A's data (not overwritten by B's tick)
    assert session_a.bo3_buf_raw is not None, "Session A should have buffer"
    assert session_a.bo3_buf_raw.get("game_number") == 1, "Session A buffer must be match 1 snapshot (game_number=1)"
    assert session_a.bo3_buf_ts is not None, "Session A should have bo3_buf_ts set"
    assert session_a.bo3_buf_last_success_epoch is not None

    assert session_b.bo3_buf_raw is not None, "Session B should have buffer"
    assert session_b.bo3_buf_raw.get("game_number") == 2, "Session B buffer must be match 2 snapshot (game_number=2)"

    # Freshness gate state must be distinct: each session has its own gate instance
    gate_a = session_a.ensure_bo3_gate()
    gate_b = session_b.ensure_bo3_gate()
    assert gate_a is not gate_b, "Each session must have its own Bo3FreshnessGate instance"


def test_bo3_multisession_isolation() -> None:
    """Entry point: tick A then B then A; assert A's session buffer and gate not overwritten by B."""
    asyncio.run(_run_isolation_test())

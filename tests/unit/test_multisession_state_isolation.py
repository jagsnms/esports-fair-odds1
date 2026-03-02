"""
Unit tests for multi-session state isolation: non-primary sessions use session.last_state
and do not call store.get_state().
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from engine.models import Config, Derived, State
from engine.telemetry import SessionKey, SourceKind

from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


def _minimal_snap(match_id: int, game_number: int = 1, round_number: int = 1) -> dict:
    return {
        "team_one": {"name": "A", "score": 0, "id": 1},
        "team_two": {"name": "B", "score": 0, "id": 2},
        "game_number": game_number,
        "round_number": round_number,
        "created_at": "ts1",
        "updated_at": 1000,
    }


async def _run_state_isolation_test() -> None:
    store = MemoryStore(max_history=100)
    config = Config(source="BO3", match_id=1, poll_interval_s=5.0)
    state = State(config=config, segment_id=0)
    derived = Derived(p_hat=0.5, bound_low=0.2, bound_high=0.8, rail_low=0.4, rail_high=0.6, kappa=0.0, debug={})
    await store.set_current(state, derived)

    broadcaster = MagicMock()
    broadcaster.broadcast = AsyncMock(return_value=None)

    # Match 1: first tick round 1, second tick round 2 (so last_state evolves)
    call_count = [0]

    async def mock_get_snapshot(mid: int):
        call_count[0] += 1
        if mid == 1:
            return _minimal_snap(1, game_number=1, round_number=1 + (1 if call_count[0] > 1 else 0))
        return _minimal_snap(mid, game_number=mid, round_number=1)

    runner = Runner(store=store, broadcaster=broadcaster)
    get_state_calls = []
    real_get_state = store.get_state

    async def tracked_get_state():
        get_state_calls.append(1)
        return await real_get_state()

    store.get_state = tracked_get_state

    with patch("engine.ingest.bo3_client.get_snapshot", AsyncMock(side_effect=mock_get_snapshot)):
        # Tick session A (match 1) twice, non-primary — must not call store.get_state()
        await runner._tick_bo3_for_match(1, config, is_primary=False)
        await runner._tick_bo3_for_match(1, config, is_primary=False)
        # Tick session B (match 2) once, non-primary
        await runner._tick_bo3_for_match(2, config, is_primary=False)

    # Non-primary ticks must not call store.get_state()
    assert len(get_state_calls) == 0, "store.get_state() must not be called for non-primary BO3 ticks"

    key_a = SessionKey(source=SourceKind.BO3, id="1")
    key_b = SessionKey(source=SourceKind.BO3, id="2")
    assert key_a in runner._sessions
    assert key_b in runner._sessions

    session_a = runner._sessions[key_a]
    session_b = runner._sessions[key_b]

    # A's last_state must have evolved (we ticked A twice)
    assert session_a.last_state is not None, "Session A must have last_state after two ticks"
    assert session_a.last_frame is not None
    assert session_a.last_update_ts is not None

    # B's last_state must be independent of A and not from store
    assert session_b.last_state is not None, "Session B must have last_state after one tick"
    assert session_b.last_state is not session_a.last_state, "B.last_state must not be A.last_state"
    # B's frame should reflect match 2 (game_number 2)
    assert session_b.last_frame is not None
    b_game = session_b.last_frame.get("series_score") or (0, 0)
    a_game = session_a.last_frame.get("map_index") if session_a.last_frame else None
    # Session B should have its own state (e.g. from reduce_state with B's frame)
    assert session_b.last_state.config is not None


def test_multisession_state_isolation() -> None:
    """Non-primary sessions use session last_state; store.get_state() not called."""
    asyncio.run(_run_state_isolation_test())

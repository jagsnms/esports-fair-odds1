"""
Unit tests for multi-session types: SessionKey, SessionRuntime, registry. No FastAPI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.telemetry.core import MatchContext, SourceKind
from engine.telemetry.session import SessionKey, SessionRuntime, SessionRegistry


def test_session_key_display() -> None:
    """SessionKey.display() returns source:id."""
    k1 = SessionKey(source=SourceKind.BO3, id="123")
    assert k1.display() == "BO3:123"
    k2 = SessionKey(source=SourceKind.GRID, id="abc-series")
    assert k2.display() == "GRID:abc-series"


def test_session_key_hashable() -> None:
    """SessionKey is hashable for use as dict key."""
    k1 = SessionKey(source=SourceKind.BO3, id="1")
    k2 = SessionKey(source=SourceKind.BO3, id="1")
    reg: SessionRegistry = {}
    reg[k1] = SessionRuntime(ctx=MatchContext(match_id=1))
    assert k2 in reg
    assert reg[k2].ctx.match_id == 1


def test_session_runtime_holds_ctx() -> None:
    """SessionRuntime holds MatchContext and optional last_*."""
    ctx = MatchContext(match_id=42)
    rt = SessionRuntime(ctx=ctx, last_update_ts=100.0, last_error=None)
    assert rt.ctx.match_id == 42
    assert rt.last_update_ts == 100.0
    rt.last_state = {"scores": (5, 4)}
    rt.last_frame = {"alive_counts": (4, 3)}
    assert rt.last_state["scores"] == (5, 4)
    assert rt.last_frame["alive_counts"] == (4, 3)


def test_grid_session_runtime_from_fixture() -> None:
    """Feed GRID fixture into reducer; update a SessionRuntime (no live calls)."""
    from engine.ingest.grid_reducer import GridState, reduce_event, grid_state_to_canonical_frame
    from engine.telemetry import compute_monotonic_key_from_grid_state

    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "grid_sample.jsonl"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")
    lines: list[dict] = []
    with open(fixture_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    assert len(lines) >= 1
    ss = (lines[0].get("data") or {}).get("seriesState")
    assert isinstance(ss, dict)
    state = GridState()
    state = reduce_event(state, ss)
    ctx = MatchContext(match_id=hash("2901816") % (2**31))
    runtime = SessionRuntime(ctx=ctx, grid_state=state)
    frame_d = grid_state_to_canonical_frame(runtime.grid_state, team_a_is_team_one=True)
    runtime.last_frame = frame_d
    runtime.last_update_ts = 1000.0
    runtime.last_error = None
    assert runtime.last_frame.get("alive_counts") == (4, 3)
    assert runtime.last_frame.get("scores") == (5, 4)
    key = compute_monotonic_key_from_grid_state(state, 1000.0)
    assert key.game_number == 1
    assert runtime.ctx.match_id is not None

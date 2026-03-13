"""
Unit tests for multi-session types: SessionKey, SessionRuntime, registry. No FastAPI.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.telemetry.core import MatchContext, SourceKind
from engine.telemetry.session import SessionKey, SessionRuntime, SessionRegistry

from backend.services.runner import Runner
from backend.store.memory_store import MemoryStore


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


def test_runner_sessions_diag_exposes_bo3_pipeline_for_bo3_session() -> None:
    store = MemoryStore()
    runner = Runner(store=store, broadcaster=MagicMock())
    runtime = SessionRuntime(ctx=MatchContext(match_id=123), last_update_ts=995.0, last_error=None)
    runtime.last_fetch_ts = 996.0
    runtime.last_good_ts = 997.0
    runtime.telemetry_ok = True
    runtime.telemetry_reason = None
    runtime.bo3_buf_raw = {"updated_at": "2026-03-13T05:00:00Z"}
    runtime.bo3_buf_snapshot_ts = "2026-03-13T05:00:00Z"
    runtime.bo3_last_err = None
    runtime.bo3_buf_consecutive_failures = 1
    runtime.bo3_buf_last_success_epoch = 998.0
    runtime.bo3_pipeline_diag = {
        "fetch_attempt_count": 4,
        "fetch_success_count": 3,
        "last_fetch_attempt_ts": 999.0,
        "last_fetch_success_ts": 998.5,
        "last_stage": "selector_denied",
        "last_stage_reason": "inactive_source",
        "last_stage_ts": 998.25,
        "last_snapshot_status": "live",
        "last_snapshot_fresh": True,
        "same_snapshot_polls": 2,
        "last_selector_reason": "inactive_source",
        "last_emit_decision": "no_emit",
        "last_emit_reason": "selector_denied",
        "last_source_updated_at": "2026-03-13T05:00:00Z",
        "last_source_provider_event_id": "evt-123",
    }
    runner._sessions[SessionKey(source=SourceKind.BO3, id="123")] = runtime

    with patch("backend.services.runner.time.time", return_value=1000.0):
        out = runner.get_sessions_diag()

    assert len(out["sessions"]) == 1
    session = out["sessions"][0]
    pipe = session["bo3_pipeline"]
    assert pipe["fetch_attempt_count"] == 4
    assert pipe["fetch_success_count"] == 3
    assert pipe["last_stage"] == "selector_denied"
    assert pipe["last_stage_reason"] == "inactive_source"
    assert pipe["last_selector_reason"] == "inactive_source"
    assert pipe["last_emit_decision"] == "no_emit"
    assert pipe["last_emit_reason"] == "selector_denied"
    assert pipe["buffer_has_snapshot"] is True
    assert pipe["buffer_snapshot_ts"] == "2026-03-13T05:00:00Z"
    assert pipe["buffer_consecutive_failures"] == 1
    assert pipe["last_fetch_attempt_age_s"] == 1.0
    assert pipe["last_fetch_success_age_s"] == 1.5
    assert pipe["last_stage_age_s"] == 1.8
    assert pipe["buffer_last_success_age_s"] == 2.0

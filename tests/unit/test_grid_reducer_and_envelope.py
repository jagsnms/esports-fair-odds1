"""
Unit tests for GRID reducer + canonical frame + process_canonical_envelope (no FastAPI).
Uses tests/fixtures/grid_sample.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.ingest.grid_reducer import (
    GridState,
    grid_state_to_canonical_frame,
    grid_state_to_frame,
    reduce_event,
)
from engine.telemetry.core import CanonicalFrameEnvelope, MatchContext, SourceKind
from engine.telemetry.envelope import process_canonical_envelope
from engine.telemetry.monotonic import compute_monotonic_key_from_grid_state, MonotonicKey

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "grid_sample.jsonl"


def _load_grid_sample() -> list[dict]:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
    lines: list[dict] = []
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def test_grid_reducer_from_fixture_canonical_frame_fields() -> None:
    """Feed fixture events through reducer; canonical frame has non-null alive, hp, round phase."""
    payloads = _load_grid_sample()
    assert len(payloads) >= 1
    state = GridState()
    for rec in payloads:
        data = rec.get("data") or {}
        ss = data.get("seriesState")
        assert isinstance(ss, dict), "fixture line must have data.seriesState"
        state = reduce_event(state, ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["alive_counts"] == (4, 3)
    assert frame_d["scores"] == (5, 4)
    assert frame_d["hp_totals"][0] > 0 and frame_d["hp_totals"][1] > 0
    assert frame_d["map_name"] == "mirage"
    assert frame_d["round_time_remaining_s"] is not None or frame_d.get("round_time_s") is not None
    assert frame_d["team_one_provider_id"] is not None
    assert frame_d["team_two_provider_id"] is not None


def test_grid_reducer_monotonic_key_increases_over_sample() -> None:
    """Monotonic keys from fixture sequence never regress."""
    payloads = _load_grid_sample()
    assert len(payloads) >= 2
    state = GridState()
    keys: list[MonotonicKey] = []
    for i, rec in enumerate(payloads):
        data = rec.get("data") or {}
        ss = data.get("seriesState")
        state = reduce_event(state, ss)
        observed_ts = 1000.0 + i * 1.0
        key = compute_monotonic_key_from_grid_state(state, observed_ts)
        keys.append(key)
    for i in range(1, len(keys)):
        assert keys[i] >= keys[i - 1], f"key at {i} should be >= previous"


def test_grid_process_canonical_envelope_accepts_sequence() -> None:
    """Process canonical envelope accepts the fixture sequence (no regression)."""
    payloads = _load_grid_sample()
    assert len(payloads) >= 2
    ctx = MatchContext(match_id=1)
    state = GridState()
    accepted = 0
    for i, rec in enumerate(payloads):
        data = rec.get("data") or {}
        ss = data.get("seriesState")
        state = reduce_event(state, ss)
        observed_ts = 1000.0 + i * 1.0
        key = compute_monotonic_key_from_grid_state(state, observed_ts)
        frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
        env = CanonicalFrameEnvelope(
            match_id=1,
            source=SourceKind.GRID,
            observed_ts=observed_ts,
            key=key,
            frame=frame_d,
            valid=True,
        )
        ok, reason = process_canonical_envelope(ctx, env)
        if ok:
            accepted += 1
    assert accepted >= 1
    assert ctx.accepted_count == accepted
    assert ctx.last_accepted_key is not None


def test_grid_state_to_frame_has_frame_attrs() -> None:
    """grid_state_to_frame returns Frame with scores, alive_counts, map_index."""
    payloads = _load_grid_sample()
    assert payloads
    state = GridState()
    for rec in payloads:
        ss = (rec.get("data") or {}).get("seriesState")
        if isinstance(ss, dict):
            state = reduce_event(state, ss)
            break
    frame = grid_state_to_frame(state, team_a_is_team_one=True, timestamp=1000.0)
    assert frame.scores == (5, 4)
    assert frame.alive_counts == (4, 3)
    assert frame.map_index in (0, 1)
    assert frame.map_name == "mirage"

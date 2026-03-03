"""
Unit tests for GRID reducer + canonical frame + process_canonical_envelope (no FastAPI).
Uses tests/fixtures/grid_sample.jsonl.
Patch 3: timestamp from updatedAt, armor_totals alive-only, series_fmt.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.ingest.grid_reducer import (
    DEFAULT_SERIES_FMT,
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


def _parse_iso8601_to_unix(s: str) -> float:
    """Parse ISO8601 (e.g. 2026-02-22T09:27:35.047Z) to Unix seconds for test expectations."""
    dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def test_fixture_has_required_fields() -> None:
    """Fixture contains seriesState.updatedAt, player currentArmor/alive, and (on line 2) format."""
    payloads = _load_grid_sample()
    assert len(payloads) >= 2
    for i, rec in enumerate(payloads):
        ss = (rec.get("data") or {}).get("seriesState")
        assert isinstance(ss, dict), f"line {i+1} must have data.seriesState"
        assert "updatedAt" in ss and ss["updatedAt"], f"line {i+1} must have seriesState.updatedAt"
        games = ss.get("games") or []
        assert games, f"line {i+1} must have games"
        teams = games[0].get("teams") or []
        assert len(teams) >= 2, f"line {i+1} must have at least 2 game teams"
        for t in teams[:2]:
            players = t.get("players") or []
            for p in players[:2]:
                assert "alive" in p, f"line {i+1} player must have alive"
                assert "currentArmor" in p, f"line {i+1} player must have currentArmor"
    assert "format" in payloads[1].get("data", {}).get("seriesState", {}), "line 2 must have format for series_fmt test"


def test_timestamp_from_updated_at() -> None:
    """Frame.timestamp equals parsed seriesState.updatedAt (within float tolerance)."""
    payloads = _load_grid_sample()
    assert payloads
    ss = payloads[0].get("data", {}).get("seriesState", {})
    state = reduce_event(GridState(), ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    expected_ts = _parse_iso8601_to_unix(ss["updatedAt"])
    assert frame_d["timestamp"] == pytest.approx(expected_ts, abs=0.001)


def test_armor_totals_alive_only() -> None:
    """armor_totals is (float, float) and equals alive-only sum of currentArmor from fixture."""
    payloads = _load_grid_sample()
    ss = payloads[0].get("data", {}).get("seriesState", {})
    state = reduce_event(GridState(), ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert isinstance(frame_d["armor_totals"], tuple)
    assert len(frame_d["armor_totals"]) == 2
    assert frame_d["armor_totals"][0] == pytest.approx(250.0)  # A: 100+50+0+100 (alive only)
    assert frame_d["armor_totals"][1] == pytest.approx(50.0)   # B: 0+50+0+0+0 (alive only)


def test_series_fmt_from_format() -> None:
    """series_fmt is bo1/bo3/bo5 derived from fixture; line 2 has format BO5 -> bo5."""
    payloads = _load_grid_sample()
    assert len(payloads) >= 2
    state = GridState()
    for rec in payloads:
        ss = (rec.get("data") or {}).get("seriesState")
        if isinstance(ss, dict):
            state = reduce_event(state, ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["series_fmt"] == "bo5"


def test_series_fmt_default_when_missing() -> None:
    """When format is missing, series_fmt defaults to bo3."""
    payloads = _load_grid_sample()
    ss = payloads[0].get("data", {}).get("seriesState", {})
    assert "format" not in ss or ss.get("format") is None, "line 1 has no format"
    state = reduce_event(GridState(), ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["series_fmt"] == DEFAULT_SERIES_FMT == "bo3"


def test_fallback_timestamp_when_updated_at_missing() -> None:
    """When updatedAt is missing, canonical timestamp is 0.0; grid_state_to_frame uses fallback."""
    state = GridState()
    ss = {"id": "1", "valid": True, "games": [{"sequenceNumber": 1, "started": True, "finished": False, "map": {"name": "m"}, "clock": {"currentSeconds": 60, "ticking": True, "type": "gameClock", "ticksBackwards": True}, "segments": [{"sequenceNumber": 1, "teams": [{"id": "a", "won": False}, {"id": "b", "won": False}]}], "teams": [{"id": "a", "score": 0, "side": "CT", "money": 0, "loadoutValue": 0, "players": []}, {"id": "b", "score": 0, "side": "T", "money": 0, "loadoutValue": 0, "players": []}]}]}
    assert "updatedAt" not in ss
    state = reduce_event(state, ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["timestamp"] == 0.0
    frame = grid_state_to_frame(state, team_a_is_team_one=True, timestamp=999.0)
    assert frame.timestamp == 999.0


def test_fallback_armor_totals_when_missing() -> None:
    """When player armor fields missing or no players, armor_totals is (0.0, 0.0)."""
    state = GridState()
    ss = {"id": "1", "valid": True, "games": [{"sequenceNumber": 1, "started": True, "finished": False, "map": {"name": "m"}, "clock": {}, "segments": [], "teams": [{"id": "a", "score": 0, "side": "CT", "money": 0, "loadoutValue": 0, "players": []}, {"id": "b", "score": 0, "side": "T", "money": 0, "loadoutValue": 0, "players": []}]}]}
    state = reduce_event(state, ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["armor_totals"] == (0.0, 0.0)


def test_fallback_series_fmt_when_format_missing() -> None:
    """When format is missing, series_fmt is bo3 (document fallback behavior)."""
    state = GridState()
    ss = {"id": "1", "valid": True, "games": [{"sequenceNumber": 1, "started": True, "finished": False, "map": {"name": "m"}, "clock": {}, "segments": [], "teams": [{"id": "a", "score": 0, "side": "CT", "money": 0, "loadoutValue": 0, "players": []}, {"id": "b", "score": 0, "side": "T", "money": 0, "loadoutValue": 0, "players": []}]}]}
    state = reduce_event(state, ss)
    frame_d = grid_state_to_canonical_frame(state, team_a_is_team_one=True)
    assert frame_d["series_fmt"] == "bo3"

"""
Unit tests for canonical envelope processing (process_canonical_envelope). No FastAPI.
"""
from __future__ import annotations

import pytest

from engine.telemetry.core import (
    CanonicalFrameEnvelope,
    MatchContext,
    SourceHealth,
    SourceKind,
)
from engine.telemetry.envelope import process_canonical_envelope
from engine.telemetry.monotonic import MonotonicKey


def test_accept_when_last_key_none() -> None:
    """First frame: last_accepted_key is None -> accept and update ctx."""
    ctx = MatchContext(match_id=1)
    key = MonotonicKey(game_number=1, round_number=1, seq_index=0, ts=100.0)
    env = CanonicalFrameEnvelope(
        match_id=1,
        source=SourceKind.BO3,
        observed_ts=100.0,
        key=key,
        frame={},
        valid=True,
    )
    accepted, reason = process_canonical_envelope(ctx, env)
    assert accepted is True
    assert reason is None
    assert ctx.accepted_count == 1
    assert ctx.rejected_count == 0
    assert ctx.last_accepted_key == key
    h = ctx.per_source_health.get(SourceKind.BO3)
    assert h is not None
    assert h.ok_count == 1
    assert h.last_ok_ts == 100.0


def test_reject_regression() -> None:
    """New key not strictly after last -> reject, update err_count and last_reason."""
    ctx = MatchContext(match_id=1)
    ctx.last_accepted_key = MonotonicKey(1, 2, 0, 200.0)
    ctx.accepted_count = 1
    older_key = MonotonicKey(1, 1, 0, 100.0)
    env = CanonicalFrameEnvelope(
        match_id=1,
        source=SourceKind.BO3,
        observed_ts=150.0,
        key=older_key,
        frame={},
        valid=True,
    )
    accepted, reason = process_canonical_envelope(ctx, env)
    assert accepted is False
    assert reason == "regression"
    assert ctx.accepted_count == 1
    assert ctx.rejected_count == 1
    assert ctx.last_reject_reason == "regression"
    assert ctx.last_accepted_key == MonotonicKey(1, 2, 0, 200.0)
    h = ctx.per_source_health.get(SourceKind.BO3)
    assert h is not None
    assert h.err_count == 1
    assert h.last_err_ts == 150.0
    assert h.last_reason == "regression"


def test_reject_missing_key() -> None:
    """env.key is None -> reject with missing_key, update health."""
    ctx = MatchContext(match_id=1)
    env = CanonicalFrameEnvelope(
        match_id=1,
        source=SourceKind.BO3,
        observed_ts=100.0,
        key=None,
        frame={},
        valid=True,
    )
    accepted, reason = process_canonical_envelope(ctx, env)
    assert accepted is False
    assert reason == "missing_key"
    assert ctx.rejected_count == 1
    assert ctx.last_reject_reason == "missing_key"
    h = ctx.per_source_health.get(SourceKind.BO3)
    assert h is not None
    assert h.err_count == 1
    assert h.last_reason == "missing_key"


def test_updates_source_health_ok_and_err() -> None:
    """Accept increments ok_count/last_ok_ts; reject increments err_count/last_err_ts/last_reason."""
    ctx = MatchContext(match_id=1)
    key1 = MonotonicKey(1, 1, 0, 100.0)
    env1 = CanonicalFrameEnvelope(1, SourceKind.BO3, 100.0, key1, {}, True)
    process_canonical_envelope(ctx, env1)
    h = ctx.per_source_health[SourceKind.BO3]
    assert h.ok_count == 1
    assert h.last_ok_ts == 100.0
    assert h.err_count == 0
    assert h.last_reason is None

    # Reject next (regression)
    key0 = MonotonicKey(1, 0, 0, 50.0)
    env0 = CanonicalFrameEnvelope(1, SourceKind.BO3, 120.0, key0, {}, True)
    process_canonical_envelope(ctx, env0)
    assert h.err_count == 1
    assert h.last_err_ts == 120.0
    assert h.last_reason == "regression"
    assert ctx.rejected_count == 1


def test_active_source_bootstrap() -> None:
    """When ctx.active_source is None and we accept, it is set to env.source (e.g. BO3)."""
    ctx = MatchContext(match_id=1)
    assert ctx.active_source is None
    key = MonotonicKey(1, 1, 0, 100.0)
    env = CanonicalFrameEnvelope(1, SourceKind.BO3, 100.0, key, {}, True)
    process_canonical_envelope(ctx, env)
    assert ctx.active_source == SourceKind.BO3


def test_last_accepted_env_summary_set_on_accept() -> None:
    """On accept, ctx.last_accepted_env_summary has source, match_id, key_display, observed_ts."""
    ctx = MatchContext(match_id=1)
    key = MonotonicKey(1, 2, 0, 200.0)
    env = CanonicalFrameEnvelope(
        match_id=42,
        source=SourceKind.BO3,
        observed_ts=200.0,
        key=key,
        frame={},
        valid=True,
    )
    process_canonical_envelope(ctx, env)
    summary = ctx.last_accepted_env_summary
    assert summary is not None
    assert summary["source"] == "BO3"
    assert summary["match_id"] == 42
    assert summary["key_display"] == key.to_display()
    assert summary["observed_ts"] == 200.0

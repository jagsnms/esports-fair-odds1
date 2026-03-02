"""
Unit tests for source selector: staleness, decide(), maybe_switch(), anti-thrash (no FastAPI).
"""
from __future__ import annotations

import pytest

from engine.telemetry.core import MatchContext, SourceHealth, SourceKind
from engine.telemetry.selector import (
    SourceSelectorPolicy,
    allowed_to_drive,
    decide,
    is_fresh,
    maybe_switch,
)


def _ctx_with_health(
    match_id: int = 1,
    bo3_ok_ts: float | None = None,
    bo3_ok_count: int = 1,
    grid_ok_ts: float | None = None,
    grid_ok_count: int = 0,
    active_source: SourceKind | None = None,
    last_switch_ts: float | None = None,
) -> MatchContext:
    ctx = MatchContext(match_id=match_id)
    if bo3_ok_ts is not None or bo3_ok_count:
        h = ctx.get_or_create_source_health(SourceKind.BO3)
        if bo3_ok_ts is not None:
            h.last_ok_ts = bo3_ok_ts
        h.ok_count = bo3_ok_count
    if grid_ok_ts is not None or grid_ok_count:
        h = ctx.get_or_create_source_health(SourceKind.GRID)
        if grid_ok_ts is not None:
            h.last_ok_ts = grid_ok_ts
        h.ok_count = grid_ok_count
    if active_source is not None:
        ctx.active_source = active_source
    if last_switch_ts is not None:
        ctx.last_switch_ts = last_switch_ts
    return ctx


def test_is_fresh_true_when_within_stale_window() -> None:
    now = 1000.0
    health = SourceHealth(last_ok_ts=998.0, ok_count=1)
    assert is_fresh(health, now, 5.0) is True


def test_is_fresh_false_when_outside_stale_window() -> None:
    now = 1000.0
    health = SourceHealth(last_ok_ts=990.0, ok_count=1)
    assert is_fresh(health, now, 5.0) is False


def test_is_fresh_false_when_no_last_ok_ts() -> None:
    now = 1000.0
    health = SourceHealth(ok_count=0)
    assert is_fresh(health, now, 5.0) is False


def test_decide_chooses_grid_over_bo3_when_both_fresh() -> None:
    """When prefer_order is GRID then BO3 and both are fresh, chosen_source is GRID."""
    now = 1000.0
    ctx = _ctx_with_health(bo3_ok_ts=999.0, grid_ok_ts=999.0, grid_ok_count=1)
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["GRID", "BO3", "REPLAY", "PASSTHROUGH"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, None)
    assert decision.chosen_source == "GRID"
    assert decision.reason == "fresh"
    names_and_reasons = dict(decision.considered)
    assert names_and_reasons.get("GRID") == "fresh"
    assert names_and_reasons.get("BO3") == "fresh"


def test_decide_sticks_to_bo3_when_bo3_fresh_and_prefer_bo3() -> None:
    """Default policy prefers BO3 first; with only BO3 health and fresh, chosen is BO3."""
    now = 1000.0
    ctx = _ctx_with_health(bo3_ok_ts=999.0)
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["BO3", "GRID", "REPLAY", "PASSTHROUGH"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, None)
    assert decision.chosen_source == "BO3"


def test_maybe_switch_sticks_to_bo3_when_bo3_fresh_even_if_grid_fresh() -> None:
    """Active source BO3 is fresh; do not switch to GRID even if GRID is also fresh (active_fresh)."""
    now = 1000.0
    ctx = _ctx_with_health(
        bo3_ok_ts=999.0,
        grid_ok_ts=999.0,
        grid_ok_count=1,
        active_source=SourceKind.BO3,
    )
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["GRID", "BO3"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, None)
    assert decision.chosen_source == "GRID"
    switched, reason = maybe_switch(ctx, decision, now, policy, None)
    assert switched is False
    assert reason == "active_fresh"
    assert ctx.active_source == SourceKind.BO3


def test_maybe_switch_mid_round_no_switch_when_allow_false() -> None:
    """Mid-round and allow_mid_round_switch False: do not switch from BO3 to GRID even if BO3 stale and GRID fresh."""
    now = 1000.0
    ctx = _ctx_with_health(
        bo3_ok_ts=990.0,  # stale (10s ago)
        grid_ok_ts=999.0,
        grid_ok_count=1,
        active_source=SourceKind.BO3,
    )
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["GRID", "BO3"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, "live")
    assert decision.chosen_source == "GRID"
    switched, reason = maybe_switch(ctx, decision, now, policy, "live")
    assert switched is False
    assert reason == "mid_round_no_switch"
    assert ctx.active_source == SourceKind.BO3


def test_maybe_switch_switches_from_bo3_to_grid_when_bo3_stale_grid_fresh() -> None:
    """Between rounds: BO3 stale, GRID fresh -> switch to GRID."""
    now = 1000.0
    ctx = _ctx_with_health(
        bo3_ok_ts=990.0,
        grid_ok_ts=999.0,
        grid_ok_count=1,
        active_source=SourceKind.BO3,
    )
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["GRID", "BO3"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, None)  # not mid-round
    assert decision.chosen_source == "GRID"
    switched, reason = maybe_switch(ctx, decision, now, policy, None)
    assert switched is True
    assert "switch_to_GRID" in reason
    assert ctx.active_source == SourceKind.GRID
    assert ctx.last_switch_ts == now
    assert ctx.last_switch_reason is not None


def test_maybe_switch_cooldown_prevents_rapid_flip_flop() -> None:
    """After switching to GRID, within cooldown we do not switch back to BO3 even when GRID is stale and BO3 fresh."""
    now = 1000.0
    ctx = _ctx_with_health(
        bo3_ok_ts=999.0,   # BO3 fresh
        grid_ok_ts=990.0,  # GRID stale (10s ago)
        grid_ok_count=1,
        active_source=SourceKind.GRID,  # we're on GRID after a prior switch
        last_switch_ts=995.0,  # switched 5s ago; cooldown 10s
    )
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["BO3", "GRID"],
        allow_mid_round_switch=True,
    )
    decision = decide(ctx, now, policy, None)
    assert decision.chosen_source == "BO3"  # only BO3 is fresh
    switched, reason = maybe_switch(ctx, decision, now, policy, None)
    assert switched is False
    assert reason == "cooldown"
    assert ctx.active_source == SourceKind.GRID


def test_decide_no_fresh_returns_none_chosen() -> None:
    """When all sources are stale or missing, chosen_source is None."""
    now = 1000.0
    ctx = _ctx_with_health(bo3_ok_ts=980.0)  # 20s ago
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["BO3", "GRID"],
        allow_mid_round_switch=False,
    )
    decision = decide(ctx, now, policy, None)
    assert decision.chosen_source is None
    assert "stale" in decision.reason or "missing" in decision.reason or "all" in decision.reason


def test_source_decision_to_diag() -> None:
    """to_diag() returns serializable dict for debug output."""
    from engine.telemetry.selector import SourceDecision

    d = SourceDecision(chosen_source="BO3", reason="fresh", considered=[("BO3", "fresh"), ("GRID", "missing")])
    diag = d.to_diag()
    assert diag["chosen_source"] == "BO3"
    assert diag["reason"] == "fresh"
    assert diag["considered"] == [("BO3", "fresh"), ("GRID", "missing")]


# --- allowed_to_drive (authoritative selector: execution must match decision) ---


def test_allowed_to_drive_bootstrap() -> None:
    """When ctx.active_source is None, allow and set active_source = env_source (bootstrap)."""
    ctx = MatchContext(match_id=1)
    assert ctx.active_source is None
    allow, reason = allowed_to_drive(ctx, SourceKind.BO3, 1000.0, None)
    assert allow is True
    assert reason == "bootstrap"
    assert ctx.active_source == SourceKind.BO3


def test_allowed_to_drive_active_matches() -> None:
    """When ctx.active_source == env_source, allow."""
    ctx = _ctx_with_health(bo3_ok_ts=999.0, active_source=SourceKind.BO3)
    allow, reason = allowed_to_drive(ctx, SourceKind.BO3, 1000.0, None)
    assert allow is True
    assert reason == "active"
    assert ctx.active_source == SourceKind.BO3


def test_allowed_to_drive_deny_inactive_source() -> None:
    """When ctx.active_source is BO3 and env_source is GRID (and BO3 still preferred/fresh), deny (inactive_source)."""
    ctx = _ctx_with_health(bo3_ok_ts=999.0, grid_ok_ts=999.0, grid_ok_count=1, active_source=SourceKind.BO3)
    allow, reason = allowed_to_drive(ctx, SourceKind.GRID, 1000.0, None)
    assert allow is False
    assert reason == "inactive_source"
    assert ctx.active_source == SourceKind.BO3


def test_allowed_to_drive_allow_after_switch_when_active_stale() -> None:
    """When active is BO3 (stale), env_source is GRID (fresh), and not mid-round: maybe_switch switches to GRID, then allow."""
    ctx = _ctx_with_health(
        bo3_ok_ts=990.0,
        grid_ok_ts=999.0,
        grid_ok_count=1,
        active_source=SourceKind.BO3,
    )
    policy = SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0},
        switch_cooldown_s=10.0,
        prefer_order=["GRID", "BO3"],
        allow_mid_round_switch=False,
    )
    allow, reason = allowed_to_drive(ctx, SourceKind.GRID, 1000.0, None, policy)
    assert allow is True
    assert "switch_to_GRID" in reason
    assert ctx.active_source == SourceKind.GRID


def test_allowed_to_drive_session_runtime_grid_does_not_advance() -> None:
    """Integration-ish: ctx.active_source=BO3, env.source=GRID => denied; session last_update_ts unchanged (runner would not run reduce/write)."""
    from engine.telemetry.session import SessionRuntime

    ctx = _ctx_with_health(bo3_ok_ts=999.0, grid_ok_ts=999.0, grid_ok_count=1, active_source=SourceKind.BO3)
    runtime = SessionRuntime(ctx=ctx, last_state=None, last_frame=None, last_update_ts=100.0, last_error=None, grid_state=None)
    allow, reason = allowed_to_drive(ctx, SourceKind.GRID, 1000.0, None)
    assert allow is False
    assert reason == "inactive_source"
    assert runtime.last_update_ts == 100.0
    assert runtime.last_state is None
    assert runtime.last_frame is None

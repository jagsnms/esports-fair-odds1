"""
Source selector: staleness, validity, and anti-thrash switching policy.
Pure Python; no FastAPI or backend dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.telemetry.core import MatchContext, SourceHealth, SourceKind


@dataclass
class SourceSelectorPolicy:
    """Policy for source staleness and switching."""

    stale_after_s: dict[str, float]  # SourceKind name -> seconds (e.g. BO3 5, GRID 3, REPLAY 30)
    switch_cooldown_s: float  # e.g. 10
    prefer_order: list[str]  # SourceKind names in priority order (e.g. ["GRID", "BO3", "REPLAY", "PASSTHROUGH"])
    allow_mid_round_switch: bool = False


@dataclass
class SourceDecision:
    """Result of decide(): chosen source and per-source reasons."""

    chosen_source: str | None  # SourceKind name or None
    reason: str
    considered: list[tuple[str, str]]  # (source_name, reason) e.g. ("BO3", "fresh"), ("GRID", "stale")

    def to_diag(self) -> dict:
        """Serializable dict for debug/telemetry output."""
        return {
            "chosen_source": self.chosen_source,
            "reason": self.reason,
            "considered": list(self.considered),
        }


def _source_name(sk: "SourceKind") -> str:
    return getattr(sk, "name", str(sk))


def default_source_selector_policy() -> SourceSelectorPolicy:
    """Default policy: BO3 first (production), then GRID, REPLAY, PASSTHROUGH; no mid-round switch."""
    return SourceSelectorPolicy(
        stale_after_s={"BO3": 5.0, "GRID": 3.0, "REPLAY": 30.0, "PASSTHROUGH": 60.0},
        switch_cooldown_s=10.0,
        prefer_order=["BO3", "GRID", "REPLAY", "PASSTHROUGH"],
        allow_mid_round_switch=False,
    )


def is_fresh(health: "SourceHealth | None", now_ts: float, stale_after_s: float) -> bool:
    """True if source has reported OK within stale_after_s seconds."""
    if health is None or stale_after_s <= 0:
        return False
    last_ok = getattr(health, "last_ok_ts", None)
    if last_ok is None:
        return False
    return (now_ts - last_ok) <= stale_after_s


def _is_valid(health: "SourceHealth | None") -> bool:
    """True if source has ever been OK (ok_count > 0 or last_ok_ts set)."""
    if health is None:
        return False
    ok_count = getattr(health, "ok_count", 0) or 0
    last_ok = getattr(health, "last_ok_ts", None)
    return ok_count > 0 or last_ok is not None


def _is_mid_round(round_phase: str | None) -> bool:
    """True if we are in a round (not ended/halftime); conservative for no mid-round switch."""
    if not round_phase or not str(round_phase).strip():
        return False
    p = str(round_phase).strip().lower()
    return p not in ("ended", "halftime", "finished")


def decide(
    ctx: "MatchContext",
    now_ts: float,
    policy: SourceSelectorPolicy,
    round_phase: str | None,
) -> SourceDecision:
    """
    Choose highest-priority source that is fresh and valid.
    Returns SourceDecision with chosen_source (name), reason, and considered list.
    """
    from engine.telemetry.core import SourceKind

    considered: list[tuple[str, str]] = []
    chosen_name: str | None = None
    reason = "none_available"

    for name in policy.prefer_order:
        try:
            sk = SourceKind[name]
        except KeyError:
            considered.append((name, "unknown_source"))
            continue
        health = ctx.per_source_health.get(sk) if getattr(ctx, "per_source_health", None) else None
        stale_s = policy.stale_after_s.get(name, 0.0) or 0.0

        if health is None:
            considered.append((name, "missing"))
            continue
        if not _is_valid(health):
            considered.append((name, "invalid"))
            continue
        if is_fresh(health, now_ts, stale_s):
            considered.append((name, "fresh"))
            if chosen_name is None:
                chosen_name = name
                reason = "fresh"
        else:
            considered.append((name, "stale"))
            if chosen_name is None:
                reason = "all_stale_or_missing"

    if chosen_name is None:
        return SourceDecision(chosen_source=None, reason=reason, considered=considered)
    return SourceDecision(chosen_source=chosen_name, reason=reason, considered=considered)


def maybe_switch(
    ctx: "MatchContext",
    decision: SourceDecision,
    now_ts: float,
    policy: SourceSelectorPolicy,
    round_phase: str | None,
) -> tuple[bool, str]:
    """
    Apply anti-thrash: switch only if active is stale/invalid and another is fresh, respecting cooldown and mid_round.
    Updates ctx.active_source, ctx.last_switch_ts, ctx.last_switch_reason when switching.
    Returns (switched: bool, reason: str).
    """
    from engine.telemetry.core import SourceKind

    active = getattr(ctx, "active_source", None)
    active_name = _source_name(active) if active is not None else None
    last_switch_ts = getattr(ctx, "last_switch_ts", None)
    chosen = decision.chosen_source

    # No switch if current active is fresh
    if active is not None and chosen is not None and active_name == chosen:
        return (False, "active_fresh")

    if active is not None:
        health = ctx.per_source_health.get(active) if getattr(ctx, "per_source_health", None) else None
        stale_s = policy.stale_after_s.get(active_name, 0.0) or 0.0
        if health is not None and is_fresh(health, now_ts, stale_s):
            return (False, "active_fresh")

    # Cooldown
    if last_switch_ts is not None and (now_ts - last_switch_ts) < policy.switch_cooldown_s:
        return (False, "cooldown")

    # Mid-round block
    if _is_mid_round(round_phase) and not policy.allow_mid_round_switch:
        return (False, "mid_round_no_switch")

    # Switch only if we have a chosen source (fresh) and it's different from active
    if chosen is None:
        return (False, "no_fresh_source")
    try:
        new_source = SourceKind[chosen]
    except KeyError:
        return (False, "invalid_chosen")

    if active is not None and new_source == active:
        return (False, "same_source")

    ctx.active_source = new_source
    ctx.last_switch_ts = now_ts
    ctx.last_switch_reason = f"switch_to_{chosen}"
    return (True, ctx.last_switch_reason)

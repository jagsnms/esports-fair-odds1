"""
Source-agnostic telemetry core: enums and dataclasses for match context, identity, envelopes.
No FastAPI or backend dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engine.telemetry.monotonic import MonotonicKey


class SourceKind(str, Enum):
    """Telemetry source identifier."""

    BO3 = "BO3"
    GRID = "GRID"
    REPLAY = "REPLAY"
    PASSTHROUGH = "PASSTHROUGH"


@dataclass
class SourceHealth:
    """Per-source health metrics (last ok/err timestamps, counts, last reject reason)."""

    last_ok_ts: float | None = None
    last_err_ts: float | None = None
    stale_since_ts: float | None = None
    ok_count: int = 0
    err_count: int = 0
    last_reason: str | None = None


@dataclass
class IdentityEntry:
    """Canonical identity per match: team A/B mapping and provider ids. Stub for Patch 1."""

    team_a_is_team_one: bool | None = None
    provider_team_a_id: str | None = None
    provider_team_b_id: str | None = None
    team_a_display_name: str | None = None
    team_b_display_name: str | None = None


@dataclass
class CanonicalFrameEnvelope:
    """Provider-agnostic frame wrapper: match_id, source, observed_ts, monotonic key, payload."""

    match_id: int
    source: SourceKind
    observed_ts: float
    key: MonotonicKey | None
    frame: dict[str, Any]
    valid: bool = True
    invalid_reason: str | None = None


@dataclass
class MatchContext:
    """Per-match telemetry context: last accepted key, identity stub, per-source health, counters, selector state."""

    match_id: int
    active_source: SourceKind | None = None
    last_accepted_key: MonotonicKey | None = None
    identity: IdentityEntry | None = None
    per_source_health: dict[SourceKind, SourceHealth] = field(default_factory=dict)
    accepted_count: int = 0
    rejected_count: int = 0
    last_reject_reason: str | None = None
    last_switch_ts: float | None = None
    last_switch_reason: str | None = None
    last_accepted_env_summary: dict | None = None  # {source, match_id, key_display, observed_ts} for debug

    def get_or_create_source_health(self, source: SourceKind) -> SourceHealth:
        if source not in self.per_source_health:
            self.per_source_health[source] = SourceHealth()
        return self.per_source_health[source]

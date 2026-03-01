"""
Multi-session telemetry: session key and runtime. No cross-provider merging.
No FastAPI deps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.telemetry.core import MatchContext, SourceKind


@dataclass(frozen=True)
class SessionKey:
    """Identifies one telemetry session (BO3 match or GRID series). Hashable for registry."""

    source: SourceKind
    id: str  # BO3 match_id or GRID series_id as string

    def display(self) -> str:
        return f"{self.source.value}:{self.id}"


@dataclass
class SessionRuntime:
    """Per-session runtime: context, last reducer state, frame, and error."""

    ctx: MatchContext
    last_state: Any | None = None
    last_frame: dict | None = None
    last_update_ts: float | None = None
    last_error: str | None = None
    grid_state: Any | None = None  # GRID only: GridState for this series


# Registry: session key -> runtime (type alias for clarity)
SessionRegistry = dict[SessionKey, SessionRuntime]

"""
Multi-session telemetry: session key and runtime. No cross-provider merging.
No FastAPI deps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from engine.telemetry.core import MatchContext, SourceKind

if TYPE_CHECKING:
    from engine.ingest.bo3_freshness import Bo3FreshnessGate


@dataclass(frozen=True)
class SessionKey:
    """Identifies one telemetry session (BO3 match or GRID series). Hashable for registry."""

    source: SourceKind
    id: str  # BO3 match_id or GRID series_id as string

    def display(self) -> str:
        return f"{self.source.value}:{self.id}"


@dataclass
class SessionRuntime:
    """Per-session runtime: context, last reducer state, frame, and error.
    BO3 sessions also carry per-session buffer, freshness gate, and outcome trackers.
    """

    ctx: MatchContext
    last_state: Any | None = None
    last_frame: dict | None = None
    last_update_ts: float | None = None
    last_error: str | None = None
    grid_state: Any | None = None  # GRID only: GridState for this series

    # Feed vs telemetry distinction (BO3): fetch_ts = last successful HTTP 200; good_ts = last time telemetry was valid
    last_fetch_ts: float | None = None
    last_good_ts: float | None = None
    telemetry_ok: bool = True
    telemetry_reason: str | None = None  # e.g. "missing_teams", "missing_microstate", "clock_invalid"

    # --- BO3 buffer (per-session to avoid cross-session leakage) ---
    bo3_buf_raw: dict | None = None
    bo3_buf_frame: dict | None = None
    bo3_buf_ts: float | None = None  # last successful snapshot epoch
    bo3_buf_snapshot_ts: str | None = None
    bo3_buf_last_success_epoch: float | None = None
    bo3_buf_last_attempt_epoch: float | None = None
    bo3_last_err: str | None = None
    bo3_buf_consecutive_failures: int = 0
    bo3_fresh_gate: "Bo3FreshnessGate | None" = None

    # BO3 "last accepted" snapshot state (for status/dedupe)
    bo3_last_raw_snapshot: dict | None = None
    bo3_last_snapshot_ts: Any = None
    bo3_last_scores: tuple[int, int] | None = None
    bo3_same_snapshot_polls: int = 0
    bo3_last_change_epoch: float | None = None

    # BO3 outcome/dedupe trackers (per-session)
    bo3_last_seen_round_number: int | None = None
    bo3_last_emitted_round_number: int | None = None
    bo3_last_emitted_round_winner_team_id: int | None = None
    bo3_last_seen_segment_id_for_result: int | None = None
    bo3_last_seen_map_winner_team_id: int | None = None
    bo3_last_seen_scores: tuple[int, int] | None = None
    bo3_last_seen_score_team_one: int | None = None
    bo3_last_seen_score_team_two: int | None = None
    bo3_last_seen_game_number: int | None = None
    bo3_last_seen_match_score_team_one: int | None = None
    bo3_last_seen_match_score_team_two: int | None = None
    bo3_last_seen_match_score_by_game: dict[int, tuple[int | None, int | None]] = field(default_factory=dict)
    map_identity_cache: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    bo3_raw_last_sig: tuple[Any, ...] | None = None  # dedupe for raw snapshot recording

    def ensure_bo3_gate(self) -> "Bo3FreshnessGate":
        """Return per-session Bo3FreshnessGate; create and store if missing."""
        if self.bo3_fresh_gate is None:
            from engine.ingest.bo3_freshness import Bo3FreshnessGate
            self.bo3_fresh_gate = Bo3FreshnessGate()
        return self.bo3_fresh_gate


# Registry: session key -> runtime (type alias for clarity)
SessionRegistry = dict[SessionKey, SessionRuntime]

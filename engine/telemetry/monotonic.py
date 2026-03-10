"""
Source-independent monotonic acceptance gate: MonotonicKey ordering and should_accept.
BO3 key extraction uses same logical fields as Bo3FreshnessGate (game_number, map_index, round_number, seq, clock).
No FastAPI or backend dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MonotonicKey:
    """
    Ordered key for monotonic acceptance. None-safe comparison: None sorts before any value.
    Ordering: game_number asc, round_number asc, seq_index asc, ts asc.
    """

    game_number: int | None
    round_number: int | None
    seq_index: int | None
    ts: float | None  # unix seconds

    def __lt__(self, other: MonotonicKey) -> bool:
        if not isinstance(other, MonotonicKey):
            return NotImplemented
        if self.game_number is None and other.game_number is not None:
            return True
        if self.game_number is not None and other.game_number is None:
            return False
        if (self.game_number or 0) != (other.game_number or 0):
            return (self.game_number or 0) < (other.game_number or 0)
        if self.round_number is None and other.round_number is not None:
            return True
        if self.round_number is not None and other.round_number is None:
            return False
        if (self.round_number or 0) != (other.round_number or 0):
            return (self.round_number or 0) < (other.round_number or 0)
        if self.seq_index is None and other.seq_index is not None:
            return True
        if self.seq_index is not None and other.seq_index is None:
            return False
        if (self.seq_index or 0) != (other.seq_index or 0):
            return (self.seq_index or 0) < (other.seq_index or 0)
        if self.ts is None and other.ts is not None:
            return True
        if self.ts is not None and other.ts is None:
            return False
        return (self.ts or 0.0) < (other.ts or 0.0)

    def __le__(self, other: MonotonicKey) -> bool:
        return self < other or self == other

    def __gt__(self, other: MonotonicKey) -> bool:
        return not (self <= other)

    def __ge__(self, other: MonotonicKey) -> bool:
        return not (self < other)

    def to_display(self) -> str:
        """Stringify for diagnostics."""
        return f"g={self.game_number!r}r={self.round_number!r}seq={self.seq_index!r}ts={self.ts!r}"


def _coerce_ts_seconds(raw_ts: Any) -> float | None:
    """Normalize timestamp to float seconds. Accepts int (ms or sec), float, string."""
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        try:
            v = float(raw_ts)
        except (TypeError, ValueError):
            return None
        if v != v:
            return None
        if isinstance(raw_ts, int) and raw_ts > 1e12:
            return v / 1000.0  # assume ms
        return v if v >= 0 else None
    if isinstance(raw_ts, str):
        s = raw_ts.strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        if v != v:
            return None
        return v if v >= 0 else None
    return None


def compute_monotonic_key_from_bo3_snapshot(
    frame: Any,
    raw_snap: dict[str, Any] | None,
) -> MonotonicKey | None:
    """
    Build MonotonicKey from BO3 snapshot using same logical fields as Bo3FreshnessGate:
    game_number, map_index (as seq-like), round_number, and timestamp/clock for ordering.
    Returns None if insufficient data to form a key.
    """
    game_number: int | None = 1
    round_number: int | None = 0
    seq_index: int | None = 0  # map_index as intra-game sequence
    ts: float | None = None

    if isinstance(raw_snap, dict):
        try:
            g = raw_snap.get("game_number")
            game_number = int(g) if g is not None else 1
        except (TypeError, ValueError):
            pass
        try:
            r = raw_snap.get("round_number")
            round_number = int(r) if r is not None else 0
        except (TypeError, ValueError):
            pass
        ts = _coerce_ts_seconds(
            raw_snap.get("updated_at")
            or raw_snap.get("created_at")
            or raw_snap.get("sent_time")
            or raw_snap.get("ts")
        )

    if frame is not None:
        map_idx = getattr(frame, "map_index", 0)
        if isinstance(map_idx, int):
            seq_index = map_idx
        else:
            try:
                seq_index = int(map_idx)
            except (TypeError, ValueError):
                seq_index = 0
        if ts is None and hasattr(frame, "timestamp"):
            ts = _coerce_ts_seconds(frame.timestamp)

    return MonotonicKey(
        game_number=game_number,
        round_number=round_number,
        seq_index=seq_index,
        ts=ts,
    )


def compute_monotonic_key_from_grid_state(state: Any, observed_ts: float) -> MonotonicKey:
    """
    Build MonotonicKey from GRID GridState (or dict with game_index, round_index).
    Uses observed_ts for ordering; game_index/round_index for progression.
    """
    game_index = getattr(state, "game_index", None) or (state.get("game_index") if isinstance(state, dict) else None)
    round_index = getattr(state, "round_index", None) or (state.get("round_index") if isinstance(state, dict) else None)
    game_number = int(game_index) if game_index is not None else 1
    round_number = int(round_index) if round_index is not None else 0
    return MonotonicKey(
        game_number=game_number,
        round_number=round_number,
        seq_index=0,
        ts=observed_ts,
    )


def should_accept(
    last_accepted_key: MonotonicKey | None,
    new_key: MonotonicKey | None,
    *,
    reject_missing_key: bool = True,
) -> tuple[bool, str | None]:
    """
    Source-independent monotonic gate: accept iff new_key is strictly after last_accepted_key.
    Returns (accept: bool, reason: str | None). reason is None when accepted; else e.g. "missing_key", "regression".
    If reject_missing_key is True (default), None new_key yields (False, "missing_key").
    """
    if new_key is None:
        return (False, "missing_key") if reject_missing_key else (True, None)
    if last_accepted_key is None:
        return (True, None)
    if new_key > last_accepted_key:
        return (True, None)
    return (False, "regression")

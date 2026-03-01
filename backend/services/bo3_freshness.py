"""
BO3 tick monotonic gating: reject stale/out-of-order frames before they update state.
Prevents "time rewind" (e.g. 1v1 -> 2v3 -> back to 1v1) in live dashboard.
Key: (game_number, map_index, round_number). Reject on ts_backwards, clock_rewind, alive_sig_rewind.
"""
from __future__ import annotations

from typing import Any

# Allow ~0.25s provider jitter before treating round_time_remaining increase as rewind
CLOCK_REWIND_EPS_S = 0.25


def coerce_ts_ms(raw_ts: Any) -> int | None:
    """
    Normalize timestamp to integer milliseconds.
    Accepts: float (seconds), int (ms or seconds if < 1e12), string numeric.
    Returns None if unusable.
    """
    if raw_ts is None:
        return None
    if isinstance(raw_ts, (int, float)):
        try:
            v = float(raw_ts)
        except (TypeError, ValueError):
            return None
        if v != v:  # NaN
            return None
        # Assume int > 1e12 is ms; else seconds
        if isinstance(raw_ts, int) and raw_ts > 1e12:
            return int(raw_ts)
        return int(round(v * 1000)) if v >= 0 else None
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
        return int(round(v * 1000)) if v >= 0 else None
    return None


def _get_ts_ms_from_raw(raw: dict[str, Any] | None) -> int | None:
    """Best-effort ts_ms from raw BO3 snapshot (updated_at, created_at, sent_time, ts)."""
    if not raw or not isinstance(raw, dict):
        return None
    for key in ("updated_at", "created_at", "sent_time", "ts"):
        val = raw.get(key)
        if val is None:
            continue
        ms = coerce_ts_ms(val)
        if ms is not None:
            return ms
    return None


def _key_from_frame_and_raw(
    frame: Any,
    raw_snap: dict[str, Any] | None,
) -> tuple[int, int, int]:
    """(game_number, map_index, round_number) for cache key."""
    game = 1
    round_no = 0
    if isinstance(raw_snap, dict):
        try:
            g = raw_snap.get("game_number")
            game = int(g) if g is not None else 1
        except (TypeError, ValueError):
            pass
        try:
            r = raw_snap.get("round_number")
            round_no = int(r) if r is not None else 0
        except (TypeError, ValueError):
            pass
    map_idx = getattr(frame, "map_index", 0)
    if not isinstance(map_idx, int):
        try:
            map_idx = int(map_idx)
        except (TypeError, ValueError):
            map_idx = 0
    return (game, map_idx, round_no)


def _alive_sig(frame: Any) -> str:
    """Alive signature string for rewind detection."""
    alive = getattr(frame, "alive_counts", (0, 0))
    if not alive or len(alive) < 2:
        return "0v0"
    a, b = int(alive[0]) if alive[0] is not None else 0, int(alive[1]) if alive[1] is not None else 0
    return f"{a}v{b}"


class Bo3FreshnessGate:
    """
    Per-(game_number, map_index, round_number) monotonic gating.
    Rejects frames that would rewind time, round clock, or repeat an older alive state.
    """

    def __init__(self) -> None:
        # key -> {last_ts_ms, last_round_time_remaining_s, alive_sig_ts: {sig: ts_ms}}
        self._cache: dict[tuple[int, int, int], dict[str, Any]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def accept_frame(
        self,
        frame: Any,
        raw_snap: dict[str, Any] | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Decide whether to accept this frame (monotonic gating).
        Returns (accept, reason, diag). reason is None if accepted; else "ts_backwards" | "clock_rewind" | "alive_sig_rewind".
        """
        key = _key_from_frame_and_raw(frame, raw_snap)
        ts_ms = _get_ts_ms_from_raw(raw_snap)
        if ts_ms is None:
            ts_ms = coerce_ts_ms(getattr(frame, "timestamp", None))
        if ts_ms is None and hasattr(frame, "timestamp"):
            try:
                ts_ms = int(round(float(frame.timestamp) * 1000))
            except (TypeError, ValueError):
                pass
        round_time_s: float | None = getattr(frame, "round_time_remaining_s", None)
        if round_time_s is not None:
            try:
                round_time_s = float(round_time_s)
            except (TypeError, ValueError):
                round_time_s = None
        alive_sig = _alive_sig(frame)

        diag: dict[str, Any] = {
            "key": list(key),
            "ts_ms": ts_ms,
            "round_time_remaining_s": round_time_s,
            "alive_sig": alive_sig,
        }

        entry = self._cache.get(key)
        if entry is None:
            self._cache[key] = {
                "last_ts_ms": ts_ms,
                "last_round_time_remaining_s": round_time_s,
                "alive_sig_ts": {alive_sig: ts_ms} if (ts_ms is not None and alive_sig) else {},
            }
            diag["last_ts_ms"] = ts_ms
            diag["last_round_time_remaining_s"] = round_time_s
            diag["reason"] = None
            return (True, None, diag)

        last_ts_ms = entry.get("last_ts_ms")
        last_round_s = entry.get("last_round_time_remaining_s")
        alive_sig_ts = entry.get("alive_sig_ts") or {}

        diag["last_ts_ms"] = last_ts_ms
        diag["last_round_time_remaining_s"] = last_round_s

        if ts_ms is not None and last_ts_ms is not None and ts_ms < last_ts_ms:
            diag["reason"] = "ts_backwards"
            return (False, "ts_backwards", diag)

        if (
            round_time_s is not None
            and last_round_s is not None
            and round_time_s > last_round_s + CLOCK_REWIND_EPS_S
        ):
            diag["reason"] = "clock_rewind"
            return (False, "clock_rewind", diag)

        if ts_ms is not None and alive_sig and alive_sig in alive_sig_ts and ts_ms < alive_sig_ts[alive_sig]:
            diag["reason"] = "alive_sig_rewind"
            return (False, "alive_sig_rewind", diag)

        # Accept: update cache
        if ts_ms is not None:
            entry["last_ts_ms"] = ts_ms
        if round_time_s is not None:
            entry["last_round_time_remaining_s"] = round_time_s
        if ts_ms is not None and alive_sig:
            if "alive_sig_ts" not in entry:
                entry["alive_sig_ts"] = {}
            entry["alive_sig_ts"][alive_sig] = ts_ms
        diag["reason"] = None
        return (True, None, diag)

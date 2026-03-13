"""
BO3 tick monotonic gating: reject stale/out-of-order frames before they update state.
Prevents "time rewind" (e.g. 1v1 -> 2v3 -> back to 1v1) in live dashboard.
Key: (game_number, map_index, round_number). Reject on ts_backwards, alive_sig_rewind,
and only accept clock rewind when explicit meaningful live advancement is present.
No backend dependencies; used by SessionRuntime and runner.
"""
from __future__ import annotations

from typing import Any

# Allow ~0.25s provider jitter before treating round_time_remaining increase as rewind
CLOCK_REWIND_EPS_S = 0.25

_PHASE_RANKS: dict[str, int] = {
    "freeze": 0,
    "freezetime": 0,
    "buy": 0,
    "warmup": 0,
    "live": 1,
    "in_progress": 1,
    "playing": 1,
    "bomb_planted": 2,
    "planted": 2,
    "post_plant": 2,
    "round_end": 3,
    "ended": 3,
    "finished": 3,
}


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


def _alive_counts(frame: Any) -> tuple[int, int]:
    alive = getattr(frame, "alive_counts", (0, 0))
    if not alive or len(alive) < 2:
        return (0, 0)
    a = int(alive[0]) if alive[0] is not None else 0
    b = int(alive[1]) if alive[1] is not None else 0
    return (a, b)


def _scores(frame: Any) -> tuple[int, int]:
    scores = getattr(frame, "scores", (0, 0))
    if not scores or len(scores) < 2:
        return (0, 0)
    a = int(scores[0]) if scores[0] is not None else 0
    b = int(scores[1]) if scores[1] is not None else 0
    return (a, b)


def _bomb_phase(frame: Any) -> tuple[str | None, bool | None]:
    bomb_phase = getattr(frame, "bomb_phase_time_remaining", None)
    if not isinstance(bomb_phase, dict):
        return (None, None)
    phase_raw = bomb_phase.get("round_phase")
    phase = str(phase_raw).strip().lower() if isinstance(phase_raw, str) and phase_raw.strip() else None
    planted = bomb_phase.get("is_bomb_planted")
    planted_bool = planted if isinstance(planted, bool) else None
    return (phase, planted_bool)


def _meaningful_advancement(
    entry: dict[str, Any],
    frame: Any,
) -> list[str]:
    reasons: list[str] = []

    prev_scores = entry.get("last_scores")
    cur_scores = _scores(frame)
    if isinstance(prev_scores, tuple) and len(prev_scores) >= 2:
        if cur_scores[0] >= prev_scores[0] and cur_scores[1] >= prev_scores[1]:
            if cur_scores[0] > prev_scores[0] or cur_scores[1] > prev_scores[1]:
                reasons.append("score_progression")

    prev_alive = entry.get("last_alive_counts")
    cur_alive = _alive_counts(frame)
    if isinstance(prev_alive, tuple) and len(prev_alive) >= 2:
        if cur_alive[0] <= prev_alive[0] and cur_alive[1] <= prev_alive[1]:
            if cur_alive[0] < prev_alive[0] or cur_alive[1] < prev_alive[1]:
                reasons.append("alive_count_drop")

    prev_phase = entry.get("last_round_phase")
    prev_planted = entry.get("last_bomb_planted")
    cur_phase, cur_planted = _bomb_phase(frame)
    if prev_planted is False and cur_planted is True:
        reasons.append("bomb_planted_transition")
    if (
        isinstance(prev_phase, str)
        and isinstance(cur_phase, str)
        and prev_phase in _PHASE_RANKS
        and cur_phase in _PHASE_RANKS
        and _PHASE_RANKS[cur_phase] > _PHASE_RANKS[prev_phase]
    ):
        reasons.append("round_phase_progression")

    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    return deduped


class Bo3FreshnessGate:
    """
    Per-(game_number, map_index, round_number) monotonic gating.
    Rejects frames that would rewind time, round clock, or repeat an older alive state.
    """

    def __init__(self) -> None:
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
        Returns (accept, reason, diag). reason is None if accepted; else
        "ts_backwards" | "clock_rewind" | "alive_sig_rewind".
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
        scores = _scores(frame)
        round_phase, bomb_planted = _bomb_phase(frame)

        diag: dict[str, Any] = {
            "key": list(key),
            "ts_ms": ts_ms,
            "round_time_remaining_s": round_time_s,
            "alive_sig": alive_sig,
            "scores": list(scores),
            "round_phase": round_phase,
            "bomb_planted": bomb_planted,
        }

        entry = self._cache.get(key)
        if entry is None:
            self._cache[key] = {
                "last_ts_ms": ts_ms,
                "last_round_time_remaining_s": round_time_s,
                "alive_sig_ts": {alive_sig: ts_ms} if (ts_ms is not None and alive_sig) else {},
                "last_alive_counts": _alive_counts(frame),
                "last_scores": scores,
                "last_round_phase": round_phase,
                "last_bomb_planted": bomb_planted,
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
            advancement_reasons = _meaningful_advancement(entry, frame)
            diag["clock_rewind_meaningful_advancement"] = advancement_reasons
            if not advancement_reasons:
                diag["reason"] = "clock_rewind"
                return (False, "clock_rewind", diag)

        if ts_ms is not None and alive_sig and alive_sig in alive_sig_ts and ts_ms < alive_sig_ts[alive_sig]:
            diag["reason"] = "alive_sig_rewind"
            return (False, "alive_sig_rewind", diag)

        if ts_ms is not None:
            entry["last_ts_ms"] = ts_ms
        if round_time_s is not None:
            entry["last_round_time_remaining_s"] = round_time_s
        if ts_ms is not None and alive_sig:
            if "alive_sig_ts" not in entry:
                entry["alive_sig_ts"] = {}
            entry["alive_sig_ts"][alive_sig] = ts_ms
        entry["last_alive_counts"] = _alive_counts(frame)
        entry["last_scores"] = scores
        entry["last_round_phase"] = round_phase
        entry["last_bomb_planted"] = bomb_planted
        diag["reason"] = None
        return (True, None, diag)

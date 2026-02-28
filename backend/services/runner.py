"""
Async Runner: loop at poll_interval_s; BO3 live, REPLAY from JSONL, or dummy.
Market: poll Kalshi when enabled, push to delay buffer, attach market_mid to points.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from engine.models import Config, Derived, Frame, HistoryPoint, State

from backend.services.trade_episodes import TradeEpisodeManager
from backend.store.memory_store import _history_point_to_wire

logger = logging.getLogger(__name__)

# Raw BO3 snapshot recording (FastAPI runner); replay input no longer depends on legacy poller bo3_pulls.jsonl
_BO3_RAW_RECORD_ENABLED = os.environ.get("BO3_RAW_RECORD_ENABLED", "true").strip().lower() in ("1", "true", "yes")
_BO3_RAW_RECORD_DIR = (os.environ.get("BO3_RAW_RECORD_DIR", "logs") or "logs").strip()
_BO3_RAW_RECORD_PER_MATCH = os.environ.get("BO3_RAW_RECORD_PER_MATCH", "true").strip().lower() in ("1", "true", "yes")


def _minimal_explain(
    phase: str,
    rail_low: float,
    rail_high: float,
    p_hat: float,
    clamp_reason: str = "no_compute",
) -> dict[str, Any]:
    """Placeholder explain for non-compute paths so wire/history always has explain (no thin-wire-only points)."""
    cw = rail_high - rail_low if rail_high >= rail_low else 0.0
    return {
        "phase": phase,
        "p_base_map": None,
        "p_base_series": None,
        "midround_weight": 0.0,
        "q_intra_total": None,
        "q_terms": {},
        "micro_adj": {"alive_adj": 0.0, "hp_adj": 0.0, "econ_adj": 0.0},
        "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": cw},
        "final": {"p_hat_final": p_hat, "clamp_reason": clamp_reason},
    }


def _bo3_raw_record_signature(payload: dict, match_id: int) -> tuple[Any, ...] | None:
    """Build dedupe signature from key fields. Returns None if payload invalid."""
    if not isinstance(payload, dict):
        return None
    t1 = payload.get("team_one") if isinstance(payload.get("team_one"), dict) else {}
    t2 = payload.get("team_two") if isinstance(payload.get("team_two"), dict) else {}
    try:
        s1 = int(t1.get("score", 0) or 0)
        s2 = int(t2.get("score", 0) or 0)
        ms1 = int(t1.get("match_score", 0) or 0)
        ms2 = int(t2.get("match_score", 0) or 0)
    except (TypeError, ValueError):
        return None
    gn = payload.get("game_number")
    rn = payload.get("round_number")
    phase = payload.get("round_phase") or payload.get("phase")
    try:
        gn = int(gn) if gn is not None else None
        rn = int(rn) if rn is not None else None
    except (TypeError, ValueError):
        gn = None
        rn = None
    return (match_id, gn, rn, phase, s1, s2, ms1, ms2)


if TYPE_CHECKING:
    from backend.services.broadcaster import Broadcaster
    from backend.store.memory_store import MemoryStore


def _last_derived_values(cur: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """
    Extract p_hat, bound_low, bound_high, rail_low, rail_high, kappa from cur["derived"].
    Used on failure/non-live paths so we never overwrite with zeros.
    Fallbacks when missing: (0.5, 0.01, 0.99, 0.01, 0.99, 0) to avoid zero corridors.
    """
    d = cur.get("derived") or {}
    if not isinstance(d, dict):
        d = {}
    def _f(key: str, default: float) -> float:
        v = d.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return (
        _f("p_hat", 0.5),
        _f("bound_low", 0.01),
        _f("bound_high", 0.99),
        _f("rail_low", 0.01),
        _f("rail_high", 0.99),
        _f("kappa", 0.0),
    )


def _is_raw_bo3_snapshot(payload: Any) -> bool:
    """True if payload looks like a raw BO3 snapshot (team_one/team_two, match_fixture/round_phase etc.)."""
    if not isinstance(payload, dict):
        return False
    t1 = payload.get("team_one")
    t2 = payload.get("team_two")
    return (
        isinstance(t1, dict)
        or isinstance(t2, dict)
    ) and (
        "match_fixture" in payload
        or "round_phase" in payload
        or "phase" in payload
    )


def _infer_round_winner_by_score_delta(
    prev_s1: int | None,
    prev_s2: int | None,
    s1: int,
    s2: int,
    team_one_id: int,
    team_two_id: int,
) -> int:
    """Infer round winner from score delta (winning_team_id is often 0 in BO3 payloads). Returns team id or 0."""
    if prev_s1 is not None and s1 > prev_s1:
        return team_one_id
    if prev_s2 is not None and s2 > prev_s2:
        return team_two_id
    return 0


def _is_point_like_payload(payload: Any) -> bool:
    """True if payload looks like an already-processed point (t, p, lo, hi, rail_low/rail_high, seg; lacks raw keys)."""
    if not isinstance(payload, dict):
        return False
    has_point_keys = (
        "t" in payload
        and "p" in payload
        and ("lo" in payload or "series_low" in payload)
        and ("hi" in payload or "series_high" in payload)
    )
    lacks_raw = not _is_raw_bo3_snapshot(payload)
    return bool(has_point_keys and lacks_raw)


def _bo3_extract_snapshot_ts(snap: dict[str, Any] | None) -> Any:
    """Extract best ts field for change detection (updated_at/created_at/sent_time/ts)."""
    if not isinstance(snap, dict):
        return None
    return snap.get("updated_at") or snap.get("created_at") or snap.get("sent_time") or snap.get("ts")


def _bo3_snapshot_status(
    snap: dict[str, Any] | None,
    frame: Frame,
    last_snapshot_ts: Any,
    last_scores: tuple[int, int] | None,
    same_snapshot_polls: int,
) -> tuple[str, str | None, Any, bool]:
    """
    Compute app35-style bo3_snapshot_status and bo3_feed_error.
    Returns (status, feed_error, snapshot_ts, is_fresh).
    status: "live" | "stale" | "invalid_clock" | "empty"
    is_fresh: True if snapshot_ts advanced or scores changed (caller resets same_snapshot_polls).
    """
    snapshot_ts = None
    if isinstance(snap, dict):
        snapshot_ts = snap.get("created_at") or snap.get("updated_at") or snap.get("sent_time") or snap.get("ts")

    if not snap or (isinstance(snap, dict) and not snap.get("team_one") and not snap.get("team_two")):
        return ("empty", "missing snapshot or essential keys (team_one/team_two)", snapshot_ts, False)

    from engine.normalize.time_norm import normalize_round_time
    rtr_raw = snap.get("round_time_remaining") if isinstance(snap, dict) else None
    if rtr_raw is not None:
        rtr_norm = normalize_round_time(rtr_raw)
        if rtr_norm.get("invalid_reason") or rtr_norm.get("seconds") is None:
            return (
                "invalid_clock",
                rtr_norm.get("invalid_reason") or "round_time_remaining out of range",
                snapshot_ts,
                False,
            )

    scores = getattr(frame, "scores", (0, 0))
    ts_advanced = snapshot_ts is not None and snapshot_ts != last_snapshot_ts
    scores_changed = last_scores is not None and scores != last_scores
    is_fresh = ts_advanced or scores_changed
    if is_fresh:
        return ("live", None, snapshot_ts, True)
    new_count = same_snapshot_polls + 1
    if new_count > 3:
        return ("stale", f"snapshot unchanged for {new_count} polls", snapshot_ts, False)
    return ("live", None, snapshot_ts, False)


# BO3 health: label-only observability (no gating).
BO3_STALE_THRESHOLD_SEC = 30
BO3_BUFFER_GOOD_AGE_SEC = 20  # GOOD if buffer_age_s <= this
BO3_FETCH_RETRY_DELAYS = (0.5, 1.0)  # seconds between retries (2 retries after first attempt = 3 total)
BO3_PAUSED_PHASES = frozenset({
    "TIMEOUT", "TECH_TIMEOUT", "PAUSED", "HALFTIME", "INTERMISSION",
    "POSTGAME", "MAP_END", "WARMUP", "FREEZETIME",
})


def _bo3_health(
    snap: dict[str, Any] | None,
    frame: Frame | None,
    last_change_epoch: float | None,
    now: float,
) -> tuple[str, str | None, float | None]:
    """
    Compute BO3 health label: GOOD | PAUSED | STALE | ERROR (caller sets ERROR on exception).
    Returns (bo3_health, bo3_health_reason, bo3_health_age_s).
    """
    round_phase = None
    if isinstance(frame, Frame) and isinstance(getattr(frame, "bomb_phase_time_remaining", None), dict):
        round_phase = frame.bomb_phase_time_remaining.get("round_phase")
    if round_phase is None and isinstance(snap, dict):
        round_phase = snap.get("round_phase") or snap.get("phase")
    phase_str = str(round_phase).strip().upper() if round_phase is not None else ""
    if phase_str and phase_str in BO3_PAUSED_PHASES:
        return ("PAUSED", phase_str, None)
    if last_change_epoch is None:
        return ("GOOD", None, None)
    age = now - last_change_epoch
    if age > BO3_STALE_THRESHOLD_SEC:
        return ("STALE", f"no change {int(age)}s", age)
    return ("GOOD", None, None)


def compute_corridor_invariants(
    *,
    series_low: float,
    series_high: float,
    map_low: float,
    map_high: float,
    p_hat: float,
    eps: float = 1e-9,
) -> dict[str, Any]:
    """Return non-fatal invariant flags for corridor ordering and p_hat inclusion."""
    order_ok = (series_low - eps) <= map_low <= map_high <= (series_high + eps)
    p_in_map_ok = (map_low - eps) <= p_hat <= (map_high + eps)
    violations: list[str] = []
    if not order_ok:
        violations.append("series_map_order")
    if not p_in_map_ok:
        violations.append("p_hat_outside_map")
    return {
        "invariant_series_map_order_ok": order_ok,
        "invariant_p_hat_in_map_ok": p_in_map_ok,
        "invariant_violations": violations,
    }


def compute_breach_flags(
    market_mid: float | None,
    series_low: float,
    series_high: float,
    map_low: float,
    map_high: float,
) -> tuple[bool, bool, bool, bool, float | None, str | None]:
    """
    Compute breach flags: market above/below map and series corridors.
    Returns (breach_map_hi, breach_map_lo, breach_series_hi, breach_series_lo, breach_mag, breach_type).
    breach_type and breach_mag are the single primary breach (max magnitude); None when no breach.
    """
    breach_map_hi = breach_map_lo = breach_series_hi = breach_series_lo = False
    mag_map_hi = mag_map_lo = mag_series_hi = mag_series_lo = -1.0
    if market_mid is None:
        return (False, False, False, False, None, None)
    if market_mid > map_high:
        breach_map_hi = True
        mag_map_hi = market_mid - map_high
    if market_mid < map_low:
        breach_map_lo = True
        mag_map_lo = map_low - market_mid
    if market_mid > series_high:
        breach_series_hi = True
        mag_series_hi = market_mid - series_high
    if market_mid < series_low:
        breach_series_lo = True
        mag_series_lo = series_low - market_mid
    best_mag = -1.0
    best_type: str | None = None
    for name, mag in [("map_hi", mag_map_hi), ("map_lo", mag_map_lo), ("series_hi", mag_series_hi), ("series_lo", mag_series_lo)]:
        if mag >= 0 and mag > best_mag:
            best_mag = mag
            best_type = name
    breach_mag = best_mag if best_mag >= 0 else None
    return (breach_map_hi, breach_map_lo, breach_series_hi, breach_series_lo, breach_mag, best_type)


class Runner:
    """Runs tick loop: BO3 snapshot, REPLAY from JSONL, or dummy; append_point, broadcast."""

    def __init__(self, store: MemoryStore, broadcaster: Broadcaster) -> None:
        self._store = store
        self._broadcaster = broadcaster
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Replay cache (lazy-loaded when source=REPLAY)
        self._replay_payloads: list[dict[str, Any]] = []
        self._replay_index: int = 0
        self._replay_path: str | None = None
        self._replay_match_id: int | None = None
        self._replay_format: str | None = None  # "raw" | "point" from first payload
        self._replay_skipped_point_like_count: int = 0  # warning counter when in raw mode
        self._dummy_snapshot_sent = False
        # Market delay buffer + poll throttle
        from backend.services.market_buffer import MarketDelayBuffer
        self._market_buffer = MarketDelayBuffer(maxlen=5000)
        self._last_market_poll_ts: float = 0.0
        self._last_market_error: str | None = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="runner_market")
        # BO3 liveness diagnostics
        self._bo3_success_counter: int = 0
        self._bo3_last_success_epoch: float | None = None
        self._bo3_last_snapshot_ts: Any = None
        self._bo3_last_scores: tuple[int, int] | None = None
        self._bo3_same_snapshot_polls: int = 0
        self._bo3_last_change_epoch: float | None = None  # updated when snapshot_ts or scores change
        self._bo3_last_raw_snapshot: dict | None = None  # last raw BO3 snapshot dict (for debug dump)
        self._last_breach_type: str | None = None
        # BO3 in-memory snapshot buffer (writer updates, consumer reads)
        self._bo3_buf_raw: dict | None = None
        self._bo3_buf_snapshot_ts: str | None = None
        self._bo3_buf_last_success_epoch: float | None = None
        self._bo3_buf_last_attempt_epoch: float | None = None
        self._bo3_buf_last_error: str | None = None
        self._bo3_buf_consecutive_failures: int = 0
        self._bo3_buf_match_id: int | None = None  # clear buffer when match_id changes
        # Outcome label events: avoid duplicate round_result / segment_result emits
        self._bo3_last_seen_round_number: int | None = None  # updated every tick
        self._bo3_last_emitted_round_number: int | None = None
        self._bo3_last_emitted_round_winner_team_id: int | None = None
        self._bo3_last_seen_segment_id_for_result: int | None = None
        self._bo3_last_seen_map_winner_team_id: int | None = None
        self._bo3_last_seen_scores: tuple[int, int] | None = None  # (team_one_score, team_two_score) for inferring winner when winning_team_id==0
        self._bo3_last_seen_score_team_one: int | None = None
        self._bo3_last_seen_score_team_two: int | None = None
        self._bo3_last_seen_game_number: int | None = None  # for segment_result: emit only on map transition
        self._bo3_last_seen_match_score_team_one: int | None = None
        self._bo3_last_seen_match_score_team_two: int | None = None
        # Per-game match_score for credible segment_result emission (map end = +1 to one team only)
        self._bo3_last_seen_match_score_by_game: dict[int, tuple[int | None, int | None]] = {}
        # Raw BO3 snapshot recording dedupe: last signature per match to skip duplicate lines
        self._bo3_raw_last_sig_by_match: dict[int, tuple[Any, ...]] = {}
        # ML-ready series-line-dislocation episode logger (paper only; emits setup_trigger, episode_start/end/outcome)
        self._trade_episode_manager = TradeEpisodeManager()

    def _reset_outcome_trackers(self) -> None:
        """Reset dedupe trackers for outcome event emission."""
        self._bo3_last_seen_round_number = None
        self._bo3_last_emitted_round_number = None
        self._bo3_last_emitted_round_winner_team_id = None
        self._bo3_last_seen_segment_id_for_result = None
        self._bo3_last_seen_map_winner_team_id = None
        self._bo3_last_seen_scores = None
        self._bo3_last_seen_score_team_one = None
        self._bo3_last_seen_score_team_two = None
        self._bo3_last_seen_game_number = None
        self._bo3_last_seen_match_score_team_one = None
        self._bo3_last_seen_match_score_team_two = None
        self._bo3_last_seen_match_score_by_game = {}

    def get_replay_progress(self) -> dict[str, int] | None:
        """Return {index, total} when replay is active and list is loaded."""
        if not self._replay_payloads:
            return None
        return {"index": self._replay_index, "total": len(self._replay_payloads)}

    async def _maybe_poll_market(self, config: Config) -> None:
        """If market_enabled and kalshi_ticker and poll interval elapsed, fetch and push to buffer."""
        if not getattr(config, "market_enabled", True):
            return
        ticker = getattr(config, "kalshi_ticker", None) or ""
        if not ticker or not ticker.strip():
            return
        poll_sec = max(1, int(getattr(config, "market_poll_sec", 5)))
        now = time.time()
        if now - self._last_market_poll_ts < poll_sec:
            return
        self._last_market_poll_ts = now
        ticker = ticker.strip()

        def _fetch() -> None:
            from engine.market.kalshi_client import fetch_kalshi_bid_ask
            bid, ask, mid, ts_epoch = fetch_kalshi_bid_ask(ticker)
            self._market_buffer.push({
                "ts_epoch": ts_epoch,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "ticker": ticker,
            })
            self._last_market_error = None

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, _fetch)
        except Exception as e:
            self._last_market_error = str(e)

    def _get_market_for_point(self, config: Config) -> tuple[float | None, dict[str, Any]]:
        """Return (market_mid, extra_debug). market_mid from delayed buffer; extra_debug has market_error if set."""
        out_debug: dict[str, Any] = {}
        if self._last_market_error:
            out_debug["market_error"] = self._last_market_error
        if not getattr(config, "market_enabled", True) or not getattr(config, "kalshi_ticker", None):
            return (None, out_debug)
        delay_sec = max(0, int(getattr(config, "market_delay_sec", 120)))
        snap = self._market_buffer.get_delayed(delay_sec)
        if snap is None:
            return (None, out_debug)
        mid = snap.get("mid")
        if mid is not None:
            out_debug["market_bid"] = snap.get("bid")
            out_debug["market_ask"] = snap.get("ask")
        return (float(mid) if mid is not None else None, out_debug)

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()

    async def _broadcast_point(self, point: HistoryPoint) -> None:
        """Broadcast chart point message and current frame so HUD can update."""
        wire = _history_point_to_wire(point)
        await self._broadcaster.broadcast({"type": "point", "point": wire})
        await self._broadcast_frame()

    async def _broadcast_frame(self) -> None:
        """Broadcast HUD-only frame message with the latest last_frame (including players)."""
        state = await self._store.get_state()
        last_frame = getattr(state, "last_frame", None)
        if last_frame is None:
            return
        if isinstance(last_frame, dict):
            frame_dict = last_frame
        elif hasattr(last_frame, "__dataclass_fields__"):
            try:
                frame_dict = asdict(last_frame)
            except (TypeError, ValueError):
                frame_dict = {}
        else:
            frame_dict = {}
        if frame_dict:
            await self._broadcaster.broadcast({"type": "frame", "frame": frame_dict})

    async def _bo3_fetch_into_buffer(self, match_id: int) -> None:
        """
        Writer: attempt fetch up to 3 times with short delays; update buffer only.
        On success: set _bo3_buf_raw, _bo3_buf_snapshot_ts, _bo3_buf_last_success_epoch, clear error/failures.
        On failure: keep _bo3_buf_raw, set _bo3_buf_last_error, increment _bo3_buf_consecutive_failures.
        Clear buffer when match_id changes.
        """
        mid = int(match_id)
        now = time.time()
        self._bo3_buf_last_attempt_epoch = now
        if mid != self._bo3_buf_match_id:
            self._bo3_buf_raw = None
            self._bo3_buf_snapshot_ts = None
            self._bo3_buf_last_success_epoch = None
            self._bo3_buf_last_error = None
            self._bo3_buf_consecutive_failures = 0
            self._bo3_buf_match_id = mid
            self._reset_outcome_trackers()
            self._trade_episode_manager = TradeEpisodeManager()
            self._bo3_raw_last_sig_by_match.clear()
        try:
            from engine.ingest.bo3_client import get_snapshot
        except ImportError:
            self._bo3_buf_last_error = "bo3_client not available"
            self._bo3_buf_consecutive_failures += 1
            return
        last_err: str | None = None
        for attempt in range(3):
            try:
                snap = await get_snapshot(mid)
                if snap and isinstance(snap, dict) and (snap.get("team_one") or snap.get("team_two")):
                    self._bo3_buf_raw = snap
                    self._bo3_buf_snapshot_ts = _bo3_extract_snapshot_ts(snap)
                    self._bo3_buf_last_success_epoch = time.time()
                    self._bo3_buf_last_error = None
                    self._bo3_buf_consecutive_failures = 0
                    return
                last_err = "snapshot empty or missing team keys"
            except Exception as e:
                last_err = str(e)
            if attempt < 2:
                delay = BO3_FETCH_RETRY_DELAYS[attempt]
                await asyncio.sleep(delay)
        self._bo3_buf_last_error = last_err
        self._bo3_buf_consecutive_failures += 1

    async def _maybe_emit_outcome_events_from_bo3_payload(
        self,
        *,
        raw: dict[str, Any],
        config: Config,
        new_state: State,
        t: float,
        p_hat: float,
        bound_low: float,
        bound_high: float,
        rail_low: float,
        rail_high: float,
        market_mid: float | None,
        dbg: dict[str, Any],
        team_a_is_team_one: bool,
        match_id_used: int | None,
    ) -> None:
        """
        Emit outcome label events from a BO3 payload: round_result and segment_result.
        Appends+broadcasts event HistoryPoints before the main point. Dedupes via runner trackers.
        """
        if not isinstance(raw, dict):
            return
        t1 = raw.get("team_one") or {}
        t2 = raw.get("team_two") or {}
        team_one_id = int(t1.get("id", 0) or 0)
        team_two_id = int(t2.get("id", 0) or 0)
        team_a_id = team_one_id if team_a_is_team_one else team_two_id
        rtr = raw.get("round_time_remaining")
        try:
            rtr = int(rtr) if rtr is not None else None
        except (TypeError, ValueError):
            rtr = None
        rp = raw.get("round_phase") or raw.get("phase")
        phase_upper = str(rp).strip().upper() if rp is not None else ""
        winning_team_id = raw.get("winning_team_id")
        try:
            winning_team_id = int(winning_team_id) if winning_team_id is not None else 0
        except (TypeError, ValueError):
            winning_team_id = 0
        rn = raw.get("round_number")
        try:
            rn = int(rn) if rn is not None else None
        except (TypeError, ValueError):
            rn = None
        s1 = int(t1.get("score", 0) or 0)
        s2 = int(t2.get("score", 0) or 0)
        last_s1 = self._bo3_last_seen_score_team_one
        last_s2 = self._bo3_last_seen_score_team_two

        async def maybe_emit_round_result(round_to_label: int, winner_team_id: int) -> None:
            if winner_team_id == 0:
                return
            if (
                self._bo3_last_emitted_round_number == round_to_label
                and self._bo3_last_emitted_round_winner_team_id == winner_team_id
            ):
                return
            round_event = {
                "event_type": "round_result",
                "round_number": round_to_label,
                "round_winner_team_id": winner_team_id,
                "round_winner_is_team_a": bool(team_a_id and winner_team_id == team_a_id),
            }
            round_point = HistoryPoint(
                time=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                segment_id=new_state.segment_id,
                explain=None,
                event=round_event,
            )
            derived_evt = Derived(
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                kappa=0.0,
                debug={**dbg, "event": round_event, "match_id_used": match_id_used},
            )
            await self._store.append_point(round_point, new_state, derived_evt)
            await self._broadcast_point(round_point)
            self._bo3_last_emitted_round_number = round_to_label
            self._bo3_last_emitted_round_winner_team_id = winner_team_id

        # Infer winner by score delta (winning_team_id is always 0 in our BO3 replay payloads)
        winner_from_delta = _infer_round_winner_by_score_delta(
            last_s1, last_s2, s1, s2, team_one_id, team_two_id
        )

        # A) phase == "FINISHED": label the round that just finished (rn)
        if phase_upper == "FINISHED" and rn is not None:
            winner = winning_team_id if winning_team_id != 0 else winner_from_delta
            await maybe_emit_round_result(rn, winner)

        # B) rn advanced: previous round ended when round number increased
        last_rn = self._bo3_last_seen_round_number
        if last_rn is not None and rn is not None and rn > last_rn:
            prev_round = last_rn
            winner = winning_team_id if winning_team_id != 0 else winner_from_delta
            await maybe_emit_round_result(prev_round, winner)

        # Update last-seen round and scores every tick
        self._bo3_last_seen_round_number = rn
        self._bo3_last_seen_score_team_one = s1
        self._bo3_last_seen_score_team_two = s2
        self._bo3_last_seen_scores = (s1, s2)

        # --- segment_result: emit on credible match_score increment only (+1 for one team, 0 for other) ---
        mf = raw.get("match_fixture") or {}
        game_number_raw = raw.get("game_number")
        if game_number_raw is None:
            pass
        else:
            try:
                game_number = int(game_number_raw)
            except (TypeError, ValueError):
                game_number = 1
            map_index = game_number - 1

            # match_score from team_one/team_two with match_fixture fallback
            ms1_raw = t1.get("match_score")
            ms2_raw = t2.get("match_score")
            if ms1_raw is None:
                ms1_raw = mf.get("team_one_score")
            if ms2_raw is None:
                ms2_raw = mf.get("team_two_score")
            try:
                ms1 = int(ms1_raw) if ms1_raw is not None else 0
            except (TypeError, ValueError):
                ms1 = 0
            try:
                ms2 = int(ms2_raw) if ms2_raw is not None else 0
            except (TypeError, ValueError):
                ms2 = 0

            prev = self._bo3_last_seen_match_score_by_game.get(game_number)
            if prev is not None:
                prev_ms1, prev_ms2 = prev[0], prev[1]
                d1 = ms1 - (prev_ms1 if prev_ms1 is not None else 0)
                d2 = ms2 - (prev_ms2 if prev_ms2 is not None else 0)
                # Credible map end: exactly one team gained one map (no decreases/resets)
                credible = (d1 == 1 and d2 == 0) or (d1 == 0 and d2 == 1)
                if credible:
                    winner_team_id = team_one_id if d1 == 1 else team_two_id
                    finished_map_index = map_index
                    # Dedupe on (finished_map_index, winner_team_id)
                    if (
                        self._bo3_last_seen_segment_id_for_result != finished_map_index
                        or self._bo3_last_seen_map_winner_team_id != winner_team_id
                    ):
                        scores = (
                            getattr(new_state.last_frame, "scores", (0, 0))
                            if new_state.last_frame
                            else (0, 0)
                        )
                        final_rounds_a = int(scores[0]) if scores and len(scores) > 0 else 0
                        final_rounds_b = int(scores[1]) if scores and len(scores) > 1 else 0
                        segment_event = {
                            "event_type": "segment_result",
                            "segment_id": finished_map_index,
                            "map_index": finished_map_index,
                            "map_winner_team_id": winner_team_id,
                            "map_winner_is_team_a": bool(team_a_id and winner_team_id == team_a_id),
                            "final_rounds_a": final_rounds_a,
                            "final_rounds_b": final_rounds_b,
                        }
                        segment_point = HistoryPoint(
                            time=t,
                            p_hat=p_hat,
                            bound_low=bound_low,
                            bound_high=bound_high,
                            rail_low=rail_low,
                            rail_high=rail_high,
                            market_mid=market_mid,
                            segment_id=finished_map_index,
                            explain=None,
                            event=segment_event,
                        )
                        derived_seg = Derived(
                            p_hat=p_hat,
                            bound_low=bound_low,
                            bound_high=bound_high,
                            rail_low=rail_low,
                            rail_high=rail_high,
                            kappa=0.0,
                            debug={**dbg, "event": segment_event, "match_id_used": match_id_used},
                        )
                        await self._store.append_point(segment_point, new_state, derived_seg)
                        await self._broadcast_point(segment_point)
                        self._bo3_last_seen_segment_id_for_result = finished_map_index
                        self._bo3_last_seen_map_winner_team_id = winner_team_id

            # Always update stored match_score for this game (overwrite on resets/decreases)
            self._bo3_last_seen_match_score_by_game[game_number] = (ms1, ms2)
            self._bo3_last_seen_game_number = game_number
            self._bo3_last_seen_match_score_team_one = ms1
            self._bo3_last_seen_match_score_team_two = ms2

    def _bo3_buffer_debug(self, now: float) -> dict[str, Any]:
        """Build buffer observability dict for derived.debug."""
        age = None
        if self._bo3_buf_last_success_epoch is not None:
            age = now - self._bo3_buf_last_success_epoch
        return {
            "bo3_buffer_has_snapshot": self._bo3_buf_raw is not None,
            "bo3_buffer_age_s": age,
            "bo3_buffer_last_error": self._bo3_buf_last_error,
            "bo3_buffer_consecutive_failures": self._bo3_buf_consecutive_failures,
            "bo3_buffer_snapshot_ts": self._bo3_buf_snapshot_ts,
            "bo3_buffer_last_success_epoch": self._bo3_buf_last_success_epoch,
        }

    def _bo3_health_from_buffer(self, frame: Frame | None, now: float) -> tuple[str, str | None, float | None]:
        """
        Health from buffer age + phase: GOOD if age <= 20s, STALE if > 20, PAUSED from phase, ERROR only if no snapshot and last_error set.
        """
        if frame is not None:
            round_phase = None
            if isinstance(getattr(frame, "bomb_phase_time_remaining", None), dict):
                round_phase = frame.bomb_phase_time_remaining.get("round_phase")
            if round_phase is not None:
                phase_str = str(round_phase).strip().upper()
                if phase_str in BO3_PAUSED_PHASES:
                    return ("PAUSED", phase_str, None)
        if self._bo3_buf_raw is None and self._bo3_buf_last_error is not None:
            return ("ERROR", self._bo3_buf_last_error, None)
        if self._bo3_buf_last_success_epoch is None:
            return ("GOOD", None, None)
        age = now - self._bo3_buf_last_success_epoch
        if age <= BO3_BUFFER_GOOD_AGE_SEC:
            return ("GOOD", None, None)
        return ("STALE", f"buffer age {int(age)}s", age)

    def _maybe_record_raw_bo3_snapshot(
        self,
        payload: dict[str, Any],
        match_id: int,
        team_a_is_team_one: bool | None,
    ) -> None:
        """
        If raw BO3 recording is enabled, validate payload, dedupe by signature, and append one line
        to logs/bo3_raw_match_<match_id>.jsonl (or logs/bo3_raw.jsonl if per-match disabled).
        Does not change engine computations; recording only.
        """
        if not _BO3_RAW_RECORD_ENABLED:
            return
        if not isinstance(payload, dict):
            return
        t1 = payload.get("team_one")
        t2 = payload.get("team_two")
        if not isinstance(t1, dict) or not isinstance(t2, dict):
            return
        # Optional: game_number and round_number present; no strict requirement
        sig = _bo3_raw_record_signature(payload, match_id)
        if sig is None:
            return
        last_sig = self._bo3_raw_last_sig_by_match.get(match_id)
        if last_sig is not None and last_sig == sig:
            return
        self._bo3_raw_last_sig_by_match[match_id] = sig

        rec = {
            "schema": "bo3_raw_snapshot_v1",
            "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "match_id": match_id,
            "team_a_is_team_one": team_a_is_team_one,
            "payload": payload,
            "source": "BO3",
            "ok": True,
        }
        dirpath = _BO3_RAW_RECORD_DIR
        if _BO3_RAW_RECORD_PER_MATCH:
            filename = f"bo3_raw_match_{match_id}.jsonl"
        else:
            filename = "bo3_raw.jsonl"
        filepath = os.path.join(dirpath, filename)
        try:
            os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("BO3 raw record write failed %s: %s", filepath, e)

    async def _tick_bo3(self, config: Config) -> bool:
        """If source=BO3 and match_id set: fetch snapshot, normalize, append_point, broadcast. Return True if did BO3."""
        src = getattr(config, "source", None)
        match_id = getattr(config, "match_id", None)
        if src != "BO3" or match_id is None:
            return False
        mid = int(match_id)
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        try:
            from engine.ingest.bo3_client import get_snapshot
            from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
        except ImportError:
            return False

        await self._bo3_fetch_into_buffer(mid)
        snap = self._bo3_buf_raw
        now = time.time()
        if snap is None:
            cur = await self._store.get_current()
            state = await self._store.get_state()
            if not isinstance(cur, dict):
                cur = {}
            p_hat, bound_low, bound_high, rail_low, rail_high, kappa = _last_derived_values(cur)
            cur_derived = cur.get("derived") or {}
            new_debug = dict(cur_derived.get("debug") or {}) if isinstance(cur_derived, dict) else {}
            new_debug["bo3_fetch_ok"] = False
            new_debug["bo3_snapshot_status"] = "empty"
            new_debug["bo3_feed_error"] = self._bo3_buf_last_error
            new_debug["bo3_snapshot_ts"] = None
            new_debug["bo3_match_id_used"] = mid
            new_debug.update(self._bo3_buffer_debug(now))
            health, health_reason, health_age_s = self._bo3_health_from_buffer(None, now)
            new_debug["bo3_health"] = health
            new_debug["bo3_health_reason"] = health_reason
            new_debug["bo3_health_age_s"] = health_age_s
            fail_derived = Derived(
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                kappa=kappa,
                debug=new_debug,
            )
            await self._store.set_current(state, fail_derived)
            return True
        if isinstance(snap, dict):
            self._bo3_last_raw_snapshot = snap
            self._maybe_record_raw_bo3_snapshot(snap, mid, team_a_is_team_one)
        frame = bo3_snapshot_to_frame(snap, team_a_is_team_one=team_a_is_team_one)
        status, feed_error, snapshot_ts, is_fresh = _bo3_snapshot_status(
            snap, frame,
            self._bo3_last_snapshot_ts,
            self._bo3_last_scores,
            self._bo3_same_snapshot_polls,
        )
        if status != "live":
            now = time.time()
            health, health_reason, health_age_s = self._bo3_health_from_buffer(frame, now)
            if status == "invalid_clock":
                health = "PAUSED" if health == "PAUSED" else "GOOD"
                health_reason = "invalid_clock"
            cur = await self._store.get_current()
            state = await self._store.get_state()
            if not isinstance(cur, dict):
                cur = {}
            p_hat, bound_low, bound_high, rail_low, rail_high, kappa = _last_derived_values(cur)
            cur_derived = cur.get("derived") or {}
            new_debug = dict(cur_derived.get("debug") or {}) if isinstance(cur_derived, dict) else {}
            new_debug["bo3_snapshot_status"] = status
            new_debug["bo3_feed_error"] = feed_error
            new_debug["bo3_snapshot_ts"] = snapshot_ts
            new_debug["bo3_match_id_used"] = mid
            new_debug.update(self._bo3_buffer_debug(now))
            new_debug["bo3_health"] = health
            new_debug["bo3_health_reason"] = health_reason
            new_debug["bo3_health_age_s"] = health_age_s
            if status == "invalid_clock":
                new_debug["time_term_used"] = False
            from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
            new_debug["raw"] = build_raw_debug(frame)
            new_debug["fragility"] = compute_fragility_debug(frame)
            fail_derived = Derived(
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                kappa=kappa,
                debug=new_debug,
            )
            market_mid, _ = self._get_market_for_point(config)
            explain_phase = "stale_snapshot" if status == "stale" else ("paused" if status == "invalid_clock" else "no_compute")
            hold_explain = _minimal_explain(
                explain_phase,
                fail_derived.rail_low,
                fail_derived.rail_high,
                fail_derived.p_hat,
                clamp_reason="no_compute",
            )
            hold_point = HistoryPoint(
                time=now,
                p_hat=fail_derived.p_hat,
                bound_low=fail_derived.bound_low,
                bound_high=fail_derived.bound_high,
                rail_low=fail_derived.rail_low,
                rail_high=fail_derived.rail_high,
                market_mid=market_mid,
                segment_id=getattr(state, "segment_id", 0),
                explain=hold_explain,
            )
            await self._store.append_point(hold_point, state, fail_derived)
            await self._broadcast_point(hold_point)
            if status == "stale":
                self._bo3_same_snapshot_polls += 1
            return True

        if is_fresh:
            self._bo3_same_snapshot_polls = 0
            self._bo3_last_snapshot_ts = snapshot_ts
            self._bo3_last_scores = getattr(frame, "scores", (0, 0))
            self._bo3_last_change_epoch = time.time()
        else:
            self._bo3_same_snapshot_polls += 1

        t = time.time()
        health, health_reason, health_age_s = self._bo3_health_from_buffer(frame, t)
        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state
        from engine.diagnostics.inter_map_break import detect_inter_map_break

        old_state = await self._store.get_state()
        new_state = reduce_state(old_state, frame, config)
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}

        # Detect inter-map break (between maps) and keep corridors/p_hat stable.
        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            series_width = bound_high - bound_low
            center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
            # Use a modest map width inside series; avoid collapsing to zero.
            map_width = min(series_width * 0.6, 0.30)
            half = 0.5 * map_width
            rail_low = max(bound_low, min(bound_high, center - half))
            rail_high = max(bound_low, min(bound_high, center + half))
            # Try to reuse last p_hat if available; else midpoint.
            cur = await self._store.get_current()
            d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
            last_p = d.get("p_hat")
            if isinstance(last_p, (int, float)):
                p_hat = float(last_p)
            else:
                p_hat = center
            p_hat = max(rail_low, min(rail_high, p_hat))
            dbg: dict[str, Any] = {
                "inter_map_break": True,
                "inter_map_break_reason": break_reason,
                "p_hat_old": last_p,
                "p_hat_final": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
            }
            dbg.update(bounds_debug)
            cw = rail_high - rail_low if rail_high >= rail_low else 0.0
            dbg["explain"] = {
                "phase": "inter_map_break",
                "p_base_map": None,
                "p_base_series": None,
                "midround_weight": 0.0,
                "q_intra_total": None,
                "q_terms": {},
                "micro_adj": {"alive_adj": 0.0, "hp_adj": 0.0, "econ_adj": 0.0},
                "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": cw},
                "final": {"p_hat_final": p_hat, "clamp_reason": "inter_map_break"},
            }
        else:
            bounds = (bound_low, bound_high)
            rails_result = compute_rails(frame, config, new_state, bounds)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}

        from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
        dbg["raw"] = build_raw_debug(frame)
        dbg["fragility"] = compute_fragility_debug(frame)
        market_mid, market_dbg = self._get_market_for_point(config)
        dbg.update(market_dbg)
        # Breach detection: market vs corridors
        breach_map_hi, breach_map_lo, breach_series_hi, breach_series_lo, breach_mag, breach_type = compute_breach_flags(
            market_mid, bound_low, bound_high, rail_low, rail_high
        )
        dbg["breach_map_hi"] = breach_map_hi
        dbg["breach_map_lo"] = breach_map_lo
        dbg["breach_series_hi"] = breach_series_hi
        dbg["breach_series_lo"] = breach_series_lo
        dbg["breach_mag"] = breach_mag
        inv = compute_corridor_invariants(
            series_low=bound_low,
            series_high=bound_high,
            map_low=rail_low,
            map_high=rail_high,
            p_hat=p_hat,
        )
        dbg.update(inv)
        if inv["invariant_violations"]:
            logger.warning(
                "corridor invariant violation",
                extra={"violations": inv["invariant_violations"], "seg": getattr(new_state, "segment_id", 0)},
            )
        if breach_type is not None and (self._last_breach_type is None or self._last_breach_type != breach_type):
            scores = (0, 0)
            if new_state.last_frame is not None:
                scores = getattr(new_state.last_frame, "scores", (0, 0))
            series_score = getattr(new_state, "last_series_score", None) or (0, 0)
            breach_evt = {
                "ts_epoch": t,
                "match_id": mid,
                "seg": getattr(new_state, "segment_id", 0),
                "scores": list(scores),
                "series_score": list(series_score),
                "map_index": getattr(new_state, "map_index", 0),
                "market_mid": market_mid,
                "p_hat": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
                "breach_type": breach_type,
                "breach_mag": breach_mag,
            }
            await self._store.append_breach_event(breach_evt)
            self._last_breach_type = breach_type
        if breach_type is None:
            self._last_breach_type = None
        # Outcome label events (round_result, segment_result) from raw snapshot — emit before main point, no duplicate spam
        if isinstance(snap, dict):
            await self._maybe_emit_outcome_events_from_bo3_payload(
                raw=snap,
                config=config,
                new_state=new_state,
                t=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                dbg=dbg,
                team_a_is_team_one=team_a_is_team_one,
                match_id_used=mid,
            )
        # ML-ready series-line-dislocation episode logger: emit setup_trigger, episode_start/end/outcome into history
        bid_yes = market_dbg.get("market_bid")
        ask_yes = market_dbg.get("market_ask")
        round_phase_ep: str | None = None
        if hasattr(frame, "bomb_phase_time_remaining") and isinstance(frame.bomb_phase_time_remaining, dict):
            round_phase_ep = frame.bomb_phase_time_remaining.get("round_phase")
        if round_phase_ep is None and isinstance(snap, dict):
            round_phase_ep = snap.get("round_phase") or snap.get("phase")
        game_number_ep = None
        round_number_ep = None
        if isinstance(snap, dict):
            try:
                if snap.get("game_number") is not None:
                    game_number_ep = int(snap["game_number"])
            except (TypeError, ValueError):
                pass
            try:
                if snap.get("round_number") is not None:
                    round_number_ep = int(snap["round_number"])
            except (TypeError, ValueError):
                pass
        mf_ep = (snap.get("match_fixture") or {}) if isinstance(snap, dict) else {}
        game_ended_ep = bool(snap.get("game_ended") or mf_ep.get("game_ended")) if isinstance(snap, dict) else False
        episode_events = self._trade_episode_manager.on_tick(
            t=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            bid_yes=float(bid_yes) if bid_yes is not None else None,
            ask_yes=float(ask_yes) if ask_yes is not None else None,
            round_phase=round_phase_ep,
            game_number=game_number_ep,
            segment_id=new_state.segment_id,
            round_number=round_number_ep,
            explain=dbg.get("explain"),
            game_ended=game_ended_ep,
        )
        for evt in episode_events:
            ep_explain = _minimal_explain(
                "episode_event",
                rail_low,
                rail_high,
                p_hat,
                clamp_reason="setup_logger",
            )
            ep_point = HistoryPoint(
                time=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                segment_id=new_state.segment_id,
                explain=ep_explain,
                event=evt,
            )
            await self._store.append_point(ep_point, new_state, Derived(p_hat=p_hat, bound_low=bound_low, bound_high=bound_high, rail_low=rail_low, rail_high=rail_high, kappa=0.0, debug={**dbg, "event": evt}))
            await self._broadcast_point(ep_point)
        # BO3 liveness diagnostics (success path)
        self._bo3_success_counter += 1
        self._bo3_last_success_epoch = time.time()
        dbg["bo3_fetch_ok"] = True
        dbg["bo3_snapshot_status"] = "live"
        dbg["bo3_feed_error"] = None
        dbg["bo3_match_id_used"] = mid
        dbg["bo3_success_counter"] = self._bo3_success_counter
        dbg["bo3_snapshot_ts"] = self._bo3_buf_snapshot_ts
        dbg.update(self._bo3_buffer_debug(t))
        dbg["bo3_health"] = health
        dbg["bo3_health_reason"] = health_reason
        dbg["bo3_health_age_s"] = health_age_s
        explain = dbg.get("explain")
        if explain is None:
            explain = _minimal_explain(
                "live",
                rail_low,
                rail_high,
                p_hat,
                clamp_reason="no_explain_from_resolve",
            )
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=new_state.segment_id,
            explain=explain,
        )
        derived = Derived(
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug=dbg,
        )
        await self._store.append_point(point, new_state, derived)
        await self._broadcast_point(point)
        return True

    async def _tick_replay_point_passthrough(self, payload: dict[str, Any], config: Config) -> bool:
        """Point replay: append wire point as-is, no normalize/resolve; backfill explain if missing."""
        t = float(payload.get("t", 0) or 0)
        p = float(payload.get("p", 0.5) or 0.5)
        lo = float(payload.get("lo") or payload.get("series_low", 0) or 0)
        hi = float(payload.get("hi") or payload.get("series_high", 1) or 1)
        rail_low = float(payload.get("rail_low") or payload.get("map_low", lo) or lo)
        rail_high = float(payload.get("rail_high") or payload.get("map_high", hi) or hi)
        seg = int(payload.get("seg", 0) or 0)
        m = payload.get("m")
        market_mid = float(m) if m is not None and m != "" else None
        explain = payload.get("explain")
        if explain is None or not isinstance(explain, dict):
            explain = {
                "phase": "replay_passthrough",
                "q_terms": {},
                "micro_adj": {"alive_adj": 0, "hp_adj": 0, "econ_adj": 0},
                "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": rail_high - rail_low},
                "final": {"p_hat_final": p, "clamp_reason": "passthrough"},
            }
        event = payload.get("event") if isinstance(payload.get("event"), dict) else None
        point = HistoryPoint(
            time=t,
            p_hat=p,
            bound_low=lo,
            bound_high=hi,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=seg,
            explain=explain,
            event=event,
        )
        state = await self._store.get_state()
        derived = Derived(
            p_hat=p,
            bound_low=lo,
            bound_high=hi,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug={"explain": explain},
        )
        await self._store.append_point(point, state, derived)
        await self._broadcast_point(point)
        self._replay_index += 1
        return True

    async def _tick_replay(self, config: Config) -> bool:
        """If source=REPLAY: feed next JSONL payload to normalize+compute, append_point, broadcast. Return True if did replay."""
        src = getattr(config, "source", None)
        if src != "REPLAY":
            return False
        path = getattr(config, "replay_path", None) or "logs/bo3_pulls.jsonl"
        match_id = getattr(config, "match_id", None)
        if match_id is not None:
            try:
                match_id = int(match_id)
            except (TypeError, ValueError):
                match_id = None
        replay_loop = getattr(config, "replay_loop", True)
        replay_speed = max(0.1, float(getattr(config, "replay_speed", 1.0)))

        # Lazy-load and cache when path or match_id changes
        if path != self._replay_path or match_id != self._replay_match_id:
            try:
                from engine.replay.bo3_jsonl import load_bo3_jsonl_entries, iter_payloads, load_generic_jsonl
            except ImportError:
                return False
            entries = load_bo3_jsonl_entries(path)
            self._replay_payloads = [p for _, p in iter_payloads(entries, match_id)]
            if not self._replay_payloads:
                self._replay_payloads = load_generic_jsonl(path)
            self._replay_index = 0
            self._replay_path = path
            self._replay_match_id = match_id
            self._replay_skipped_point_like_count = 0
            if self._replay_payloads:
                self._replay_format = "raw" if _is_raw_bo3_snapshot(self._replay_payloads[0]) else "point"
            else:
                self._replay_format = None
            self._reset_outcome_trackers()
            self._trade_episode_manager = TradeEpisodeManager()

        if not self._replay_payloads:
            return True  # no entries; sleep and keep trying

        if self._replay_index >= len(self._replay_payloads):
            # End of replay: loop resets index and bumps segment; else keep last state on screen
            if replay_loop:
                state = await self._store.get_state()
                seg = getattr(state, "segment_id", 0) + 1
                from engine.models import State as StateModel
                loop_state = StateModel(
                    config=config,
                    last_frame=state.last_frame,
                    team_mapping=state.team_mapping,
                    map_index=getattr(state, "map_index", 0),
                    last_total_rounds=getattr(state, "last_total_rounds", 0),
                    segment_id=seg,
                    last_series_score=getattr(state, "last_series_score", None),
                    last_map_index=getattr(state, "last_map_index", None),
                )
                cur = await self._store.get_current()
                if not isinstance(cur, dict):
                    cur = {}
                p_hat, bound_low, bound_high, rail_low, rail_high, kappa = _last_derived_values(cur)
                d = cur.get("derived") or {}
                derived_obj = Derived(
                    p_hat=p_hat,
                    bound_low=bound_low,
                    bound_high=bound_high,
                    rail_low=rail_low,
                    rail_high=rail_high,
                    kappa=kappa,
                    debug=d.get("debug", {}) if isinstance(d, dict) else {},
                )
                loop_explain = _minimal_explain(
                    "replay_loop_boundary",
                    derived_obj.rail_low,
                    derived_obj.rail_high,
                    derived_obj.p_hat,
                    clamp_reason="replay_loop",
                )
                pt = HistoryPoint(
                    time=time.time(),
                    p_hat=derived_obj.p_hat,
                    bound_low=derived_obj.bound_low,
                    bound_high=derived_obj.bound_high,
                    rail_low=derived_obj.rail_low,
                    rail_high=derived_obj.rail_high,
                    market_mid=None,
                    segment_id=seg,
                    explain=loop_explain,
                )
                await self._store.append_point(pt, loop_state, derived_obj)
                await self._broadcast_point(pt)
                self._replay_index = 0
            return True

        payload = self._replay_payloads[self._replay_index]

        # Set replay format from first payload (raw vs point)
        if self._replay_format is None:
            self._replay_format = "raw" if _is_raw_bo3_snapshot(payload) else "point"

        # Raw snapshot replay: skip any point-like lines (mixed input)
        if self._replay_format == "raw" and _is_point_like_payload(payload):
            self._replay_skipped_point_like_count += 1
            logger.warning(
                "replay raw mode: skipping point-like line (mixed format)",
                extra={"index": self._replay_index, "skipped_total": self._replay_skipped_point_like_count},
            )
            self._replay_index += 1
            return True

        # Point replay: passthrough, no normalize/resolve; backfill explain if missing
        if self._replay_format == "point":
            return await self._tick_replay_point_passthrough(payload, config)

        # Raw snapshot replay: full pipeline
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        try:
            from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
        except ImportError:
            return False
        frame = bo3_snapshot_to_frame(payload, team_a_is_team_one=team_a_is_team_one)
        t = time.time()
        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state
        from engine.diagnostics.inter_map_break import detect_inter_map_break

        old_state = await self._store.get_state()
        new_state = reduce_state(old_state, frame, config)
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}

        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            series_width = bound_high - bound_low
            center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
            map_width = min(series_width * 0.6, 0.30)
            half = 0.5 * map_width
            rail_low = max(bound_low, min(bound_high, center - half))
            rail_high = max(bound_low, min(bound_high, center + half))
            cur = await self._store.get_current()
            d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
            last_p = d.get("p_hat")
            if isinstance(last_p, (int, float)):
                p_hat = float(last_p)
            else:
                p_hat = center
            p_hat = max(rail_low, min(rail_high, p_hat))
            dbg: dict[str, Any] = {
                "inter_map_break": True,
                "inter_map_break_reason": break_reason,
                "p_hat_old": last_p,
                "p_hat_final": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
            }
            dbg.update(bounds_debug)
            cw = rail_high - rail_low if rail_high >= rail_low else 0.0
            dbg["explain"] = {
                "phase": "inter_map_break",
                "p_base_map": None,
                "p_base_series": None,
                "midround_weight": 0.0,
                "q_intra_total": None,
                "q_terms": {},
                "micro_adj": {"alive_adj": 0.0, "hp_adj": 0.0, "econ_adj": 0.0},
                "rails": {"rail_low": rail_low, "rail_high": rail_high, "corridor_width": cw},
                "final": {"p_hat_final": p_hat, "clamp_reason": "inter_map_break"},
            }
        else:
            bounds = (bound_low, bound_high)
            rails_result = compute_rails(frame, config, new_state, bounds)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}

        from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
        dbg["raw"] = build_raw_debug(frame)
        dbg["fragility"] = compute_fragility_debug(frame)
        market_mid, market_dbg = self._get_market_for_point(config)
        dbg.update(market_dbg)
        breach_map_hi, breach_map_lo, breach_series_hi, breach_series_lo, breach_mag, breach_type = compute_breach_flags(
            market_mid, bound_low, bound_high, rail_low, rail_high
        )
        dbg["breach_map_hi"] = breach_map_hi
        dbg["breach_map_lo"] = breach_map_lo
        dbg["breach_series_hi"] = breach_series_hi
        dbg["breach_series_lo"] = breach_series_lo
        dbg["breach_mag"] = breach_mag
        inv = compute_corridor_invariants(
            series_low=bound_low,
            series_high=bound_high,
            map_low=rail_low,
            map_high=rail_high,
            p_hat=p_hat,
        )
        dbg.update(inv)
        if inv["invariant_violations"]:
            logger.warning(
                "corridor invariant violation",
                extra={"violations": inv["invariant_violations"], "seg": getattr(new_state, "segment_id", 0)},
            )
        if breach_type is not None and (self._last_breach_type is None or self._last_breach_type != breach_type):
            scores = (0, 0)
            if new_state.last_frame is not None:
                scores = getattr(new_state.last_frame, "scores", (0, 0))
            series_score = getattr(new_state, "last_series_score", None) or (0, 0)
            breach_evt = {
                "ts_epoch": t,
                "match_id": self._replay_match_id,
                "seg": getattr(new_state, "segment_id", 0),
                "scores": list(scores),
                "series_score": list(series_score),
                "map_index": getattr(new_state, "map_index", 0),
                "market_mid": market_mid,
                "p_hat": p_hat,
                "series_low": bound_low,
                "series_high": bound_high,
                "map_low": rail_low,
                "map_high": rail_high,
                "breach_type": breach_type,
                "breach_mag": breach_mag,
            }
            await self._store.append_breach_event(breach_evt)
            self._last_breach_type = breach_type
        if breach_type is None:
            self._last_breach_type = None
        # Outcome label events (round_result, segment_result) from replay payload — same logic as live
        if isinstance(payload, dict):
            await self._maybe_emit_outcome_events_from_bo3_payload(
                raw=payload,
                config=config,
                new_state=new_state,
                t=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                dbg=dbg,
                team_a_is_team_one=team_a_is_team_one,
                match_id_used=self._replay_match_id,
            )
        replay_explain = dbg.get("explain")
        if replay_explain is None:
            replay_explain = _minimal_explain(
                "replay",
                rail_low,
                rail_high,
                p_hat,
                clamp_reason="no_explain_from_resolve",
            )
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=new_state.segment_id,
            explain=replay_explain,
        )
        derived = Derived(
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug=dbg,
        )
        await self._store.append_point(point, new_state, derived)
        await self._broadcast_point(point)
        self._replay_index += 1
        return True

    async def _loop(self) -> None:
        """Tick: get config; REPLAY or BO3 or dummy, append_point, broadcast."""
        while not self._stop.is_set():
            try:
                config = await self._store.get_config()
                interval = max(5.0, float(getattr(config, "poll_interval_s", 5.0)))
                if getattr(config, "source", None) != "DUMMY":
                    self._dummy_snapshot_sent = False
                await self._maybe_poll_market(config)
            except Exception:
                interval = 5.0
            did_replay = await self._tick_replay(config)
            if did_replay:
                speed = max(0.1, float(getattr(config, "replay_speed", 1.0)))
                sleep_interval = interval / speed
            else:
                sleep_interval = interval
                did_bo3 = await self._tick_bo3(config)
                if did_bo3 and getattr(self, "_bo3_buf_consecutive_failures", 0) >= 3:
                    sleep_interval += 5.0  # mild backoff when BO3 fetch keeps failing
                if not did_bo3:
                    src = getattr(config, "source", None)
                    if src == "DUMMY":
                        # Idle: no history append, no point broadcasts. Optionally one snapshot on first entry.
                        if not self._dummy_snapshot_sent:
                            neutral_derived = Derived(
                                p_hat=0.5,
                                bound_low=0.01,
                                bound_high=0.99,
                                rail_low=0.01,
                                rail_high=0.99,
                                kappa=0.0,
                            )
                            neutral_state = State(
                                config=config,
                                last_frame=Frame(timestamp=time.time(), teams=("A", "B"), scores=(0, 0)),
                                map_index=0,
                                last_total_rounds=0,
                            )
                            await self._store.set_current(neutral_state, neutral_derived)
                            current = await self._store.get_current()
                            history = await self._store.get_history(limit=500)
                            await self._broadcaster.broadcast(
                                {"type": "snapshot", "current": current, "history": history}
                            )
                            self._dummy_snapshot_sent = True
                    else:
                        # Legacy fallback when not BO3/REPLAY and source != DUMMY (e.g. BO3 with no match_id)
                        t = time.time()
                        p_hat = 0.5 + 0.4 * math.sin(t * 0.1)
                        p_hat = max(0.1, min(0.9, p_hat))
                        lo = max(0.0, p_hat - 0.1)
                        hi = min(1.0, p_hat + 0.1)
                        market_mid = p_hat - 0.02 * math.sin(t * 0.15)
                        market_mid = max(0.0, min(1.0, market_mid)) if market_mid is not None else None
                        idle_explain = _minimal_explain("idle", lo, hi, p_hat, clamp_reason="no_source")
                        point = HistoryPoint(
                            time=t,
                            p_hat=p_hat,
                            bound_low=lo,
                            bound_high=hi,
                            rail_low=lo,
                            rail_high=hi,
                            market_mid=market_mid,
                            segment_id=0,
                            explain=idle_explain,
                        )
                        state = State(
                            config=config,
                            last_frame=Frame(timestamp=t, teams=("A", "B"), scores=(0, 0)),
                            map_index=0,
                            last_total_rounds=0,
                        )
                        derived = Derived(p_hat=p_hat, bound_low=lo, bound_high=hi, rail_low=lo, rail_high=hi, kappa=0.0)
                        await self._store.append_point(point, state, derived)
                        await self._broadcast_point(point)
            try:
                await asyncio.sleep(sleep_interval)
            except asyncio.CancelledError:
                break
        self._task = None

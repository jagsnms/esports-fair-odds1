"""
Async Runner: loop at poll_interval_s; BO3 live, REPLAY from JSONL, or GRID.
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

from engine.diagnostics.invariants import compute_corridor_invariants
from engine.models import Config, Derived, Frame, HistoryPoint, State

from backend.services.bo3_capture_contract import (
    append_bo3_live_capture_record,
    build_bo3_live_capture_record,
)
from backend.services.trade_episodes import TradeEpisodeManager
from backend.store.memory_store import _history_point_to_wire

logger = logging.getLogger(__name__)

# Raw BO3 snapshot recording (FastAPI runner); replay input no longer depends on legacy poller bo3_pulls.jsonl
_BO3_RAW_RECORD_ENABLED = os.environ.get("BO3_RAW_RECORD_ENABLED", "true").strip().lower() in ("1", "true", "yes")
_BO3_RAW_RECORD_DIR = (os.environ.get("BO3_RAW_RECORD_DIR", "logs") or "logs").strip()
_BO3_RAW_RECORD_PER_MATCH = os.environ.get("BO3_RAW_RECORD_PER_MATCH", "true").strip().lower() in ("1", "true", "yes")
# Monotonic gating: reject out-of-order BO3 frames (time rewind)
_BO3_TICK_MONOTONIC_DEBUG = os.environ.get("BO3_TICK_MONOTONIC_DEBUG", "false").strip().lower() in ("1", "true", "yes")
# Point-emit instrumentation: log each candidate and warn on discontinuities (teams/map/round/ts/corridor jump)
_RUNNER_POINT_EMIT_DEBUG = os.environ.get("RUNNER_POINT_EMIT_DEBUG", "0").strip().lower() in ("1", "true", "yes")
_POINT_JUMP_THRESHOLD = float(os.environ.get("POINT_JUMP_THRESHOLD", "0.25"))
BO3_RATE_DEBUG = os.environ.get("BO3_RATE_DEBUG", "") == "1"

REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE = "reject_point_like"

REPLAY_POINT_POLICY_DECISION_REJECT = "reject"
REPLAY_POINT_POLICY_DECISION_TRANSITION_PASSTHROUGH = "transition_passthrough"

REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY = "POINT_REPLAY_REJECTED_DEFAULT_POLICY"
REPLAY_POINT_REJECT_REASON_UNSUPPORTED_POLICY = "POINT_REPLAY_REJECTED_UNSUPPORTED_POLICY"
REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_MISSING = "POINT_REPLAY_REJECTED_TRANSITION_SUNSET_MISSING"
REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED = "POINT_REPLAY_REJECTED_TRANSITION_SUNSET_EXPIRED"


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


def _inter_map_break_phat_and_dbg(
    bound_low: float,
    bound_high: float,
    break_reason: str,
    last_p: float | None,
    bounds_debug: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Build canonical inter_map_break p_hat and debug payload (Stage 3B). Callers add source-specific keys after."""
    series_width = bound_high - bound_low
    center = max(bound_low, min(bound_high, 0.5 * (bound_low + bound_high)))
    map_width = min(series_width * 0.6, 0.30)
    half = 0.5 * map_width
    rail_low = max(bound_low, min(bound_high, center - half))
    rail_high = max(bound_low, min(bound_high, center + half))
    p_hat = float(last_p) if isinstance(last_p, (int, float)) else center
    p_hat = max(rail_low, min(rail_high, p_hat))
    cw = rail_high - rail_low if rail_high >= rail_low else 0.0
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
    return (p_hat, dbg)


def _compute_dominance_features(frame: Frame | None) -> dict[str, Any]:
    """
    Compute compact round dominance / win-quality features from Frame only.
    No side effects; caller merges into debug (and optionally explain).
    """
    if frame is None:
        return {}

    def _logistic(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    # Alive dominance: difference in alive_counts (Team A - Team B)
    alive = getattr(frame, "alive_counts", None)
    dom_alive = None
    if isinstance(alive, (tuple, list)) and len(alive) >= 2:
        try:
            da = float(alive[0]) if alive[0] is not None else 0.0
            db = float(alive[1]) if alive[1] is not None else 0.0
        except (TypeError, ValueError):
            da = db = 0.0
        # Scale: 2 players ~ strong edge; map to [0,1]
        dom_alive = _logistic((da - db) / 2.0)

    # HP dominance: total HP difference
    hp = getattr(frame, "hp_totals", None)
    dom_hp = None
    if isinstance(hp, (tuple, list)) and len(hp) >= 2:
        try:
            ha = float(hp[0]) if hp[0] is not None else 0.0
            hb_val = hp[1]
            hb = float(hb_val) if hb_val is not None else 0.0
        except (TypeError, ValueError):
            ha = hb = 0.0
        # Scale: 50 HP difference ~ noticeable edge
        dom_hp = _logistic((ha - hb) / 50.0)

    # Loadout dominance: loadout_totals difference
    loadout = getattr(frame, "loadout_totals", None)
    dom_loadout = None
    if isinstance(loadout, (tuple, list)) and len(loadout) >= 2:
        try:
            la = float(loadout[0]) if loadout[0] is not None else 0.0
            lb = float(loadout[1]) if loadout[1] is not None else 0.0
        except (TypeError, ValueError):
            la = lb = 0.0
        # Scale: 5000 value difference ~ strong buy vs weak
        dom_loadout = _logistic((la - lb) / 5000.0)

    # Bomb dominance: planted flag / round time remaining (if normalized bomb_phase_time_remaining is present)
    bomb = getattr(frame, "bomb_phase_time_remaining", None)
    dom_bomb = None
    if isinstance(bomb, dict):
        is_planted = bomb.get("is_bomb_planted")
        # Use normalized round_time_remaining seconds when available
        rtr = bomb.get("round_time_remaining")
        if isinstance(is_planted, bool) and rtr is not None:
            try:
                rtr_s = float(rtr)
            except (TypeError, ValueError):
                rtr_s = None
            if rtr_s is not None:
                # Early plant (more time left) -> higher dominance for T (Team A assumed offense on plant)
                # Map rtr_s (0..40+) into [0,1] with gentle saturation.
                dom_raw = max(0.0, min(1.0, rtr_s / 40.0))
                dom_bomb = 0.5 + 0.5 * dom_raw if is_planted else 0.5 - 0.5 * dom_raw

    # Combine into a single dominance_score in [0,1] (simple weighted average of present components).
    components: list[float] = []
    weights: list[float] = []
    if dom_alive is not None:
        components.append(dom_alive)
        weights.append(1.5)
    if dom_hp is not None:
        components.append(dom_hp)
        weights.append(1.0)
    if dom_loadout is not None:
        components.append(dom_loadout)
        weights.append(1.0)
    if dom_bomb is not None:
        components.append(dom_bomb)
        weights.append(1.0)

    dominance_score = None
    if components and weights:
        w_sum = sum(weights)
        if w_sum > 0:
            dominance_score = sum(c * w for c, w in zip(components, weights)) / w_sum

    return {
        "dominance_alive": dom_alive,
        "dominance_hp": dom_hp,
        "dominance_loadout": dom_loadout,
        "dominance_bomb": dom_bomb,
        "dominance_score": dominance_score,
    }


def _team_identity_for_point(frame: Frame | None, config: Config) -> dict[str, Any]:
    """Build team identity kwargs for HistoryPoint (score_diag_v2 / witness CSV). Canonical: Team A == team_one when team_a_is_team_one True."""
    out: dict[str, Any] = {}
    if config is not None:
        ta = getattr(config, "team_a_is_team_one", None)
        if ta is not None:
            out["team_a_is_team_one"] = bool(ta)
    if frame is None:
        return out
    t1_id = getattr(frame, "team_one_id", None)
    t2_id = getattr(frame, "team_two_id", None)
    if t1_id is not None:
        out["team_one_id"] = int(t1_id)
    if t2_id is not None:
        out["team_two_id"] = int(t2_id)
    for key in ("team_one_provider_id", "team_two_provider_id", "a_side"):
        val = getattr(frame, key, None)
        if val is not None:
            out[key] = str(val)
    return out


def _team_identity_from_cache(entry: dict[str, Any] | None) -> dict[str, Any]:
    """Build HistoryPoint team identity kwargs from a canonical per-map cache entry."""
    if not entry:
        return {}
    out: dict[str, Any] = {}
    for key in ("team_one_id", "team_two_id", "team_one_provider_id", "team_two_provider_id", "team_a_is_team_one", "a_side"):
        val = entry.get(key)
        if val is None:
            continue
        if key == "team_a_is_team_one":
            out[key] = bool(val)
        elif key in ("team_one_id", "team_two_id"):
            try:
                out[key] = int(val)
            except (TypeError, ValueError):
                pass
        else:
            out[key] = str(val)
    return out


def _cached_map_identity_entry(
    session_runtime: Any,
    game_number: int | None,
    map_index: int | None,
) -> dict[str, Any] | None:
    """Return canonical per-session BO3 map identity once raw payload has established it."""
    if session_runtime is None or game_number is None or map_index is None:
        return None
    try:
        key = (int(game_number), int(map_index))
    except (TypeError, ValueError):
        return None
    cache = getattr(session_runtime, "map_identity_cache", None)
    if not isinstance(cache, dict):
        return None
    entry = cache.get(key)
    return entry if isinstance(entry, dict) else None


def _normalize_side_raw(side: Any) -> str | None:
    """Normalize side to T or CT for cache."""
    if side is None:
        return None
    s = str(side).strip().upper()
    return s if s in ("T", "CT") else None


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


def _bo3_raw_record_path(match_id: int) -> str:
    filename = f"bo3_raw_match_{match_id}.jsonl" if _BO3_RAW_RECORD_PER_MATCH else "bo3_raw.jsonl"
    return os.path.join(_BO3_RAW_RECORD_DIR, filename)


def _match_context_diag(ctx: Any, selector_decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build diagnostics dict from MatchContext for debug/dbg. Safe if ctx is None."""
    if ctx is None:
        return {}
    per_source: dict[str, dict[str, Any]] = {}
    for sk, h in getattr(ctx, "per_source_health", {}).items():
        name = getattr(sk, "name", str(sk))
        per_source[name] = {
            "last_ok_ts": getattr(h, "last_ok_ts", None),
            "last_err_ts": getattr(h, "last_err_ts", None),
            "ok_count": getattr(h, "ok_count", 0),
            "err_count": getattr(h, "err_count", 0),
            "last_reason": getattr(h, "last_reason", None),
        }
    active = getattr(ctx, "active_source", None)
    out: dict[str, Any] = {
        "accepted_count": getattr(ctx, "accepted_count", 0),
        "rejected_count": getattr(ctx, "rejected_count", 0),
        "last_reject_reason": getattr(ctx, "last_reject_reason", None),
        "last_accepted_key": ctx.last_accepted_key.to_display() if getattr(ctx, "last_accepted_key", None) else None,
        "per_source_health": per_source,
        "active_source": getattr(active, "name", str(active)) if active is not None else None,
        "last_switch_ts": getattr(ctx, "last_switch_ts", None),
        "last_switch_reason": getattr(ctx, "last_switch_reason", None),
        "last_env": getattr(ctx, "last_accepted_env_summary", None),
        "last_drive_deny_reason": getattr(ctx, "last_drive_deny_reason", None),
    }
    if selector_decision is not None:
        out["selector_decision"] = selector_decision
    return out


def _bo3_snapshot_diag(snap: dict[str, Any] | None) -> dict[str, Any]:
    """Extract compact source identifiers for BO3 pipeline diagnostics."""
    if not isinstance(snap, dict):
        return {}
    out: dict[str, Any] = {}
    snapshot_ts = _bo3_extract_snapshot_ts(snap)
    if snapshot_ts is not None:
        out["last_source_snapshot_ts"] = snapshot_ts
    for src_key, out_key in (
        ("updated_at", "last_source_updated_at"),
        ("created_at", "last_source_created_at"),
        ("sent_time", "last_source_sent_time"),
        ("provider_event_id", "last_source_provider_event_id"),
        ("seq_index", "last_source_seq_index"),
        ("game_number", "last_source_game_number"),
        ("round_number", "last_source_round_number"),
    ):
        val = snap.get(src_key)
        if val is not None:
            out[out_key] = val
    return out


def _set_bo3_pipeline_diag(
    session_runtime: Any,
    stage: str,
    *,
    reason: str | None = None,
    at_ts: float | None = None,
    **fields: Any,
) -> None:
    """Update compact per-session BO3 pipeline diagnostics for debug/telemetry output."""
    if session_runtime is None:
        return
    diag = getattr(session_runtime, "bo3_pipeline_diag", None)
    if not isinstance(diag, dict):
        diag = {}
    ts = float(at_ts) if at_ts is not None else time.time()
    diag["last_stage"] = stage
    diag["last_stage_reason"] = reason
    diag["last_stage_ts"] = ts
    for key, value in fields.items():
        diag[key] = value
    setattr(session_runtime, "bo3_pipeline_diag", diag)


def _bo3_pipeline_diag_view(session_runtime: Any, now: float) -> dict[str, Any]:
    """Return operator-facing BO3 fetch->ingest->propagation diagnostics for one session."""
    diag = getattr(session_runtime, "bo3_pipeline_diag", None)
    if not isinstance(diag, dict):
        diag = {}
    out = dict(diag)
    out["buffer_has_snapshot"] = session_runtime.bo3_buf_raw is not None
    out["buffer_snapshot_ts"] = session_runtime.bo3_buf_snapshot_ts
    out["buffer_last_error"] = session_runtime.bo3_last_err
    out["buffer_consecutive_failures"] = session_runtime.bo3_buf_consecutive_failures
    if session_runtime.bo3_buf_last_success_epoch is not None:
        out["buffer_last_success_age_s"] = round(now - session_runtime.bo3_buf_last_success_epoch, 1)
    for ts_key, age_key in (
        ("last_fetch_attempt_ts", "last_fetch_attempt_age_s"),
        ("last_fetch_success_ts", "last_fetch_success_age_s"),
        ("last_stage_ts", "last_stage_age_s"),
        ("last_emit_ts", "last_emit_age_s"),
        ("last_store_append_ts", "last_store_append_age_s"),
        ("last_broadcast_ts", "last_broadcast_age_s"),
    ):
        ts = out.get(ts_key)
        if isinstance(ts, (int, float)):
            out[age_key] = round(now - float(ts), 1)
    return out


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


def _bo3_telemetry_ok(snap: dict[str, Any] | None, frame: Frame) -> tuple[bool, str | None]:
    """
    Same criteria as readiness: teams present + required fields (microstate, clock).
    Returns (telemetry_ok, telemetry_reason). reason is set when ok is False.
    """
    if not snap or not (snap.get("team_one") or snap.get("team_two")):
        return (False, "missing_teams")
    from engine.diagnostics.fragility_cs2 import compute_fragility_debug
    fragility = compute_fragility_debug(frame)
    if fragility.get("missing_microstate_flag"):
        return (False, "missing_microstate")
    if fragility.get("clock_invalid_flag"):
        return (False, "clock_invalid")
    return (True, None)


# BO3 health: label-only observability (no gating).
BO3_STALE_THRESHOLD_SEC = 30
BO3_BUFFER_GOOD_AGE_SEC = 20  # GOOD if buffer_age_s <= this
BO3_FETCH_RETRY_DELAYS = (0.5, 1.0)  # seconds between retries (2 retries after first attempt = 3 total)
# Pinned mode: candidate list refresh vs validation probe cadence
BO3_CANDIDATES_REFRESH_S = 120.0
BO3_VALIDATION_PROBE_INTERVAL_S = 60.0   # low frequency (was 8s); no pointless re-probing
BO3_VALIDATION_PROBE_BUDGET = 5
# After 404, do not re-probe for 10 minutes (long cooldown)
BO3_TELEM_RETRY_COOLDOWN_S = 600.0
BO3_INVALID_BLACKLIST_TTL_S = BO3_TELEM_RETRY_COOLDOWN_S  # alias for 404 cooldown
# Live-only: only consider candidates with status LIVE/current/in_progress or live_coverage
BO3_PARSED_STATUS_LIVE = frozenset({"current", "live", "in_progress", "running", "in progress"})
BO3_PAUSED_PHASES = frozenset({
    "TIMEOUT", "TECH_TIMEOUT", "PAUSED", "HALFTIME", "INTERMISSION",
    "POSTGAME", "MAP_END", "WARMUP", "FREEZETIME",
})

# Telemetry status tri-state: thresholds and derivation (sessions diagnostics)
TELEMETRY_FEED_DEAD_S = 90.0   # no fetch/update for this long -> feed considered dead (TELEMETRY_LOST)
TELEMETRY_STALLED_S = 60.0     # no good telemetry for this long -> stalled (TELEMETRY_LOST)


def derive_telemetry_status(
    now: float,
    last_update_ts: float | None,
    last_fetch_ts: float | None,
    last_good_ts: float | None,
    telemetry_ok: bool,
    telemetry_reason: str | None,
    *,
    last_error: str | None = None,
    grid_rate_limit_reason: str | None = None,
) -> dict[str, Any]:
    """
    Pure helper: derive telemetry_status (FEED_ALIVE | TELEMETRY_LOST | NO_DATA), combined reason, and ages.
    Used by get_sessions_diag. Does not mutate any state.
    """
    age_s = round(now - last_update_ts, 1) if last_update_ts is not None else None
    fetch_age_s = round(now - last_fetch_ts, 1) if last_fetch_ts is not None else age_s
    good_age_s = round(now - last_good_ts, 1) if last_good_ts is not None else None

    reasons: list[str] = []
    if last_error:
        reasons.append(last_error)
    if telemetry_reason:
        reasons.append(telemetry_reason)
    if grid_rate_limit_reason:
        reasons.append(f"grid:{grid_rate_limit_reason}")
    combined_reason = "; ".join(reasons) if reasons else None

    if last_update_ts is None:
        return {
            "telemetry_status": "NO_DATA",
            "telemetry_reason": combined_reason or "no_update_yet",
            "age_s": age_s,
            "fetch_age_s": fetch_age_s,
            "good_age_s": good_age_s,
        }

    fetch_age = (now - last_fetch_ts) if last_fetch_ts is not None else (now - last_update_ts)
    good_age = (now - last_good_ts) if last_good_ts is not None else float("inf")

    if fetch_age >= TELEMETRY_FEED_DEAD_S:
        return {
            "telemetry_status": "TELEMETRY_LOST",
            "telemetry_reason": combined_reason or f"feed_stalled_{int(fetch_age)}s",
            "age_s": age_s,
            "fetch_age_s": fetch_age_s,
            "good_age_s": good_age_s,
        }
    if not telemetry_ok or good_age >= TELEMETRY_STALLED_S:
        return {
            "telemetry_status": "TELEMETRY_LOST",
            "telemetry_reason": combined_reason or (telemetry_reason or f"telem_stalled_{int(good_age)}s"),
            "age_s": age_s,
            "fetch_age_s": fetch_age_s,
            "good_age_s": good_age_s,
        }

    return {
        "telemetry_status": "FEED_ALIVE",
        "telemetry_reason": combined_reason,
        "age_s": age_s,
        "fetch_age_s": fetch_age_s,
        "good_age_s": good_age_s,
    }


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
    """Runs tick loop: BO3 snapshot, REPLAY from JSONL, or GRID; append_point, broadcast."""

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
        self._replay_point_inputs_seen: int = 0
        self._replay_point_rejected_count: int = 0
        self._replay_point_transition_passthrough_count: int = 0
        self._replay_point_reject_reason_counts: dict[str, int] = {}
        self._replay_last_point_policy_decision: str | None = None
        self._replay_last_point_policy_reason: str | None = None
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
        # Canonical Team A identity per (game_number, map_index) so round_result labels and score ticks use same definition
        self._map_identity_cache: dict[tuple[int, int], dict[str, Any]] = {}
        # BO3 tick monotonic gating: reject stale/out-of-order frames (time rewind)
        from backend.services.bo3_freshness import Bo3FreshnessGate
        self._bo3_freshness_gate = Bo3FreshnessGate()
        self._bo3_monotonic_rejected_count: int = 0
        # Source-agnostic telemetry: one MatchContext for active match_id (single-match stage)
        self._match_context: Any = None
        self._match_context_match_id: int | None = None
        # GRID: reduced state and last series id (single-match)
        self._grid_state: Any = None
        self._grid_series_id: str | None = None
        # GRID per-series polling schedule (Patch 5 multi-session can reuse)
        self._grid_next_fetch_ts: dict[str, float] = {}
        self._grid_last_rate_limit_reason: dict[str, str] = {}
        # Multi-session registry (Patch 5A): SessionKey -> SessionRuntime
        self._sessions: dict[Any, Any] = {}
        # Sticky primary session (source, id) when primary_session_source/id not explicitly set
        self._last_primary_session: tuple[str, str] | None = None
        # Last effective primary for change logging (pinned / sticky / fallback)
        self._last_eff_primary_session: tuple[str, str] | None = None
        # GRID auto-track (runtime only; manual grid_series_ids overrides)
        self._grid_auto_last_refresh_ts: float = 0.0
        self._grid_auto_series_ids: list[str] = []
        # BO3 auto-track (runtime only; manual bo3_match_ids overrides)
        self._bo3_auto_last_refresh_ts: float = 0.0
        self._bo3_auto_match_ids: list[int] = []
        self._bo3_readiness_cache: dict[int, dict[str, Any]] = {}
        # Candidate discovery + validation: separate cadences; blacklist 404s
        self._bo3_next_candidates_refresh_at: float = 0.0
        self._bo3_next_validation_probe_at: float = 0.0
        self._bo3_candidates_cache: list[dict[str, Any]] = []
        self._bo3_invalid_snapshot_ids: dict[int, float] = {}  # match_id -> expires_at (unix)
        # ML-ready series-line-dislocation episode logger (paper only; emits setup_trigger, episode_start/end/outcome)
        self._trade_episode_manager = TradeEpisodeManager()
        # Per-session last emitted signature for point-emit discontinuity detection (key: (source, id))
        self._point_emit_last_sig: dict[tuple[str, str], dict[str, Any]] = {}

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
        self._map_identity_cache.clear()
        self._bo3_freshness_gate.clear_cache()
        self._match_context = None
        self._match_context_match_id = None

    def _point_emit_log(
        self,
        *,
        session_source: str,
        session_id: str,
        match_key: str,
        teams: tuple[str, str],
        map_i: int | None,
        round_i: int | None,
        ts: float,
        drv_summary: dict[str, Any],
        p_hat: float,
        bounds: tuple[float, float],
        rails: tuple[float, float],
        clamp_ctx: str,
        emit_allowed: bool,
        emit_reason: str,
    ) -> None:
        """Log per-candidate point and warn on discontinuities. No-op unless RUNNER_POINT_EMIT_DEBUG=1."""
        if not _RUNNER_POINT_EMIT_DEBUG:
            return
        sig_key = (session_source, session_id)
        last = self._point_emit_last_sig.get(sig_key)
        bound_low, bound_high = bounds
        rail_low, rail_high = rails
        teams_tuple = (teams[0] if len(teams) > 0 else "", teams[1] if len(teams) > 1 else "")
        log_line = (
            f"point_emit session={session_source}:{session_id} match={match_key} teams={teams_tuple!r} "
            f"map_i={map_i} round_i={round_i} ts={ts} p_hat={p_hat:.4f} "
            f"bounds=({bound_low:.4f},{bound_high:.4f}) rails=({rail_low:.4f},{rail_high:.4f}) "
            f"clamp={clamp_ctx} emit_allowed={emit_allowed} reason={emit_reason}"
        )
        logger.info(log_line, extra={"drv_summary": drv_summary})
        if last is not None:
            warn: list[str] = []
            if last.get("teams") != teams_tuple:
                warn.append(f"teams_change {last.get('teams')!r} -> {teams_tuple!r}")
            if last.get("map_i") is not None and map_i is not None and last["map_i"] > map_i:
                warn.append(f"map_regress {last['map_i']} -> {map_i}")
            if last.get("round_i") is not None and round_i is not None and last["round_i"] > round_i:
                warn.append(f"round_regress {last['round_i']} -> {round_i}")
            if last.get("ts") is not None and last["ts"] > ts:
                warn.append(f"ts_regress {last['ts']} -> {ts}")
            for key, (lo, hi) in [("bounds", bounds), ("rails", rails)]:
                old_lo = last.get(f"{key}_lo")
                old_hi = last.get(f"{key}_hi")
                if old_lo is not None and old_hi is not None:
                    delta_lo = abs(lo - old_lo)
                    delta_hi = abs(hi - old_hi)
                    if delta_lo > _POINT_JUMP_THRESHOLD or delta_hi > _POINT_JUMP_THRESHOLD:
                        warn.append(f"{key}_jump ({old_lo},{old_hi}) -> ({lo},{hi})")
            if last.get("p_hat") is not None and abs(p_hat - last["p_hat"]) > _POINT_JUMP_THRESHOLD:
                warn.append(f"p_hat_jump {last['p_hat']} -> {p_hat}")
            if warn:
                logger.warning("point_emit discontinuity: %s", "; ".join(warn), extra={"session": sig_key})
        if emit_allowed:
            self._point_emit_last_sig[sig_key] = {
                "teams": teams_tuple,
                "map_i": map_i,
                "round_i": round_i,
                "ts": ts,
                "p_hat": p_hat,
                "bounds_lo": bound_low,
                "bounds_hi": bound_high,
                "rails_lo": rail_low,
                "rails_hi": rail_high,
            }

    def _ensure_map_identity_from_raw(
        self,
        raw: dict[str, Any],
        config: Config,
        game_number: int | None,
        map_index: int | None,
        session_runtime: Any = None,
    ) -> dict[str, Any] | None:
        """Populate canonical identity for (game_number, map_index) from raw payload; return cache entry or None."""
        if game_number is None or map_index is None:
            return None
        key = (int(game_number), int(map_index))
        cache = session_runtime.map_identity_cache if session_runtime is not None else self._map_identity_cache
        if key in cache:
            return cache[key]
        t1 = raw.get("team_one") or {}
        t2 = raw.get("team_two") or {}
        try:
            team_one_id = int(t1.get("id", 0) or 0)
        except (TypeError, ValueError):
            team_one_id = 0
        try:
            team_two_id = int(t2.get("id", 0) or 0)
        except (TypeError, ValueError):
            team_two_id = 0
        # Require distinct non-zero raw team ids before a fresh BO3 map identity is trusted.
        if team_one_id <= 0 or team_two_id <= 0 or team_one_id == team_two_id:
            return None
        pid1 = t1.get("provider_id")
        team_one_provider_id = (str(pid1).strip() or None) if pid1 is not None else None
        pid2 = t2.get("provider_id")
        team_two_provider_id = (str(pid2).strip() or None) if pid2 is not None else None
        team_a_is_team_one = bool(getattr(config, "team_a_is_team_one", True))
        a_side = _normalize_side_raw(t1.get("side") if team_a_is_team_one else t2.get("side"))
        entry: dict[str, Any] = {
            "team_one_id": team_one_id,
            "team_two_id": team_two_id,
            "team_one_provider_id": team_one_provider_id,
            "team_two_provider_id": team_two_provider_id,
            "team_a_is_team_one": team_a_is_team_one,
            "a_side": a_side,
        }
        cache[key] = entry
        return entry

    def _ensure_map_identity_from_frame(
        self,
        frame: Frame | None,
        config: Config,
        game_number: int | None,
        map_index: int | None,
        session_runtime: Any = None,
    ) -> dict[str, Any] | None:
        """Populate canonical identity for (game_number, map_index) from frame; return cache entry or None."""
        if game_number is None or map_index is None:
            return None
        key = (int(game_number), int(map_index))
        cache = session_runtime.map_identity_cache if session_runtime is not None else self._map_identity_cache
        if key in cache:
            return cache[key]
        if frame is None:
            return None
        team_one_id = getattr(frame, "team_one_id", None)
        team_two_id = getattr(frame, "team_two_id", None)
        if team_one_id is not None:
            try:
                team_one_id = int(team_one_id)
            except (TypeError, ValueError):
                team_one_id = 0
        else:
            team_one_id = 0
        if team_two_id is not None:
            try:
                team_two_id = int(team_two_id)
            except (TypeError, ValueError):
                team_two_id = 0
        else:
            team_two_id = 0
        team_one_provider_id = getattr(frame, "team_one_provider_id", None)
        if team_one_provider_id is not None:
            team_one_provider_id = str(team_one_provider_id).strip() or None
        team_two_provider_id = getattr(frame, "team_two_provider_id", None)
        if team_two_provider_id is not None:
            team_two_provider_id = str(team_two_provider_id).strip() or None
        team_a_is_team_one = bool(getattr(config, "team_a_is_team_one", True))
        a_side_raw = getattr(frame, "a_side", None)
        a_side = _normalize_side_raw(a_side_raw) if a_side_raw is not None else None
        entry = {
            "team_one_id": team_one_id,
            "team_two_id": team_two_id,
            "team_one_provider_id": team_one_provider_id,
            "team_two_provider_id": team_two_provider_id,
            "team_a_is_team_one": team_a_is_team_one,
            "a_side": a_side,
        }
        cache[key] = entry
        return entry

    def get_replay_progress(self) -> dict[str, int] | None:
        """Return {index, total} when replay is active and list is loaded."""
        if not self._replay_payloads:
            return None
        return {"index": self._replay_index, "total": len(self._replay_payloads)}

    def get_replay_contract_status(self) -> dict[str, Any]:
        """Return replay contract-gate counters and last decision metadata."""
        return {
            "point_like_inputs_seen": self._replay_point_inputs_seen,
            "point_like_inputs_rejected": self._replay_point_rejected_count,
            "point_like_inputs_transition_passthrough": self._replay_point_transition_passthrough_count,
            "point_like_reject_reason_counts": dict(self._replay_point_reject_reason_counts),
            "last_point_like_policy_decision": self._replay_last_point_policy_decision,
            "last_point_like_policy_reason": self._replay_last_point_policy_reason,
            "raw_mode_point_like_skipped": self._replay_skipped_point_like_count,
        }

    def _coerce_replay_contract_policy(self, config: Config) -> str:
        """Normalize replay contract policy. Stage 1 supports reject_point_like only."""
        raw = getattr(config, "replay_contract_policy", REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE)
        policy = str(raw).strip().lower() if raw is not None else REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE
        if policy != REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE:
            return REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE
        return policy

    def _coerce_replay_transition_sunset_epoch(self, config: Config) -> float | None:
        raw = getattr(config, "replay_point_transition_sunset_epoch", None)
        if raw in (None, ""):
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _decide_point_like_replay_policy(self, config: Config) -> tuple[bool, str | None, dict[str, Any]]:
        """
        Stage 1 gate:
        - default reject_point_like
        - optional transition passthrough only when explicit and sunset-bound.
        """
        policy = self._coerce_replay_contract_policy(config)
        transition_enabled = bool(getattr(config, "replay_point_transition_enabled", False))
        sunset_epoch = self._coerce_replay_transition_sunset_epoch(config)
        now = time.time()

        policy_meta: dict[str, Any] = {
            "replay_contract_policy": policy,
            "replay_point_transition_enabled": transition_enabled,
            "replay_point_transition_sunset_epoch": sunset_epoch,
            "replay_point_transition_window_valid": bool(transition_enabled and sunset_epoch is not None and sunset_epoch > now),
        }

        if policy != REPLAY_CONTRACT_POLICY_REJECT_POINT_LIKE:
            return (False, REPLAY_POINT_REJECT_REASON_UNSUPPORTED_POLICY, policy_meta)
        if not transition_enabled:
            return (False, REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY, policy_meta)
        if sunset_epoch is None:
            return (False, REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_MISSING, policy_meta)
        if sunset_epoch <= now:
            return (False, REPLAY_POINT_REJECT_REASON_TRANSITION_SUNSET_EXPIRED, policy_meta)
        return (True, None, policy_meta)

    def _record_point_like_policy_decision(self, allow_transition_passthrough: bool, reason_code: str | None) -> None:
        """Update deterministic policy counters for replay contract observability."""
        self._replay_point_inputs_seen += 1
        if allow_transition_passthrough:
            self._replay_point_transition_passthrough_count += 1
            self._replay_last_point_policy_decision = REPLAY_POINT_POLICY_DECISION_TRANSITION_PASSTHROUGH
            self._replay_last_point_policy_reason = None
            return
        self._replay_point_rejected_count += 1
        code = reason_code or REPLAY_POINT_REJECT_REASON_DEFAULT_POLICY
        self._replay_point_reject_reason_counts[code] = self._replay_point_reject_reason_counts.get(code, 0) + 1
        self._replay_last_point_policy_decision = REPLAY_POINT_POLICY_DECISION_REJECT
        self._replay_last_point_policy_reason = code

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

    def get_match_context_diag(self) -> dict[str, Any]:
        """Return source-agnostic match context diagnostics for debug/telemetry endpoint."""
        ctx = self._match_context
        selector_decision: dict[str, Any] | None = None
        if ctx is not None:
            from engine.telemetry.selector import decide, default_source_selector_policy
            _policy = default_source_selector_policy()
            _dec = decide(ctx, time.time(), _policy, None)
            selector_decision = _dec.to_diag() if _dec else None
        out: dict[str, Any] = {
            "match_id": self._match_context_match_id,
            "context": _match_context_diag(ctx, selector_decision),
        }
        grid_series_id = getattr(self, "_grid_series_id", None)
        if grid_series_id:
            from engine.ingest.grid_schedule import next_fetch_in_s
            now = time.time()
            next_fetch_ts = getattr(self, "_grid_next_fetch_ts", {})
            out["grid_schedule"] = {
                "series_id": grid_series_id,
                "next_fetch_in_s": round(next_fetch_in_s(grid_series_id, next_fetch_ts, now), 1),
            }
            last_reason = getattr(self, "_grid_last_rate_limit_reason", {}).get(grid_series_id)
            if last_reason:
                out["grid_schedule"]["last_rate_limit_reason"] = last_reason
        return out

    def get_sessions_diag(self, config: Config | None = None) -> dict[str, Any]:
        """Return multi-session diagnostics for GET /debug/telemetry/sessions."""
        now = time.time()
        sessions_out: list[dict[str, Any]] = []
        for key, runtime in getattr(self, "_sessions", {}).items():
            display = key.display() if hasattr(key, "display") else str(key)
            source = getattr(key, "source", None)
            source_name = getattr(source, "value", str(source)) if source else "?"
            id_str = getattr(key, "id", "")
            last_ts = getattr(runtime, "last_update_ts", None)
            age_s = round(now - last_ts, 1) if last_ts is not None else None
            last_fetch_ts = getattr(runtime, "last_fetch_ts", None)
            last_good_ts = getattr(runtime, "last_good_ts", None)
            fetch_age_s = round(now - last_fetch_ts, 1) if last_fetch_ts is not None else (age_s if last_ts is not None else None)
            good_age_s = round(now - last_good_ts, 1) if last_good_ts is not None else None
            telemetry_ok = getattr(runtime, "telemetry_ok", True)
            telemetry_reason = getattr(runtime, "telemetry_reason", None)
            ctx_diag = _match_context_diag(getattr(runtime, "ctx", None), None)
            # Optional: expose last_frame teams for UI match label (defensive).
            try:
                last_frame = getattr(runtime, "last_frame", None)
                teams = None
                if isinstance(last_frame, dict):
                    teams = last_frame.get("teams")
                else:
                    teams = getattr(last_frame, "teams", None)
                if isinstance(teams, (list, tuple)) and len(teams) == 2 and all(isinstance(t, str) for t in teams):
                    ctx_diag = dict(ctx_diag or {})
                    ctx_diag["last_frame"] = {"teams": [teams[0], teams[1]]}
            except Exception:
                # Never let diagnostics break the endpoint.
                pass
            row: dict[str, Any] = {
                "session_key": display,
                "source": source_name,
                "id": id_str,
                "last_update_ts": last_ts,
                "age_s": age_s,
                "fetch_age_s": fetch_age_s,
                "good_age_s": good_age_s,
                "telemetry_ok": telemetry_ok,
                "telemetry_reason": telemetry_reason,
                "ctx": ctx_diag,
            }
            if source_name == "BO3":
                row["bo3_pipeline"] = _bo3_pipeline_diag_view(runtime, now)
            if source_name == "GRID" and id_str:
                next_ts = getattr(self, "_grid_next_fetch_ts", {}).get(id_str, 0.0)
                row["grid_schedule"] = {
                    "next_fetch_in_s": round(max(0.0, next_ts - now), 1),
                }
                last_reason = getattr(self, "_grid_last_rate_limit_reason", {}).get(id_str)
                if last_reason:
                    row["grid_schedule"]["last_rate_limit_reason"] = last_reason
            last_error = getattr(runtime, "last_error", None)
            grid_reason = row.get("grid_schedule", {}).get("last_rate_limit_reason") if isinstance(row.get("grid_schedule"), dict) else None
            derived = derive_telemetry_status(
                now,
                last_ts,
                last_fetch_ts,
                last_good_ts,
                telemetry_ok,
                telemetry_reason,
                last_error=last_error,
                grid_rate_limit_reason=grid_reason,
            )
            row["telemetry_status"] = derived["telemetry_status"]
            row["telemetry_reason"] = derived["telemetry_reason"]
            row["age_s"] = derived["age_s"]
            row["fetch_age_s"] = derived["fetch_age_s"]
            row["good_age_s"] = derived["good_age_s"]
            sessions_out.append(row)
        out: dict[str, Any] = {"now_ts": now, "sessions": sessions_out}
        if config is not None and getattr(config, "bo3_auto_track", False):
            last_ts = getattr(self, "_bo3_auto_last_refresh_ts", 0.0)
            out["bo3_auto_track_enabled"] = True
            out["bo3_auto_match_ids"] = list(getattr(self, "_bo3_auto_match_ids", []))
            out["bo3_auto_last_refresh_age_s"] = round(now - last_ts, 1) if last_ts else None
            cache = getattr(self, "_bo3_readiness_cache", {})
            out["bo3_readiness_cache_size"] = len(cache)
        if config is not None and getattr(config, "grid_auto_track", False):
            last_ts = getattr(self, "_grid_auto_last_refresh_ts", 0.0)
            out["grid_auto_track_enabled"] = True
            out["grid_auto_track_limit"] = max(0, min(50, int(getattr(config, "grid_auto_track_limit", 5) or 5)))
            out["grid_auto_series_ids"] = list(getattr(self, "_grid_auto_series_ids", []))
            out["grid_auto_last_refresh_age_s"] = round(now - last_ts, 1) if last_ts else None
        return out

    def clear_sessions(self) -> None:
        """
        Runtime-only cleanup: clear session registry, auto-track lists, readiness cache, grid schedule,
        and pinned primary state so the app behaves like a fresh launch.
        Does NOT modify persistent config. Used by POST /debug/telemetry/clear_sessions and POST /debug/reset.
        """
        self._sessions.clear()
        if hasattr(self, "_session_points") and isinstance(getattr(self, "_session_points"), dict):
            getattr(self, "_session_points").clear()
        self._bo3_auto_match_ids = []
        self._grid_auto_series_ids = []
        self._bo3_auto_last_refresh_ts = 0.0
        self._grid_auto_last_refresh_ts = 0.0
        self._bo3_readiness_cache.clear()
        self._bo3_next_candidates_refresh_at = 0.0
        self._bo3_next_validation_probe_at = 0.0
        self._bo3_candidates_cache = []
        self._bo3_invalid_snapshot_ids.clear()
        self._grid_next_fetch_ts.clear()
        self._grid_last_rate_limit_reason.clear()
        self._last_primary_session = None
        self._last_breach_type = None

    def _is_multi_session(self, config: Config) -> bool:
        """True if bo3_match_ids, grid_series_ids (manual), or bo3/grid auto_track is used."""
        bo3 = getattr(config, "bo3_match_ids", None)
        grid = getattr(config, "grid_series_ids", None)
        bo3_auto = getattr(config, "bo3_auto_track", False)
        grid_auto = getattr(config, "grid_auto_track", False)
        return (
            bool(bo3 and len(bo3) > 0)
            or bool(grid and len(grid) > 0)
            or bool(bo3_auto)
            or bool(grid_auto)
        )

    def _maybe_refresh_grid_auto_ids(self, config: Config, now_ts: float) -> None:
        """Refresh _grid_auto_series_ids from Central Data when grid_auto_track and refresh interval elapsed."""
        if not getattr(config, "grid_auto_track", False):
            return
        refresh_s = max(10.0, float(getattr(config, "grid_auto_track_refresh_s", 60.0)))
        if now_ts - getattr(self, "_grid_auto_last_refresh_ts", 0.0) < refresh_s:
            return
        try:
            from engine.ingest.grid_central_data import get_cs2_series_candidates, select_best_series_ids
            candidates = get_cs2_series_candidates(limit=100, order_direction="ASC")
            limit = getattr(config, "grid_auto_track_limit", 5)
            try:
                limit = max(0, min(50, int(limit)))
            except (TypeError, ValueError):
                limit = 5
            self._grid_auto_series_ids = select_best_series_ids(candidates, limit=limit, min_rank=2, allow_unknown_fallback=True)
            self._grid_auto_last_refresh_ts = now_ts
        except Exception:
            pass

    async def _maybe_refresh_bo3_candidates(self, config: Config, now_ts: float) -> None:
        """Fetch candidate list only (no probing). Cadence: BO3_CANDIDATES_REFRESH_S (120s)."""
        if not getattr(config, "bo3_auto_track", False):
            return
        if now_ts < getattr(self, "_bo3_next_candidates_refresh_at", 0.0):
            return
        try:
            from engine.ingest.bo3_client import fetch_candidates
            candidates = await fetch_candidates()
            self._bo3_candidates_cache = candidates
            self._bo3_next_candidates_refresh_at = now_ts + BO3_CANDIDATES_REFRESH_S
            if BO3_RATE_DEBUG:
                logger.info(
                    "BO3 candidates refresh: count=%s next_refresh_in_s=%.0f",
                    len(candidates),
                    BO3_CANDIDATES_REFRESH_S,
                )
        except Exception:
            self._bo3_next_candidates_refresh_at = now_ts + 60.0

    def _is_bo3_candidate_live(self, c: dict[str, Any]) -> bool:
        """True if candidate is labeled LIVE/current/in_progress or has live_coverage (no start_date window)."""
        status = (c.get("parsed_status") or "").strip().lower()
        if status in BO3_PARSED_STATUS_LIVE:
            return True
        if c.get("live_coverage"):
            return True
        return False

    def _bo3_live_candidates(self) -> list[dict[str, Any]]:
        """Candidates that are LIVE/current/in_progress or live_coverage; used for validation and sessions list."""
        candidates = getattr(self, "_bo3_candidates_cache", [])
        return [c for c in candidates if self._is_bo3_candidate_live(c)]

    def _bo3_validation_probe_candidates(self, now_ts: float) -> tuple[list[int], int, int]:
        """Ids to probe this cycle: live-only, skip cooldown, skip already telemetry_ready. Returns (to_probe, live_count, cooldown_skips)."""
        next_probe_after = getattr(self, "_bo3_invalid_snapshot_ids", {})  # id -> timestamp after which we may probe again
        live_candidates = self._bo3_live_candidates()
        cache = self._bo3_readiness_cache
        eligible: list[int] = []
        cooldown_skips = 0
        for c in live_candidates:
            mid = c.get("id") or c.get("match_id")
            if mid is None:
                continue
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                continue
            if mid in next_probe_after and now_ts < next_probe_after[mid]:
                cooldown_skips += 1
                continue
            entry = cache.get(mid)
            if entry and entry.get("telemetry_ready"):
                continue
            eligible.append(mid)
        to_probe = eligible[:BO3_VALIDATION_PROBE_BUDGET]
        if BO3_RATE_DEBUG:
            logger.info(
                "BO3 probe: live_candidates_count=%s probes_attempted=%s cooldown_skips=%s",
                len(live_candidates),
                len(to_probe),
                cooldown_skips,
            )
        return (to_probe, len(live_candidates), cooldown_skips)

    async def _maybe_run_bo3_validation_probes(self, config: Config, now_ts: float) -> None:
        """Probe at most N candidates (skip blacklisted). On 404 blacklist 600s; on 200 mark valid. No probes when pinned."""
        if not getattr(config, "bo3_auto_track", False):
            return
        # Hard lock: if config has BO3 primary set, never run validation probes (avoid 404 storms)
        cfg_src = (str(getattr(config, "primary_session_source", None) or "").strip().upper()) or None
        cfg_id = (str(getattr(config, "primary_session_id", None) or "").strip()) or None
        if cfg_src == "BO3" and cfg_id:
            return
        if now_ts < getattr(self, "_bo3_next_validation_probe_at", 0.0):
            return
        try:
            from engine.ingest.bo3_client import probe_snapshot_readiness
            from engine.ingest.bo3_readiness_cache import select_telemetry_ready_match_ids, update_cache_from_results
            to_probe, _live_count, _cooldown_skips = self._bo3_validation_probe_candidates(now_ts)
            self._bo3_next_validation_probe_at = now_ts + BO3_VALIDATION_PROBE_INTERVAL_S
            if not to_probe:
                return
            results: list[dict[str, Any]] = []
            for mid in to_probe:
                try:
                    r = await probe_snapshot_readiness(mid)
                    results.append(r)
                    if not r.get("telemetry_ready") and r.get("status_code") == 404:
                        self._bo3_invalid_snapshot_ids[mid] = now_ts + BO3_TELEM_RETRY_COOLDOWN_S
                        if BO3_RATE_DEBUG:
                            logger.info("BO3 validation 404 cooldown match_id=%s cooldown_s=%.0f", mid, BO3_TELEM_RETRY_COOLDOWN_S)
                except Exception as e:
                    results.append({
                        "match_id": mid,
                        "telemetry_ready": False,
                        "status_code": 502,
                        "reason": str(e),
                        "last_probe_ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    })
            update_cache_from_results(self._bo3_readiness_cache, results, now_ts)
            limit = max(0, min(50, int(getattr(config, "bo3_auto_track_limit", 5) or 5)))
            live_candidates = self._bo3_live_candidates()
            self._bo3_auto_match_ids = select_telemetry_ready_match_ids(live_candidates, self._bo3_readiness_cache, limit=limit)
            self._bo3_auto_last_refresh_ts = now_ts
        except Exception:
            self._bo3_next_validation_probe_at = now_ts + BO3_VALIDATION_PROBE_INTERVAL_S

    def _effective_bo3_match_ids(
        self,
        config: Config,
        *,
        primary_pinned: bool = False,
        primary_src: str | None = None,
        primary_id: str | None = None,
    ) -> list[int]:
        """BO3 match ids to tick. When primary is pinned to BO3, return only that id (single snapshot)."""
        manual = getattr(config, "bo3_match_ids", None) or []
        if manual and len(manual) > 0:
            return [int(x) for x in manual if x is not None]
        # Hard lock: read from config so pinned BO3 always yields exactly one id
        cfg_src = (str(getattr(config, "primary_session_source", None) or "").strip().upper()) or None
        cfg_id = (str(getattr(config, "primary_session_id", None) or "").strip()) or None
        if cfg_src == "BO3" and cfg_id:
            try:
                return [int(cfg_id)]
            except (TypeError, ValueError):
                pass
        if primary_pinned and primary_src == "BO3" and primary_id:
            try:
                return [int(primary_id)]
            except (TypeError, ValueError):
                pass
        if getattr(config, "bo3_auto_track", False):
            return list(getattr(self, "_bo3_auto_match_ids", []))
        mid = getattr(config, "match_id", None)
        return [int(mid)] if mid is not None else []

    def _effective_grid_series_ids(self, config: Config, now_ts: float | None = None) -> list[str]:
        """GRID series ids to tick: manual grid_series_ids overrides; else auto-track list or single grid_series_id."""
        manual = getattr(config, "grid_series_ids", None) or []
        if manual and len(manual) > 0:
            return [str(x).strip() for x in manual if x is not None and str(x).strip()]
        if getattr(config, "grid_auto_track", False):
            self._maybe_refresh_grid_auto_ids(config, now_ts or time.time())
            return list(getattr(self, "_grid_auto_series_ids", []))
        gid = getattr(config, "grid_series_id", None)
        s = str(gid).strip() if gid else ""
        return [s] if s else []

    def _get_or_create_session(self, key: Any, match_id_int: int) -> Any:
        """Get or create SessionRuntime for SessionKey; match_id_int used for MatchContext."""
        from engine.telemetry import MatchContext, SessionKey, SessionRuntime
        if key not in self._sessions:
            ctx = MatchContext(match_id=match_id_int)
            self._sessions[key] = SessionRuntime(ctx=ctx)
        return self._sessions[key]

    def _ingest_canonical_envelope(
        self,
        ctx: Any,
        env: Any,
        snap_meta: dict[str, Any] | None = None,
    ) -> bool:
        """
        Source-agnostic entrypoint: validate key, run should_accept, update ctx and SourceHealth.
        Does NOT call reduce/compute. Returns True if envelope accepted else False.
        If env.key is None and snap_meta provides frame/raw for BO3, key is computed.
        """
        from engine.telemetry import SourceKind, compute_monotonic_key_from_bo3_snapshot
        from engine.telemetry.envelope import process_canonical_envelope

        if env.key is None and snap_meta and env.source == SourceKind.BO3:
            frame_obj = snap_meta.get("frame")
            raw = snap_meta.get("raw")
            if frame_obj is not None or raw is not None:
                env.key = compute_monotonic_key_from_bo3_snapshot(frame_obj, raw)
        accepted, _ = process_canonical_envelope(ctx, env)
        return accepted

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

    async def _bo3_fetch_into_buffer(self, match_id: int, session_runtime: Any) -> None:
        """
        Writer: attempt fetch up to 3 times with short delays; update buffer only.
        Writes into session_runtime.bo3_buf_* (per-session, no cross-session leakage).
        On success: set bo3_buf_raw, bo3_buf_snapshot_ts, bo3_buf_last_success_epoch, clear error/failures.
        On failure: keep bo3_buf_raw, set bo3_last_err, increment bo3_buf_consecutive_failures.
        """
        mid = int(match_id)
        now = time.time()
        session_runtime.bo3_buf_last_attempt_epoch = now
        diag = getattr(session_runtime, "bo3_pipeline_diag", None)
        prev_attempt_ts = diag.get("last_fetch_attempt_ts") if isinstance(diag, dict) else None
        prev_success_ts = diag.get("last_fetch_success_ts") if isinstance(diag, dict) else None
        _set_bo3_pipeline_diag(
            session_runtime,
            "fetch_attempt",
            at_ts=now,
            fetch_attempt_count=int(diag.get("fetch_attempt_count", 0) if isinstance(diag, dict) else 0) + 1,
            last_fetch_attempt_ts=now,
            last_fetch_attempt_gap_s=(round(now - float(prev_attempt_ts), 3) if isinstance(prev_attempt_ts, (int, float)) else None),
        )
        t0 = time.perf_counter()
        try:
            from engine.ingest.bo3_client import get_snapshot
        except ImportError:
            session_runtime.bo3_last_err = "bo3_client not available"
            session_runtime.bo3_buf_consecutive_failures += 1
            _set_bo3_pipeline_diag(
                session_runtime,
                "fetch_failure",
                reason="bo3_client not available",
                at_ts=time.time(),
                last_fetch_duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                buffer_consecutive_failures=session_runtime.bo3_buf_consecutive_failures,
            )
            return
        last_err: str | None = None
        backoff_so_far = 0.0
        for attempt in range(3):
            try:
                try:
                    snap = await get_snapshot(
                        mid,
                        _rate_debug_retry_count=attempt,
                        _rate_debug_backoff_s=backoff_so_far if backoff_so_far else None,
                    )
                except TypeError as e:
                    # Compatibility path for simplified mocks/clients that only accept match_id.
                    msg = str(e)
                    if "_rate_debug_retry_count" not in msg and "_rate_debug_backoff_s" not in msg:
                        raise
                    snap = await get_snapshot(mid)
                if snap and isinstance(snap, dict) and (snap.get("team_one") or snap.get("team_two")):
                    success_ts = time.time()
                    session_runtime.bo3_buf_raw = snap
                    session_runtime.bo3_buf_snapshot_ts = _bo3_extract_snapshot_ts(snap)
                    session_runtime.bo3_buf_ts = success_ts
                    session_runtime.bo3_buf_last_success_epoch = success_ts
                    session_runtime.bo3_last_err = None
                    session_runtime.bo3_buf_consecutive_failures = 0
                    _set_bo3_pipeline_diag(
                        session_runtime,
                        "fetch_success",
                        at_ts=success_ts,
                        fetch_success_count=int(diag.get("fetch_success_count", 0) if isinstance(diag, dict) else 0) + 1,
                        last_fetch_success_ts=success_ts,
                        last_fetch_success_gap_s=(round(success_ts - float(prev_success_ts), 3) if isinstance(prev_success_ts, (int, float)) else None),
                        last_fetch_duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                        buffer_consecutive_failures=session_runtime.bo3_buf_consecutive_failures,
                        **_bo3_snapshot_diag(snap),
                    )
                    return
                last_err = "snapshot empty or missing team keys"
            except Exception as e:
                last_err = str(e)
            if attempt < 2:
                delay = BO3_FETCH_RETRY_DELAYS[attempt]
                await asyncio.sleep(delay)
                backoff_so_far += delay
        session_runtime.bo3_last_err = last_err
        session_runtime.bo3_buf_consecutive_failures += 1
        _set_bo3_pipeline_diag(
            session_runtime,
            "fetch_failure",
            reason=last_err,
            at_ts=time.time(),
            last_fetch_duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            buffer_consecutive_failures=session_runtime.bo3_buf_consecutive_failures,
        )

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
        session_runtime: Any = None,
    ) -> None:
        """
        Emit outcome label events from a BO3 payload: round_result and segment_result.
        Appends+broadcasts event HistoryPoints before the main point. Dedupes via session or runner trackers.
        """
        s = session_runtime

        def _tr(n: str, default: Any = None) -> Any:
            if s is not None:
                return getattr(s, "bo3_" + n, default)
            return getattr(self, "_bo3_" + n, default)

        def _set_tr(n: str, v: Any) -> None:
            if s is not None:
                setattr(s, "bo3_" + n, v)
            else:
                setattr(self, "_bo3_" + n, v)

        if not isinstance(raw, dict):
            return
        t1 = raw.get("team_one") or {}
        t2 = raw.get("team_two") or {}
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
        game_number_raw = raw.get("game_number")
        game_number: int | None = None
        map_index: int | None = None
        if game_number_raw is not None:
            try:
                game_number = int(game_number_raw)
                map_index = game_number - 1
            except (TypeError, ValueError):
                pass
        if game_number is None and _tr("last_seen_game_number") is not None:
            game_number = _tr("last_seen_game_number")
            map_index = game_number - 1 if game_number is not None else None
        if game_number_raw is not None and game_number is not None:
            _set_tr("last_seen_game_number", game_number)
        # Canonical Team A per (game_number, map_index): same definition for labels and score ticks
        map_identity_entry = self._ensure_map_identity_from_raw(
            raw, config, game_number, map_index, session_runtime=session_runtime
        )
        team_one_id = int(t1.get("id", 0) or 0)
        team_two_id = int(t2.get("id", 0) or 0)
        if map_identity_entry:
            team_a_id = (
                map_identity_entry["team_one_id"]
                if map_identity_entry["team_a_is_team_one"]
                else map_identity_entry["team_two_id"]
            )
        else:
            team_a_id = team_one_id if team_a_is_team_one else team_two_id
        s1 = int(t1.get("score", 0) or 0)
        s2 = int(t2.get("score", 0) or 0)
        last_s1 = _tr("last_seen_score_team_one")
        last_s2 = _tr("last_seen_score_team_two")

        async def maybe_emit_round_result(round_to_label: int, winner_team_id: int) -> None:
            if winner_team_id == 0:
                return
            if (
                _tr("last_emitted_round_number") == round_to_label
                and _tr("last_emitted_round_winner_team_id") == winner_team_id
            ):
                return
            round_winner_is_team_a = bool(team_a_id and winner_team_id == team_a_id)
            round_event = {
                "event_type": "round_result",
                "round_number": round_to_label,
                "round_winner_team_id": winner_team_id,
                "round_winner_is_team_a": round_winner_is_team_a,
            }
            if map_index is not None:
                round_event["map_index"] = map_index
            if game_number is not None:
                round_event["game_number"] = game_number
            identity_kw = (
                _team_identity_from_cache(map_identity_entry)
                if map_identity_entry
                else _team_identity_for_point(getattr(new_state, "last_frame", None), config)
            )
            round_point = HistoryPoint(
                time=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                segment_id=new_state.segment_id,
                match_id=match_id_used,
                map_index=map_index,
                round_number=round_to_label,
                game_number=game_number,
                explain=None,
                event=round_event,
                **identity_kw,
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
            _set_tr("last_emitted_round_number", round_to_label)
            _set_tr("last_emitted_round_winner_team_id", winner_team_id)

        # Infer winner by score delta (winning_team_id is always 0 in our BO3 replay payloads)
        winner_from_delta = _infer_round_winner_by_score_delta(
            last_s1, last_s2, s1, s2, team_one_id, team_two_id
        )

        # A) phase == "FINISHED": label the round that just finished (rn)
        if phase_upper == "FINISHED" and rn is not None:
            winner = winning_team_id if winning_team_id != 0 else winner_from_delta
            await maybe_emit_round_result(rn, winner)

        # B) rn advanced: previous round ended when round number increased
        last_rn = _tr("last_seen_round_number")
        if last_rn is not None and rn is not None and rn > last_rn:
            prev_round = last_rn
            winner = winning_team_id if winning_team_id != 0 else winner_from_delta
            await maybe_emit_round_result(prev_round, winner)

        # Update last-seen round and scores every tick
        _set_tr("last_seen_round_number", rn)
        _set_tr("last_seen_score_team_one", s1)
        _set_tr("last_seen_score_team_two", s2)
        _set_tr("last_seen_scores", (s1, s2))

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

            prev = (_tr("last_seen_match_score_by_game") or {}).get(game_number)
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
                        _tr("last_seen_segment_id_for_result") != finished_map_index
                        or _tr("last_seen_map_winner_team_id") != winner_team_id
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
                        seg_identity = (
                            _team_identity_from_cache(map_identity_entry)
                            if map_identity_entry
                            else _team_identity_for_point(getattr(new_state, "last_frame", None), config)
                        )
                        segment_point = HistoryPoint(
                            time=t,
                            p_hat=p_hat,
                            bound_low=bound_low,
                            bound_high=bound_high,
                            rail_low=rail_low,
                            rail_high=rail_high,
                            market_mid=market_mid,
                            segment_id=finished_map_index,
                            map_index=finished_map_index,
                            round_number=None,
                            game_number=game_number,
                            explain=None,
                            event=segment_event,
                            **seg_identity,
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
                        _set_tr("last_seen_segment_id_for_result", finished_map_index)
                        _set_tr("last_seen_map_winner_team_id", winner_team_id)

            # Always update stored match_score for this game (overwrite on resets/decreases)
            d = dict(_tr("last_seen_match_score_by_game") or {})
            d[game_number] = (ms1, ms2)
            _set_tr("last_seen_match_score_by_game", d)
            _set_tr("last_seen_game_number", game_number)
            _set_tr("last_seen_match_score_team_one", ms1)
            _set_tr("last_seen_match_score_team_two", ms2)

    def _bo3_buffer_debug(self, session_runtime: Any, now: float) -> dict[str, Any]:
        """Build buffer observability dict for derived.debug from session_runtime BO3 buffer."""
        age = None
        if session_runtime.bo3_buf_last_success_epoch is not None:
            age = now - session_runtime.bo3_buf_last_success_epoch
        return {
            "bo3_buffer_has_snapshot": session_runtime.bo3_buf_raw is not None,
            "bo3_buffer_age_s": age,
            "bo3_buffer_last_error": session_runtime.bo3_last_err,
            "bo3_buffer_consecutive_failures": session_runtime.bo3_buf_consecutive_failures,
            "bo3_buffer_snapshot_ts": session_runtime.bo3_buf_snapshot_ts,
            "bo3_buffer_last_success_epoch": session_runtime.bo3_buf_last_success_epoch,
        }

    def _log_bo3_session_update(
        self, match_id: int, session_runtime: Any, last_update_ts: float
    ) -> None:
        """When BO3_RATE_DEBUG=1, log SessionRuntime update: match_id, last_update_ts, provider_ts, age_s."""
        if not BO3_RATE_DEBUG:
            return
        now = time.time()
        provider_ts = getattr(session_runtime, "bo3_buf_snapshot_ts", None)
        age_s = round(now - last_update_ts, 1) if last_update_ts else None
        logger.info(
            "BO3 session update match_id=%s last_update_ts=%.3f provider_ts=%s age_s=%s",
            match_id,
            last_update_ts,
            provider_ts,
            age_s,
            extra={
                "match_id": match_id,
                "last_update_ts": last_update_ts,
                "provider_ts": provider_ts,
                "age_s": age_s,
            },
        )

    def _bo3_health_from_buffer(
        self, session_runtime: Any, frame: Frame | None, now: float
    ) -> tuple[str, str | None, float | None]:
        """
        Health from session buffer age + phase: GOOD if age <= 20s, STALE if > 20, PAUSED from phase, ERROR only if no snapshot and last_error set.
        """
        if frame is not None:
            round_phase = None
            if isinstance(getattr(frame, "bomb_phase_time_remaining", None), dict):
                round_phase = frame.bomb_phase_time_remaining.get("round_phase")
            if round_phase is not None:
                phase_str = str(round_phase).strip().upper()
                if phase_str in BO3_PAUSED_PHASES:
                    return ("PAUSED", phase_str, None)
        if session_runtime.bo3_buf_raw is None and session_runtime.bo3_last_err is not None:
            return ("ERROR", session_runtime.bo3_last_err, None)
        if session_runtime.bo3_buf_last_success_epoch is None:
            return ("GOOD", None, None)
        age = now - session_runtime.bo3_buf_last_success_epoch
        if age <= BO3_BUFFER_GOOD_AGE_SEC:
            return ("GOOD", None, None)
        return ("STALE", f"buffer age {int(age)}s", age)

    def _maybe_record_raw_bo3_snapshot(
        self,
        payload: dict[str, Any],
        match_id: int,
        team_a_is_team_one: bool | None,
        session_runtime: Any = None,
    ) -> str | None:
        """
        If raw BO3 recording is enabled, validate payload, dedupe by signature, and append one line
        to logs/bo3_raw_match_<match_id>.jsonl (or logs/bo3_raw.jsonl if per-match disabled).
        Does not change engine computations; recording only.
        """
        if not _BO3_RAW_RECORD_ENABLED:
            return None
        if not isinstance(payload, dict):
            return None
        t1 = payload.get("team_one")
        t2 = payload.get("team_two")
        if not isinstance(t1, dict) or not isinstance(t2, dict):
            return None
        # Optional: game_number and round_number present; no strict requirement
        sig = _bo3_raw_record_signature(payload, match_id)
        if sig is None:
            return None
        if session_runtime is not None:
            last_sig = session_runtime.bo3_raw_last_sig
            if last_sig is not None and last_sig == sig:
                return _bo3_raw_record_path(match_id)
            session_runtime.bo3_raw_last_sig = sig
        else:
            last_sig = self._bo3_raw_last_sig_by_match.get(match_id)
            if last_sig is not None and last_sig == sig:
                return _bo3_raw_record_path(match_id)
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
        filepath = _bo3_raw_record_path(match_id)
        try:
            os.makedirs(dirpath, exist_ok=True)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("BO3 raw record write failed %s: %s", filepath, e)
            return None
        return filepath

    async def _tick_bo3(
        self,
        config: Config,
        effective_match_id: int | None = None,
        session_runtime: Any = None,
        write_to_store: bool = True,
    ) -> bool:
        """If source=BO3 and match_id set: fetch snapshot, normalize, append_point, broadcast. Return True if did BO3.
        When session_runtime is provided, use its ctx and per-session buffer/gate; when None, get-or-create one for mid (single-session)."""
        from engine.telemetry import SessionKey, SourceKind

        src = getattr(config, "source", None)
        match_id = effective_match_id if effective_match_id is not None else getattr(config, "match_id", None)
        if match_id is None:
            return False
        if effective_match_id is None and src != "BO3":
            return False
        mid = int(match_id)
        if session_runtime is None:
            session_runtime = self._get_or_create_session(SessionKey(source=SourceKind.BO3, id=str(mid)), mid)
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        try:
            from engine.ingest.bo3_client import get_snapshot
            from engine.normalize.bo3_normalize import bo3_snapshot_to_frame
        except ImportError:
            return False

        await self._bo3_fetch_into_buffer(mid, session_runtime)
        snap = session_runtime.bo3_buf_raw
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
            new_debug["bo3_feed_error"] = session_runtime.bo3_last_err
            new_debug["bo3_snapshot_ts"] = None
            new_debug["bo3_match_id_used"] = mid
            new_debug.update(self._bo3_buffer_debug(session_runtime, now))
            health, health_reason, health_age_s = self._bo3_health_from_buffer(session_runtime, None, now)
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
            _set_bo3_pipeline_diag(
                session_runtime,
                "fetch_empty",
                reason=session_runtime.bo3_last_err or "snapshot empty or missing team keys",
                at_ts=now,
                last_snapshot_status="empty",
                last_emit_decision="no_emit",
                last_emit_reason="fetch_empty",
            )
            self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
            return True
        session_runtime.last_fetch_ts = now
        raw_record_path = None
        if isinstance(snap, dict):
            session_runtime.bo3_last_raw_snapshot = snap
            raw_record_path = self._maybe_record_raw_bo3_snapshot(
                snap, mid, team_a_is_team_one, session_runtime=session_runtime
            )
        frame = bo3_snapshot_to_frame(snap, team_a_is_team_one=team_a_is_team_one)
        status, feed_error, snapshot_ts, is_fresh = _bo3_snapshot_status(
            snap, frame,
            session_runtime.bo3_last_snapshot_ts,
            session_runtime.bo3_last_scores,
            session_runtime.bo3_same_snapshot_polls,
        )
        if status != "live":
            now = time.time()
            health, health_reason, health_age_s = self._bo3_health_from_buffer(session_runtime, frame, now)
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
            new_debug.update(self._bo3_buffer_debug(session_runtime, now))
            new_debug["bo3_health"] = health
            new_debug["bo3_health_reason"] = health_reason
            new_debug["bo3_health_age_s"] = health_age_s
            if status == "invalid_clock":
                new_debug["time_term_used"] = False
            from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
            new_debug["raw"] = build_raw_debug(frame)
            new_debug["fragility"] = compute_fragility_debug(frame)
            telem_ok, telem_reason = _bo3_telemetry_ok(snap if isinstance(snap, dict) else None, frame)
            session_runtime.telemetry_ok = telem_ok
            session_runtime.telemetry_reason = telem_reason
            if telem_ok:
                session_runtime.last_good_ts = now
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
            store_append_ts = time.time()
            await self._store.append_point(hold_point, state, fail_derived)
            broadcast_ts = time.time()
            await self._broadcast_point(hold_point)
            if status == "stale":
                session_runtime.bo3_same_snapshot_polls += 1
            emit_ts = time.time()
            _set_bo3_pipeline_diag(
                session_runtime,
                "hold_point_emitted",
                reason=status,
                at_ts=emit_ts,
                last_snapshot_status=status,
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_emit_decision="hold_point",
                last_emit_reason=status,
                last_emit_ts=emit_ts,
                last_store_append_ts=store_append_ts,
                last_broadcast_ts=broadcast_ts,
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
            self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
            return True

        if is_fresh:
            session_runtime.bo3_same_snapshot_polls = 0
            session_runtime.bo3_last_snapshot_ts = snapshot_ts
            session_runtime.bo3_last_scores = getattr(frame, "scores", (0, 0))
            session_runtime.bo3_last_change_epoch = time.time()
        else:
            session_runtime.bo3_same_snapshot_polls += 1

        t = time.time()
        health, health_reason, health_age_s = self._bo3_health_from_buffer(session_runtime, frame, t)
        # Monotonic gating: per-session gate (no cross-session leakage)
        gate = session_runtime.ensure_bo3_gate()
        accept, gate_reason, gate_diag = gate.accept_frame(frame, snap if isinstance(snap, dict) else None)
        if not accept:
            self._bo3_monotonic_rejected_count += 1
            if _BO3_TICK_MONOTONIC_DEBUG:
                logger.warning(
                    "BO3 monotonic gate rejected frame: %s",
                    gate_reason,
                    extra={"diag": gate_diag, "rejected_total": self._bo3_monotonic_rejected_count},
                )
            _set_bo3_pipeline_diag(
                session_runtime,
                "freshness_gate_reject",
                reason=gate_reason,
                at_ts=t,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_freshness_gate_reason=gate_reason,
                last_emit_decision="no_emit",
                last_emit_reason="freshness_gate",
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
            self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
            return True
        # Canonical envelope path: BO3 as first-class SourceKind
        from engine.telemetry import (
            CanonicalFrameEnvelope,
            MatchContext,
            SourceKind,
            compute_monotonic_key_from_bo3_snapshot,
        )
        from engine.telemetry.core import IdentityEntry
        ctx = session_runtime.ctx
        frame_dict = asdict(frame)
        new_key = compute_monotonic_key_from_bo3_snapshot(frame, snap if isinstance(snap, dict) else None)
        env = CanonicalFrameEnvelope(
            match_id=mid,
            source=SourceKind.BO3,
            observed_ts=t,
            key=new_key,
            frame=frame_dict,
            valid=True,
        )
        snap_meta = {"frame": frame, "raw": snap if isinstance(snap, dict) else None}
        if isinstance(snap, dict):
            snap_meta["round_phase"] = snap.get("round_phase") or snap.get("phase")
        if not self._ingest_canonical_envelope(ctx, env, snap_meta):
            reject_reason = getattr(ctx, "last_reject_reason", None)
            _set_bo3_pipeline_diag(
                session_runtime,
                "canonical_envelope_reject",
                reason=reject_reason,
                at_ts=t,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_envelope_reject_reason=reject_reason,
                last_emit_decision="no_emit",
                last_emit_reason="canonical_envelope",
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
            self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
            return True
        round_phase_drive = snap_meta.get("round_phase") if snap_meta else None
        from engine.telemetry.selector import allowed_to_drive, default_source_selector_policy
        allow_drive, drive_reason = allowed_to_drive(ctx, env.source, t, round_phase_drive, default_source_selector_policy())
        if not allow_drive:
            ctx.last_drive_deny_reason = drive_reason
            _set_bo3_pipeline_diag(
                session_runtime,
                "selector_denied",
                reason=drive_reason,
                at_ts=t,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_selector_reason=drive_reason,
                last_emit_decision="no_emit",
                last_emit_reason="selector_denied",
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
            self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
            return True
        # Identity cache stub: per-match team_a_is_team_one and provider ids from map identity
        _gn: int | None = None
        if isinstance(snap, dict):
            try:
                g = snap.get("game_number")
                _gn = int(g) if g is not None else None
            except (TypeError, ValueError):
                pass
        _mi = getattr(frame, "map_index", 0)
        map_id_entry = self._ensure_map_identity_from_raw(snap if isinstance(snap, dict) else {}, config, _gn, _mi, session_runtime=session_runtime)
        if map_id_entry:
            ta_one = map_id_entry.get("team_a_is_team_one", True)
            ctx.identity = IdentityEntry(
                team_a_is_team_one=bool(map_id_entry.get("team_a_is_team_one", True)),
                provider_team_a_id=map_id_entry.get("team_one_provider_id") if ta_one else map_id_entry.get("team_two_provider_id"),
                provider_team_b_id=map_id_entry.get("team_two_provider_id") if ta_one else map_id_entry.get("team_one_provider_id"),
            )
        # Selector diagnostics only (no ingest change): decide + maybe_switch for telemetry output
        round_phase_sel = snap_meta.get("round_phase") if isinstance(snap, dict) else None
        from engine.telemetry.selector import decide, default_source_selector_policy, maybe_switch
        _sel_policy = default_source_selector_policy()
        _sel_decision = decide(ctx, t, _sel_policy, round_phase_sel)
        maybe_switch(ctx, _sel_decision, t, _sel_policy, round_phase_sel)
        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state
        from engine.diagnostics.inter_map_break import detect_inter_map_break
        from engine.telemetry.initial_state import initial_state

        if write_to_store:
            old_state = await self._store.get_state()
        else:
            old_state = session_runtime.last_state if (session_runtime.last_state is not None) else initial_state(config)
        new_state = reduce_state(old_state, frame, config)
        session_runtime.last_state = new_state
        session_runtime.last_frame = asdict(frame)
        session_runtime.last_update_ts = t
        session_runtime.telemetry_ok = True
        session_runtime.last_good_ts = t
        session_runtime.telemetry_reason = None
        self._log_bo3_session_update(mid, session_runtime, t)
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}

        # Detect inter-map break (between maps) and keep corridors/p_hat stable.
        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            # Stage 3B: continuity from store only when write_to_store; else session-local only.
            if write_to_store:
                cur = await self._store.get_current()
                d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
                last_p = d.get("p_hat")
            else:
                last_p = getattr(session_runtime, "last_p_hat", None)
            p_hat, dbg = _inter_map_break_phat_and_dbg(
                bound_low, bound_high, break_reason, last_p, dict(bounds_debug)
            )
            rail_low = dbg["map_low"]
            rail_high = dbg["map_high"]
            dbg["bo3_monotonic_gate"] = gate_diag
            dbg["match_context_diag"] = _match_context_diag(ctx, _sel_decision.to_diag() if _sel_decision else None)
        else:
            bounds = (bound_low, bound_high)
            _src = getattr(config, "source", None)
            rails_result = compute_rails(frame, config, new_state, bounds, source=_src, replay_kind=None)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False))
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}
            dbg["bo3_monotonic_gate"] = gate_diag
            dbg["match_context_diag"] = _match_context_diag(ctx, _sel_decision.to_diag() if _sel_decision else None)

        if session_runtime is not None:
            setattr(session_runtime, "last_p_hat", p_hat)

        from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
        dbg["raw"] = build_raw_debug(frame)
        dbg["fragility"] = compute_fragility_debug(frame)
        dbg.update(_compute_dominance_features(frame))
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
        # BO3 live emission gate: do not append/broadcast when driver validity is false (legacy-style gating).
        fragility = dbg.get("fragility") or {}
        drv_valid_microstate = not fragility.get("missing_microstate_flag", False)
        drv_valid_roundstate = not fragility.get("clock_invalid_flag", False)
        if not (drv_valid_microstate and drv_valid_roundstate):
            dbg["drv_emit_skipped"] = True
            dbg["drv_emit_reason"] = "drv_invalid"
            dbg["drv_valid_microstate"] = drv_valid_microstate
            dbg["drv_valid_roundstate"] = drv_valid_roundstate
            if session_runtime is not None:
                session_runtime.last_state = new_state
                session_runtime.last_frame = asdict(frame)
                session_runtime.last_update_ts = t
                session_runtime.last_error = "drv_invalid"
                session_runtime.telemetry_ok = False
                session_runtime.telemetry_reason = (
                    "missing_microstate" if not drv_valid_microstate else "clock_invalid"
                )
                self._log_bo3_session_update(mid, session_runtime, t)
            teams_bo3 = getattr(frame, "teams", None) or ("", "")
            if isinstance(teams_bo3, (list, tuple)) and len(teams_bo3) >= 2:
                teams_bo3 = (str(teams_bo3[0]), str(teams_bo3[1]))
            else:
                teams_bo3 = ("", "")
            clamp_reason = "unknown"
            if isinstance(dbg.get("explain"), dict) and isinstance(dbg["explain"].get("final"), dict):
                clamp_reason = dbg["explain"]["final"].get("clamp_reason", "unknown")
            self._point_emit_log(
                session_source="BO3",
                session_id=str(mid),
                match_key=str(mid),
                teams=teams_bo3,
                map_i=getattr(frame, "map_index", None),
                round_i=getattr(new_state, "round_index", None),
                ts=t,
                drv_summary={
                    "drv_valid_microstate": drv_valid_microstate,
                    "drv_valid_roundstate": drv_valid_roundstate,
                },
                p_hat=p_hat,
                bounds=(bound_low, bound_high),
                rails=(rail_low, rail_high),
                clamp_ctx=clamp_reason,
                emit_allowed=False,
                emit_reason="drv_invalid",
            )
            _set_bo3_pipeline_diag(
                session_runtime,
                "driver_emit_denied",
                reason=session_runtime.telemetry_reason,
                at_ts=t,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_emit_decision="no_emit",
                last_emit_reason="drv_invalid",
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
            return True
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
        # Outcome label events (round_result, segment_result) from raw snapshot Ã¢â‚¬â€ emit before main point, no duplicate spam
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
                session_runtime=session_runtime,
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
            map_index_ep = (game_number_ep - 1) if game_number_ep is not None else None
            ep_point = HistoryPoint(
                time=t,
                p_hat=p_hat,
                bound_low=bound_low,
                bound_high=bound_high,
                rail_low=rail_low,
                rail_high=rail_high,
                market_mid=market_mid,
                segment_id=new_state.segment_id,
                map_index=map_index_ep,
                round_number=round_number_ep,
                game_number=game_number_ep,
                explain=ep_explain,
                event=evt,
                **_team_identity_for_point(getattr(new_state, "last_frame", None), config),
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
        dbg["bo3_snapshot_ts"] = session_runtime.bo3_buf_snapshot_ts
        dbg.update(self._bo3_buffer_debug(session_runtime, t))
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
        map_index_bo3 = (game_number_ep - 1) if game_number_ep is not None else None
        map_identity_bo3 = _cached_map_identity_entry(session_runtime, game_number_ep, map_index_bo3)
        bo3_identity_kw = (
            _team_identity_from_cache(map_identity_bo3)
            if map_identity_bo3
            else _team_identity_for_point(frame, config)
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
            map_index=map_index_bo3,
            round_number=round_number_ep,
            game_number=game_number_ep,
            explain=explain,
            **bo3_identity_kw,
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
        if write_to_store and isinstance(snap, dict):
            try:
                if map_identity_bo3 is None:
                    dbg["bo3_backend_capture_skipped_reason"] = "identity_not_yet_trustworthy"
                    logger.warning(
                        "BO3 backend capture skipped: identity_not_yet_trustworthy match_id=%s game_number=%s map_index=%s",
                        mid,
                        game_number_ep,
                        map_index_bo3,
                    )
                else:
                    capture_record = build_bo3_live_capture_record(
                        match_id=mid,
                        team_a_is_team_one=team_a_is_team_one,
                        raw_snapshot=snap,
                        raw_record_path=raw_record_path,
                        frame=frame,
                        point=point,
                        derived=derived,
                    )
                    capture_path = append_bo3_live_capture_record(capture_record)
                    if capture_path:
                        dbg["bo3_backend_capture_path"] = capture_path
            except OSError as e:
                logger.warning("BO3 backend capture contract write failed: %s", e)
        if session_runtime is not None:
            session_runtime.last_state = new_state
            session_runtime.last_frame = asdict(frame)
            session_runtime.last_update_ts = t
            session_runtime.last_error = None
            session_runtime.telemetry_ok = True
            session_runtime.last_good_ts = t
            session_runtime.telemetry_reason = None
            self._log_bo3_session_update(mid, session_runtime, t)
        clamp_reason_bo3 = "ok"
        if isinstance(explain, dict) and isinstance(explain.get("final"), dict):
            clamp_reason_bo3 = explain["final"].get("clamp_reason", "ok")
        teams_bo3_emit = getattr(frame, "teams", None) or ("", "")
        if isinstance(teams_bo3_emit, (list, tuple)) and len(teams_bo3_emit) >= 2:
            teams_bo3_emit = (str(teams_bo3_emit[0]), str(teams_bo3_emit[1]))
        else:
            teams_bo3_emit = ("", "")
        self._point_emit_log(
            session_source="BO3",
            session_id=str(mid),
            match_key=str(mid),
            teams=teams_bo3_emit,
            map_i=map_index_bo3,
            round_i=round_number_ep,
            ts=t,
            drv_summary={"drv_valid_microstate": True, "drv_valid_roundstate": True},
            p_hat=p_hat,
            bounds=(bound_low, bound_high),
            rails=(rail_low, rail_high),
            clamp_ctx=clamp_reason_bo3,
            emit_allowed=write_to_store,
            emit_reason="ok",
        )
        if write_to_store:
            store_append_ts = time.time()
            await self._store.append_point(point, new_state, derived)
            broadcast_ts = time.time()
            await self._broadcast_point(point)
            emit_ts = time.time()
            self._bo3_last_raw_snapshot = session_runtime.bo3_last_raw_snapshot
            _set_bo3_pipeline_diag(
                session_runtime,
                "point_emitted",
                reason="ok",
                at_ts=emit_ts,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_emit_decision="point_emitted",
                last_emit_reason="ok",
                last_emit_ts=emit_ts,
                last_store_append_ts=store_append_ts,
                last_broadcast_ts=broadcast_ts,
                last_capture_skip_reason=dbg.get("bo3_backend_capture_skipped_reason"),
                last_capture_path=dbg.get("bo3_backend_capture_path"),
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
        else:
            _set_bo3_pipeline_diag(
                session_runtime,
                "accepted_not_propagated",
                reason="write_to_store_false",
                at_ts=t,
                last_snapshot_status="live",
                last_snapshot_fresh=is_fresh,
                same_snapshot_polls=session_runtime.bo3_same_snapshot_polls,
                last_emit_decision="accepted_not_propagated",
                last_emit_reason="write_to_store_false",
                last_emit_ts=t,
                last_capture_skip_reason=dbg.get("bo3_backend_capture_skipped_reason"),
                last_capture_path=dbg.get("bo3_backend_capture_path"),
                **_bo3_snapshot_diag(snap if isinstance(snap, dict) else None),
            )
        self._bo3_buf_consecutive_failures = session_runtime.bo3_buf_consecutive_failures
        return True

    async def _tick_bo3_for_match(
        self, match_id: int, config: Config, is_primary: bool
    ) -> bool:
        """Multi-session: tick one BO3 match; use session ctx; only primary writes to store."""
        # Pinned BO3 hard lock: only allow snapshot for the pinned id
        cfg_src = (str(getattr(config, "primary_session_source", None) or "").strip().upper()) or None
        cfg_id = (str(getattr(config, "primary_session_id", None) or "").strip()) or None
        if cfg_src == "BO3" and cfg_id and str(match_id) != cfg_id:
            return False
        from engine.telemetry import SessionKey, SourceKind
        key = SessionKey(source=SourceKind.BO3, id=str(match_id))
        session = self._get_or_create_session(key, match_id)
        return await self._tick_bo3(
            config,
            effective_match_id=match_id,
            session_runtime=session,
            write_to_store=is_primary,
        )

    async def _tick_grid(
        self,
        config: Config,
        effective_series_id: str | None = None,
        session_runtime: Any = None,
        write_to_store: bool = True,
    ) -> bool:
        """If source=GRID and grid_series_id set: fetch series state, reduce, envelope, compute. Return True if did GRID.
        When session_runtime is provided, use its ctx and grid_state; when write_to_store is False, skip append/broadcast."""
        from engine.ingest.grid_client import fetch_series_state
        from engine.ingest.grid_schedule import (
            after_rate_limit,
            after_success,
            is_rate_limit_response,
            next_fetch_allowed,
        )
        from engine.ingest.grid_reducer import GridState, grid_state_to_frame, reduce_event
        from engine.telemetry import (
            CanonicalFrameEnvelope,
            MatchContext,
            SourceKind,
            compute_monotonic_key_from_grid_state,
        )
        from engine.telemetry.core import IdentityEntry
        from engine.telemetry.selector import decide, default_source_selector_policy, maybe_switch

        src = getattr(config, "source", None)
        grid_series_id = (effective_series_id or "").strip() or (getattr(config, "grid_series_id", None) or "")
        grid_series_id = str(grid_series_id).strip() if grid_series_id else ""
        if not grid_series_id:
            return False
        if effective_series_id is None and src != "GRID":
            return False
        mid = getattr(config, "match_id", None)
        if mid is not None:
            try:
                mid = int(mid)
            except (TypeError, ValueError):
                mid = None
        if mid is None:
            mid = hash(grid_series_id) % (2**31)
        team_a_is_team_one = getattr(config, "team_a_is_team_one", True)
        t = time.time()

        # Per-series schedule: skip fetch until next_fetch_ts
        if not next_fetch_allowed(grid_series_id, self._grid_next_fetch_ts, t):
            return True

        if session_runtime is not None:
            ctx = session_runtime.ctx
        else:
            if self._match_context is None or self._match_context_match_id != mid:
                self._match_context = MatchContext(match_id=mid)
                self._match_context_match_id = mid
            ctx = self._match_context

        payload = fetch_series_state(grid_series_id)

        if is_rate_limit_response(payload):
            after_rate_limit(grid_series_id, self._grid_next_fetch_ts, t)
            self._grid_last_rate_limit_reason[grid_series_id] = "rate_limit_429"
            h = ctx.get_or_create_source_health(SourceKind.GRID)
            h.err_count = getattr(h, "err_count", 0) + 1
            h.last_err_ts = t
            h.last_reason = "rate_limit_429"
            return True

        if payload.get("errors"):
            return True
        data = payload.get("data") or {}
        ss = data.get("seriesState")
        if not isinstance(ss, dict):
            return True
        after_success(grid_series_id, self._grid_next_fetch_ts, t)
        if session_runtime is not None:
            if session_runtime.grid_state is None:
                session_runtime.grid_state = GridState()
            session_runtime.grid_state = reduce_event(session_runtime.grid_state, ss)
            state = session_runtime.grid_state
        else:
            if self._grid_state is None or self._grid_series_id != grid_series_id:
                self._grid_state = GridState()
                self._grid_series_id = grid_series_id
            self._grid_state = reduce_event(self._grid_state, ss)
            state = self._grid_state
        frame = grid_state_to_frame(state, team_a_is_team_one=team_a_is_team_one, timestamp=t)
        new_key = compute_monotonic_key_from_grid_state(state, t)
        frame_dict = asdict(frame)
        env = CanonicalFrameEnvelope(
            match_id=mid,
            source=SourceKind.GRID,
            observed_ts=t,
            key=new_key,
            frame=frame_dict,
            valid=True,
        )
        if not self._ingest_canonical_envelope(ctx, env, None):
            return True
        round_phase_drive = getattr(state, "round_phase", None)
        from engine.telemetry.selector import allowed_to_drive, default_source_selector_policy
        allow_drive, drive_reason = allowed_to_drive(ctx, env.source, t, round_phase_drive, default_source_selector_policy())
        if not allow_drive:
            ctx.last_drive_deny_reason = drive_reason
            return True
        ctx.identity = IdentityEntry(
            team_a_is_team_one=team_a_is_team_one,
            provider_team_a_id=state.team_a_id,
            provider_team_b_id=state.team_b_id,
        )
        round_phase_sel = state.round_phase
        _sel_policy = default_source_selector_policy()
        _sel_decision = decide(ctx, t, _sel_policy, round_phase_sel)
        maybe_switch(ctx, _sel_decision, t, _sel_policy, round_phase_sel)

        from engine.compute.bounds import compute_bounds
        from engine.compute.rails import compute_rails
        from engine.compute.resolve import resolve_p_hat
        from engine.state.reducer import reduce_state
        from engine.diagnostics.inter_map_break import detect_inter_map_break
        from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
        from engine.telemetry.initial_state import initial_state

        if write_to_store:
            old_state = await self._store.get_state()
        else:
            old_state = (
                session_runtime.last_state
                if (session_runtime is not None and session_runtime.last_state is not None)
                else initial_state(config)
            )
        new_state = reduce_state(old_state, frame, config)
        if session_runtime is not None:
            session_runtime.last_state = new_state
            session_runtime.last_frame = asdict(frame)
            session_runtime.last_update_ts = t
        bounds_result = compute_bounds(frame, config, new_state)
        bound_low, bound_high = bounds_result[0], bounds_result[1]
        bounds_debug = bounds_result[2] if len(bounds_result) > 2 else {}
        is_break, break_reason = detect_inter_map_break(frame, new_state)
        if is_break:
            # Stage 3B: continuity from store only when write_to_store; else session-local only.
            if write_to_store:
                cur = await self._store.get_current()
                d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
                last_p = d.get("p_hat")
            else:
                last_p = getattr(session_runtime, "last_p_hat", None)
            p_hat, dbg = _inter_map_break_phat_and_dbg(
                bound_low, bound_high, break_reason, last_p, dict(bounds_debug)
            )
            rail_low = dbg["map_low"]
            rail_high = dbg["map_high"]
            dbg["source"] = "GRID"
            dbg["grid_series_id"] = grid_series_id
            dbg["match_context_diag"] = _match_context_diag(ctx, _sel_decision.to_diag() if _sel_decision else None)
        else:
            bounds = (bound_low, bound_high)
            rails_result = compute_rails(frame, config, new_state, bounds, source="GRID", replay_kind=None)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False))
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}
            dbg["source"] = "GRID"
            dbg["grid_series_id"] = grid_series_id
            dbg["match_context_diag"] = _match_context_diag(ctx, _sel_decision.to_diag() if _sel_decision else None)
        if session_runtime is not None:
            setattr(session_runtime, "last_p_hat", p_hat)
        dbg["raw"] = build_raw_debug(frame)
        dbg["fragility"] = compute_fragility_debug(frame)
        dbg.update(_compute_dominance_features(frame))
        market_mid, market_dbg = self._get_market_for_point(config)
        dbg.update(market_dbg)
        from engine.compute.breach import compute_breach_flags
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
        explain = dbg.get("explain") or _minimal_explain("live", rail_low, rail_high, p_hat, clamp_reason="grid")
        identity_kw = _team_identity_for_point(frame, config)
        point = HistoryPoint(
            time=t,
            p_hat=p_hat,
            bound_low=bound_low,
            bound_high=bound_high,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=new_state.segment_id,
            map_index=getattr(frame, "map_index", 0),
            round_number=getattr(state, "round_index", None),
            game_number=getattr(state, "game_index", None),
            explain=explain,
            **identity_kw,
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
        if session_runtime is not None:
            session_runtime.last_state = new_state
            session_runtime.last_frame = asdict(frame)
            session_runtime.last_update_ts = t
            session_runtime.last_error = None
        clamp_reason_grid = "grid"
        if isinstance(explain, dict) and isinstance(explain.get("final"), dict):
            clamp_reason_grid = explain["final"].get("clamp_reason", "grid")
        teams_grid = getattr(frame, "teams", None) or ("", "")
        if isinstance(teams_grid, (list, tuple)) and len(teams_grid) >= 2:
            teams_grid = (str(teams_grid[0]), str(teams_grid[1]))
        else:
            teams_grid = ("", "")
        self._point_emit_log(
            session_source="GRID",
            session_id=grid_series_id or "",
            match_key=grid_series_id or "",
            teams=teams_grid,
            map_i=getattr(frame, "map_index", None),
            round_i=getattr(state, "round_index", None),
            ts=t,
            drv_summary=dbg.get("fragility") or {},
            p_hat=p_hat,
            bounds=(bound_low, bound_high),
            rails=(rail_low, rail_high),
            clamp_ctx=clamp_reason_grid,
            emit_allowed=write_to_store,
            emit_reason="ok",
        )
        if write_to_store:
            await self._store.append_point(point, new_state, derived)
            await self._broadcast_point(point)
        return True

    async def _tick_grid_for_series(
        self, series_id: str, config: Config, is_primary: bool
    ) -> bool:
        """Multi-session: tick one GRID series; use session ctx and grid_state; only primary writes to store."""
        from engine.telemetry import SessionKey, SourceKind
        mid = hash(series_id) % (2**31)
        key = SessionKey(source=SourceKind.GRID, id=series_id)
        session = self._get_or_create_session(key, mid)
        return await self._tick_grid(
            config,
            effective_series_id=series_id,
            session_runtime=session,
            write_to_store=is_primary,
        )

    async def _tick_replay_point_passthrough(
        self,
        payload: dict[str, Any],
        config: Config,
        policy_meta: dict[str, Any] | None = None,
    ) -> bool:
        """Point replay transition mode: append point as-is, explicitly tagged as non-canonical."""
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
        _mi, _rn, _gn = payload.get("map_index"), payload.get("round_number"), payload.get("game_number")
        try:
            map_index_pt = int(_mi) if _mi is not None else None
        except (TypeError, ValueError):
            map_index_pt = None
        try:
            round_number_pt = int(_rn) if _rn is not None else None
        except (TypeError, ValueError):
            round_number_pt = None
        try:
            game_number_pt = int(_gn) if _gn is not None else None
        except (TypeError, ValueError):
            game_number_pt = None
        quarantine_reason = "point_payload_bypasses_canonical_pipeline"
        quarantine_status = "transition_passthrough_allowed"
        emit_quarantined_points = bool(getattr(config, "replay_emit_quarantined_points", True))
        state = await self._store.get_state()
        point = HistoryPoint(
            time=t,
            p_hat=p,
            bound_low=lo,
            bound_high=hi,
            rail_low=rail_low,
            rail_high=rail_high,
            market_mid=market_mid,
            segment_id=seg,
            map_index=map_index_pt,
            round_number=round_number_pt,
            game_number=game_number_pt,
            explain=explain,
            event=event,
            **_team_identity_for_point(getattr(state, "last_frame", None), config),
        )
        derived = Derived(
            p_hat=p,
            bound_low=lo,
            bound_high=hi,
            rail_low=rail_low,
            rail_high=rail_high,
            kappa=0.0,
            debug={
                "explain": explain,
                "replay_mode": "point_passthrough",
                "replay_contract_class": "non_canonical_point",
                "replay_quarantine_status": quarantine_status,
                "replay_quarantine_reason": quarantine_reason,
                "replay_point_policy_decision": REPLAY_POINT_POLICY_DECISION_TRANSITION_PASSTHROUGH,
                **(policy_meta or {}),
            },
        )
        await self._store.append_point(point, state, derived)
        if emit_quarantined_points:
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
            self._replay_point_inputs_seen = 0
            self._replay_point_rejected_count = 0
            self._replay_point_transition_passthrough_count = 0
            self._replay_point_reject_reason_counts = {}
            self._replay_last_point_policy_decision = None
            self._replay_last_point_policy_reason = None
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

        # Point replay: contract gate (default reject; optional explicit transition passthrough only).
        if self._replay_format == "point":
            allow_transition_passthrough, reason_code, policy_meta = self._decide_point_like_replay_policy(config)
            self._record_point_like_policy_decision(allow_transition_passthrough, reason_code)
            if not allow_transition_passthrough:
                logger.warning(
                    "replay point-like payload rejected by contract gate",
                    extra={
                        "index": self._replay_index,
                        "reason_code": reason_code,
                        "replay_contract_policy": policy_meta.get("replay_contract_policy"),
                        "transition_enabled": policy_meta.get("replay_point_transition_enabled"),
                        "transition_sunset_epoch": policy_meta.get("replay_point_transition_sunset_epoch"),
                    },
                )
                self._replay_index += 1
                return True
            return await self._tick_replay_point_passthrough(payload, config, policy_meta=policy_meta)

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
            # Stage 3B: REPLAY is single-session; continuity from store.
            cur = await self._store.get_current()
            d = (cur.get("derived") or {}) if isinstance(cur, dict) else {}
            last_p = d.get("p_hat")
            p_hat, dbg = _inter_map_break_phat_and_dbg(
                bound_low, bound_high, break_reason, last_p, dict(bounds_debug)
            )
            rail_low = dbg["map_low"]
            rail_high = dbg["map_high"]
        else:
            bounds = (bound_low, bound_high)
            _src = getattr(config, "source", None) or "REPLAY"
            _kind = getattr(self, "_replay_format", None) or "raw"
            rails_result = compute_rails(frame, config, new_state, bounds, source=_src, replay_kind=_kind)
            rail_low, rail_high = rails_result[0], rails_result[1]
            rails_debug = rails_result[2] if len(rails_result) > 2 else {}
            setattr(config, "contract_testing_mode", getattr(config, "invariant_diagnostics", False))
            p_hat, dbg = resolve_p_hat(frame, config, new_state, (rail_low, rail_high))
            dbg = {**dbg, **bounds_debug, **rails_debug}
        dbg["replay_mode"] = "raw_contract"

        from engine.diagnostics.fragility_cs2 import build_raw_debug, compute_fragility_debug
        dbg["raw"] = build_raw_debug(frame)
        dbg["fragility"] = compute_fragility_debug(frame)
        dbg.update(_compute_dominance_features(frame))
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
        # Outcome label events (round_result, segment_result) from replay payload Ã¢â‚¬â€ same logic as live
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
        game_number_replay = None
        round_number_replay = None
        if isinstance(payload, dict):
            try:
                if payload.get("game_number") is not None:
                    game_number_replay = int(payload["game_number"])
            except (TypeError, ValueError):
                pass
            try:
                if payload.get("round_number") is not None:
                    round_number_replay = int(payload["round_number"])
            except (TypeError, ValueError):
                pass
        map_index_replay = (game_number_replay - 1) if game_number_replay is not None else getattr(new_state, "map_index", None)
        map_identity_replay = self._ensure_map_identity_from_frame(frame, config, game_number_replay, map_index_replay)
        replay_identity_kw = (
            _team_identity_from_cache(map_identity_replay)
            if map_identity_replay
            else _team_identity_for_point(frame, config)
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
            map_index=map_index_replay,
            round_number=round_number_replay,
            game_number=game_number_replay,
            explain=replay_explain,
            **replay_identity_kw,
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
        """Tick: get config; REPLAY or BO3 or GRID, append_point, broadcast."""
        while not self._stop.is_set():
            try:
                config = await self._store.get_config()
                interval = max(5.0, float(getattr(config, "poll_interval_s", 5.0)))
                await self._maybe_poll_market(config)
            except Exception:
                interval = 5.0
            did_replay = await self._tick_replay(config)
            if did_replay:
                speed = max(0.1, float(getattr(config, "replay_speed", 1.0)))
                sleep_interval = interval / speed
            else:
                sleep_interval = interval
                did_tick = False
                if self._is_multi_session(config):
                    now_ts = time.time()
                    # Normalized primary from config (needed for pinned-only BO3 and probe skip)
                    _src = getattr(config, "primary_session_source", None)
                    _id = getattr(config, "primary_session_id", None)
                    primary_src = (str(_src).strip().upper() if _src else None) or None
                    primary_id = (str(_id).strip() if _id is not None else None) or None
                    primary_pinned = bool(primary_src and primary_id)
                    # When pinned to BO3: do NOT run list refresh (protects primary snapshot cadence)
                    if getattr(config, "bo3_auto_track", False):
                        if not (primary_pinned and primary_src == "BO3"):
                            await self._maybe_run_bo3_validation_probes(config, now_ts)
                        # Prune expired blacklist entries
                        invalid = getattr(self, "_bo3_invalid_snapshot_ids", {})
                        if invalid:
                            self._bo3_invalid_snapshot_ids = {k: v for k, v in invalid.items() if v > now_ts}
                    bo3_ids = self._effective_bo3_match_ids(
                        config,
                        primary_pinned=primary_pinned,
                        primary_src=primary_src,
                        primary_id=primary_id,
                    )
                    grid_ids = self._effective_grid_series_ids(config, now_ts=now_ts)

                    # Proof log: tick decision (BO3_RATE_DEBUG)
                    if BO3_RATE_DEBUG:
                        bo3_auto = getattr(config, "bo3_auto_track", False)
                        validation_probes_this_tick = bo3_auto and not (primary_pinned and primary_src == "BO3")
                        logger.info(
                            "BO3 tick decision: primary_src=%s primary_id=%s primary_pinned=%s bo3_auto_track=%s effective_bo3_ids_count=%s bo3_ids_prefix=%s validation_probes_this_tick=%s",
                            primary_src,
                            primary_id,
                            primary_pinned,
                            bo3_auto,
                            len(bo3_ids),
                            bo3_ids[:5] if bo3_ids else [],
                            validation_probes_this_tick,
                        )

                    # Determine effective primary: pinned from config, else sticky, else fallback. Do not rotate on reorder.
                    eff_primary_src: str | None
                    eff_primary_id: str | None
                    reason: str
                    if primary_pinned:
                        eff_primary_src = primary_src
                        eff_primary_id = primary_id
                        reason = "pinned"
                    else:
                        eff_primary_src = None
                        eff_primary_id = None
                        reason = "fallback"
                        sticky = getattr(self, "_last_primary_session", None)
                        if sticky is not None:
                            sticky_src, sticky_id = sticky
                            in_bo3 = sticky_src == "BO3" and any(str(mid) == sticky_id for mid in bo3_ids)
                            in_grid = sticky_src == "GRID" and any(sid == sticky_id for sid in grid_ids)
                            if in_bo3 or in_grid:
                                eff_primary_src, eff_primary_id = sticky_src, sticky_id
                                reason = "sticky"
                            else:
                                reason = "sticky_gone"
                        if eff_primary_src is None:
                            if bo3_ids:
                                eff_primary_src, eff_primary_id = "BO3", str(bo3_ids[0])
                            elif grid_ids:
                                eff_primary_src, eff_primary_id = "GRID", grid_ids[0] if grid_ids else None
                            if reason != "sticky_gone":
                                reason = "fallback"

                    # Low-volume log when effective primary changes
                    last_eff = getattr(self, "_last_eff_primary_session", None)
                    new_eff = (eff_primary_src, eff_primary_id) if (eff_primary_src and eff_primary_id) else None
                    if last_eff != new_eff:
                        logger.info(
                            "primary_session change: old=%s new=%s reason=%s",
                            last_eff,
                            new_eff,
                            reason,
                            extra={"old": last_eff, "new": new_eff, "reason": reason},
                        )
                    self._last_eff_primary_session = new_eff

                    # Keep sticky in sync: when pinned, remember so unpin stays stable
                    if primary_pinned and new_eff:
                        self._last_primary_session = new_eff

                    last_primary_this_tick: tuple[str, str] | None = None

                    for mid in bo3_ids:
                        is_primary = eff_primary_src == "BO3" and eff_primary_id == str(mid)
                        did = await self._tick_bo3_for_match(mid, config, is_primary)
                        if did and is_primary:
                            last_primary_this_tick = ("BO3", str(mid))
                        did_tick = did or did_tick
                        if did and getattr(self, "_bo3_buf_consecutive_failures", 0) >= 3:
                            sleep_interval += 5.0

                    for sid in grid_ids:
                        is_primary = eff_primary_src == "GRID" and eff_primary_id == sid
                        did = await self._tick_grid_for_series(sid, config, is_primary)
                        if did and is_primary:
                            last_primary_this_tick = ("GRID", sid)
                        did_tick = did or did_tick

                    if last_primary_this_tick is not None:
                        self._last_primary_session = last_primary_this_tick
                    # List refresh only when NOT pinned to BO3, and after primary snapshot (protects cadence)
                    if getattr(config, "bo3_auto_track", False) and not (primary_pinned and primary_src == "BO3"):
                        await self._maybe_refresh_bo3_candidates(config, time.time())
                else:
                    src = getattr(config, "source", None)
                    if src == "GRID":
                        did_tick = await self._tick_grid(config)
                    elif src == "BO3":
                        did_tick = await self._tick_bo3(config)
                        if did_tick and getattr(self, "_bo3_buf_consecutive_failures", 0) >= 3:
                            sleep_interval += 5.0
                # When no tick (no replay, no BO3/GRID update): do not inject synthetic points
            try:
                await asyncio.sleep(sleep_interval)
            except asyncio.CancelledError:
                break
        self._task = None





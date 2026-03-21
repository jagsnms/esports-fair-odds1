"""
Thread-safe in-memory store for State, Derived, and history (ring buffer).
Optional persistent JSONL recording of HistoryPoints via env HISTORY_RECORD_ENABLED / HISTORY_RECORD_JSONL_PATH.

Score-space diagnostics (separate file): schema score_diag_v2 (v1 deprecated); NOT backward-compatible with old files;
kept separate intentionally so existing history_points.jsonl and calibration data remain comparable.
Pre-asymptote raw score, term contribs, residual_contrib (exact reconstruction), term_raw/term_coef for learning true weights.
Uses same mechanism as history: HISTORY_SCORE_RECORD_ENABLED (default true) and HISTORY_SCORE_RECORD_JSONL_PATH.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import asdict
from typing import Any

from engine.config import merge_config
from engine.models import Config, Derived, Frame, HistoryPoint, State

# Persistent history recording (env): default on, path default logs/history_points.jsonl
_HISTORY_RECORD_ENABLED = os.environ.get("HISTORY_RECORD_ENABLED", "true").strip().lower() in ("1", "true", "yes")
_HISTORY_RECORD_JSONL_PATH = os.environ.get("HISTORY_RECORD_JSONL_PATH", "logs/history_points.jsonl").strip() or "logs/history_points.jsonl"

# Score-space diagnostics (same mechanism as history): default on, path default logs/history_score_points.jsonl
_HISTORY_SCORE_RECORD_ENABLED = os.environ.get("HISTORY_SCORE_RECORD_ENABLED", "true").strip().lower() in ("1", "true", "yes")
_HISTORY_SCORE_RECORD_JSONL_PATH = (
    os.environ.get("HISTORY_SCORE_RECORD_JSONL_PATH") or os.environ.get("HISTORY_SCORE_RECORD_PATH") or "logs/history_score_points.jsonl"
).strip() or "logs/history_score_points.jsonl"

_logger = logging.getLogger(__name__)


def _state_derived_to_dict(state: State, derived: Derived) -> dict[str, Any]:
    """Serialize State + Derived to JSON-serializable dict."""
    state_d = asdict(state)
    derived_d = asdict(derived)
    # Corridor naming aliases for series-winner contract:
    # - series_low/high: series corridor (formerly bound_low/high)
    # - map_low/high: map corridor (formerly rail_low/high)
    if "bound_low" in derived_d and "bound_high" in derived_d:
        derived_d.setdefault("series_low", derived_d["bound_low"])
        derived_d.setdefault("series_high", derived_d["bound_high"])
    if "rail_low" in derived_d and "rail_high" in derived_d:
        derived_d.setdefault("map_low", derived_d["rail_low"])
        derived_d.setdefault("map_high", derived_d["rail_high"])
    # Config/Frame are nested dataclasses; asdict recurses. Tuples become lists.
    return {"state": state_d, "derived": derived_d}


def _history_point_to_wire(p: HistoryPoint) -> dict[str, Any]:
    """Canonical wire format for WS point, GET /api/v1/state/history, and JSONL recording.
    Includes: t, p, lo, hi, m, seg, rail_low/rail_high, series_low/series_high, map_low/map_high,
    and when present: explain, event (round_result/segment_result).
    """
    out: dict[str, Any] = {
        "t": p.time,
        "p": p.p_hat,
        "lo": p.bound_low,
        "hi": p.bound_high,
        "m": p.market_mid,
        "series_low": p.bound_low,
        "series_high": p.bound_high,
    }
    if hasattr(p, "rail_low"):
        out["rail_low"] = p.rail_low
        out.setdefault("map_low", p.rail_low)
    if hasattr(p, "rail_high"):
        out["rail_high"] = p.rail_high
        out.setdefault("map_high", p.rail_high)
    if hasattr(p, "segment_id"):
        out["seg"] = p.segment_id
    if getattr(p, "map_index", None) is not None:
        out["map_index"] = p.map_index
    if getattr(p, "round_number", None) is not None:
        out["round_number"] = p.round_number
    if getattr(p, "game_number", None) is not None:
        out["game_number"] = p.game_number
    # Full payload: include explain and event when present (same as WS / GET history)
    expl = getattr(p, "explain", None)
    if expl is not None:
        out["explain"] = expl
    ev = getattr(p, "event", None)
    if ev is not None:
        out["event"] = ev
    for key in ("match_id", "team_one_id", "team_two_id", "team_one_provider_id", "team_two_provider_id", "team_a_is_team_one", "a_side"):
        val = getattr(p, key, None)
        if val is not None:
            out[key] = val
    return out


def _append_jsonl(path: str, obj: dict[str, Any]) -> None:
    """Append one JSON object as a single line to path. Creates parent dir if needed. Swallows OSError."""
    try:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _make_score_diag_record(point: HistoryPoint) -> dict[str, Any] | None:
    """Build one score-space diagnostic record for history_score_points.jsonl, or None if skip.
    Schema score_diag_v2: adds residual_contrib (exact reconstruction), contrib_sum, term_raw, term_coef.
    Only emits for compute ticks with explain/final present, phase != idle, and score_raw present.
    """
    explain = getattr(point, "explain", None)
    if not isinstance(explain, dict):
        return None
    final = explain.get("final")
    if not isinstance(final, dict):
        return None
    phase = explain.get("phase")
    if phase == "idle":
        return None
    score_raw = explain.get("score_raw")
    if score_raw is None:
        return None
    term_contribs = explain.get("term_contribs")
    if not isinstance(term_contribs, dict):
        term_contribs = {}
    base_intercept = explain.get("base_intercept")
    if base_intercept is None:
        base_intercept = 0.0
    contrib_sum = float(base_intercept) + sum(float(v) for v in term_contribs.values())
    residual_contrib = float(score_raw) - contrib_sum
    p_hat_final = final.get("p_hat_final")
    if p_hat_final is None:
        p_hat_final = getattr(point, "p_hat", 0.5)
    record: dict[str, Any] = {
        "schema": "score_diag_v2",
        "ts_ms": int(round(getattr(point, "time", 0.0) * 1000)),
        "t": getattr(point, "time", 0.0),
        "match_id": getattr(point, "match_id", None),
        "game_number": getattr(point, "game_number", None),
        "map_index": getattr(point, "map_index", None),
        "round_number": getattr(point, "round_number", None),
        "phase": phase,
        "round_phase": explain.get("round_phase"),
        "clamp_reason": final.get("clamp_reason"),
        "p_hat_final": float(p_hat_final),
        "rail_low": getattr(point, "rail_low", None),
        "rail_high": getattr(point, "rail_high", None),
        "series_low": getattr(point, "bound_low", None),
        "series_high": getattr(point, "bound_high", None),
        "target_p_hat": explain.get("target_p_hat"),
        "p_hat_prev": explain.get("p_hat_prev"),
        "movement_confidence": explain.get("movement_confidence"),
        "expected_p_hat_after_movement": explain.get("expected_p_hat_after_movement"),
        "movement_gap_abs": explain.get("movement_gap_abs"),
        "alive_counts": explain.get("alive_counts"),
        "hp_totals": explain.get("hp_totals"),
        "loadout_totals": explain.get("loadout_totals"),
        "q_intra_total": explain.get("q_intra_total"),
        "score_raw": float(score_raw),
        "p_unshaped": explain.get("p_unshaped"),
        "term_contribs": {k: float(v) for k, v in term_contribs.items()},
        "base_intercept": float(base_intercept),
        "contrib_sum": contrib_sum,
        "residual_contrib": residual_contrib,
    }
    term_raw = explain.get("term_raw")
    term_coef = explain.get("term_coef")
    if isinstance(term_raw, dict) and isinstance(term_coef, dict):
        record["term_raw"] = {k: float(v) for k, v in term_raw.items() if isinstance(v, (int, float))}
        record["term_coef"] = {k: float(v) for k, v in term_coef.items() if isinstance(v, (int, float))}
    # Team identity for label-alignment audit (when present on point)
    for key in ("team_one_id", "team_two_id", "team_one_provider_id", "team_two_provider_id", "a_side", "team_a_is_team_one"):
        val = getattr(point, key, None)
        if val is not None:
            record[key] = val
    return record


class MemoryStore:
    """Thread-safe in-memory store. Uses asyncio.Lock."""

    def __init__(self, max_history: int = 2000, max_breach_events: int = 200) -> None:
        self._lock = asyncio.Lock()
        self._state = State(config=Config(poll_interval_s=5.0))
        self._derived = Derived()
        self._history: deque[HistoryPoint] = deque(maxlen=max_history)
        self._breach_events: deque[dict[str, Any]] = deque(maxlen=max_breach_events)
        self._points_without_explain_count: int = 0
        self._score_diag_skipped_count: int = 0  # compute ticks skipped (e.g. score_raw missing)
        # Touch score diag file on startup when enabled (same idea as history; file exists before first tick)
        if _HISTORY_SCORE_RECORD_ENABLED and _HISTORY_SCORE_RECORD_JSONL_PATH:
            try:
                dirpath = os.path.dirname(_HISTORY_SCORE_RECORD_JSONL_PATH)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                open(_HISTORY_SCORE_RECORD_JSONL_PATH, "a", encoding="utf-8").close()
            except OSError:
                pass

    async def get_current(self) -> dict[str, Any]:
        """Serialize current State + Derived."""
        async with self._lock:
            return _state_derived_to_dict(self._state, self._derived)

    async def clear_display(self) -> None:
        """Clear in-memory history and reset current state/derived for display. Keeps config. No JSONL wipe."""
        async with self._lock:
            self._history.clear()
            self._state = State(
                config=self._state.config,
                last_frame=None,
                team_mapping=self._state.team_mapping,
                map_index=0,
                last_total_rounds=0,
                segment_id=0,
                last_series_score=None,
                last_map_index=None,
            )
            self._derived = Derived()

    async def get_history(self, limit: int = 2000) -> list[dict[str, Any]]:
        """Last `limit` points in wire format."""
        async with self._lock:
            points = list(self._history)[-limit:]
            return [_history_point_to_wire(p) for p in points]

    async def append_point(
        self,
        point: HistoryPoint,
        state: State,
        derived: Derived,
    ) -> None:
        """Append one history point and update current state/derived. If enabled, append wire payload to JSONL."""
        async with self._lock:
            has_explain = getattr(point, "explain", None) is not None
            has_event = getattr(point, "event", None) is not None
            if not has_explain and not has_event:
                self._points_without_explain_count += 1
                _logger.warning(
                    "point appended without explain or event (count=%d); thin-wire points poison calibration",
                    self._points_without_explain_count,
                )
            self._history.append(point)
            self._state = state
            self._derived = derived
            if _HISTORY_RECORD_ENABLED and _HISTORY_RECORD_JSONL_PATH:
                # Full wire payload (same as WS and GET /api/v1/state/history), including explain and event
                wire = _history_point_to_wire(point)
                try:
                    dirpath = os.path.dirname(_HISTORY_RECORD_JSONL_PATH)
                    if dirpath:
                        os.makedirs(dirpath, exist_ok=True)
                    with open(_HISTORY_RECORD_JSONL_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(wire, ensure_ascii=False) + "\n")
                except OSError:
                    pass
            # Score-space diagnostics (separate file; same mechanism as history)
            if _HISTORY_SCORE_RECORD_ENABLED and _HISTORY_SCORE_RECORD_JSONL_PATH:
                rec = _make_score_diag_record(point)
                if rec is not None:
                    _append_jsonl(_HISTORY_SCORE_RECORD_JSONL_PATH, rec)
                else:
                    expl = getattr(point, "explain", None)
                    if isinstance(expl, dict) and isinstance(expl.get("final"), dict) and expl.get("phase") != "idle":
                        self._score_diag_skipped_count += 1

    async def set_current(self, state: State, derived: Derived) -> None:
        """Set current state and derived without appending to history."""
        async with self._lock:
            self._state = state
            self._derived = derived

    async def update_config(self, partial: dict[str, Any]) -> None:
        """Merge partial config update. If source or match_id changed (old != new), clear history
        and reset state.last_total_rounds to avoid mixing points between sources or between matches."""
        async with self._lock:
            current = self._state.config
            new_config = merge_config(current, partial)
            source_changed = getattr(current, "source", None) != getattr(new_config, "source", None)
            match_id_changed = getattr(current, "match_id", None) != getattr(new_config, "match_id", None)
            if source_changed or match_id_changed:
                self._history.clear()
                last_frame = None
                last_total_rounds = 0
                segment_id = 0
                last_series_score = None
                last_map_index = None
                map_index = 0
            else:
                last_frame = self._state.last_frame
                last_total_rounds = self._state.last_total_rounds
                segment_id = getattr(self._state, "segment_id", 0)
                last_series_score = getattr(self._state, "last_series_score", None)
                last_map_index = getattr(self._state, "last_map_index", None)
                map_index = self._state.map_index
            self._state = State(
                config=new_config,
                last_frame=last_frame,
                team_mapping=self._state.team_mapping,
                map_index=map_index,
                last_total_rounds=last_total_rounds,
                segment_id=segment_id,
                last_series_score=last_series_score,
                last_map_index=last_map_index,
            )

    async def get_config(self) -> Config:
        """Return current config."""
        async with self._lock:
            return self._state.config

    async def get_state(self) -> State:
        """Return current state (read-only; do not mutate)."""
        async with self._lock:
            return self._state

    async def append_breach_event(self, event: dict[str, Any]) -> None:
        """Append one breach event to the ring-buffer. Caller holds event dict keys."""
        async with self._lock:
            self._breach_events.append(event)

    async def get_breach_events(self, limit: int = 200) -> list[dict[str, Any]]:
        """Return most recent breach events (newest last)."""
        async with self._lock:
            events = list(self._breach_events)[-limit:]
            return events

    async def clear_breach_events(self) -> None:
        """Clear the breach events ring-buffer. Used by POST /debug/reset."""
        async with self._lock:
            self._breach_events.clear()

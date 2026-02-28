"""
ML-ready trading setup logger: PHAT-targeted "series line dislocation" episodes.
Paper only. No execution. Emits setup_trigger, episode_start, episode_end, episode_outcome
into history via HistoryPoint.event for calibration/ML.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

# Setup parameters (constants)
DELTA_ENTRY = 0.07
MIN_SECONDS_BETWEEN_SAME_DIR = 60.0
MAX_EPISODES_PER_MAP = 1
SETUP_NAME = "series_line_dislocation_phat_target"
PROGRESS_CLAMP_LO = -1.0
PROGRESS_CLAMP_HI = 2.0

# Trailing reversal entry mode (env TRADE_EPISODE_ENTRY_MODE = "immediate" | "trailing_reversal")
def _entry_mode() -> str:
    return (os.environ.get("TRADE_EPISODE_ENTRY_MODE") or "trailing_reversal").strip().lower()


MIN_PHAT_GAP = 0.05
TRAIL_PTS = 0.03
CONFIRM_TICKS = 2


def _safe_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _p_hat_from_explain(explain: dict | None, fallback_p: float) -> float:
    if explain and isinstance(explain.get("final"), dict):
        p = explain["final"].get("p_hat_final")
        if p is not None:
            return _safe_float(p, fallback_p)
    return fallback_p


class TradeEpisodeManager:
    """
    Tracks series-line-dislocation episodes: trigger on executable price vs S_A_line / S_B_line,
    track progress toward PHAT, end at next map start or match end.
    Emits events for Runner to append as HistoryPoint.event (no execution).
    """

    def __init__(self) -> None:
        self._last_trigger_time: dict[str, float] = {}  # direction -> ts
        self._episodes_this_map: dict[str, int] = {}  # direction -> count
        self._current_map_id: int | None = None  # segment_id or game_number
        self._last_seen_map_id: int | None = None
        self._last_seen_round_phase: str | None = None
        self._active: dict[str, Any] | None = None  # active episode state
        self._episode_counter = 0
        # Trailing reversal: per direction { armed, P_extreme, extreme_gap, armed_ts, extreme_ts,
        #   confirm_count_price, confirm_count_gap, candidate_price_emitted, candidate_gap_emitted }
        self._trail_state: dict[str, dict[str, Any]] = {
            "LONG_A": {
                "armed": False, "P_extreme": 0.0, "extreme_gap": 0.0, "armed_ts": 0.0, "extreme_ts": 0.0,
                "confirm_count_price": 0, "confirm_count_gap": 0,
                "candidate_price_emitted": False, "candidate_gap_emitted": False,
            },
            "SHORT_A": {
                "armed": False, "P_extreme": 0.0, "extreme_gap": 0.0, "armed_ts": 0.0, "extreme_ts": 0.0,
                "confirm_count_price": 0, "confirm_count_gap": 0,
                "candidate_price_emitted": False, "candidate_gap_emitted": False,
            },
        }

    def _map_id(self, segment_id: int | None, game_number: int | None) -> int:
        if game_number is not None:
            return int(game_number)
        return int(segment_id) if segment_id is not None else 0

    def _is_map_start(
        self,
        segment_id: int | None,
        game_number: int | None,
        round_phase: str | None,
    ) -> bool:
        """True if this tick is the first IN_PROGRESS of a new map."""
        phase_upper = (round_phase or "").strip().upper()
        if phase_upper != "IN_PROGRESS":
            return False
        mid = self._map_id(segment_id, game_number)
        if self._last_seen_map_id is not None and mid > self._last_seen_map_id:
            return True
        return False

    def _reset_map_counts(self, map_id: int) -> None:
        """On map change: reset _episodes_this_map and optionally trail state.
        Do NOT reset P_extreme, gap_extreme, confirm_count_*, candidate_*_emitted, or armed
        when there is an active episode or any armed window. Only full reset when
        we have no active episode and no armed state (or definitively new map with no episode).
        """
        if self._current_map_id == map_id:
            return
        # Map changed
        has_active_episode = self._active is not None
        has_armed = any(
            self._trail_state[k].get("armed") for k in self._trail_state
        )
        if has_active_episode or has_armed:
            # Don't reset trail state; only advance map id and reset per-map episode count
            self._current_map_id = map_id
            self._episodes_this_map = {}
            return
        # No active episode and no armed window: full reset
        self._current_map_id = map_id
        self._episodes_this_map = {}
        for key in self._trail_state:
            self._trail_state[key] = {
                "armed": False, "P_extreme": 0.0, "extreme_gap": 0.0, "armed_ts": 0.0, "extreme_ts": 0.0,
                "confirm_count_price": 0, "confirm_count_gap": 0,
                "candidate_price_emitted": False, "candidate_gap_emitted": False,
            }

    def _explain_snapshot(self, explain: dict | None) -> dict[str, Any]:
        """Compact subset of explain for ML entry context."""
        if not explain or not isinstance(explain, dict):
            return {}
        out: dict[str, Any] = {}
        if "q_terms" in explain:
            out["q_terms"] = explain["q_terms"]
        if "micro_adj" in explain:
            out["micro_adj"] = explain["micro_adj"]
        if isinstance(explain.get("final"), dict):
            out["clamp_reason"] = explain["final"].get("clamp_reason")
        if "rails" in explain and isinstance(explain["rails"], dict):
            out["corridor_width"] = explain["rails"].get("corridor_width")
        return out

    def _try_trailing_reversal(
        self,
        t: float,
        p_hat: float,
        S_A_line: float,
        S_B_line: float,
        H0: float,
        bid_yes_f: float,
        ask_yes_f: float,
        ask_no: float,
        map_id: int,
        game_number: int | None,
        segment_id: int | None,
        round_number: int | None,
        explain: dict | None,
    ) -> list[dict[str, Any]]:
        """
        Trailing reversal entry: arm when boundary + abs(gap)>=MIN_PHAT_GAP; track extreme_gap;
        fire when gap reverses by TRAIL_PTS from extreme for CONFIRM_TICKS consecutive ticks.
        S_A_line = Team A series boundary (bound_high); S_B_line = Team B series boundary (bound_low).
        """
        events: list[dict[str, Any]] = []
        no_threshold = (1.0 - S_A_line) + DELTA_ENTRY   # SHORT_A: ask_no >= (1 - S_A_line) + DELTA_ENTRY
        yes_threshold = S_B_line - DELTA_ENTRY          # LONG_A: ask_yes <= S_B_line - DELTA_ENTRY

        # LONG_A: P = ask_yes, gap = P - H0. Track P_extreme = min(P), gap_extreme = min(gap).
        P_long = ask_yes_f
        gap_long = P_long - H0
        arm_long = ask_yes_f <= yes_threshold and abs(gap_long) >= MIN_PHAT_GAP
        state_long = self._trail_state["LONG_A"]
        was_armed_long = state_long["armed"]
        if arm_long:
            if not state_long["armed"]:
                state_long["armed"] = True
                state_long["P_extreme"] = P_long
                state_long["extreme_gap"] = gap_long
                state_long["armed_ts"] = t
                state_long["extreme_ts"] = t
                state_long["confirm_count_price"] = 0
                state_long["confirm_count_gap"] = 0
                state_long["candidate_price_emitted"] = False
                state_long["candidate_gap_emitted"] = False
            else:
                state_long["P_extreme"] = min(state_long["P_extreme"], P_long)
                state_long["extreme_gap"] = min(state_long["extreme_gap"], gap_long)
                state_long["extreme_ts"] = t
            # PRICE_REVERSAL: P >= P_extreme + TRAIL_PTS
            price_trigger = P_long >= state_long["P_extreme"] + TRAIL_PTS
            if price_trigger:
                state_long["confirm_count_price"] = state_long.get("confirm_count_price", 0) + 1
            else:
                state_long["confirm_count_price"] = 0
            # GAP_REVERSAL: gap >= gap_extreme + TRAIL_PTS
            gap_trigger = gap_long >= state_long["extreme_gap"] + TRAIL_PTS
            if gap_trigger:
                state_long["confirm_count_gap"] = state_long.get("confirm_count_gap", 0) + 1
            else:
                state_long["confirm_count_gap"] = 0

            # Emit entry_candidate at most once per type per armed window
            armed_id = f"armed_long_{state_long['armed_ts']}"
            time_since_extreme = t - state_long["extreme_ts"]
            if not state_long.get("candidate_price_emitted") and state_long["confirm_count_price"] >= CONFIRM_TICKS:
                state_long["candidate_price_emitted"] = True
                events.append({
                    "event_type": "entry_candidate",
                    "candidate_type": "PRICE_REVERSAL",
                    "armed_id": armed_id,
                    "ts": t,
                    "P": P_long,
                    "H": H0,
                    "gap": gap_long,
                    "P_extreme": state_long["P_extreme"],
                    "gap_extreme": state_long["extreme_gap"],
                    "reversal_amount_price": P_long - state_long["P_extreme"],
                    "reversal_amount_gap": gap_long - state_long["extreme_gap"],
                    "time_since_extreme_s": time_since_extreme,
                    "direction": "LONG_A",
                    "instrument": "YES",
                })
            if not state_long.get("candidate_gap_emitted") and state_long["confirm_count_gap"] >= CONFIRM_TICKS:
                state_long["candidate_gap_emitted"] = True
                events.append({
                    "event_type": "entry_candidate",
                    "candidate_type": "GAP_REVERSAL",
                    "armed_id": armed_id,
                    "ts": t,
                    "P": P_long,
                    "H": H0,
                    "gap": gap_long,
                    "P_extreme": state_long["P_extreme"],
                    "gap_extreme": state_long["extreme_gap"],
                    "reversal_amount_price": P_long - state_long["P_extreme"],
                    "reversal_amount_gap": gap_long - state_long["extreme_gap"],
                    "time_since_extreme_s": time_since_extreme,
                    "direction": "LONG_A",
                    "instrument": "YES",
                })

            # Start episode at FIRST candidate, or record second candidate if episode already active
            active_same_dir = self._active is not None and self._active.get("direction") == "LONG_A"
            first_candidate: str | None = None
            if state_long["confirm_count_price"] >= CONFIRM_TICKS:
                first_candidate = "PRICE_REVERSAL"
            elif state_long["confirm_count_gap"] >= CONFIRM_TICKS:
                first_candidate = "GAP_REVERSAL"
            both_candidates_same_tick_long = (
                state_long["confirm_count_price"] >= CONFIRM_TICKS
                and state_long["confirm_count_gap"] >= CONFIRM_TICKS
            )

            if first_candidate and was_armed_long:
                if active_same_dir:
                    # Second candidate: record ts and P0, init metrics for the other entry
                    ep = self._active
                    if first_candidate == "PRICE_REVERSAL" and ep.get("price_candidate_P0") is None:
                        ep["price_candidate_ts"] = t
                        ep["price_candidate_P0"] = ask_yes_f
                        ep["MFE_price"] = 0.0
                        ep["MAE_price"] = 0.0
                        ep["max_progress_price"] = 0.0
                        ep["time_to_0_5_price"] = None
                        ep["time_to_1_0_price"] = None
                    elif first_candidate == "GAP_REVERSAL" and ep.get("gap_candidate_P0") is None:
                        ep["gap_candidate_ts"] = t
                        ep["gap_candidate_P0"] = ask_yes_f
                        ep["MFE_gap"] = 0.0
                        ep["MAE_gap"] = 0.0
                        ep["max_progress_gap"] = 0.0
                        ep["time_to_0_5_gap"] = None
                        ep["time_to_1_0_gap"] = None
                    # Add episode_id to entry_candidate events we already emitted this tick
                    for ev in events:
                        if ev.get("event_type") == "entry_candidate":
                            ev["episode_id"] = ep["episode_id"]
                    return events
                if (
                    self._episodes_this_map.get("LONG_A", 0) < MAX_EPISODES_PER_MAP
                    and t - self._last_trigger_time.get("LONG_A", 0) >= MIN_SECONDS_BETWEEN_SAME_DIR
                ):
                    self._last_trigger_time["LONG_A"] = t
                    self._episodes_this_map["LONG_A"] = self._episodes_this_map.get("LONG_A", 0) + 1
                    self._episode_counter += 1
                    episode_id = f"sl_{self._episode_counter}_{uuid.uuid4().hex[:8]}"
                    P0 = ask_yes_f
                    gap0 = P0 - H0
                    explain_snap = self._explain_snapshot(explain)
                    for ev in events:
                        if ev.get("event_type") == "entry_candidate":
                            ev["episode_id"] = episode_id
                            if both_candidates_same_tick_long:
                                ev["both_candidates_same_tick"] = True
                                ev["tie_break"] = "PRICE_FIRST"
                            else:
                                ev["both_candidates_same_tick"] = False
                    events.append({
                        "event_type": "setup_trigger",
                        "setup_name": SETUP_NAME,
                        "episode_id": episode_id,
                        "direction": "LONG_A",
                        "instrument": "YES",
                        "entry_mode": "trailing_reversal",
                    })
                    events.append({
                        "event_type": "episode_start",
                        "episode_id": episode_id,
                        "trade_id": episode_id,
                        "direction": "LONG_A",
                        "instrument": "YES",
                        "first_candidate_type": first_candidate,
                        "both_candidates_same_tick": both_candidates_same_tick_long,
                        "tie_break": "PRICE_FIRST" if both_candidates_same_tick_long else None,
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "gap": gap_long,
                        "P_extreme": state_long["P_extreme"],
                        "extreme_gap": state_long["extreme_gap"],
                        "trail_pts": TRAIL_PTS,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "DELTA_ENTRY": DELTA_ENTRY,
                        "map_index": (game_number - 1) if game_number is not None else (segment_id or 0),
                        "seg": segment_id,
                        "round_number": round_number,
                        "trigger_explain_snapshot": explain_snap,
                    })
                    self._active = {
                        "episode_id": episode_id,
                        "direction": "LONG_A",
                        "instrument": "YES",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "start_t": t,
                        "start_seg": segment_id,
                        "start_map": map_id,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "first_candidate_type": first_candidate,
                        "both_candidates_same_tick": both_candidates_same_tick_long,
                        "tie_break": "PRICE_FIRST" if both_candidates_same_tick_long else None,
                        "price_candidate_ts": t if first_candidate == "PRICE_REVERSAL" else None,
                        "price_candidate_P0": P0 if first_candidate == "PRICE_REVERSAL" else None,
                        "gap_candidate_ts": t if first_candidate == "GAP_REVERSAL" else None,
                        "gap_candidate_P0": P0 if first_candidate == "GAP_REVERSAL" else None,
                        "Pt": bid_yes_f,
                        "MFE": 0.0,
                        "MAE": 0.0,
                        "max_progress_static": 0.0,
                        "time_to_0_5": None,
                        "time_to_1_0": None,
                        "min_dist_static": abs(bid_yes_f - H0),
                        "min_dist_static_ts": t,
                        "min_dist_dynamic": abs(bid_yes_f - p_hat),
                        "min_dist_dynamic_ts": t,
                        "MFE_price": 0.0 if first_candidate == "PRICE_REVERSAL" else None,
                        "MAE_price": 0.0 if first_candidate == "PRICE_REVERSAL" else None,
                        "max_progress_price": 0.0 if first_candidate == "PRICE_REVERSAL" else None,
                        "time_to_0_5_price": None,
                        "time_to_1_0_price": None,
                        "MFE_gap": 0.0 if first_candidate == "GAP_REVERSAL" else None,
                        "MAE_gap": 0.0 if first_candidate == "GAP_REVERSAL" else None,
                        "max_progress_gap": 0.0 if first_candidate == "GAP_REVERSAL" else None,
                        "time_to_0_5_gap": None,
                        "time_to_1_0_gap": None,
                    }
                    # Do not disarm: second candidate may fire later in same armed window
                    return events
        else:
            state_long["armed"] = False
            state_long["confirm_count_price"] = 0
            state_long["confirm_count_gap"] = 0

        # SHORT_A: P = ask_no, gap = P - H0. Track P_extreme = max(P), gap_extreme = max(gap).
        P_short = ask_no
        gap_short = P_short - H0
        arm_short = ask_no >= no_threshold and abs(gap_short) >= MIN_PHAT_GAP
        state_short = self._trail_state["SHORT_A"]
        was_armed_short = state_short["armed"]
        if arm_short:
            if not state_short["armed"]:
                state_short["armed"] = True
                state_short["P_extreme"] = P_short
                state_short["extreme_gap"] = gap_short
                state_short["armed_ts"] = t
                state_short["extreme_ts"] = t
                state_short["confirm_count_price"] = 0
                state_short["confirm_count_gap"] = 0
                state_short["candidate_price_emitted"] = False
                state_short["candidate_gap_emitted"] = False
            else:
                state_short["P_extreme"] = max(state_short["P_extreme"], P_short)
                state_short["extreme_gap"] = max(state_short["extreme_gap"], gap_short)
                state_short["extreme_ts"] = t
            # PRICE_REVERSAL: P <= P_extreme - TRAIL_PTS
            price_trigger = P_short <= state_short["P_extreme"] - TRAIL_PTS
            if price_trigger:
                state_short["confirm_count_price"] = state_short.get("confirm_count_price", 0) + 1
            else:
                state_short["confirm_count_price"] = 0
            # GAP_REVERSAL: gap <= gap_extreme - TRAIL_PTS
            gap_trigger = gap_short <= state_short["extreme_gap"] - TRAIL_PTS
            if gap_trigger:
                state_short["confirm_count_gap"] = state_short.get("confirm_count_gap", 0) + 1
            else:
                state_short["confirm_count_gap"] = 0

            armed_id = f"armed_short_{state_short['armed_ts']}"
            time_since_extreme = t - state_short["extreme_ts"]
            if not state_short.get("candidate_price_emitted") and state_short["confirm_count_price"] >= CONFIRM_TICKS:
                state_short["candidate_price_emitted"] = True
                events.append({
                    "event_type": "entry_candidate",
                    "candidate_type": "PRICE_REVERSAL",
                    "armed_id": armed_id,
                    "ts": t,
                    "P": P_short,
                    "H": H0,
                    "gap": gap_short,
                    "P_extreme": state_short["P_extreme"],
                    "gap_extreme": state_short["extreme_gap"],
                    "reversal_amount_price": state_short["P_extreme"] - P_short,
                    "reversal_amount_gap": state_short["extreme_gap"] - gap_short,
                    "time_since_extreme_s": time_since_extreme,
                    "direction": "SHORT_A",
                    "instrument": "NO",
                })
            if not state_short.get("candidate_gap_emitted") and state_short["confirm_count_gap"] >= CONFIRM_TICKS:
                state_short["candidate_gap_emitted"] = True
                events.append({
                    "event_type": "entry_candidate",
                    "candidate_type": "GAP_REVERSAL",
                    "armed_id": armed_id,
                    "ts": t,
                    "P": P_short,
                    "H": H0,
                    "gap": gap_short,
                    "P_extreme": state_short["P_extreme"],
                    "gap_extreme": state_short["extreme_gap"],
                    "reversal_amount_price": state_short["P_extreme"] - P_short,
                    "reversal_amount_gap": state_short["extreme_gap"] - gap_short,
                    "time_since_extreme_s": time_since_extreme,
                    "direction": "SHORT_A",
                    "instrument": "NO",
                })

            first_candidate_short: str | None = None
            if state_short["confirm_count_price"] >= CONFIRM_TICKS:
                first_candidate_short = "PRICE_REVERSAL"
            elif state_short["confirm_count_gap"] >= CONFIRM_TICKS:
                first_candidate_short = "GAP_REVERSAL"
            both_candidates_same_tick_short = (
                state_short["confirm_count_price"] >= CONFIRM_TICKS
                and state_short["confirm_count_gap"] >= CONFIRM_TICKS
            )

            if first_candidate_short and was_armed_short:
                active_same_dir_short = self._active is not None and self._active.get("direction") == "SHORT_A"
                if active_same_dir_short:
                    ep = self._active
                    if first_candidate_short == "PRICE_REVERSAL" and ep.get("price_candidate_P0") is None:
                        ep["price_candidate_ts"] = t
                        ep["price_candidate_P0"] = ask_no
                        ep["MFE_price"] = 0.0
                        ep["MAE_price"] = 0.0
                        ep["max_progress_price"] = 0.0
                        ep["time_to_0_5_price"] = None
                        ep["time_to_1_0_price"] = None
                    elif first_candidate_short == "GAP_REVERSAL" and ep.get("gap_candidate_P0") is None:
                        ep["gap_candidate_ts"] = t
                        ep["gap_candidate_P0"] = ask_no
                        ep["MFE_gap"] = 0.0
                        ep["MAE_gap"] = 0.0
                        ep["max_progress_gap"] = 0.0
                        ep["time_to_0_5_gap"] = None
                        ep["time_to_1_0_gap"] = None
                    for ev in events:
                        if ev.get("event_type") == "entry_candidate":
                            ev["episode_id"] = ep["episode_id"]
                    return events
                if (
                    self._episodes_this_map.get("SHORT_A", 0) < MAX_EPISODES_PER_MAP
                    and t - self._last_trigger_time.get("SHORT_A", 0) >= MIN_SECONDS_BETWEEN_SAME_DIR
                ):
                    self._last_trigger_time["SHORT_A"] = t
                    self._episodes_this_map["SHORT_A"] = self._episodes_this_map.get("SHORT_A", 0) + 1
                    self._episode_counter += 1
                    episode_id = f"sl_{self._episode_counter}_{uuid.uuid4().hex[:8]}"
                    P0 = ask_no
                    gap0 = P0 - H0
                    bid_no = 1.0 - ask_yes_f
                    explain_snap = self._explain_snapshot(explain)
                    for ev in events:
                        if ev.get("event_type") == "entry_candidate":
                            ev["episode_id"] = episode_id
                            if both_candidates_same_tick_short:
                                ev["both_candidates_same_tick"] = True
                                ev["tie_break"] = "PRICE_FIRST"
                            else:
                                ev["both_candidates_same_tick"] = False
                    events.append({
                        "event_type": "setup_trigger",
                        "setup_name": SETUP_NAME,
                        "episode_id": episode_id,
                        "direction": "SHORT_A",
                        "instrument": "NO",
                        "entry_mode": "trailing_reversal",
                    })
                    events.append({
                        "event_type": "episode_start",
                        "episode_id": episode_id,
                        "trade_id": episode_id,
                        "direction": "SHORT_A",
                        "instrument": "NO",
                        "first_candidate_type": first_candidate_short,
                        "both_candidates_same_tick": both_candidates_same_tick_short,
                        "tie_break": "PRICE_FIRST" if both_candidates_same_tick_short else None,
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "gap": gap_short,
                        "P_extreme": state_short["P_extreme"],
                        "extreme_gap": state_short["extreme_gap"],
                        "trail_pts": TRAIL_PTS,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "DELTA_ENTRY": DELTA_ENTRY,
                        "map_index": (game_number - 1) if game_number is not None else (segment_id or 0),
                        "seg": segment_id,
                        "round_number": round_number,
                        "trigger_explain_snapshot": explain_snap,
                    })
                    self._active = {
                        "episode_id": episode_id,
                        "direction": "SHORT_A",
                        "instrument": "NO",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "start_t": t,
                        "start_seg": segment_id,
                        "start_map": map_id,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "first_candidate_type": first_candidate_short,
                        "both_candidates_same_tick": both_candidates_same_tick_short,
                        "tie_break": "PRICE_FIRST" if both_candidates_same_tick_short else None,
                        "price_candidate_ts": t if first_candidate_short == "PRICE_REVERSAL" else None,
                        "price_candidate_P0": P0 if first_candidate_short == "PRICE_REVERSAL" else None,
                        "gap_candidate_ts": t if first_candidate_short == "GAP_REVERSAL" else None,
                        "gap_candidate_P0": P0 if first_candidate_short == "GAP_REVERSAL" else None,
                        "Pt": bid_no,
                        "MFE": 0.0,
                        "MAE": 0.0,
                        "max_progress_static": 0.0,
                        "time_to_0_5": None,
                        "time_to_1_0": None,
                        "min_dist_static": abs(bid_no - H0),
                        "min_dist_static_ts": t,
                        "min_dist_dynamic": abs(bid_no - p_hat),
                        "min_dist_dynamic_ts": t,
                        "MFE_price": 0.0 if first_candidate_short == "PRICE_REVERSAL" else None,
                        "MAE_price": 0.0 if first_candidate_short == "PRICE_REVERSAL" else None,
                        "max_progress_price": 0.0 if first_candidate_short == "PRICE_REVERSAL" else None,
                        "time_to_0_5_price": None,
                        "time_to_1_0_price": None,
                        "MFE_gap": 0.0 if first_candidate_short == "GAP_REVERSAL" else None,
                        "MAE_gap": 0.0 if first_candidate_short == "GAP_REVERSAL" else None,
                        "max_progress_gap": 0.0 if first_candidate_short == "GAP_REVERSAL" else None,
                        "time_to_0_5_gap": None,
                        "time_to_1_0_gap": None,
                    }
                    return events
        else:
            state_short["armed"] = False
            state_short["confirm_count_price"] = 0
            state_short["confirm_count_gap"] = 0

        return events

    def on_tick(
        self,
        *,
        t: float,
        p_hat: float,
        bound_low: float,
        bound_high: float,
        rail_low: float,
        rail_high: float,
        bid_yes: float | None,
        ask_yes: float | None,
        round_phase: str | None,
        game_number: int | None,
        segment_id: int | None,
        round_number: int | None,
        explain: dict | None,
        game_ended: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Process one tick. Returns list of event dicts to emit (Runner appends each as HistoryPoint.event).
        """
        events: list[dict[str, Any]] = []
        map_id = self._map_id(segment_id, game_number)
        phase_upper = (round_phase or "").strip().upper()

        # Run map-change reset before processing episode end so we don't clear trail state
        # while an episode is still active (reset only when no active episode and no armed window).
        self._reset_map_counts(map_id)

        # End active episode on map start or match end
        if self._active is not None:
            end_reason: str | None = None
            if game_ended:
                end_reason = "MATCH_END"
            elif self._is_map_start(segment_id, game_number, round_phase):
                end_reason = "MAP_START"

            if end_reason:
                ep = self._active
                Pt = ep["Pt"]
                Ht = p_hat
                pnl_at_end = Pt - ep["P0"]
                hold_seconds = t - ep["start_t"]

                # Outcome blocks for price-entry and gap-entry (for ML comparison)
                outcome_price: dict[str, Any]
                if ep.get("price_candidate_P0") is not None:
                    outcome_price = {
                        "MFE": ep.get("MFE_price", 0.0),
                        "MAE": ep.get("MAE_price", 0.0),
                        "pnl_at_end": Pt - ep["price_candidate_P0"],
                        "max_progress_static": ep.get("max_progress_price"),
                        "time_to_0_5": ep.get("time_to_0_5_price"),
                        "time_to_1_0": ep.get("time_to_1_0_price"),
                    }
                else:
                    outcome_price = {"missing": True}
                outcome_gap: dict[str, Any]
                if ep.get("gap_candidate_P0") is not None:
                    outcome_gap = {
                        "MFE": ep.get("MFE_gap", 0.0),
                        "MAE": ep.get("MAE_gap", 0.0),
                        "pnl_at_end": Pt - ep["gap_candidate_P0"],
                        "max_progress_static": ep.get("max_progress_gap"),
                        "time_to_0_5": ep.get("time_to_0_5_gap"),
                        "time_to_1_0": ep.get("time_to_1_0_gap"),
                    }
                else:
                    outcome_gap = {"missing": True}

                events.append({
                    "event_type": "episode_end",
                    "episode_id": ep["episode_id"],
                    "end_reason": end_reason,
                    "end_ts": t,
                    "end_Pt": Pt,
                    "end_Ht": Ht,
                })
                events.append({
                    "event_type": "episode_outcome",
                    "episode_id": ep["episode_id"],
                    "MFE": ep["MFE"],
                    "MAE": ep["MAE"],
                    "max_progress_static": ep["max_progress_static"],
                    "time_to_0_5": ep.get("time_to_0_5"),
                    "time_to_1_0": ep.get("time_to_1_0"),
                    "min_dist_static": ep["min_dist_static"],
                    "min_dist_static_ts": ep.get("min_dist_static_ts"),
                    "min_dist_dynamic": ep.get("min_dist_dynamic"),
                    "min_dist_dynamic_ts": ep.get("min_dist_dynamic_ts"),
                    "pnl_at_end": pnl_at_end,
                    "hold_seconds": hold_seconds,
                    "outcome_price": outcome_price,
                    "outcome_gap": outcome_gap,
                    "both_candidates_same_tick": ep.get("both_candidates_same_tick", False),
                    "tie_break": ep.get("tie_break"),
                })
                self._active = None

        self._last_seen_map_id = map_id
        self._last_seen_round_phase = phase_upper

        # Update active episode metrics if in progress (we have bid/ask below)
        if self._active is not None and not events and bid_yes is not None and ask_yes is not None:
            ep = self._active
            P0, H0, gap0 = ep["P0"], ep["H0"], ep["gap0"]
            instrument = ep["instrument"]
            if instrument == "NO":
                Pt = 1.0 - float(ask_yes)  # LONG_NO MTM = bid_no = 1 - ask_yes
            else:
                Pt = float(bid_yes)  # LONG_YES MTM = bid_yes
            ep["Pt"] = Pt

            pnl_t = Pt - P0
            ep["MFE"] = max(ep["MFE"], pnl_t)
            ep["MAE"] = min(ep["MAE"], pnl_t)

            if gap0 != 0:
                if gap0 > 0:
                    progress = (P0 - Pt) / (P0 - H0)
                else:
                    progress = (Pt - P0) / (H0 - P0)
                progress_clamped = max(PROGRESS_CLAMP_LO, min(PROGRESS_CLAMP_HI, progress))
                ep["max_progress_static"] = max(ep["max_progress_static"], progress_clamped)
                if ep.get("time_to_0_5") is None and progress >= 0.5:
                    ep["time_to_0_5"] = t - ep["start_t"]
                if ep.get("time_to_1_0") is None and progress >= 1.0:
                    ep["time_to_1_0"] = t - ep["start_t"]

            # Update price-entry metrics if second candidate (price) occurred
            if ep.get("price_candidate_P0") is not None:
                P0_price = ep["price_candidate_P0"]
                pnl_price = Pt - P0_price
                ep["MFE_price"] = max(ep.get("MFE_price", 0.0), pnl_price)
                ep["MAE_price"] = min(ep.get("MAE_price", 0.0), pnl_price)
                gap0_price = P0_price - H0
                if gap0_price != 0:
                    if gap0_price > 0:
                        progress_price = (P0_price - Pt) / (P0_price - H0)
                    else:
                        progress_price = (Pt - P0_price) / (H0 - P0_price)
                    progress_price_clamped = max(PROGRESS_CLAMP_LO, min(PROGRESS_CLAMP_HI, progress_price))
                    ep["max_progress_price"] = max(ep.get("max_progress_price", 0.0), progress_price_clamped)
                    start_t_price = ep.get("price_candidate_ts") or ep["start_t"]
                    if ep.get("time_to_0_5_price") is None and progress_price >= 0.5:
                        ep["time_to_0_5_price"] = t - start_t_price
                    if ep.get("time_to_1_0_price") is None and progress_price >= 1.0:
                        ep["time_to_1_0_price"] = t - start_t_price

            # Update gap-entry metrics if second candidate (gap) occurred
            if ep.get("gap_candidate_P0") is not None:
                P0_gap = ep["gap_candidate_P0"]
                pnl_gap = Pt - P0_gap
                ep["MFE_gap"] = max(ep.get("MFE_gap", 0.0), pnl_gap)
                ep["MAE_gap"] = min(ep.get("MAE_gap", 0.0), pnl_gap)
                gap0_gap = P0_gap - H0
                if gap0_gap != 0:
                    if gap0_gap > 0:
                        progress_gap = (P0_gap - Pt) / (P0_gap - H0)
                    else:
                        progress_gap = (Pt - P0_gap) / (H0 - P0_gap)
                    progress_gap_clamped = max(PROGRESS_CLAMP_LO, min(PROGRESS_CLAMP_HI, progress_gap))
                    ep["max_progress_gap"] = max(ep.get("max_progress_gap", 0.0), progress_gap_clamped)
                    start_t_gap = ep.get("gap_candidate_ts") or ep["start_t"]
                    if ep.get("time_to_0_5_gap") is None and progress_gap >= 0.5:
                        ep["time_to_0_5_gap"] = t - start_t_gap
                    if ep.get("time_to_1_0_gap") is None and progress_gap >= 1.0:
                        ep["time_to_1_0_gap"] = t - start_t_gap

            dist_static = abs(Pt - H0)
            if ep["min_dist_static"] is None or dist_static < ep["min_dist_static"]:
                ep["min_dist_static"] = dist_static
                ep["min_dist_static_ts"] = t
            dist_dynamic = abs(Pt - p_hat)
            if ep.get("min_dist_dynamic") is None or dist_dynamic < ep["min_dist_dynamic"]:
                ep["min_dist_dynamic"] = dist_dynamic
                ep["min_dist_dynamic_ts"] = t

            return events  # no new events; avoid re-triggering below

        # No bid/ask: cannot trigger or update
        if bid_yes is None or ask_yes is None:
            return events

        bid_yes_f = float(bid_yes)
        ask_yes_f = float(ask_yes)
        ask_no = 1.0 - bid_yes_f
        bid_no = 1.0 - ask_yes_f
        S_A_line = bound_high   # Team A series boundary
        S_B_line = bound_low    # Team B series boundary
        H0 = _p_hat_from_explain(explain, p_hat)

        if _entry_mode() == "trailing_reversal":
            trail_events = self._try_trailing_reversal(
                t=t,
                p_hat=p_hat,
                S_A_line=S_A_line,
                S_B_line=S_B_line,
                H0=H0,
                bid_yes_f=bid_yes_f,
                ask_yes_f=ask_yes_f,
                ask_no=ask_no,
                map_id=map_id,
                game_number=game_number,
                segment_id=segment_id,
                round_number=round_number,
                explain=explain,
            )
            return trail_events  # do not fall through to immediate entry

        # Immediate entry: check triggers (dedupe: same dir debounce + max per map)
        # SHORT_A (LONG_NO): ask_no >= (1 - S_A_line) + DELTA_ENTRY
        no_threshold = (1.0 - S_A_line) + DELTA_ENTRY
        if ask_no >= no_threshold:
            direction = "SHORT_A"
            last_t = self._last_trigger_time.get(direction, 0)
            if t - last_t >= MIN_SECONDS_BETWEEN_SAME_DIR:
                count = self._episodes_this_map.get(direction, 0)
                if count < MAX_EPISODES_PER_MAP:
                    self._last_trigger_time[direction] = t
                    self._episodes_this_map[direction] = count + 1
                    self._episode_counter += 1
                    episode_id = f"sl_{self._episode_counter}_{uuid.uuid4().hex[:8]}"
                    P0 = ask_no
                    gap0 = P0 - H0
                    explain_snap = self._explain_snapshot(explain)
                    events.append({
                        "event_type": "setup_trigger",
                        "setup_name": SETUP_NAME,
                        "episode_id": episode_id,
                        "direction": direction,
                        "instrument": "NO",
                    })
                    events.append({
                        "event_type": "episode_start",
                        "episode_id": episode_id,
                        "trade_id": episode_id,
                        "direction": direction,
                        "instrument": "NO",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "DELTA_ENTRY": DELTA_ENTRY,
                        "map_index": (game_number - 1) if game_number is not None else (segment_id or 0),
                        "seg": segment_id,
                        "round_number": round_number,
                        "trigger_explain_snapshot": explain_snap,
                    })
                    self._active = {
                        "episode_id": episode_id,
                        "direction": direction,
                        "instrument": "NO",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "start_t": t,
                        "start_seg": segment_id,
                        "start_map": map_id,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "Pt": bid_no,
                        "MFE": 0.0,
                        "MAE": 0.0,
                        "max_progress_static": 0.0,
                        "time_to_0_5": None,
                        "time_to_1_0": None,
                        "min_dist_static": abs(bid_no - H0),
                        "min_dist_static_ts": t,
                        "min_dist_dynamic": None,
                        "min_dist_dynamic_ts": None,
                    }
                    self._active["min_dist_dynamic"] = abs(bid_no - p_hat)
                    self._active["min_dist_dynamic_ts"] = t
                    return events

        # LONG_A (LONG_YES): ask_yes <= S_B_line - DELTA_ENTRY
        yes_threshold = S_B_line - DELTA_ENTRY
        if ask_yes_f <= yes_threshold:
            direction = "LONG_A"
            last_t = self._last_trigger_time.get(direction, 0)
            if t - last_t >= MIN_SECONDS_BETWEEN_SAME_DIR:
                count = self._episodes_this_map.get(direction, 0)
                if count < MAX_EPISODES_PER_MAP:
                    self._last_trigger_time[direction] = t
                    self._episodes_this_map[direction] = count + 1
                    self._episode_counter += 1
                    episode_id = f"sl_{self._episode_counter}_{uuid.uuid4().hex[:8]}"
                    P0 = ask_yes_f
                    gap0 = P0 - H0
                    explain_snap = self._explain_snapshot(explain)
                    events.append({
                        "event_type": "setup_trigger",
                        "setup_name": SETUP_NAME,
                        "episode_id": episode_id,
                        "direction": direction,
                        "instrument": "YES",
                    })
                    events.append({
                        "event_type": "episode_start",
                        "episode_id": episode_id,
                        "trade_id": episode_id,
                        "direction": direction,
                        "instrument": "YES",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "DELTA_ENTRY": DELTA_ENTRY,
                        "map_index": (game_number - 1) if game_number is not None else (segment_id or 0),
                        "seg": segment_id,
                        "round_number": round_number,
                        "trigger_explain_snapshot": explain_snap,
                    })
                    self._active = {
                        "episode_id": episode_id,
                        "direction": direction,
                        "instrument": "YES",
                        "P0": P0,
                        "H0": H0,
                        "gap0": gap0,
                        "start_t": t,
                        "start_seg": segment_id,
                        "start_map": map_id,
                        "S_A_line": S_A_line,
                        "S_B_line": S_B_line,
                        "Pt": bid_yes_f,
                        "MFE": 0.0,
                        "MAE": 0.0,
                        "max_progress_static": 0.0,
                        "time_to_0_5": None,
                        "time_to_1_0": None,
                        "min_dist_static": abs(bid_yes_f - H0),
                        "min_dist_static_ts": t,
                        "min_dist_dynamic": abs(bid_yes_f - p_hat),
                        "min_dist_dynamic_ts": t,
                    }
                    return events

        return events

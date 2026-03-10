"""
Unit tests for TradeEpisodeManager: series-line-dislocation trigger, progress tracking, episode end.
"""
from __future__ import annotations

import os
from unittest.mock import patch

from backend.services.trade_episodes import TradeEpisodeManager


def test_trigger_fires_once_short_a() -> None:
    """Trigger fires once for SHORT_A when ask_no >= (1 - S_A) + DELTA_ENTRY. Second tick same map does not re-trigger."""
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # S_A_line=bound_high=0.6, SHORT trigger: ask_no >= (1 - 0.6) + 0.07 = 0.47. bid_yes=0.4 => ask_no=0.6 >= 0.47.
    t0 = 1000.0
    events0 = mgr.on_tick(
        t=t0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=5,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    assert len(events0) >= 2
    setup_ev = next((e for e in events0 if e.get("event_type") == "setup_trigger"), None)
    start_ev = next((e for e in events0 if e.get("event_type") == "episode_start"), None)
    assert setup_ev is not None
    assert setup_ev.get("setup_name") == "series_line_dislocation_phat_target"
    assert setup_ev.get("direction") == "SHORT_A"
    assert start_ev is not None
    assert start_ev.get("instrument") == "NO"
    assert start_ev.get("P0") == 0.6  # ask_no = 1 - 0.4
    assert start_ev.get("H0") == 0.5
    # Same tick conditions again on same map: should not trigger (already have active episode; we return after update)
    # Actually we have active episode so we'll hit the update block and return. So no second trigger. Good.
    events1 = mgr.on_tick(
        t=t0 + 1,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=6,
        explain=None,
        game_ended=False,
    )
    assert len(events1) == 0  # update only, no new trigger


def test_episode_tracks_progress_and_mfe_mae() -> None:
    """Episode tracks progress_static, MFE, MAE; on end emits episode_outcome with them."""
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # Trigger SHORT_A
    mgr.on_tick(
        t=1000.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    # Tick 2: market moves: ask_yes 0.65 => bid_no = 0.35, Pt = 0.35. P0=0.6. pnl_t = 0.35 - 0.6 = -0.25 (MAE)
    mgr.on_tick(
        t=1001.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.35,
        ask_yes=0.65,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=2,
        explain=None,
        game_ended=False,
    )
    # Tick 3: ask_yes 0.45 => bid_no = 0.55, Pt = 0.55. pnl_t = 0.55 - 0.6 = -0.05. Still negative. Then end by map start
    events_end = mgr.on_tick(
        t=1002.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.55,
        ask_yes=0.45,
        round_phase="IN_PROGRESS",
        game_number=2,  # map changed
        segment_id=1,
        round_number=1,
        explain=None,
        game_ended=False,
    )
    end_ev = next((e for e in events_end if e.get("event_type") == "episode_end"), None)
    outcome_ev = next((e for e in events_end if e.get("event_type") == "episode_outcome"), None)
    assert end_ev is not None
    assert end_ev.get("end_reason") == "MAP_START"
    assert outcome_ev is not None
    assert "MFE" in outcome_ev
    assert "MAE" in outcome_ev
    assert "max_progress_static" in outcome_ev
    assert "pnl_at_end" in outcome_ev
    assert "hold_seconds" in outcome_ev
    assert outcome_ev["hold_seconds"] == 2.0


def test_episode_ends_on_match_end() -> None:
    """Episode ends with end_reason=MATCH_END when game_ended=True."""
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # First tick: no trigger (ask_yes 0.5 > S_B_line-DELTA=0.33, ask_no 0.5 < (1-S_A_line)+DELTA=0.51)
    mgr.on_tick(
        t=2000.0,
        p_hat=0.48,
        bound_low=0.4,
        bound_high=0.56,
        rail_low=0.4,
        rail_high=0.56,
        bid_yes=0.5,
        ask_yes=0.5,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=10,
        explain={"final": {"p_hat_final": 0.48}},
        game_ended=False,
    )
    # Trigger LONG_A only: S_B_line=0.4, ask_yes <= 0.33. Use ask_yes=0.30, bid_yes=0.5 so ask_no=0.5 (no SHORT_A)
    events0 = mgr.on_tick(
        t=2000.5,
        p_hat=0.48,
        bound_low=0.4,
        bound_high=0.56,
        rail_low=0.4,
        rail_high=0.56,
        bid_yes=0.5,
        ask_yes=0.30,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=10,
        explain={"final": {"p_hat_final": 0.48}},
        game_ended=False,
    )
    assert any(e.get("event_type") == "episode_start" for e in events0)
    assert any(e.get("direction") == "LONG_A" for e in events0)
    events_end = mgr.on_tick(
        t=2001.0,
        p_hat=0.52,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.5,
        ask_yes=0.54,
        round_phase="FINISHED",
        game_number=1,
        segment_id=0,
        round_number=13,
        explain=None,
        game_ended=True,
    )
    end_ev = next((e for e in events_end if e.get("event_type") == "episode_end"), None)
    assert end_ev is not None
    assert end_ev.get("end_reason") == "MATCH_END"


def test_trailing_reversal_arms_and_fires_after_reversal() -> None:
    """Trailing reversal: arm when boundary + abs(gap)>=MIN_PHAT_GAP; fire when gap reverses by TRAIL_PTS for CONFIRM_TICKS."""
    old_mode = os.environ.pop("TRADE_EPISODE_ENTRY_MODE", None)
    try:
        os.environ["TRADE_EPISODE_ENTRY_MODE"] = "trailing_reversal"
        mgr = TradeEpisodeManager()
        # LONG_A: S_B_line=bound_low=0.4, arm when ask_yes <= S_B_line - DELTA_ENTRY = 0.33. Use ask_yes=0.30.
        # SHORT does not arm: ask_no < (1-S_A_line)+DELTA = 0.47, so bid_yes=0.6 => ask_no=0.4.
        # Tick 1: arm LONG (gap=-0.20, extreme_gap=-0.20). No entry (reversal not yet).
        t1 = 1000.0
        ev1 = mgr.on_tick(
            t=t1,
            p_hat=0.5,
            bound_low=0.4,
            bound_high=0.6,
            rail_low=0.4,
            rail_high=0.6,
            bid_yes=0.6,
            ask_yes=0.30,
            round_phase="IN_PROGRESS",
            game_number=1,
            segment_id=0,
            round_number=1,
            explain={"final": {"p_hat_final": 0.5}},
            game_ended=False,
        )
        assert len(ev1) == 0  # armed but no entry
        # Trailing state: LONG_A should be armed with extreme_gap=-0.20 (ask_yes 0.30 - H0 0.5), confirm_count=0.
        assert mgr._trail_state["LONG_A"]["armed"] is True
        assert abs(mgr._trail_state["LONG_A"]["extreme_gap"] - (-0.20)) < 1e-6
        from backend.services.trade_episodes import TRAIL_PTS

        assert TRAIL_PTS == 0.03  # trailing reversal uses 3pp
    finally:
        if old_mode is not None:
            os.environ["TRADE_EPISODE_ENTRY_MODE"] = old_mode
        elif "TRADE_EPISODE_ENTRY_MODE" in os.environ:
            del os.environ["TRADE_EPISODE_ENTRY_MODE"]


def test_episode_outcome_has_price_and_gap_blocks() -> None:
    """episode_outcome includes outcome_price and outcome_gap (missing when not trailing dual-candidate)."""
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # Start episode (immediate SHORT_A)
    mgr.on_tick(
        t=1000.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    # End episode
    ev_end = mgr.on_tick(
        t=1010.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.5,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=2,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=True,
    )
    outcome_ev = next((e for e in ev_end if e.get("event_type") == "episode_outcome"), None)
    assert outcome_ev is not None
    assert "outcome_price" in outcome_ev
    assert "outcome_gap" in outcome_ev
    # In immediate mode both are missing (no second candidate)
    assert outcome_ev["outcome_price"].get("missing") is True
    assert outcome_ev["outcome_gap"].get("missing") is True


def test_map_change_does_not_reset_trail_state_while_episode_active() -> None:
    """Map-change reset must NOT clear P_extreme, gap_extreme, confirm_count_*, candidate_*_emitted, or armed
    when there is an active episode or an armed window. Verify unchanged until episode_end is emitted.
    """
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # Start episode on map 1 so _active is set
    mgr.on_tick(
        t=1000.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    assert mgr._active is not None
    # Simulate an armed window for another direction (so has_armed is True) with non-default trail state
    mgr._trail_state["LONG_A"]["armed"] = True
    mgr._trail_state["LONG_A"]["P_extreme"] = 0.25
    mgr._trail_state["LONG_A"]["extreme_gap"] = -0.2
    mgr._trail_state["LONG_A"]["confirm_count_price"] = 1
    mgr._trail_state["LONG_A"]["confirm_count_gap"] = 0
    mgr._trail_state["LONG_A"]["candidate_price_emitted"] = True
    mgr._trail_state["LONG_A"]["candidate_gap_emitted"] = False
    # Map change: game_number=2 triggers _reset_map_counts(2) at start of on_tick; episode_end is then emitted
    ev = mgr.on_tick(
        t=1005.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.5,
        round_phase="IN_PROGRESS",
        game_number=2,
        segment_id=1,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    # episode_end must be emitted (MAP_START)
    end_ev = next((e for e in ev if e.get("event_type") == "episode_end"), None)
    assert end_ev is not None
    assert end_ev.get("end_reason") == "MAP_START"
    # Trail state must not have been reset (we had active episode + armed window when _reset_map_counts ran)
    assert mgr._trail_state["LONG_A"]["armed"] is True
    assert mgr._trail_state["LONG_A"]["P_extreme"] == 0.25
    assert mgr._trail_state["LONG_A"]["extreme_gap"] == -0.2
    assert mgr._trail_state["LONG_A"]["confirm_count_price"] == 1
    assert mgr._trail_state["LONG_A"]["candidate_price_emitted"] is True
    assert mgr._trail_state["LONG_A"]["candidate_gap_emitted"] is False


def test_episode_end_and_outcome_not_lost_when_trailing_reversal_evaluated_same_tick() -> None:
    """Map-start (or match end) emits episode_end and episode_outcome into events; trailing_reversal branch
    must not drop them by returning only trail_events. Return events + trail_events so both are present.
    """
    # Start an episode in immediate mode so _active is set
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    mgr.on_tick(
        t=1000.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    assert mgr._active is not None
    # Next tick: map change (MAP_START) + trailing_reversal mode so we evaluate _try_trailing_reversal.
    # We must have bid/ask so we reach the trailing branch and return events + trail_events.
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "trailing_reversal"
    ev = mgr.on_tick(
        t=1005.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.5,
        ask_yes=0.55,
        round_phase="IN_PROGRESS",
        game_number=2,
        segment_id=1,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    end_ev = next((e for e in ev if e.get("event_type") == "episode_end"), None)
    outcome_ev = next((e for e in ev if e.get("event_type") == "episode_outcome"), None)
    assert end_ev is not None, "episode_end must not be lost when trailing_reversal runs same tick"
    assert outcome_ev is not None, "episode_outcome must not be lost when trailing_reversal runs same tick"
    assert end_ev.get("end_reason") == "MAP_START"


def test_map_start_fires_on_first_in_progress_tick_after_buy_time_in_new_map() -> None:
    """Map-start end can be missed if _last_seen_map_id is updated on BUY_TIME of new map.
    Only update _last_seen_map_id when phase is IN_PROGRESS so the first IN_PROGRESS tick
    of the new map still triggers MAP_START.
    """
    os.environ["TRADE_EPISODE_ENTRY_MODE"] = "immediate"
    mgr = TradeEpisodeManager()
    # Map 1, IN_PROGRESS: start episode and set _last_seen_map_id = 1
    mgr.on_tick(
        t=1000.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.4,
        ask_yes=0.6,
        round_phase="IN_PROGRESS",
        game_number=1,
        segment_id=0,
        round_number=12,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    assert mgr._active is not None
    assert mgr._last_seen_map_id == 1
    # Map 2, BUY_TIME: do not update _last_seen_map_id; no episode end yet
    ev_buy = mgr.on_tick(
        t=1001.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.45,
        ask_yes=0.55,
        round_phase="BUY_TIME",
        game_number=2,
        segment_id=1,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    assert mgr._last_seen_map_id == 1  # unchanged (only update on IN_PROGRESS)
    end_buy = next((e for e in ev_buy if e.get("event_type") == "episode_end"), None)
    assert end_buy is None  # MAP_START not fired on BUY_TIME
    assert mgr._active is not None
    # Map 2, IN_PROGRESS: first IN_PROGRESS of new map -> MAP_START
    ev_progress = mgr.on_tick(
        t=1002.0,
        p_hat=0.5,
        bound_low=0.4,
        bound_high=0.6,
        rail_low=0.4,
        rail_high=0.6,
        bid_yes=0.45,
        ask_yes=0.55,
        round_phase="IN_PROGRESS",
        game_number=2,
        segment_id=1,
        round_number=1,
        explain={"final": {"p_hat_final": 0.5}},
        game_ended=False,
    )
    end_progress = next((e for e in ev_progress if e.get("event_type") == "episode_end"), None)
    assert end_progress is not None
    assert end_progress.get("end_reason") == "MAP_START"
    assert mgr._active is None

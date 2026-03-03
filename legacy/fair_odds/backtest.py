"""
In-play backtest incremental update: process one new row without re-running full backtest.
State is persisted so Add snapshot can update trades and chart without subprocess.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def _parse_ts(ts):
    if ts is None:
        return None
    if hasattr(ts, "isoformat"):
        return ts
    try:
        return pd.to_datetime(ts)
    except Exception:
        return None


def _float(v):
    if v is None or (isinstance(v, float) and (v != v or v is None)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def make_trade_id(strategy_id: str, match_id: str, side: str, entry_ts: str, entry_px: float) -> str:
    raw = f"{strategy_id}|{match_id}|{side}|{entry_ts}|{entry_px:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _serialize_position(pos: dict | None) -> dict | None:
    """Return a copy of position with entry_ts/last_ts as ISO strings for JSON."""
    if pos is None:
        return None
    out = dict(pos)
    for key in ("entry_ts", "last_ts"):
        if key in out and out[key] is not None:
            v = out[key]
            if hasattr(v, "isoformat"):
                out[key] = v.isoformat()
            else:
                out[key] = str(v)
    return out


def load_state(state_path: Path | None) -> dict:
    if state_path is None or not state_path.exists():
        return {"strategies": {}}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"strategies": {}}


def repair_state_from_trades(state: dict, trades_by_strategy: dict) -> dict:
    """
    Fix stale state: if state says a strategy has an open position in match_id M,
    but the trades CSV has a completed trade (row) for that strategy and match_id M,
    then that position was closed and we set position to None.
    Returns a new state dict (does not mutate state). Saves having state say "open"
    when trades show the position was already closed.
    """
    if not state.get("strategies") or not trades_by_strategy:
        return state
    repaired = {"strategies": {}}
    for sid, sdata in state["strategies"].items():
        repaired["strategies"][sid] = dict(sdata)
        pos = sdata.get("position")
        if pos is None:
            continue
        pos_match_id = str(pos.get("match_id", "")).strip()
        if not pos_match_id:
            continue
        df = trades_by_strategy.get(sid)
        if df is None or len(df) == 0 or "match_id" not in df.columns:
            continue
        match_trades = df[df["match_id"].astype(str).str.strip() == pos_match_id]
        if len(match_trades) > 0:
            repaired["strategies"][sid]["position"] = None
    return repaired


def save_state(state: dict, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_match_settlements(results_path: Path | None) -> dict:
    if results_path is None or not results_path.exists():
        return {}
    try:
        rdf = pd.read_csv(results_path)
    except Exception:
        return {}
    for col in ["match_id", "winner", "team_a", "team_b"]:
        if col not in rdf.columns:
            return {}
    out = {}
    for _, r in rdf.iterrows():
        mid = str(r.get("match_id", "")).strip()
        if not mid:
            continue
        winner = str(r.get("winner", "")).strip().lower()
        team_a = str(r.get("team_a", "")).strip().lower()
        team_b = str(r.get("team_b", "")).strip().lower()
        out[mid] = 1.0 if winner == team_a else 0.0
    return out


def incremental_update(
    row: dict,
    state: dict,
    config: dict,
    match_settlements: dict,
) -> tuple[dict, dict]:
    """
    Process one new row; update state and return any new closed trades.
    row: dict with timestamp, match_id, contract_scope, bid, ask, mid, spread_abs, p_fair, band_lo, band_hi.
    state: from load_state().
    config: inplay_strategies.json (filters.scope_allowlist, sizing.contracts, sizing.start_bankroll, strategies).
    match_settlements: match_id -> 1.0 or 0.0 for match-end settlement.
    Returns (new_state, new_trades_by_strategy) where new_trades_by_strategy[strategy_id] = [trade_dict, ...].
    """
    filters = config.get("filters", {})
    sizing = config.get("sizing", {})
    strategies = config.get("strategies", [])
    scope_allowlist = filters.get("scope_allowlist") or []
    contracts = int(sizing.get("contracts", 100))
    start_bankroll = float(sizing.get("start_bankroll", 2000))

    ts = _parse_ts(row.get("timestamp"))
    if ts is None:
        return state, {}
    match_id = str(row.get("match_id", "")).strip()
    contract_scope = str(row.get("contract_scope", "")).strip()
    if scope_allowlist and contract_scope not in scope_allowlist:
        return state, {}

    bid = _float(row.get("bid"))
    ask = _float(row.get("ask"))
    mid = _float(row.get("mid"))
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    spread_abs = _float(row.get("spread_abs"))
    if spread_abs is None and bid is not None and ask is not None:
        spread_abs = ask - bid if ask and bid else None
    p_fair = _float(row.get("p_fair"))
    band_lo = _float(row.get("band_lo"))
    band_hi = _float(row.get("band_hi"))

    if ask is None or bid is None or mid is None:
        return state, {}
    if ask <= bid:
        return state, {}
    spread_abs_max = 0.06
    if spread_abs is not None and spread_abs > spread_abs_max:
        return state, {}

    long_signal = band_lo is not None and mid <= (band_lo - 0.10)
    short_signal = band_hi is not None and mid >= (band_hi + 0.10)
    buffer = 0.10

    new_state = {"strategies": state.get("strategies", {}).copy()}
    new_trades_by_strategy = {}

    for strat in strategies:
        strategy_id = strat["id"]
        buffer = float(strat.get("buffer", 0.10))
        spread_abs_max = float(strat.get("spread_abs_max", 0.06))
        max_hold_minutes = float(strat.get("max_hold_minutes", 45))
        exit_mode = str(strat.get("exit_mode", "to_entry_fair"))
        entry_mode = str(strat.get("entry_mode", "edge"))  # "edge" = only on crossing into signal; "level" = whenever flat and signal true

        if spread_abs is not None and spread_abs > spread_abs_max:
            continue
        long_signal = band_lo is not None and mid <= (band_lo - buffer)
        short_signal = band_hi is not None and mid >= (band_hi + buffer)

        sstate = new_state["strategies"].get(strategy_id) or {
            "position": None,
            "prev_long_signal": {},
            "prev_short_signal": {},
            "bankroll": start_bankroll,
        }
        position = sstate.get("position")
        prev_long_signal = dict(sstate.get("prev_long_signal") or {})
        prev_short_signal = dict(sstate.get("prev_short_signal") or {})
        bankroll = float(sstate.get("bankroll", start_bankroll))
        new_trades = []

        def _close_position(pos, exit_ts, exit_px, exit_reason):
            nonlocal position, bankroll
            entry_ts = _parse_ts(pos.get("entry_ts"))
            exit_ts = _parse_ts(exit_ts) if not hasattr(exit_ts, "total_seconds") else exit_ts
            hold_min = (exit_ts - entry_ts).total_seconds() / 60.0 if (entry_ts and exit_ts) else 0.0
            pnl_price = (
                exit_px - pos["entry_ask"]
                if pos["side"] == "LONG"
                else pos["entry_bid"] - exit_px
            )
            pnl_d = pnl_price * contracts
            bankroll_before = bankroll
            bankroll_after = bankroll + pnl_d
            bankroll = bankroll_after
            ret_pct = (pnl_d / bankroll_before) if bankroll_before else 0.0
            entry_ts_str = entry_ts.isoformat() if hasattr(entry_ts, "isoformat") else str(entry_ts)
            exit_ts_str = exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts)
            trade_id = make_trade_id(strategy_id, str(pos["match_id"]), pos["side"], entry_ts_str, pos["entry_px"])
            trade = {
                "trade_id": trade_id,
                "strategy_id": strategy_id,
                "match_id": pos["match_id"],
                "side": pos["side"],
                "entry_ts": entry_ts_str,
                "entry_px": pos["entry_px"],
                "entry_mid": pos["entry_mid"],
                "entry_fair": pos["entry_fair"],
                "entry_band_lo": pos["entry_band_lo"],
                "entry_band_hi": pos["entry_band_hi"],
                "entry_spread_abs": pos["entry_spread_abs"],
                "exit_ts": exit_ts_str,
                "exit_px": exit_px,
                "exit_reason": exit_reason,
                "hold_minutes": hold_min,
                "contracts": contracts,
                "pnl_price": pnl_price,
                "pnl_$": pnl_d,
                "bankroll_before": bankroll_before,
                "bankroll_after": bankroll_after,
                "ret_pct_account": ret_pct,
            }
            new_trades.append(trade)
            position = None

        # In a position
        if position is not None:
            pos_match_id = str(position.get("match_id", "")).strip()
            if pos_match_id != match_id:
                exit_ts = _parse_ts(position.get("last_ts") or position.get("entry_ts"))
                exit_mid = float(position.get("last_mid", 0.5))
                mid_str = pos_match_id
                if mid_str in match_settlements:
                    exit_px = float(match_settlements[mid_str])
                    exit_reason = "match_end"
                else:
                    exit_px = exit_mid
                    exit_reason = "match_end_no_result"
                _close_position(position, exit_ts, exit_px, exit_reason)
                prev_long_signal[match_id] = long_signal
                prev_short_signal[match_id] = short_signal
            else:
                # Same match: exit only on rule (no time-based exit)
                exit_ts = None
                exit_px = None
                exit_reason = None
                if position["side"] == "LONG":
                    if exit_mode == "to_entry_fair" and position.get("entry_fair") is not None and mid >= position["entry_fair"]:
                        exit_ts, exit_px, exit_reason = ts, bid, "to_entry_fair"
                    elif exit_mode == "to_current_fair" and p_fair is not None and mid >= p_fair:
                        exit_ts, exit_px, exit_reason = ts, bid, "to_current_fair"
                    elif exit_mode == "back_inside_band" and band_lo is not None and mid >= band_lo:
                        exit_ts, exit_px, exit_reason = ts, bid, "back_inside_band"
                else:
                    if exit_mode == "to_entry_fair" and position.get("entry_fair") is not None and mid <= position["entry_fair"]:
                        exit_ts, exit_px, exit_reason = ts, ask, "to_entry_fair"
                    elif exit_mode == "to_current_fair" and p_fair is not None and mid <= p_fair:
                        exit_ts, exit_px, exit_reason = ts, ask, "to_current_fair"
                    elif exit_mode == "back_inside_band" and band_hi is not None and mid <= band_hi:
                        exit_ts, exit_px, exit_reason = ts, ask, "back_inside_band"
                if exit_ts is not None:
                    _close_position(position, exit_ts, exit_px, exit_reason)
                    prev_long_signal[match_id] = long_signal
                    prev_short_signal[match_id] = short_signal
                else:
                    position["last_ts"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                    position["last_mid"] = mid
                    prev_long_signal[match_id] = long_signal
                    prev_short_signal[match_id] = short_signal
                if new_trades:
                    new_trades_by_strategy[strategy_id] = new_trades
                new_state["strategies"][strategy_id] = {
                    "position": _serialize_position(position),
                    "prev_long_signal": prev_long_signal,
                    "prev_short_signal": prev_short_signal,
                    "bankroll": bankroll,
                }
                continue

        # Flat: entry (edge = on crossing into signal; level = whenever flat and signal true)
        pl = prev_long_signal.get(match_id, False)
        ps = prev_short_signal.get(match_id, False)
        long_edge = long_signal and (not pl if entry_mode == "edge" else True)
        short_edge = short_signal and (not ps if entry_mode == "edge" else True)
        if long_signal and long_edge:
            position = {
                "strategy_id": strategy_id,
                "match_id": match_id,
                "side": "LONG",
                "entry_ts": ts,
                "entry_px": ask,
                "entry_ask": ask,
                "entry_bid": bid,
                "entry_mid": mid,
                "entry_fair": p_fair,
                "entry_band_lo": band_lo,
                "entry_band_hi": band_hi,
                "entry_spread_abs": spread_abs,
                "last_ts": ts,
                "last_mid": mid,
            }
        elif short_signal and short_edge:
            position = {
                "strategy_id": strategy_id,
                "match_id": match_id,
                "side": "SHORT",
                "entry_ts": ts,
                "entry_px": bid,
                "entry_ask": ask,
                "entry_bid": bid,
                "entry_mid": mid,
                "entry_fair": p_fair,
                "entry_band_lo": band_lo,
                "entry_band_hi": band_hi,
                "entry_spread_abs": spread_abs,
                "last_ts": ts,
                "last_mid": mid,
            }
        prev_long_signal[match_id] = long_signal
        prev_short_signal[match_id] = short_signal

        new_state["strategies"][strategy_id] = {
            "position": _serialize_position(position),
            "prev_long_signal": prev_long_signal,
            "prev_short_signal": prev_short_signal,
            "bankroll": bankroll,
        }
        if new_trades:
            new_trades_by_strategy[strategy_id] = new_trades

    return new_state, new_trades_by_strategy

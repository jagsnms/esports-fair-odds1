"""
In-play P/L backtest runner. Run independently via CLI; Streamlit tab calls via subprocess.
No Streamlit dependency. Writes trades CSV per strategy and summary JSON to outdir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


def _parse_ts(s: str):
    """Parse timestamp string to datetime for ordering. Handles ISO format."""
    if pd.isna(s) or s == "":
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


def _float(row: pd.Series, col: str) -> float | None:
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _str(row: pd.Series, col: str) -> str:
    v = row.get(col)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_columns(df: pd.DataFrame, data_cfg: dict) -> None:
    required = [
        data_cfg["timestamp_col"],
        data_cfg["stream_id_col"],
        data_cfg["scope_col"],
        data_cfg["bid_col"],
        data_cfg["ask_col"],
        data_cfg["mid_col"],
        data_cfg["spread_abs_col"],
        data_cfg["fair_col"],
        data_cfg["band_lo_col"],
        data_cfg["band_hi_col"],
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        available = ", ".join(sorted(df.columns.tolist()))
        raise ValueError(
            f"Missing columns in CSV: {missing}. Available columns: {available}"
        )


def make_trade_id(strategy_id: str, match_id: str, side: str, entry_ts: str, entry_px: float) -> str:
    """Stable hash for dedup/reference."""
    raw = f"{strategy_id}|{match_id}|{side}|{entry_ts}|{entry_px:.6f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_match_results(results_path: Path | None):
    """Load match_id -> settlement (1 or 0 for YES contract = Team A wins). Returns dict match_id -> float."""
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
        # YES contract = Team A wins
        out[mid] = 1.0 if winner == team_a else 0.0
    return out


def run_backtest(
    df: pd.DataFrame,
    config: dict,
    outdir: Path,
    match_settlements: dict | None = None,
) -> None:
    match_settlements = match_settlements or {}
    data_cfg = config["data"]
    filters = config.get("filters", {})
    sizing = config.get("sizing", {})
    strategies = config.get("strategies", [])
    scope_allowlist = filters.get("scope_allowlist") or []
    contracts = int(sizing.get("contracts", 100))
    start_bankroll = float(sizing.get("start_bankroll", 2000))

    ts_col = data_cfg["timestamp_col"]
    stream_col = data_cfg["stream_id_col"]
    scope_col = data_cfg["scope_col"]
    bid_col = data_cfg["bid_col"]
    ask_col = data_cfg["ask_col"]
    mid_col = data_cfg["mid_col"]
    spread_col = data_cfg["spread_abs_col"]
    fair_col = data_cfg["fair_col"]
    band_lo_col = data_cfg["band_lo_col"]
    band_hi_col = data_cfg["band_hi_col"]

    # Filter by scope
    if scope_allowlist:
        df = df[df[scope_col].astype(str).str.strip().isin(scope_allowlist)].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col)

    summary = {}

    for strat in strategies:
        strategy_id = strat["id"]
        buffer = float(strat.get("buffer", 0.10))
        spread_abs_max = float(strat.get("spread_abs_max", 0.06))
        max_hold_minutes = float(strat.get("max_hold_minutes", 45))
        exit_mode = str(strat.get("exit_mode", "to_entry_fair"))

        all_trades = []
        # One open position per strategy globally (not per match)
        position = None  # {"match_id", "side", "entry_*", ...}
        prev_long_signal = {}  # match_id -> bool
        prev_short_signal = {}  # match_id -> bool

        # Process all rows in global time order so we never have >1 open trade per strategy
        df_glob = df.sort_values(ts_col).reset_index(drop=True)
        for i in range(len(df_glob)):
            row = df_glob.iloc[i]
            match_id = row[stream_col]
            ts = row[ts_col]
            bid = _float(row, bid_col)
            ask = _float(row, ask_col)
            mid = _float(row, mid_col)
            if mid is None and bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            spread_abs = _float(row, spread_col)
            if spread_abs is None and bid is not None and ask is not None:
                spread_abs = ask - bid
            p_fair = _float(row, fair_col)
            band_lo = _float(row, band_lo_col)
            band_hi = _float(row, band_hi_col)

            if ask is None or bid is None or mid is None:
                continue
            if ask <= bid:
                continue
            if spread_abs is not None and spread_abs > spread_abs_max:
                continue

            long_signal = (
                band_lo is not None
                and mid <= (band_lo - buffer)
            )
            short_signal = (
                band_hi is not None
                and mid >= (band_hi + buffer)
            )

            # ---- In a position: only handle exit/timeout for the same match ----
            if position is not None:
                if position["match_id"] != match_id:
                    # Match ended (we moved on to another match): force-close so we can trade in the new match
                    exit_ts = position.get("last_ts", position["entry_ts"])
                    exit_mid = position.get("last_mid", 0.5)
                    mid_str = str(position["match_id"]).strip()
                    if mid_str in match_settlements:
                        exit_px = float(match_settlements[mid_str])
                        exit_reason = "match_end"
                    else:
                        exit_px = exit_mid
                        exit_reason = "match_end_no_result"
                    hold_min = (exit_ts - position["entry_ts"]).total_seconds() / 60.0
                    pnl_price = (
                        exit_px - position["entry_ask"]
                        if position["side"] == "LONG"
                        else position["entry_bid"] - exit_px
                    )
                    all_trades.append({
                        **position,
                        "exit_ts": exit_ts,
                        "exit_px": exit_px,
                        "exit_reason": exit_reason,
                        "hold_minutes": hold_min,
                        "contracts": contracts,
                        "pnl_price": pnl_price,
                        "pnl_$": pnl_price * contracts,
                    })
                    position = None
                    prev_long_signal[match_id] = long_signal
                    prev_short_signal[match_id] = short_signal
                    # Fall through to entry check for current row (new match)
                else:
                    # Same match: check timeout first
                    entry_ts = position["entry_ts"]
                    hold_min = (ts - entry_ts).total_seconds() / 60.0
                    if hold_min >= max_hold_minutes:
                        exit_bid = bid
                        exit_ask = ask
                        if position["side"] == "LONG":
                            exit_px = exit_bid
                        else:
                            exit_px = exit_ask
                        pnl_price = (
                            exit_px - position["entry_ask"]
                            if position["side"] == "LONG"
                            else position["entry_bid"] - exit_ask
                        )
                        all_trades.append({
                            **position,
                            "exit_ts": ts,
                            "exit_px": exit_px,
                            "exit_reason": "timeout",
                            "hold_minutes": hold_min,
                            "contracts": contracts,
                            "pnl_price": pnl_price,
                            "pnl_$": pnl_price * contracts,
                        })
                        position = None
                        prev_long_signal[match_id] = long_signal
                        prev_short_signal[match_id] = short_signal
                        continue
                    # Regular exit checks (same match, in position)
                    exit_ts = None
                    exit_px = None
                    exit_reason = None
                    if position["side"] == "LONG":
                        if exit_mode == "to_entry_fair":
                            if mid >= position["entry_fair"]:
                                exit_ts, exit_px = ts, bid
                                exit_reason = "to_entry_fair"
                        elif exit_mode == "to_current_fair":
                            if p_fair is not None and mid >= p_fair:
                                exit_ts, exit_px = ts, bid
                                exit_reason = "to_current_fair"
                        elif exit_mode == "back_inside_band":
                            if band_lo is not None and mid >= band_lo:
                                exit_ts, exit_px = ts, bid
                                exit_reason = "back_inside_band"
                    else:
                        if exit_mode == "to_entry_fair":
                            if mid <= position["entry_fair"]:
                                exit_ts, exit_px = ts, ask
                                exit_reason = "to_entry_fair"
                        elif exit_mode == "to_current_fair":
                            if p_fair is not None and mid <= p_fair:
                                exit_ts, exit_px = ts, ask
                                exit_reason = "to_current_fair"
                        elif exit_mode == "back_inside_band":
                            if band_hi is not None and mid <= band_hi:
                                exit_ts, exit_px = ts, ask
                                exit_reason = "back_inside_band"

                    if exit_ts is not None:
                        hold_min = (exit_ts - position["entry_ts"]).total_seconds() / 60.0
                        pnl_price = (
                            exit_px - position["entry_ask"]
                            if position["side"] == "LONG"
                            else position["entry_bid"] - exit_px
                        )
                        all_trades.append({
                            **position,
                            "exit_ts": exit_ts,
                            "exit_px": exit_px,
                            "exit_reason": exit_reason,
                            "hold_minutes": hold_min,
                            "contracts": contracts,
                            "pnl_price": pnl_price,
                            "pnl_$": pnl_price * contracts,
                        })
                        position = None
                    else:
                        position["last_ts"] = ts
                        position["last_mid"] = mid
                    prev_long_signal[match_id] = long_signal
                    prev_short_signal[match_id] = short_signal
                    continue

            # ---- Flat: entry only if edge-triggered for this match ----
            pl = prev_long_signal.get(match_id, False)
            ps = prev_short_signal.get(match_id, False)
            if long_signal and not pl:
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
            elif short_signal and not ps:
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

        # Force-close at match end if position still open (contract settles at 1 or 0)
        if position is not None:
            exit_ts = position.get("last_ts", position["entry_ts"])
            exit_mid = position.get("last_mid", 0.5)
            mid_str = str(position["match_id"]).strip()
            if mid_str in match_settlements:
                exit_px = float(match_settlements[mid_str])
                exit_reason = "match_end"
            else:
                exit_px = exit_mid
                exit_reason = "match_end_no_result"
            hold_min = (exit_ts - position["entry_ts"]).total_seconds() / 60.0
            pnl_price = (
                exit_px - position["entry_ask"]
                if position["side"] == "LONG"
                else position["entry_bid"] - exit_px
            )
            all_trades.append({
                **position,
                "exit_ts": exit_ts,
                "exit_px": exit_px,
                "exit_reason": exit_reason,
                "hold_minutes": hold_min,
                "contracts": contracts,
                "pnl_price": pnl_price,
                "pnl_$": pnl_price * contracts,
            })
            position = None

        # Sort trades by entry_ts for bankroll sequence
        all_trades.sort(key=lambda t: t["entry_ts"])
        bankroll = start_bankroll
        for t in all_trades:
            t["bankroll_before"] = bankroll
            pnl_d = t["pnl_$"]
            t["bankroll_after"] = bankroll + pnl_d
            t["ret_pct_account"] = (pnl_d / bankroll) if bankroll else 0.0
            bankroll = t["bankroll_after"]

        # Build trade rows for CSV
        trade_rows = []
        for t in all_trades:
            entry_ts_str = t["entry_ts"].isoformat() if hasattr(t["entry_ts"], "isoformat") else str(t["entry_ts"])
            trade_id = make_trade_id(
                strategy_id, str(t["match_id"]), t["side"], entry_ts_str, t["entry_px"]
            )
            exit_ts = t["exit_ts"]
            exit_ts_str = exit_ts.isoformat() if hasattr(exit_ts, "isoformat") else str(exit_ts)
            trade_rows.append({
                "trade_id": trade_id,
                "strategy_id": strategy_id,
                "match_id": t["match_id"],
                "side": t["side"],
                "entry_ts": entry_ts_str,
                "entry_px": t["entry_px"],
                "entry_mid": t["entry_mid"],
                "entry_fair": t["entry_fair"],
                "entry_band_lo": t["entry_band_lo"],
                "entry_band_hi": t["entry_band_hi"],
                "entry_spread_abs": t["entry_spread_abs"],
                "exit_ts": exit_ts_str,
                "exit_px": t["exit_px"],
                "exit_reason": t["exit_reason"],
                "hold_minutes": t["hold_minutes"],
                "contracts": t["contracts"],
                "pnl_price": t["pnl_price"],
                "pnl_$": t["pnl_$"],
                "bankroll_before": t["bankroll_before"],
                "bankroll_after": t["bankroll_after"],
                "ret_pct_account": t["ret_pct_account"],
            })

        # Equity curve and max drawdown
        if trade_rows:
            equity = [t["bankroll_after"] for t in trade_rows]
            running_max = equity[0]
            max_dd = 0.0
            for e in equity:
                running_max = max(running_max, e)
                dd = (e - running_max) / running_max if running_max else 0.0
                max_dd = min(max_dd, dd)
        else:
            equity = []
            max_dd = 0.0

        n = len(trade_rows)
        total_pnl = sum(t["pnl_$"] for t in trade_rows)
        end_br = bankroll
        total_ret = (end_br - start_bankroll) / start_bankroll if start_bankroll else 0.0
        wins = sum(1 for t in trade_rows if t["pnl_$"] > 0)
        win_rate = (wins / n) if n else 0.0
        avg_pnl = (total_pnl / n) if n else 0.0
        pnls = [t["pnl_$"] for t in trade_rows]
        median_pnl = float(pd.Series(pnls).median()) if pnls else 0.0
        best = max(pnls) if pnls else 0.0
        worst = min(pnls) if pnls else 0.0
        avg_hold = (sum(t["hold_minutes"] for t in trade_rows) / n) if n else 0.0

        summary[strategy_id] = {
            "trades": n,
            "total_pnl_$": total_pnl,
            "end_bankroll": end_br,
            "total_return_pct": total_ret,
            "win_rate": win_rate,
            "avg_pnl_$": avg_pnl,
            "median_pnl_$": median_pnl,
            "best_trade_$": best,
            "worst_trade_$": worst,
            "avg_hold_minutes": avg_hold,
            "max_drawdown_pct": max_dd,
        }

        # Write trades CSV (overwrite)
        trades_path = outdir / f"inplay_backtest_trades__{strategy_id}.csv"
        trade_columns = [
            "trade_id", "strategy_id", "match_id", "side",
            "entry_ts", "entry_px", "entry_mid", "entry_fair", "entry_band_lo", "entry_band_hi", "entry_spread_abs",
            "exit_ts", "exit_px", "exit_reason", "hold_minutes", "contracts",
            "pnl_price", "pnl_$", "bankroll_before", "bankroll_after", "ret_pct_account",
        ]
        if trade_rows:
            pd.DataFrame(trade_rows).to_csv(trades_path, index=False)
        else:
            pd.DataFrame(columns=trade_columns).to_csv(trades_path, index=False)

    # Write summary JSON
    summary_path = outdir / "inplay_backtest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="In-play P/L backtest runner")
    parser.add_argument(
        "--inplay",
        type=Path,
        required=True,
        help="Path to in-play kappa log CSV",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inplay_strategies.json"),
        help="Path to strategy config JSON (default: configs/inplay_strategies.json)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("logs"),
        help="Output directory for trades CSV and summary JSON (default: logs)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Optional path to match results CSV (match_id, winner, team_a, team_b) for match_end settlement at 1 or 0",
    )
    args = parser.parse_args()

    inplay_path = args.inplay.resolve()
    config_path = args.config.resolve()
    outdir = args.outdir.resolve()
    results_path = args.results.resolve() if args.results else (outdir / "inplay_match_results_clean.csv")

    if not inplay_path.exists():
        print(f"Error: in-play CSV not found: {inplay_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    match_settlements = load_match_results(results_path) if results_path else {}

    config = load_config(config_path)
    df = pd.read_csv(inplay_path)
    validate_columns(df, config["data"])
    run_backtest(df, config, outdir, match_settlements=match_settlements)
    print(f"Wrote summary to {outdir / 'inplay_backtest_summary.json'}")


if __name__ == "__main__":
    main()

# === PART 1/3 ===
# app.py — Esports Fair Odds (CS2 + Dota2)
# Run: streamlit run app.py

import os
import sys
import re
import json
import math
import subprocess
import asyncio
from pathlib import Path
from difflib import SequenceMatcher
import unicodedata
import string
from datetime import datetime
import csv
from typing import Optional  # 3.9-compatible Optional[...] for type hints

import streamlit as st

# Streamlit layout
st.set_page_config(layout="wide", page_title="Esports Fair Odds")

import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist
import requests

# ---- Windows asyncio fix: ensure subprocess support for Playwright ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ==========================
# Refactored modules
# ==========================
from fair_odds.paths import (
    PROJECT_ROOT, APP_DIR, LOG_PATH, LOG_COLUMNS,
    INPLAY_LOG_PATH, INPLAY_RESULTS_PATH, INPLAY_MAP_RESULTS_PATH,
    INPLAY_LOG_COLUMNS, INPLAY_RESULTS_COLUMNS, INPLAY_MAP_RESULTS_COLUMNS,
    KAPPA_CALIB_PATH, P_CALIB_PATH, P_CALIB_REPORT_PATH,
)
from fair_odds.odds import (
    american_to_decimal, decimal_to_american, implied_prob_from_american,
    calculate_fair_odds_curve, calculate_fair_odds_from_p, logistic_mapping,
    color_ev, prob_to_fair_american, compute_bo2_probs, ev_pct_decimal,
    decide_bet, decide_bo2_3way,
)
from fair_odds.calibration import (
    load_kappa_calibration, load_p_calibration_json, load_p_calibration,
    apply_p_calibration, get_kappa_multiplier,
    run_kappa_trainer, run_prob_trainer,
)
from fair_odds.data import (
    load_cs2_teams, load_dota_teams, get_team_tier,
    normalize_name, gosu_name_from_slug, read_csv_tolerant, sniff_bad_csv,
    piecewise_recent_weights,
)
from fair_odds.logs import (
    migrate_log_schema, recompute_metrics_from_logs, load_persisted_logs,
    persist_log_row, init_metrics, update_metrics_binary, update_metrics_3way,
    log_row, export_logs_df,
    persist_inplay_row, persist_inplay_result, persist_inplay_map_result,
    show_inplay_log_paths,
)
from fair_odds.scrapers import scrape_cs2_matches, scrape_dota_matches_gosu_subprocess, fetch_dota_matches
from fair_odds.scoring import (
    calculate_score, base_points_from_tier, normalize_series_result_from_fields,
    series_score_modifier_5tier,
    SERIES_CLAMP, SERIES_WEIGHTS, SERIES_PCT_OF_BASE_CAP,
)
from fair_odds.backtest import (
    load_state as load_backtest_state,
    save_state as save_backtest_state,
    load_match_settlements,
    incremental_update as backtest_incremental_update,
    repair_state_from_trades,
)

# ==========================
# App UI + remaining code (markets, in-play logic) — logic from fair_odds.*
# ==========================
# NOTE: calculate_score needs UI overrides for series modifier — pass explicitly
USE_SERIES_SCORE_MOD = True  # toggled by UI
# === PART 3/3 ===
# ==========================
# Streamlit App
# ==========================
st.title("Esports Fair Odds Calculator (CS2 + Dota2)")
init_metrics()

# Global calibration + decision controls (applies to both tabs)
with st.expander("Calibration & Mapping (optional)"):
    use_clip = st.checkbox("Clip extreme adjusted gaps before mapping", value=False,
                           help="Prevents absurd probabilities on outlier gaps.")
    clip_limit = st.slider("Clip limit (|gap|)", 10, 40, 25, 1)
    use_logistic = st.checkbox("Use calibrated logistic mapping (gap → p)", value=False,
                               help="OFF = original curve. ON = p = 1/(1+exp(-(a+b*gap))).")
    colA, colB = st.columns(2)
    with colA:
        a_param = st.number_input("Logistic a (intercept)",
                                  value=st.session_state.get("a_param", 0.0),
                                  step=0.01, format="%.4f")
    with colB:
        b_param = st.number_input("Logistic b (slope)",
                                  value=st.session_state.get("b_param", 0.18),
                                  step=0.01, format="%.4f",
                                  help="Fit this from your logged data later. Placeholder default.")

    st.markdown("---")
    # --- Decision layer controls ---
    min_edge_pct = st.slider(
        "Minimum EV to bet (%)", 0, 15, 5, 1,
        help="Edges below this are auto 'No bet' for logging/decision. Model EVs remain unchanged."
    )
    prob_gap_pp = st.slider(
        "Minimum model vs market probability gap (percentage points)", 0, 5, 3, 1,
        help="Require |p_model - p_market| ≥ this many percentage points to consider a bet."
    )
    shrink_target_matches = st.slider(
        "Effective matches for full confidence (shrinkage target)", 6, 20, 12, 1,
        help="Blends model p toward 50% when data is thin. Set higher = stricter."
    )

    st.markdown("---")
    # --- Series modifier controls (UI override of globals) ---
    use_series_mod_ui = st.checkbox(
        "Use series score modifier (map-count aware, tier-adjusted)",
        value=True,
        help="Rewards weaker teams when they take maps/wins; penalizes stronger teams for dropping maps. No bonuses for 0–2 losses."
    )
    series_clamp_ui = st.slider("Series modifier absolute clamp (points)", 0.25, 2.00, float(SERIES_CLAMP), 0.05)
    series_pct_cap_ui = st.slider("Series modifier cap as % of base", 0.10, 0.60, float(SERIES_PCT_OF_BASE_CAP), 0.05)

    # apply UI overrides to globals
    USE_SERIES_SCORE_MOD = bool(use_series_mod_ui)
    SERIES_CLAMP = float(series_clamp_ui)
    SERIES_PCT_OF_BASE_CAP = float(series_pct_cap_ui)


# ==========================
# Market data helpers (Kalshi / Polymarket)
# ==========================
from urllib.parse import urlparse, parse_qs

def _try_extract_kalshi_ticker(raw: str) -> str:
    """
    Best-effort extraction of a Kalshi market ticker from a pasted URL or raw input.
    If raw already looks like a ticker, returns it unchanged.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    # If it's already a simple ticker-like token, return it
    if re.fullmatch(r"[A-Z0-9_-]{3,64}", raw):
        return raw
    try:
        u = urlparse(raw)
        path_parts = [p for p in (u.path or "").split("/") if p]
        # Common patterns: /markets/{TICKER} or /trade/{TICKER}
        for i, part in enumerate(path_parts[:-1]):
            if part.lower() in ("markets", "market", "trade", "event"):
                cand = path_parts[i+1]
                if re.fullmatch(r"[A-Z0-9_-]{3,64}", cand):
                    return cand
        # Fallback: last segment
        if path_parts:
            cand = path_parts[-1]
            if re.fullmatch(r"[A-Z0-9_-]{3,64}", cand):
                return cand
    except Exception:
        pass
    return ""

def _try_extract_polymarket_token_id(raw: str) -> str:
    """
    Best-effort extraction of a Polymarket token_id from raw input or URL query (?token_id=...).
    NOTE: Polymarket website URLs often do NOT contain token_id; you may need to paste token_id directly.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    if re.fullmatch(r"\d{6,32}", raw):
        return raw
    try:
        u = urlparse(raw)
        qs = parse_qs(u.query or "")
        if "token_id" in qs and qs["token_id"]:
            cand = qs["token_id"][0].strip()
            if re.fullmatch(r"\d{6,32}", cand):
                return cand
    except Exception:
        pass
    return ""

@st.cache_data(ttl=3, show_spinner=False)
def fetch_kalshi_bid_ask(ticker: str):
    """
    Fetch best bid/ask (YES side) for a Kalshi market.

    Preferred source: GET /markets/{ticker} (often includes yes_bid/yes_ask/no_bid/no_ask).
    Fallback source:  GET /markets/{ticker}/orderbook (bids only; asks implied via 100 - opposite bid).

    Returns: (bid, ask, meta_dict) where bid/ask are in 0..1 (probability/price in $1 scale).
    """
    t = _try_extract_kalshi_ticker(ticker)
    if not t:
        raise ValueError("Missing Kalshi ticker")

    bases = [
        "https://api.kalshi.com/trade-api/v2",
        "https://api.elections.kalshi.com/trade-api/v2",
    ]

    def _to_int(x):
        if x is None:
            return None
        try:
            return int(float(x))
        except Exception:
            return None

    def _best_bid(levels):
        best_p = None
        best_sz = None

        if not levels:
            return None, None

        for lvl in levels:
            p = None
            q = None

            # Common format in Kalshi docs: [price_cents, quantity]
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 1:
                p = lvl[0]
                q = lvl[1] if len(lvl) >= 2 else None

            # Alternate formats: {"price": 63, "quantity": 46} or {"price": "0.63", "size": "46"}
            elif isinstance(lvl, dict):
                p = lvl.get("price", None)
                q = lvl.get("quantity", lvl.get("size", None))

            if p is None:
                continue

            try:
                p_f = float(p)
            except Exception:
                continue

            # If p looks like 0..1, scale to cents; else treat as cents.
            p_cents = (p_f * 100.0) if p_f <= 1.0 else p_f

            if best_p is None or p_cents > best_p:
                best_p = p_cents
                try:
                    best_sz = float(q) if q is not None else None
                except Exception:
                    best_sz = None

        return best_p, best_sz

    last_err = None

    for base in bases:
        # 1) Try market endpoint (simplest: already provides bid/ask in cents)
        try:
            url_m = f"{base}/markets/{t}"
            r = requests.get(url_m, timeout=5)
            r.raise_for_status()
            data = r.json()
            market = data.get("market", data)

            yb = _to_int(market.get("yes_bid", market.get("yesBid")))
            ya = _to_int(market.get("yes_ask", market.get("yesAsk")))
            nb = _to_int(market.get("no_bid",  market.get("noBid")))
            na = _to_int(market.get("no_ask",  market.get("noAsk")))
            lp = _to_int(market.get("last_price", market.get("lastPrice")))

            # If one ask missing but opposite bid exists, imply it
            if ya is None and nb is not None:
                ya = 100 - nb
            if na is None and yb is not None:
                na = 100 - yb

            if yb is not None or ya is not None:
                bid = (yb / 100.0) if yb is not None else None
                ask = (ya / 100.0) if ya is not None else None
                meta = {
                    "ticker": t,
                    "base_url": base,
                    "source": "market",
                    "yes_bid_cents": yb,
                    "yes_ask_cents": ya,
                    "no_bid_cents": nb,
                    "no_ask_cents": na,
                    "last_price_cents": lp,
                }
                return bid, ask, meta

        except Exception as e:
            last_err = e  # keep, but still attempt orderbook on this base

        # 2) Fallback to orderbook endpoint (bids only)
        try:
            url = f"{base}/markets/{t}/orderbook"
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            data = r.json()
            ob = data.get("orderbook", data)

            yes_bids = ob.get("yes", []) or ob.get("yes_bids", []) or ob.get("yesOrders", [])
            no_bids  = ob.get("no",  []) or ob.get("no_bids",  []) or ob.get("noOrders",  [])

            best_yes_c, best_yes_sz = _best_bid(yes_bids)
            best_no_c,  best_no_sz  = _best_bid(no_bids)

            if best_yes_c is None and best_no_c is None:
                raise RuntimeError("No orderbook levels returned")

            bid = (best_yes_c / 100.0) if best_yes_c is not None else None
            ask = ((100.0 - best_no_c) / 100.0) if best_no_c is not None else None

            meta = {
                "ticker": t,
                "base_url": base,
                "source": "orderbook",
                "best_yes_bid_cents": best_yes_c,
                "best_yes_bid_size": best_yes_sz,
                "best_no_bid_cents": best_no_c,
                "best_no_bid_size": best_no_sz,
            }
            return bid, ask, meta

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Kalshi fetch failed for {t}: {last_err}")
def _kalshi_parse_event_ticker(url_or_ticker: str) -> Optional[str]:
    """Best-effort parse of a Kalshi *event_ticker* from a Kalshi page URL or ticker.

    - If given a Kalshi site URL like .../kxvalorantgame-26jan29blgfpx, returns KXVALORANTGAME-26JAN29BLGFPX.
    - If given a market ticker like KXVALORANTGAME-26JAN29BLGFPX-BLG, returns KXVALORANTGAME-26JAN29BLGFPX.
    - If given an event ticker already, returns it uppercased.
    """
    if not url_or_ticker:
        return None
    s = str(url_or_ticker).strip()
    if not s:
        return None

    # If it's a URL, take the last path segment.
    try:
        from urllib.parse import urlparse
        if "://" in s:
            parsed = urlparse(s)
            path = (parsed.path or "").rstrip("/")
            if path:
                s = path.split("/")[-1]
    except Exception:
        pass

    s = s.strip().strip("/")
    if not s:
        return None

    s_up = s.upper()

    # If user pasted a market ticker (event + suffix), strip suffix.
    # Example: KXVALORANTGAME-26JAN29BLGFPX-BLG -> KXVALORANTGAME-26JAN29BLGFPX
    if re.match(r"^[A-Z0-9]+(?:-[A-Z0-9]+)*-[A-Z]{2,6}$", s_up):
        parts = s_up.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:-1])

    return s_up


@st.cache_data(ttl=30, show_spinner=False)
def kalshi_list_markets_for_event(event_ticker: str):
    """Return a list of markets for a Kalshi event_ticker. Unauthenticated read."""
    ev = _kalshi_parse_event_ticker(event_ticker)
    if not ev:
        raise ValueError("Missing/invalid Kalshi event ticker")
    bases = [
        "https://api.kalshi.com/trade-api/v2",
        "https://api.elections.kalshi.com/trade-api/v2",
    ]
    last_err = None
    for base in bases:
        try:
            url = f"{base}/markets"
            r = requests.get(url, params={"event_ticker": ev}, timeout=6)
            r.raise_for_status()
            data = r.json() or {}
            markets = data.get("markets", []) or []
            out = []
            for m in markets:
                out.append({
                    "ticker": m.get("ticker"),
                    "title": m.get("title") or "",
                    "subtitle": m.get("subtitle") or "",
                })
            out = [x for x in out if x.get("ticker")]
            out.sort(key=lambda x: (x["ticker"], x.get("title","")))
            if not out:
                raise RuntimeError("No markets returned for that event_ticker")
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Kalshi market list failed: {last_err}")

@st.cache_data(ttl=3, show_spinner=False)
def fetch_polymarket_bid_ask(token_id: str):
    """
    Fetch best bid/ask from Polymarket CLOB order book for a specific token_id.
    Returns: (bid, ask, meta_dict) where bid/ask are in 0..1.
    """
    tid = _try_extract_polymarket_token_id(token_id)
    if not tid:
        raise ValueError("Missing Polymarket token_id (asset_id)")
    url = f"https://clob.polymarket.com/book?token_id={tid}"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    data = r.json()
    bids = data.get("bids", []) or []
    asks = data.get("asks", []) or []

    def _best_bid(bids_list):
        best_p = None
        best_sz = None
        for lvl in bids_list:
            try:
                p = float(lvl.get("price"))
            except Exception:
                continue
            try:
                sz = float(lvl.get("size", 0))
            except Exception:
                sz = None
            if best_p is None or p > best_p:
                best_p = p
                best_sz = sz
        return best_p, best_sz

    def _best_ask(asks_list):
        best_p = None
        best_sz = None
        for lvl in asks_list:
            try:
                p = float(lvl.get("price"))
            except Exception:
                continue
            try:
                sz = float(lvl.get("size", 0))
            except Exception:
                sz = None
            if best_p is None or p < best_p:
                best_p = p
                best_sz = sz
        return best_p, best_sz

    bid, bid_sz = _best_bid(bids)
    ask, ask_sz = _best_ask(asks)
    if bid is None and ask is None:
        raise RuntimeError("No bids/asks returned")
    meta = {
        "token_id": tid,
        "bid_size": bid_sz,
        "ask_size": ask_sz,
    }
    return bid, ask, meta


def _run_inplay_backtest_and_refresh_session():
    """Run backtest with default paths; update session state so backtest tab shows latest trades. Called from Add snapshot (CS2/Valorant)."""
    default_inplay = str(INPLAY_LOG_PATH) if INPLAY_LOG_PATH else str(PROJECT_ROOT / "logs" / "inplay_kappa_logs_clean.csv")
    default_config = str(PROJECT_ROOT / "configs" / "inplay_strategies.json")
    default_outdir = str(PROJECT_ROOT / "logs")
    script_path = PROJECT_ROOT / "scripts" / "inplay_backtest_runner.py"
    if not script_path.exists():
        return
    try:
        subprocess.run(
            [sys.executable, str(script_path), "--inplay", default_inplay, "--config", default_config, "--outdir", default_outdir],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception):
        return
    paths_key = (default_inplay, default_config, default_outdir)
    st.session_state["inplay_bt_paths"] = paths_key
    summary_path = Path(default_outdir) / "inplay_backtest_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                st.session_state["inplay_bt_summary"] = json.load(f)
        except Exception:
            pass
    summary = st.session_state.get("inplay_bt_summary") or {}
    trades = {}
    for sid in summary.keys():
        tp = Path(default_outdir) / f"inplay_backtest_trades__{sid}.csv"
        if tp.exists():
            try:
                trades[sid] = pd.read_csv(tp)
            except Exception:
                pass
    st.session_state["inplay_bt_trades"] = trades


def _run_inplay_incremental_and_refresh_session(row_dict: dict) -> None:
    """Run incremental backtest on one new row; update state file, append trades to CSV, refresh session. No subprocess."""
    outdir = PROJECT_ROOT / "logs"
    state_path = outdir / "inplay_backtest_state.json"
    results_path = outdir / "inplay_match_results_clean.csv"
    config_path = PROJECT_ROOT / "configs" / "inplay_strategies.json"
    default_inplay = str(INPLAY_LOG_PATH) if INPLAY_LOG_PATH else str(outdir / "inplay_kappa_logs_clean.csv")
    default_config = str(config_path)
    default_outdir = str(outdir)
    if not config_path.exists():
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return
    state = load_backtest_state(state_path)
    trades_for_repair = {}
    for strat in (config.get("strategies") or []):
        sid = strat["id"] if isinstance(strat, dict) else strat
        tp = outdir / f"inplay_backtest_trades__{sid}.csv"
        if tp.exists():
            try:
                trades_for_repair[sid] = pd.read_csv(tp)
            except Exception:
                pass
    if trades_for_repair:
        state = repair_state_from_trades(state, trades_for_repair)
        save_backtest_state(state, state_path)
    match_settlements = load_match_settlements(results_path)
    new_state, new_trades_by_strategy = backtest_incremental_update(row_dict, state, config, match_settlements)
    save_backtest_state(new_state, state_path)
    st.session_state["inplay_bt_state"] = new_state
    st.session_state["inplay_bt_paths"] = (default_inplay, default_config, default_outdir)

    trade_columns = [
        "trade_id", "strategy_id", "match_id", "side",
        "entry_ts", "entry_px", "entry_mid", "entry_fair", "entry_band_lo", "entry_band_hi", "entry_spread_abs",
        "exit_ts", "exit_px", "exit_reason", "hold_minutes", "contracts",
        "pnl_price", "pnl_$", "bankroll_before", "bankroll_after", "ret_pct_account",
    ]
    trades = dict(st.session_state.get("inplay_bt_trades") or {})
    for sid, trades_list in new_trades_by_strategy.items():
        if not trades_list:
            continue
        tp = outdir / f"inplay_backtest_trades__{sid}.csv"
        existing = pd.DataFrame()
        if tp.exists():
            try:
                existing = pd.read_csv(tp)
            except Exception:
                pass
        new_df = pd.DataFrame(trades_list)
        if not new_df.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined[[c for c in trade_columns if c in combined.columns]]
            combined.to_csv(tp, index=False)
        if sid not in trades:
            trades[sid] = pd.read_csv(tp) if tp.exists() else pd.DataFrame(trades_list)
        else:
            trades[sid] = pd.concat([trades[sid], pd.DataFrame(trades_list)], ignore_index=True)
    st.session_state["inplay_bt_trades"] = trades

    # Recompute summary from full session trades so KPI table stays current
    sizing = config.get("sizing", {})
    start_bankroll = float(sizing.get("start_bankroll", 2000))
    summary = {}
    for sid, df in trades.items():
        if df is None or len(df) == 0:
            summary[sid] = {
                "trades": 0, "total_pnl_$": 0, "end_bankroll": start_bankroll,
                "total_return_pct": 0, "win_rate": 0, "avg_pnl_$": 0,
                "median_pnl_$": 0, "best_trade_$": 0, "worst_trade_$": 0,
                "avg_hold_minutes": 0, "max_drawdown_pct": 0,
            }
            continue
        n = len(df)
        total_pnl = float(df["pnl_$"].sum()) if "pnl_$" in df.columns else 0
        end_br = float(df["bankroll_after"].iloc[-1]) if "bankroll_after" in df.columns else start_bankroll
        total_ret = (end_br - start_bankroll) / start_bankroll if start_bankroll else 0
        wins = int((df["pnl_$"] > 0).sum()) if "pnl_$" in df.columns else 0
        win_rate = wins / n if n else 0
        avg_pnl = total_pnl / n if n else 0
        pnls = df["pnl_$"].tolist() if "pnl_$" in df.columns else []
        median_pnl = float(pd.Series(pnls).median()) if pnls else 0
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0
        avg_hold = float(df["hold_minutes"].mean()) if "hold_minutes" in df.columns and n else 0
        equity = df["bankroll_after"].tolist() if "bankroll_after" in df.columns else []
        running_max = equity[0] if equity else 0
        max_dd = 0.0
        for e in equity:
            running_max = max(running_max, e)
            dd = (e - running_max) / running_max if running_max else 0.0
            max_dd = min(max_dd, dd)
        summary[sid] = {
            "trades": n, "total_pnl_$": total_pnl, "end_bankroll": end_br,
            "total_return_pct": total_ret, "win_rate": win_rate, "avg_pnl_$": avg_pnl,
            "median_pnl_$": median_pnl, "best_trade_$": best, "worst_trade_$": worst,
            "avg_hold_minutes": avg_hold, "max_drawdown_pct": max_dd,
        }
    st.session_state["inplay_bt_summary"] = summary


tabs = st.tabs([
    "CS2", "Dota2", "Diagnostics / Export",
    "CS2 In-Play Indicator (MVP)", "Valorant In-Play Indicator (MVP)", "Calibration",
    "In-Play Backtest (P/L Tracker)",
])

# --------------------------
# CS2 TAB
# --------------------------
with tabs[0]:
    st.header("CS2 Fair Odds")
    df_cs2 = load_cs2_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (CS2)", df_cs2["team"].tolist(), key="cs2_a")
        row_a = df_cs2.loc[df_cs2["team"] == team_a].iloc[0]
        team_a_slug = row_a["slug"]
        hltv_a_val = row_a["hltv_id"]
        team_a_id = str(int(hltv_a_val)) if pd.notna(hltv_a_val) else None

        if team_a_id is None:
            st.caption("Team A: no hltv_id — scraping disabled (calculations still work).")

        if st.button("Scrape Team A (CS2)"):
            if team_a_id is None or not isinstance(team_a_slug, str) or not team_a_slug:
                st.error("Cannot scrape Team A: missing hltv_id or slug for this team.")
                st.info("You can still Calculate using the tiers; scraping just won’t work for this team.")
                matches_a = []
            else:
                matches_a = scrape_cs2_matches(team_a_id, team_a_slug)
            st.session_state["cs2_matches_a"] = matches_a
        matches_a = st.session_state.get("cs2_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (CS2)", df_cs2["team"].tolist(), key="cs2_b")
        row_b = df_cs2.loc[df_cs2["team"] == team_b].iloc[0]
        team_b_slug = row_b["slug"]
        hltv_b_val = row_b["hltv_id"]
        team_b_id = str(int(hltv_b_val)) if pd.notna(hltv_b_val) else None

        if team_b_id is None:
            st.caption("Team B: no hltv_id — scraping disabled (calculations still work).")

        if st.button("Scrape Team B (CS2)"):
            if team_b_id is None or not isinstance(team_b_slug, str) or not team_b_slug:
                st.error("Cannot scrape Team B: missing hltv_id or slug for this team.")
                st.info("You can still Calculate using the tiers; scraping just won’t work for this team.")
                matches_b = []
            else:
                matches_b = scrape_cs2_matches(team_b_id, team_b_slug)
            st.session_state["cs2_matches_b"] = matches_b
        matches_b = st.session_state.get("cs2_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (CS2) — American", value=-140, key="cs2_odds_a")
    odds_b = st.number_input("Team B Market Odds (CS2) — American", value=+120, key="cs2_odds_b")

    st.subheader("Recency Weighting (CS2)")
    K_cs2 = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_cs2")
    decay_cs2 = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_cs2")
    floor_cs2 = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_cs2")

    if st.button("Calculate (CS2)"):
        if matches_a and matches_b:
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            team_a_tier = float(row_a["tier"])
            team_b_tier = float(row_b["tier"])

            raw_a, adj_a, breakdown_a = calculate_score(
                matches_a, df_cs2, current_opponent_tier=team_b_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True,
                draw_policy="loss", self_team_tier=team_a_tier,
                use_series_mod=use_series_mod_ui, series_clamp=series_clamp_ui,
                series_pct_cap=series_pct_cap_ui, series_weights=SERIES_WEIGHTS
            )
            raw_b, adj_b, breakdown_b = calculate_score(
                matches_b, df_cs2, current_opponent_tier=team_a_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True,
                draw_policy="loss", self_team_tier=team_b_tier,
                use_series_mod=use_series_mod_ui, series_clamp=series_clamp_ui,
                series_pct_cap=series_pct_cap_ui, series_weights=SERIES_WEIGHTS
            )

            raw_gap = raw_a - raw_b
            adj_gap = adj_a - adj_b
            if use_clip:
                adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

            p = logistic_mapping(adj_gap, a_param, b_param) if use_logistic else calculate_fair_odds_curve(adj_gap)
            dec_a = american_to_decimal(odds_a); dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100
            fair_a, fair_b = calculate_fair_odds_from_p(p)

            st.subheader("Summary (CS2)")
            st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
            st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.caption("EV below reflects raw model p (pre-decision filters).")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a: st.text(line)
            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b: st.text(line)

            decision = decide_bet(
                p_model=p,
                odds_a=odds_a, odds_b=odds_b,
                n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                min_edge_pct=min_edge_pct,
                prob_gap_pp=prob_gap_pp,
                shrink_target=shrink_target_matches,
            )
            if decision["choice"] is None:
                st.markdown(f"**Decision:** ❌ No bet** — {decision['reason']}")
                ev_a_eff = ev_b_eff = None
            else:
                pick_team = team_a if decision["choice"] == "A" else team_b
                pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else None
                ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else None

            update_metrics_binary(ev_a_eff, ev_b_eff, odds_a, odds_b)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "game": "CS2",
                "series_format": "BO3",
                "market_type": "BINARY",
                "team_a": team_a, "team_b": team_b,
                "tier_a": team_a_tier, "tier_b": team_b_tier,
                "K": K_cs2, "decay": decay_cs2, "floor": floor_cs2,
                "raw_a": raw_a, "raw_b": raw_b,
                "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                "use_logistic": use_logistic, "a": a_param, "b": b_param,
                "use_clip": use_clip, "clip_limit": clip_limit,
                "p_model": p,
                "odds_a": odds_a, "odds_b": odds_b,
                "fair_a": fair_a, "fair_b": fair_b,
                "ev_a_pct": ev_a, "ev_b_pct": ev_b,
                "p_decide": decision["p_decide"],
                "ev_a_dec": decision["ev_a_dec"],
                "ev_b_dec": decision["ev_b_dec"],
                "decision": decision["choice"] or "NO_BET",
                "decision_reason": decision["reason"],
                "min_edge_pct": min_edge_pct,
                "prob_gap_pp": prob_gap_pp,
                "shrink_target": shrink_target_matches,
                "p_map_model":"", "p_map_decide":"", "draw_k":"",
                "p_a20":"", "p_draw":"", "p_b02":"",
                "odds_a20":"", "odds_draw":"", "odds_b02":"",
                "ev_a20_pct":"", "ev_draw_pct":"", "ev_b02_pct":"",
                "selected_outcome":"", "selected_prob":"", "selected_odds":"", "selected_ev_pct":"",
                "fair_a20_us":"", "fair_draw_us":"", "fair_b02_us":""
            }
            log_row(entry)
        else:
            st.warning("Scrape both teams first.")

# --------------------------
# Dota TAB  (with BO2 3-way + graded draws)
# --------------------------
with tabs[1]:
    st.header("Dota 2 Fair Odds")
    df_dota = load_dota_teams()
    dota_ok = df_dota

    series_format = st.radio(
        "Series format",
        options=["BO3 (binary win/lose)", "BO2 (3-way with draw)"],
        horizontal=True,
        index=0,
        help="BO2 adds a proper 3-way (A 2–0 / Draw 1–1 / B 0–2) market."
    )

    src = st.radio("Data source", ["OpenDota (API)", "GosuGamers (Scrape)"], horizontal=True, index=1)
    headed_toggle = st.toggle("Show browser during Gosu scrape", value=True,
                              help="Use a visible browser window for the scrape.")
    browser_channel = st.selectbox("Browser for Gosu scrape",
        options=["bundled", "chrome", "msedge"], index=0, key="dota_browser_channel",
        help="Use 'bundled' to mimic CLI (Playwright Chromium). Or force Chrome/Edge if installed.",
    )
    zoom_pct = st.slider("Gosu page zoom (%)", min_value=60, max_value=110, value=80, step=5,
                         help="Zoom out to keep the paginator in view.")
    target_matches = st.slider("Matches to use (last N)", min_value=8, max_value=30, value=14, step=1)

    st.subheader("Recency Weighting (Dota2)")
    K_dota = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_dota")
    decay_dota = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_dota")
    floor_dota = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_dota")

    st.subheader("Bo2 Draw Handling")
    draw_mode = st.radio(
        "Draw policy",
        ["Graded by tier (recommended)", "Neutral (0 points)", "Legacy (treat as loss)"],
        horizontal=False,
        index=0,
        help="Graded: + if draw vs stronger, − if draw vs weaker; scaled by tier gap."
    )
    draw_gamma = st.slider("Draw magnitude γ (0–1)", 0.00, 1.00, 0.50, 0.05,
                           help="0 = ignore draws, 1 = half-way to a full win/loss at max gap.")
    draw_gap_cap = st.slider("Tier gap cap for draws", 1.0, 4.0, 3.0, 0.5,
                             help="How many tier steps until a draw is 'maximally' graded.")
    draw_gap_power = st.slider("Tier gap curve (power)", 0.5, 2.0, 1.0, 0.1,
                               help="<1 boosts small gaps; >1 emphasizes big gaps.")
    draw_policy = "graded" if draw_mode.startswith("Graded") else ("neutral" if draw_mode.startswith("Neutral") else "loss")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Dota2)", dota_ok["team"].tolist(), key="dota_a")
        row_a = dota_ok.loc[dota_ok["team"] == team_a].iloc[0]
        team_a_slug = row_a["slug"]

        team_a_id = None
        try:
            if "opendota_id" in row_a and pd.notna(row_a["opendota_id"]) and str(row_a["opendota_id"]).strip() != "":
                team_a_id = int(float(row_a["opendota_id"]))
        except Exception:
            team_a_id = None

        if st.button("Scrape Team A (Dota2)"):
            if src.startswith("OpenDota"):
                if team_a_id is None:
                    st.info("No OpenDota ID for this team — falling back to Gosu scraping.")
                    gosu_name_a = gosu_name_from_slug(team_a_slug)
                    matches_a = scrape_dota_matches_gosu_subprocess(
                        team_slug=team_a_slug, team_name=gosu_name_a,
                        target=target_matches, headed=headed_toggle,
                        browser_channel=browser_channel, zoom=zoom_pct,
                    )
                else:
                    matches_a = fetch_dota_matches(team_a_id, limit=target_matches)
            else:
                gosu_name_a = gosu_name_from_slug(team_a_slug)
                matches_a = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_a_slug, team_name=gosu_name_a,
                    target=target_matches, headed=headed_toggle,
                    browser_channel=browser_channel, zoom=zoom_pct,
                )
            st.session_state["dota_matches_a"] = matches_a
        matches_a = st.session_state.get("dota_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (Dota2)", dota_ok["team"].tolist(), key="dota_b")
        row_b = dota_ok.loc[dota_ok["team"] == team_b].iloc[0]
        team_b_slug = row_b["slug"]

        team_b_id = None
        try:
            if "opendota_id" in row_b and pd.notna(row_b["opendota_id"]) and str(row_b["opendota_id"]).strip() != "":
                team_b_id = int(float(row_b["opendota_id"]))
        except Exception:
            team_b_id = None

        if st.button("Scrape Team B (Dota2)"):
            if src.startswith("OpenDota"):
                if team_b_id is None:
                    st.info("No OpenDota ID for this team — falling back to Gosu scraping.")
                    gosu_name_b = gosu_name_from_slug(team_b_slug)
                    matches_b = scrape_dota_matches_gosu_subprocess(
                        team_slug=team_b_slug, team_name=gosu_name_b,
                        target=target_matches, headed=headed_toggle,
                        browser_channel=browser_channel, zoom=zoom_pct,
                    )
                else:
                    matches_b = fetch_dota_matches(team_b_id, limit=target_matches)
            else:
                gosu_name_b = gosu_name_from_slug(team_b_slug)
                matches_b = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_b_slug, team_name=gosu_name_b,
                    target=target_matches, headed=headed_toggle,
                    browser_channel=browser_channel, zoom=zoom_pct,
                )
            st.session_state["dota_matches_b"] = matches_b
        matches_b = st.session_state.get("dota_matches_b", [])
        st.write(matches_b)

    if series_format.startswith("BO3"):
        odds_a = st.number_input("Team A Market Odds (BO3) — American", value=-140, key="dota_odds_a")
        odds_b = st.number_input("Team B Market Odds (BO3) — American", value=+120, key="dota_odds_b")
    else:
        st.markdown("### BO2 — 3-Way Market (A 2–0 / Draw 1–1 / B 0–2)")
        draw_k = st.slider("Draw calibration k (fit later; 1.0 = neutral)", 0.50, 1.50, 1.00, 0.01)

        colA2, colD, colB2 = st.columns(3)
        with colA2:
            odds_a20_us = st.number_input("Odds: A 2–0 (American)", value=+220, step=1, format="%d")
        with colD:
            odds_draw_us = st.number_input("Odds: Draw 1–1 (American)", value=+110, step=1, format="%d")
        with colB2:
            odds_b02_us = st.number_input("Odds: B 0–2 (American)", value=+260, step=1, format="%d")

        odds_a20 = american_to_decimal(int(odds_a20_us))
        odds_draw = american_to_decimal(int(odds_draw_us))
        odds_b02 = american_to_decimal(int(odds_b02_us))

    if st.button(f"Calculate (Dota2 — {series_format.split()[0]})"):
        if "dota_matches_a" in st.session_state and "dota_matches_b" in st.session_state:
            matches_a = st.session_state["dota_matches_a"][:target_matches]
            matches_b = st.session_state["dota_matches_b"][:target_matches]
            if not matches_a or not matches_b:
                st.warning("Scrape both teams first (Dota2).")
            else:
                st.text(f"Using last {min(len(matches_a), len(matches_b))} matches for both teams.")

                team_a_tier = float(row_a["tier"])
                team_b_tier = float(row_b["tier"])

                raw_a, adj_a, breakdown_a = calculate_score(
                    matches_a, df_dota, current_opponent_tier=team_b_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True,
                    draw_policy=draw_policy, self_team_tier=team_a_tier,
                    draw_gamma=draw_gamma, draw_gap_cap=draw_gap_cap, draw_gap_power=draw_gap_power,
                    use_series_mod=use_series_mod_ui, series_clamp=series_clamp_ui,
                    series_pct_cap=series_pct_cap_ui, series_weights=SERIES_WEIGHTS
                )
                raw_b, adj_b, breakdown_b = calculate_score(
                    matches_b, df_dota, current_opponent_tier=team_a_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True,
                    draw_policy=draw_policy, self_team_tier=team_b_tier,
                    draw_gamma=draw_gamma, draw_gap_cap=draw_gap_cap, draw_gap_power=draw_gap_power,
                    use_series_mod=use_series_mod_ui, series_clamp=series_clamp_ui,
                    series_pct_cap=series_pct_cap_ui, series_weights=SERIES_WEIGHTS
                )

                raw_gap = raw_a - raw_b
                adj_gap = adj_a - adj_b
                if use_clip:
                    adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

                p_map = logistic_mapping(adj_gap, a_param, b_param) if use_logistic else calculate_fair_odds_curve(adj_gap)

                if series_format.startswith("BO3"):
                    dec_a = american_to_decimal(odds_a); dec_b = american_to_decimal(odds_b)
                    ev_a = ((p_map * dec_a) - 1) * 100
                    ev_b = (((1 - p_map) * dec_b) - 1) * 100
                    fair_a, fair_b = calculate_fair_odds_from_p(p_map)

                    st.subheader("Summary (Dota2 — BO3)")
                    st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
                    st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
                    st.text(f"Market Odds: {odds_a} / {odds_b}")
                    st.text(f"Fair Odds:   {fair_a} / {fair_b}")
                    st.text(f"Win Probability (series): {round(p_map * 100, 2)}%")
                    st.caption("EV below reflects raw model p (pre-decision filters).")
                    st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
                    st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

                    st.subheader(f"{team_a} Breakdown")
                    for line in breakdown_a: st.text(line)
                    st.subheader(f"{team_b} Breakdown")
                    for line in breakdown_b: st.text(line)

                    decision = decide_bet(
                        p_model=p_map,
                        odds_a=odds_a, odds_b=odds_b,
                        n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                        min_edge_pct=min_edge_pct,
                        prob_gap_pp=prob_gap_pp,
                        shrink_target=shrink_target_matches,
                    )
                    if decision["choice"] is None:
                        st.markdown(f"**Decision:** ❌ No bet** — {decision['reason']}")
                        ev_a_eff = ev_b_eff = None
                    else:
                        pick_team = team_a if decision["choice"] == "A" else team_b
                        pick_ev = decision["ev_a_dec"] if decision["choice"] == "A" else decision["ev_b_dec"]
                        st.markdown(f"**Decision:** ✅ Bet **{pick_team}** ({pick_ev:+.2f}% EV) — {decision['reason']}")
                        ev_a_eff = decision["ev_a_dec"] if decision["choice"] == "A" else None
                        ev_b_eff = decision["ev_b_dec"] if decision["choice"] == "B" else None

                    update_metrics_binary(ev_a_eff, ev_b_eff, odds_a, odds_b)
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "game": "Dota2",
                        "series_format": "BO3",
                        "market_type": "BINARY",
                        "team_a": team_a, "team_b": team_b,
                        "tier_a": team_a_tier, "tier_b": team_b_tier,
                        "K": K_dota, "decay": decay_dota, "floor": floor_dota,
                        "raw_a": raw_a, "raw_b": raw_b,
                        "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                        "use_logistic": use_logistic, "a": a_param, "b": b_param,
                        "use_clip": use_clip, "clip_limit": clip_limit,
                        "p_model": p_map,
                        "odds_a": odds_a, "odds_b": odds_b,
                        "fair_a": fair_a, "fair_b": fair_b,
                        "ev_a_pct": ev_a, "ev_b_pct": ev_b,
                        "p_decide": decision["p_decide"],
                        "ev_a_dec": decision["ev_a_dec"],
                        "ev_b_dec": decision["ev_b_dec"],
                        "decision": decision["choice"] or "NO_BET",
                        "decision_reason": decision["reason"],
                        "min_edge_pct": min_edge_pct,
                        "prob_gap_pp": prob_gap_pp,
                        "shrink_target": shrink_target_matches,
                        "p_map_model":"", "p_map_decide":"", "draw_k":"",
                        "p_a20":"", "p_draw":"", "p_b02":"",
                        "odds_a20":"", "odds_draw":"", "odds_b02":"",
                        "ev_a20_pct":"", "ev_draw_pct":"", "ev_b02_pct":"",
                        "selected_outcome":"", "selected_prob":"", "selected_odds":"", "selected_ev_pct":"",
                        "fair_a20_us":"", "fair_draw_us":"", "fair_b02_us":""
                    }
                    log_row(entry)

                else:
                    st.subheader("Summary (Dota2 — BO2 3-Way)")
                    st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
                    st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
                    st.text(f"Per-map win prob (Team A) — model: {round(p_map*100,2)}%")

                    probs_model = compute_bo2_probs(p_map, k=draw_k)
                    evs_model = {
                        "A2-0": ev_pct_decimal(probs_model["A2-0"], odds_a20),
                        "DRAW": ev_pct_decimal(probs_model["DRAW"], odds_draw),
                        "B0-2": ev_pct_decimal(probs_model["B0-2"], odds_b02),
                    }
                    fair_a20_us  = prob_to_fair_american(probs_model["A2-0"])
                    fair_draw_us = prob_to_fair_american(probs_model["DRAW"])
                    fair_b02_us  = prob_to_fair_american(probs_model["B0-2"])

                    df_preview = pd.DataFrame([
                        {"Outcome":"A 2–0","Prob%":round(probs_model["A2-0"]*100,2),"Odds (US)": int(odds_a20_us),"Fair (US)": int(fair_a20_us),"EV%":round(evs_model["A2-0"],2)},
                        {"Outcome":"Draw 1–1","Prob%":round(probs_model["DRAW"]*100,2),"Odds (US)": int(odds_draw_us),"Fair (US)": int(fair_draw_us),"EV%":round(evs_model["DRAW"],2)},
                        {"Outcome":"B 0–2","Prob%":round(probs_model["B0-2"]*100,2),"Odds (US)": int(odds_b02_us),"Fair (US)": int(fair_b02_us),"EV%":round(evs_model["B0-2"],2)},
                    ])
                    st.dataframe(df_preview, use_container_width=True)
                    st.text(f"Fair (US): A2–0 {fair_a20_us:+d} | Draw {fair_draw_us:+d} | B0–2 {fair_b02_us:+d}")

                    dec3 = decide_bo2_3way(
                        p_map_model=p_map,
                        n_matches_a=len(matches_a), n_matches_b=len(matches_b),
                        min_edge_pct=min_edge_pct,
                        prob_gap_pp=prob_gap_pp,
                        shrink_target=shrink_target_matches,
                        draw_k=draw_k,
                        odds_a20=odds_a20, odds_draw=odds_draw, odds_b02=odds_b02,
                    )

                    if dec3["selected_outcome"] is None:
                        st.markdown(f"**Decision:** ❌ No bet** — {dec3['reason']}")
                        update_metrics_3way(None, None, odds_a20, odds_draw, odds_b02)
                    else:
                        sel_us = decimal_to_american(dec3["selected_odds"])
                        st.markdown(
                            f"**Decision:** ✅ Bet **{dec3['selected_outcome']}** "
                            f"(EV {dec3['selected_ev_pct']:+.2f}%, Prob {dec3['selected_prob']*100:.2f}%, "
                            f"Odds {sel_us:+d}) — {dec3['reason']}"
                        )
                        update_metrics_3way(dec3["selected_ev_pct"], dec3["selected_outcome"],
                                            odds_a20, odds_draw, odds_b02)

                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "game": "Dota2",
                        "series_format": "BO2",
                        "market_type": "3WAY",
                        "team_a": team_a, "team_b": team_b,
                        "tier_a": team_a_tier, "tier_b": team_b_tier,
                        "K": K_dota, "decay": decay_dota, "floor": floor_dota,
                        "raw_a": raw_a, "raw_b": raw_b,
                        "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                        "use_logistic": use_logistic, "a": a_param, "b": b_param,
                        "use_clip": use_clip, "clip_limit": clip_limit,
                        "p_model": "", "odds_a": "", "odds_b": "", "fair_a": "", "fair_b": "", "ev_a_pct": "", "ev_b_pct": "",
                        "p_decide": "", "ev_a_dec": "", "ev_b_dec": "",
                        "decision": (dec3["selected_outcome"] or "NO_BET"),
                        "decision_reason": dec3["reason"],
                        "min_edge_pct": min_edge_pct,
                        "prob_gap_pp": prob_gap_pp,
                        "shrink_target": shrink_target_matches,
                        "p_map_model": p_map,
                        "p_map_decide": dec3["p_map_decide"],
                        "draw_k": draw_k,
                        "p_a20": probs_model["A2-0"], "p_draw": probs_model["DRAW"], "p_b02": probs_model["B0-2"],
                        "odds_a20": odds_a20, "odds_draw": odds_draw, "odds_b02": odds_b02,
                        "ev_a20_pct": evs_model["A2-0"], "ev_draw_pct": evs_model["DRAW"], "ev_b02_pct": evs_model["B0-2"],
                        "selected_outcome": dec3["selected_outcome"] or "",
                        "selected_prob": dec3["selected_prob"] or "",
                        "selected_odds": dec3["selected_odds"] or "",
                        "selected_ev_pct": dec3["selected_ev_pct"] or "",
                        "fair_a20_us": fair_a20_us, "fair_draw_us": fair_draw_us, "fair_b02_us": fair_b02_us,
                    }
                    log_row(entry)
        else:
            st.warning("Scrape both teams first (Dota2).")

# --------------------------
# Diagnostics / Export
# --------------------------
with tabs[2]:
    st.header("Diagnostics")
    m = st.session_state["metrics"]
    total = max(1, m["total"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Calcs", m["total"])
    col2.metric("Fav value %", f"{(m['fav_value']/total)*100:.1f}%")
    col3.metric("Dog value %", f"{(m['dog_value']/total)*100:.1f}%")
    col4.metric("No-bet %", f"{(m['no_bet']/total)*100:.1f}%")

    st.subheader("Export Logged Calculations")
    df_logs = export_logs_df()
    st.dataframe(df_logs, use_container_width=True, height=320)
    csv_bytes = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, "fair_odds_logs_snapshot.csv", "text/csv")

    if st.button("Overwrite on-disk log with current in-memory logs"):
        try:
            pd.DataFrame(st.session_state["logs"], columns=LOG_COLUMNS).to_csv(LOG_PATH, index=False, line_terminator="\n")
            st.success("Log file overwritten.")
        except Exception as e:
            st.error(f"Failed to write log file: {e}")

    st.subheader("Import / Merge Logs")
    uploaded = st.file_uploader("Upload a logs CSV to merge into history", type=["csv"])
    merge_col1, merge_col2 = st.columns(2)
    with merge_col1:
        dedup_key = st.selectbox(
            "De-duplication key",
            ["timestamp,game,team_a,team_b,adj_gap", "timestamp", "team_a,team_b,adj_gap"],
            help="Used to drop duplicates when merging"
        )
    with merge_col2:
        overwrite_disk = st.checkbox("Overwrite on-disk log with merged result", value=True,
                                     help="If off, we append only new, non-duplicate rows.")

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            cur_df = export_logs_df()

            def keyify(df, keyspec):
                cols = [c.strip() for c in keyspec.split(",")]
                for c in cols:
                    if c not in df.columns:
                        df[c] = ""
                return df.assign(_key=df[cols].astype(str).agg("|".join, axis=1))

            cur_df = keyify(cur_df, dedup_key)
            new_df = keyify(new_df, dedup_key)

            merged = pd.concat([cur_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["_key"]).drop(columns=["_key"])
            merged = merged.sort_values("timestamp", kind="stable")

            for c in LOG_COLUMNS:
                if c not in merged.columns:
                    merged[c] = ""

            st.session_state["logs"] = merged[LOG_COLUMNS].to_dict(orient="records")
            st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])

            if overwrite_disk:
                merged[LOG_COLUMNS] = merged.reindex(columns=LOG_COLUMNS, fill_value="")
                merged.to_csv(LOG_PATH, index=False, line_terminator="\n")
            else:
                on_disk = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=merged.columns)
                on_disk = keyify(on_disk, dedup_key)
                only_new = merged[~merged["_key"].isin(on_disk["_key"])] if not on_disk.empty else merged
                only_new.drop(columns=["_key"]).to_csv(LOG_PATH, mode="a", index=False, header=not LOG_PATH.exists())

            st.success(f"Merged {len(new_df)} rows; total history is now {len(st.session_state['logs'])} rows.")
        except Exception as e:
            st.error(f"Failed to import CSV: {e}")

    st.markdown("---")
    st.subheader("Migrate log schema (fix mixed old/new columns)")
    if st.button("Migrate logs to canonical schema"):
        try:
            n = migrate_log_schema(LOG_PATH)
            st.success(f"Migrated log to canonical schema. Rows written: {n}")
            df_new = pd.read_csv(LOG_PATH)
            for c in LOG_COLUMNS:
                if c not in df_new.columns:
                    df_new[c] = ""
            st.session_state["logs"] = df_new[LOG_COLUMNS].to_dict(orient="records")
            st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])
        except Exception as e:
            st.error(f"Migration failed: {e}")

    st.subheader("Repair log file (skip malformed rows and rewrite)")
    if st.button("Repair (re-write clean CSV from parsable rows)"):
        try:
            df_good = read_csv_tolerant(LOG_PATH)
            if df_good.empty:
                st.warning("No rows could be recovered to write.")
            else:
                for c in LOG_COLUMNS:
                    if c not in df_good.columns:
                        df_good[c] = ""
                df_good = df_good.reindex(columns=LOG_COLUMNS, fill_value="")
                df_good.to_csv(LOG_PATH, index=False, line_terminator="\n")
                st.success(f"Repaired and rewrote {LOG_PATH.name} with {len(df_good)} rows.")
                st.session_state["logs"] = df_good[LOG_COLUMNS].to_dict(orient="records")
                st.session_state["metrics"] = recompute_metrics_from_logs(st.session_state["logs"])
        except Exception as e:
            st.error(f"Repair failed: {e}")

    st.caption(f"Persistent log file: {LOG_PATH}")



# --------------------------
# CS2 IN-PLAY INDICATOR (MVP)
# --------------------------
def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))

def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# --------------------------
# CS2 map CT/T priors (coarse)
# Notes:
# - These are coarse priors for "side strength" on each map.
# - Values are CT win rates (0–1) assuming CT% + T% = 1.0.
# - Used only as an in-play bias term (optional).
CS2_MAP_CT_RATE = {
    "Ancient": 0.518,
    "Anubis": 0.446,
    "Cache": 0.486,
    "Cobblestone": 0.493,
    "Dust2": 0.490,
    "Inferno": 0.497,
    "Mirage": 0.528,
    "Nuke": 0.546,
    "Overpass": 0.535,
    "Season": 0.527,
    "Train": 0.548,
    "Tuscan": 0.499,
    "Vertigo": 0.491,
}
CS2_CT_RATE_AVG = float(np.mean(list(CS2_MAP_CT_RATE.values()))) if CS2_MAP_CT_RATE else 0.50



def estimate_inplay_prob(p0: float,
                         rounds_a: int,
                         rounds_b: int,
                         econ_a: float = 0.0,
                         econ_b: float = 0.0,
                         pistol_a: Optional[bool] = None,
                         pistol_b: Optional[bool] = None,
                         beta_score: float = 0.22,
                         beta_econ: float = 0.06,
                         beta_pistol: float = 0.35,
                         map_name: Optional[str] = None,
                         a_side: Optional[str] = None,
                         pistol_decay: float = 0.30,
                         beta_side: float = 0.85,
                         beta_lock: float = 0.90,
                         lock_start_offset: int = 3,
                         lock_ramp: int = 3,
                         win_target: int = 13) -> float:
    """
    In-play updater (indicator heuristic):

    - Starts at pre-match p0 (your fair map odds model).
    - Nudges log-odds by (score diff + econ diff + pistol) with:
        * stage scaling (early leads swing less than late leads)
        * pistol decay (EMA-like leak within the current half)
        * optional CT/T bias (map-aware)
        * a late-game nonlinearity (approaches 0/1 faster near match point)
        * supports overtime by letting caller pass win_target (13 reg, 16/19/... OT)
    """
    rounds_a = int(rounds_a)
    rounds_b = int(rounds_b)
    score_diff = rounds_a - rounds_b
    rp = max(0, rounds_a + rounds_b)

    # Coarse MR12 half progress (0-11). Used for pistol decay and side expectation.
    rounds_in_half = rp % 12

    # Economy diff: assumes inputs are consistent proxies; scaled to "per 1k"
    econ_diff_k = (float(econ_a) - float(econ_b)) / 1000.0

    # Stage scaling: early leads should not swing as hard as late leads.
    stage_scale = 0.65 + 0.35 * min(1.0, rp / 12.0)

    # Side-aware score expectation (map-aware, optional):
    # On CT-sided maps, being tied while on CT is *worse* than being tied while on T.
    score_diff_eff = float(score_diff)
    side = str(a_side).upper().strip() if a_side is not None else ""
    if side in ("CT", "T"):
        ct_rate = CS2_CT_RATE_AVG
        if map_name and str(map_name) in CS2_MAP_CT_RATE:
            ct_rate = float(CS2_MAP_CT_RATE.get(str(map_name), CS2_CT_RATE_AVG))
        ct_rate = float(np.clip(ct_rate, 0.05, 0.95))
        ct_adv = float(2.0 * ct_rate - 1.0)  # -1..+1 (usually small)

        expected_diff_so_far = float(rounds_in_half) * (ct_adv if side == "CT" else -ct_adv)
        score_diff_eff = float(score_diff) - expected_diff_so_far

    x = _logit(p0) + (beta_score * stage_scale) * score_diff_eff + beta_econ * econ_diff_k

    # Pistol impact decays away as the half progresses (EMA-like leak toward 0).
    if pistol_a is True or pistol_b is True:
        decay = float(np.exp(-float(pistol_decay) * float(rounds_in_half)))
        if pistol_a is True:
            x += beta_pistol * decay
        if pistol_b is True:
            x -= beta_pistol * decay

    # Future side bias (small) that fades as the half runs out.
    if side in ("CT", "T"):
        ct_rate = CS2_CT_RATE_AVG
        if map_name and str(map_name) in CS2_MAP_CT_RATE:
            ct_rate = float(CS2_MAP_CT_RATE.get(str(map_name), CS2_CT_RATE_AVG))
        ct_rate = float(np.clip(ct_rate, 0.05, 0.95))
        side_logit = float(np.log(ct_rate / (1.0 - ct_rate)))

        half_remaining = max(0, 12 - rounds_in_half)
        side_scale = float(np.clip(half_remaining / 12.0, 0.0, 1.0))

        if side == "CT":
            x += beta_side * side_logit * side_scale
        else:
            x -= beta_side * side_logit * side_scale

    # Late-game nonlinearity ("match point snap").
    lead_sign = 1 if score_diff > 0 else (-1 if score_diff < 0 else 0)
    if lead_sign != 0:
        leading_rounds = rounds_a if lead_sign > 0 else rounds_b
        # Late-game lock: dampen sensitivity once a team is near closing the map.
        # Defaults preserve prior behavior: starts at (win_target-3) and ramps over 3 rounds.
        try:
            lso = int(lock_start_offset)
        except Exception:
            lso = 3
        try:
            lr = max(1, int(lock_ramp))
        except Exception:
            lr = 3
        start_at = int(win_target) - lso
        closeness = (float(leading_rounds) - float(start_at)) / float(lr)
        closeness = float(np.clip(closeness, 0.0, 1.0))
        x += beta_lock * (closeness ** 2) * float(lead_sign)

    return _sigmoid(x)



def cs2_current_win_target(rounds_a: int,
                           rounds_b: int,
                           regulation_target: int = 13,
                           ot_block: int = 3) -> int:
    """Return the current win target for CS2 map given score.
    - Regulation: first to 13.
    - Overtime (approx): if 12-12 or beyond, targets are 16, 19, 22, ... (MR3 blocks).
    """
    ra = int(rounds_a)
    rb = int(rounds_b)
    if min(ra, rb) < 12:
        return int(regulation_target)
    sets_completed = max(0, (min(ra, rb) - 12) // int(ot_block))
    return int(regulation_target + int(ot_block) * (sets_completed + 1))


def cs2_soft_lock_map_prob(p_map: float,
                           rounds_a: int,
                           rounds_b: int,
                           win_target: int) -> float:
    """Blend p_map with a near-end state-lock so map→series doesn't cliff
    when you mark the map complete.

    Blends toward the probability implied by the *score state alone* under a neutral
    per-round assumption (q=0.5), only within ~3 rounds of winning.
    """
    try:
        p_map = float(p_map)
    except Exception:
        p_map = 0.5
    p_map = float(np.clip(p_map, 1e-6, 1.0 - 1e-6))

    ra = int(rounds_a)
    rb = int(rounds_b)
    lead = max(ra, rb)
    closeness = (float(lead) - float(win_target - 3)) / 3.0
    closeness = float(np.clip(closeness, 0.0, 1.0))
    if closeness <= 0.0:
        return p_map

    wa = max(0, int(win_target) - ra)
    wb = max(0, int(win_target) - rb)

    if wa <= 0:
        p_state = 1.0
    elif wb <= 0:
        p_state = 0.0
    else:
        p_state = series_prob_needed(int(wa), int(wb), 0.5)

    alpha = float(closeness ** 2)
    return float((1.0 - alpha) * p_map + alpha * float(p_state))






# --------------------------
# Valorant IN-PLAY (MVP) helpers

# --------------------------
# Kappa (K) confidence band helpers (ported from app22)
# --------------------------
def beta_credible_interval(p: float, kappa: float, level: float = 0.80) -> tuple[float, float]:
    """Return a Beta credible interval for probability p using concentration kappa.

    Intuition:
      - kappa = "how sure" we are about p. Higher = tighter bands.
      - level = interval mass (0.80, 0.90, 0.95, ...)

    This is deliberately pragmatic (indicator bands), not a claim of calibrated posterior certainty.
    """
    try:
        p = float(p)
    except Exception:
        p = 0.5
    p = float(np.clip(p, 1e-4, 1.0 - 1e-4))

    try:
        k = float(kappa)
    except Exception:
        k = 20.0
    k = float(max(2.0, k))

    try:
        lvl = float(level)
    except Exception:
        lvl = 0.80
    lvl = float(np.clip(lvl, 0.50, 0.99))

    a = p * k
    b = (1.0 - p) * k
    tail = (1.0 - lvl) / 2.0
    lo = float(beta_dist.ppf(tail, a, b))
    hi = float(beta_dist.ppf(1.0 - tail, a, b))

    if not np.isfinite(lo) or not np.isfinite(hi):
        # emergency fallback (should be rare)
        lo = max(0.01, p - 0.20)
        hi = min(0.99, p + 0.20)

    lo = float(np.clip(lo, 0.01, 0.99))
    hi = float(np.clip(hi, 0.01, 0.99))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi



def update_round_stream(prefix: str, rounds_a: int, rounds_b: int) -> dict:
    """Infer streak/reversal/gaps from score updates (no per-round clicking).

    Works like this:
      - If the score increases by exactly 1 total round since last run, we infer who won the round.
      - If it jumps by >1, we mark a 'gap' (you probably missed rounds / updated late).
      - If the score goes backwards (manual correction), we reset tracking.

    Returns a dict with:
      delta_total, gap_delta, streak_len, streak_winner ('A'/'B'/None), reversal, n_tracked
    """
    ra = int(rounds_a)
    rb = int(rounds_b)

    key_prev = f"{prefix}_prev_score"
    key_wins = f"{prefix}_round_winners"

    prev = st.session_state.get(key_prev, (0, 0))
    wins = list(st.session_state.get(key_wins, []))

    prev_a, prev_b = int(prev[0]), int(prev[1])

    # If user edited the score backwards, reset the derived stream.
    if (ra < prev_a) or (rb < prev_b) or (ra == 0 and rb == 0 and (prev_a != 0 or prev_b != 0)):
        prev_a, prev_b = 0, 0
        wins = []

    delta_total = (ra - prev_a) + (rb - prev_b)
    gap_delta = 0
    winner = None
    reversal = False

    if delta_total == 1:
        if ra == prev_a + 1 and rb == prev_b:
            winner = "A"
        elif rb == prev_b + 1 and ra == prev_a:
            winner = "B"

        if winner is not None:
            prev_last = wins[-1] if wins else None
            wins.append(winner)
            reversal = (prev_last is not None and winner != prev_last)
        else:
            # weird inconsistent single-step update; treat as gap
            gap_delta = 1

    elif delta_total > 1:
        gap_delta = int(delta_total)

    # Update session state
    st.session_state[key_prev] = (ra, rb)
    st.session_state[key_wins] = wins

    # Current streak
    streak_len = 0
    streak_winner = None
    if wins:
        streak_winner = wins[-1]
        streak_len = 1
        for w in reversed(wins[:-1]):
            if w == streak_winner:
                streak_len += 1
            else:
                break

    return {
        "delta_total": int(delta_total),
        "gap_delta": int(gap_delta),
        "streak_len": int(streak_len),
        "streak_winner": streak_winner,
        "reversal": bool(reversal),
        "n_tracked": int(len(wins)),
    }



def compute_kappa_cs2(p0: float,
                     rounds_a: int,
                     rounds_b: int,
                     econ_missing: bool = False,
                     econ_fragile: bool = False,
                     pistol_a: bool = False,
                     pistol_b: bool = False,
                     streak_len: int = 0,
                     streak_winner: Optional[str] = None,
                     reversal: bool = False,
                     gap_delta: int = 0,
                     chaos_boost: float = 0.0,
                     total_rounds: int = 24) -> float:
    """Heuristic concentration parameter K for CS2 bands.

    Higher K = tighter credible interval around p_hat.
    We tighten when:
      - match progresses and lock approaches,
      - score dominance and sustained streaks add information.
    We widen when:
      - context is missing (unknown buys),
      - fragile buy states,
      - score-update gaps (missed rounds),
      - reversal after a streak,
      - manual chaos widen.
    """
    try:
        p0 = float(p0)
    except Exception:
        p0 = 0.5
    p0 = float(np.clip(p0, 1e-4, 1.0 - 1e-4))

    ra = int(rounds_a)
    rb = int(rounds_b)
    rp = ra + rb
    tr = max(1, int(total_rounds))

    conf0 = min(1.0, abs(p0 - 0.5) * 2.0)  # 0 coinflip -> 1 heavy favorite
    k = 18.0 + 70.0 * conf0  # 18..88 baseline

    # Pre-match: we should be reasonably tight, not absurdly wide.
    if rp <= 0:
        k += 20.0 * conf0
        k *= max(0.55, 1.0 - 1.1 * float(chaos_boost))
        return float(np.clip(k, 12.0, 160.0))

    # Progress: slowly tighten as info accrues.
    frac = min(1.0, float(rp) / float(tr))
    k += 45.0 * (frac ** 1.25)

    # Lock: tighten hard when a team is near 13.
    lead = max(ra, rb)
    lock = float(np.clip((lead - 6) / 7.0, 0.0, 1.0))  # starts contributing after round ~6
    k += 70.0 * (lock ** 2.0)

    # Dominance: score diff scaled by how many rounds we actually observed.
    dom = abs(ra - rb) / max(1.0, float(rp))
    k += 35.0 * dom

    # Streak information: sustained run reduces uncertainty… until it snaps.
    try:
        sl = int(streak_len)
    except Exception:
        sl = 0
    if sl >= 3:
        k += min(30.0, 6.0 * float(sl - 2))
        # slightly more confident if the streak belongs to the current leader
        if (ra > rb and streak_winner == "A") or (rb > ra and streak_winner == "B"):
            k += 8.0

    if bool(reversal):
        k *= 0.85  # reversal = volatility spike

    # Pistol effect (small): favored pistol/conversion usually stabilizes.
    fav_is_a = (p0 >= 0.5)
    if pistol_a or pistol_b:
        if fav_is_a:
            if pistol_a:
                k += 8.0
            if pistol_b:
                k *= 0.92
        else:
            if pistol_b:
                k += 8.0
            if pistol_a:
                k *= 0.92

    # Missing/fragile econ -> widen.
    if bool(econ_missing):
        k *= 0.78
    if bool(econ_fragile):
        k *= 0.88

    # If you skipped rounds (score jumped), treat as less reliable context.
    gd = int(gap_delta or 0)
    if gd > 1:
        k *= 0.70
        k *= max(0.55, 1.0 - 0.06 * float(gd - 1))

    # Manual chaos widen
    k *= max(0.50, 1.0 - 1.1 * float(chaos_boost))

    return float(np.clip(k, 8.0, 240.0))



def compute_kappa_valorant(p0: float,
                          rounds_a: int,
                          rounds_b: int,
                          eco_a_level: str = "Light",
                          eco_b_level: str = "Light",
                          ults_diff: int = 0,
                          op_diff: int = 0,
                          a_on_defense: Optional[bool] = None,
                          limited_context: bool = False,
                          pistol_a: bool = False,
                          pistol_b: bool = False,
                          streak_len: int = 0,
                          streak_winner: Optional[str] = None,
                          reversal: bool = False,
                          gap_delta: int = 0,
                          chaos_boost: float = 0.0,
                          total_rounds: int = 24) -> float:
    """Heuristic concentration K for Valorant bands (same philosophy as CS2).

    Adds two Valorant-specific wideners:
      - large ult/OP imbalances can cause sharp swings (wider interval),
      - unknown side / limited context widens to avoid fake certainty.
    """
    try:
        p0 = float(p0)
    except Exception:
        p0 = 0.5
    p0 = float(np.clip(p0, 1e-4, 1.0 - 1e-4))

    ra = int(rounds_a)
    rb = int(rounds_b)
    rp = ra + rb
    tr = max(int(total_rounds), rp + 6)  # allow some slack for OT

    conf0 = min(1.0, abs(p0 - 0.5) * 2.0)
    k = 18.0 + 70.0 * conf0

    if rp <= 0:
        k += 15.0 * conf0
        if bool(limited_context):
            k *= 0.80
        if a_on_defense is None:
            k *= 0.90
        k *= max(0.55, 1.0 - 1.1 * float(chaos_boost))
        return float(np.clip(k, 12.0, 160.0))

    frac = min(1.0, float(rp) / float(tr))
    k += 42.0 * (frac ** 1.20)

    lead = max(ra, rb)
    lock = float(np.clip((lead - 6) / 7.0, 0.0, 1.0))
    k += 65.0 * (lock ** 2.0)

    dom = abs(ra - rb) / max(1.0, float(rp))
    k += 32.0 * dom

    sl = int(streak_len or 0)
    if sl >= 3:
        k += min(28.0, 5.5 * float(sl - 2))
        if (ra > rb and streak_winner == "A") or (rb > ra and streak_winner == "B"):
            k += 7.0

    if bool(reversal):
        k *= 0.86

    # Big ult/OP diff can create near-term swing risk (widen a bit).
    try:
        ud = abs(int(ults_diff))
    except Exception:
        ud = 0
    if ud >= 4:
        k *= 0.90

    try:
        od = abs(int(op_diff))
    except Exception:
        od = 0
    if od >= 2:
        k *= 0.92

    # Coarse econ mismatch / weirdness -> widen slightly (it's not a fitted model).
    if str(eco_a_level) == "Eco" or str(eco_b_level) == "Eco":
        k *= 0.94

    if bool(limited_context):
        k *= 0.78
    if a_on_defense is None:
        k *= 0.90

    # Pistol effect small
    fav_is_a = (p0 >= 0.5)
    if pistol_a or pistol_b:
        if fav_is_a:
            if pistol_a: k += 6.0
            if pistol_b: k *= 0.94
        else:
            if pistol_b: k += 6.0
            if pistol_a: k *= 0.94

    gd = int(gap_delta or 0)
    if gd > 1:
        k *= 0.72
        k *= max(0.55, 1.0 - 0.06 * float(gd - 1))

    k *= max(0.50, 1.0 - 1.1 * float(chaos_boost))

    return float(np.clip(k, 8.0, 240.0))


# --------------------------
# NEW — Series/Map probability helpers for in-play indicator
# --------------------------

# --------------------------
_VAL_ECO_MAP = {
    "Eco": -1.0,
    "Light": 0.0,
    "Full": 1.0,
}

def _eco_to_pseudo_cash(eco_level: str) -> float:
    """Map coarse Valorant buy states to pseudo 'team economy' dollars so we can reuse sigma logic."""
    lvl = str(eco_level or "Light")
    if lvl == "Full":
        return 9000.0
    if lvl == "Eco":
        return 3000.0
    return 6000.0  # Light

def estimate_inplay_prob_valorant(p0: float,
                                 rounds_a: int,
                                 rounds_b: int,
                                 eco_a_level: str = "Light",
                                 eco_b_level: str = "Light",
                                 pistol_a: Optional[bool] = None,
                                 pistol_b: Optional[bool] = None,
                                 ults_diff: int = 0,
                                 op_diff: int = 0,
                                 a_on_defense: Optional[bool] = None,
                                 beta_score: float = 0.18,
                                 beta_eco: float = 0.20,
                                 beta_pistol: float = 0.28,
                                 beta_ults: float = 0.06,
                                 beta_op: float = 0.08,
                                 beta_side: float = 0.06,
                                 beta_lock: float = 0.90,
                                 lock_start_offset: int = 3,
                                 lock_ramp: int = 3,
                                 win_target: int = 13) -> float:
    """Valorant in-play updater (MVP).

    - Keeps the same structure as CS2 (prior -> live fair -> compare to market).
    - Adds two high-impact Valorant-specific drivers you can track quickly:
        * Ult advantage (online ult count difference)
        * Operator presence (op count difference)
    - Economy is modeled as *coarse buy state* (Eco/Light/Full), not exact credits.

    This is a heuristic skeleton, not a fitted model.
    """
    score_diff = int(rounds_a) - int(rounds_b)
    eco_a = _VAL_ECO_MAP.get(str(eco_a_level), 0.0)
    eco_b = _VAL_ECO_MAP.get(str(eco_b_level), 0.0)
    eco_diff = eco_a - eco_b

    # Normalize score impact by stage so early leads don't swing too hard.
    rp = max(1, int(rounds_a) + int(rounds_b))
    stage_scale = 0.65 + 0.35 * min(1.0, rp / 12.0)  # ramps into midgame
    x = _logit(p0) + (beta_score * stage_scale) * score_diff + beta_eco * eco_diff

    if pistol_a is True:
        x += beta_pistol
    if pistol_b is True:
        x -= beta_pistol

    try:
        ud = int(ults_diff)
    except Exception:
        ud = 0
    x += beta_ults * float(ud)

    try:
        od = int(op_diff)
    except Exception:
        od = 0
    x += beta_op * float(od)

    # Optional coarse side bias.
    if a_on_defense is True:
        x += beta_side
    elif a_on_defense is False:
        x -= beta_side


    # Late-game lock (match point snap): once a team is near closing (e.g., 12+),
    # reduce spurious swings from small score/econ changes.
    lead_sign = 1 if score_diff > 0 else (-1 if score_diff < 0 else 0)
    if lead_sign != 0:
        leading_rounds = rounds_a if lead_sign > 0 else rounds_b
        try:
            lso = int(lock_start_offset)
        except Exception:
            lso = 3
        try:
            lr = max(1, int(lock_ramp))
        except Exception:
            lr = 3
        start_at = int(win_target) - lso
        closeness = (float(leading_rounds) - float(start_at)) / float(lr)
        closeness = float(np.clip(closeness, 0.0, 1.0))
        x += beta_lock * (closeness ** 2) * float(lead_sign)

    return _sigmoid(x)

def estimate_sigma_valorant(p0: float,
                           rounds_a: int,
                           rounds_b: int,
                           eco_a_level: str = "Light",
                           eco_b_level: str = "Light",
                           chaos_boost: float = 0.0,
                           missing_context_widen: float = 0.0,
                           total_rounds: int = 24) -> float:
    """Certainty band width for Valorant (MVP).

    Reuses the CS2 sigma shape (time decay + lock tightening + econ instability),
    but feeds in pseudo-cash derived from buy-state, and allows an explicit
    missing_context_widen so you don't get false precision when key info is missing.
    """
    econ_a = _eco_to_pseudo_cash(eco_a_level)
    econ_b = _eco_to_pseudo_cash(eco_b_level)
    return estimate_sigma(
        p0=float(p0),
        rounds_a=int(rounds_a),
        rounds_b=int(rounds_b),
        econ_a=float(econ_a),
        econ_b=float(econ_b),
        chaos_boost=float(chaos_boost) + float(missing_context_widen),
        total_rounds=int(total_rounds),
    )

def estimate_sigma(p0: float,
                   rounds_a: int,
                   rounds_b: int,
                   econ_a: float = 0.0,
                   econ_b: float = 0.0,
                   econ_missing: bool = False,
                   chaos_boost: float = 0.0,
                   total_rounds: int = 24,
                   base: float = 0.22,
                   min_sigma: float = 0.02,
                   max_sigma: float = 0.30) -> float:
    """
    'Certainty band' width (probability-space).

    Goals:
      - Wide early; narrows as the map approaches completion (fewer rounds left).
      - Narrows further when the match state becomes 'locked' (leader close to 13).
      - Widens during chaotic/low-econ phases; tightens in stable full-buy phases.
      - Slightly tighter for strong pre-match favorites/dogs (p0 far from 0.5).
      - Optional manual chaos_boost to widen instantly (timeouts, stand-ins, etc).

    Notes:
      - Uses only inputs you can reasonably supply during buy time.
      - Designed for MR12 (first to 13) => max 24 rounds by default.
    """
    ra = max(int(rounds_a), 0)
    rb = max(int(rounds_b), 0)
    rp = ra + rb

    # clamp total rounds (MR12 = 24). If it goes past, treat as "done".
    tr = max(1, int(total_rounds))
    frac_remaining = max(0.0, float(tr - rp) / float(tr))  # 1.0 early -> 0.0 late

    # Pre-match confidence 0..1 (how far from coinflip)
    conf0 = min(1.0, abs(float(p0) - 0.5) * 2.0)

    # Core time decay: shrink as the map nears completion
    # sqrt keeps it from collapsing too fast early, but tight late.
    time_term = np.sqrt(frac_remaining)

    # "Lock" term: if leader is close to winning, tighten bands.
    lead_score = max(ra, rb)
    rounds_to_win = max(0, 13 - lead_score)  # 0..13
    lock_term = 0.35 + 0.65 * (rounds_to_win / 13.0)  # near win => ~0.35, early => ~1.0

    # Economy stability: tight when BOTH teams can full-buy; wide when either is broke.
    # Heuristic: >=9k = stable, <=4k = unstable.
    econ_min = min(float(econ_a), float(econ_b))
    econ_stability = (econ_min - 4000.0) / 5000.0  # 4k->0, 9k->1
    econ_stability = float(min(1.0, max(0.0, econ_stability)))
    econ_unc = 0.10 * (1.0 - econ_stability) * (0.55 + 0.45 * time_term)


    # If economy inputs are missing/unknown, widen bands (avoid false precision).
    missing_unc = (0.06 * (0.55 + 0.45 * time_term)) if bool(econ_missing) else 0.0
    # Base sigma shrinks with time and a bit with pre-match confidence
    sigma = (base * time_term * (1.0 - 0.30 * conf0)) * lock_term + econ_unc + float(missing_unc) + float(chaos_boost)

    return float(min(max_sigma, max(min_sigma, sigma)))

# --------------------------
# NEW — Series/Map probability helpers for in-play indicator
# --------------------------
def _bestof_target(n_maps: int) -> int:
    return int(n_maps // 2 + 1)

def series_prob_needed(wins_needed: int, losses_allowed_plus1: int, p: float) -> float:
    """Probability to reach 'wins_needed' wins before reaching 'losses_allowed_plus1' losses
    in independent Bernoulli trials with win prob p.
    losses_allowed_plus1 = losses_needed (i.e., max losses before losing series) + 1.
    For best-of-(2t-1): wins_needed=t, losses_allowed_plus1=t.
    """
    p = float(min(1-1e-9, max(1e-9, p)))
    w = int(max(0, wins_needed))
    l_lim = int(max(0, losses_allowed_plus1))
    if w == 0:
        return 1.0
    if l_lim == 0:
        return 0.0
    # Sum over k losses (0..l_lim-1) before the final winning map
    # C(w+k-1, k) * p^w * (1-p)^k
    q = 1.0 - p
    out = 0.0
    for k in range(0, l_lim):
        out += math.comb(w + k - 1, k) * (p ** w) * (q ** k)
    return float(min(1.0, max(0.0, out)))

def series_win_prob_live(n_maps: int,
                         maps_a_won: int,
                         maps_b_won: int,
                         p_current_map: float,
                         p_future_map: float) -> float:
    """Series win probability for Team A in a best-of-n_maps series, given current series score
    (maps_a_won, maps_b_won) and probability Team A wins the *current* map (p_current_map).
    Future maps after the current one are assumed i.i.d with win prob p_future_map.
    """
    n = int(n_maps)
    target = _bestof_target(n)
    a = int(max(0, maps_a_won))
    b = int(max(0, maps_b_won))
    # If series already decided
    if a >= target:
        return 1.0
    if b >= target:
        return 0.0

    ra = target - a  # wins still needed for A including current map
    rb = target - b  # wins still needed for B including current map
    pc = float(min(1-1e-9, max(1e-9, p_current_map)))
    pf = float(min(1-1e-9, max(1e-9, p_future_map)))

    # If A wins current map -> needs (ra-1) more wins before rb losses
    win_branch = series_prob_needed(max(0, ra - 1), rb, pf)
    # If A loses current map -> needs ra wins before (rb-1) losses
    lose_branch = series_prob_needed(ra, max(0, rb - 1), pf)

    return float(pc * win_branch + (1.0 - pc) * lose_branch)

def invert_series_to_map_prob(p_series: float, n_maps: int) -> float:
    """Given a pre-match series win probability p_series for best-of-n_maps,
    infer the implied per-map win probability p (assuming i.i.d across maps).
    Uses monotone bisection on [0,1]."""
    n = int(n_maps)
    if n <= 1:
        return float(p_series)
    target = _bestof_target(n)
    ps = float(min(0.999999, max(0.000001, p_series)))

    def f(p):
        return series_prob_needed(target, target, p) - ps

    lo, hi = 1e-6, 1 - 1e-6
    flo, fhi = f(lo), f(hi)
    # Should bracket, but be defensive
    if flo >= 0:
        return float(lo)
    if fhi <= 0:
        return float(hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))

with tabs[3]:
    st.header("CS2 In-Play Indicator (MVP)")
    st.caption("Fair probability line + certainty bands. Supports BO1 / BO3 / BO5 and map-vs-series markets. MVP = manual match inputs + manual/auto market bid-ask.")

    st.markdown("### K-bands + calibration logging (always visible)")
    l1, l2, l3, l4 = st.columns([1.35, 1.10, 1.05, 1.20])
    with l1:
        cs2_inplay_match_id = st.text_input("Match ID", key="cs2_inplay_match_id")
        if "cs2_inplay_map_index" not in st.session_state:
            st.session_state["cs2_inplay_map_index"] = 0
            st.session_state["cs2_inplay_last_total_rounds"] = None
    with l2:
        cs2_inplay_persist = st.checkbox("Persist snapshots + results", value=bool(st.session_state.get("cs2_inplay_persist", False)), key="cs2_inplay_persist")
    with l3:
        cs2_band_level = st.selectbox("Band level", options=[0.68, 0.75, 0.80, 0.90, 0.95], index=2,
                                     format_func=lambda x: f"{int(x*100)}%", key="cs2_inplay_band_level")
    with l4:
        cs2_k_scale = st.slider("K scale", 0.50, 2.00, float(st.session_state.get("cs2_inplay_k_scale", 1.00)), 0.05, key="cs2_inplay_k_scale")
    cs2_inplay_notes = st.text_input("Notes (optional)", key="cs2_inplay_notes")
    show_inplay_log_paths()

    # --- K calibration controls (optional) ---
    calib = load_kappa_calibration()
    ccal1, ccal2, ccal3 = st.columns([1.2, 1.0, 1.8])
    with ccal1:
        cs2_use_calib = st.checkbox("Use calibrated K multiplier", value=bool(st.session_state.get("cs2_use_calib", False)), key="cs2_use_calib")
    with ccal2:
        if st.button("Train calibration (clean logs)", key="cs2_train_calib_btn"):
            ok, out = run_kappa_trainer()
            if ok:
                st.success("Calibration trained.")
            else:
                st.error("Calibration training failed.")
            st.session_state["kappa_train_last_out"] = out
    with ccal3:
        if st.session_state.get("kappa_train_last_out"):
            st.caption(str(st.session_state.get("kappa_train_last_out"))[:240] + ("…" if len(str(st.session_state.get("kappa_train_last_out"))) > 240 else ""))
        elif calib:
            st.caption(f"Calibration loaded: {KAPPA_CALIB_PATH.name}")
        else:
            st.caption("No calibration loaded yet.")


    # Team names (typed input so logs reflect exactly what you enter)
    # (We still load the teams list in case you want to use it elsewhere, but we don't force a dropdown here.)
    df_cs2_live = load_cs2_teams()

    colA, colB = st.columns(2)
    with colA:
        st.session_state.setdefault("cs2_live_team_a", st.session_state.get("cs2_live_team_a", ""))
        team_a = st.text_input("Team A", value=st.session_state.get("cs2_live_team_a", ""), key="cs2_live_team_a")
    with colB:
        st.session_state.setdefault("cs2_live_team_b", st.session_state.get("cs2_live_team_b", ""))
        team_b = st.text_input("Team B", value=st.session_state.get("cs2_live_team_b", ""), key="cs2_live_team_b")

    # Back-compat aliases (some downstream code expects these names)
    team_a_live = team_a
    team_b_live = team_b

    st.markdown("### Context — series format and what the contract represents")
    cctx1, cctx2, cctx3, cctx4 = st.columns([1.0, 1.5, 1.0, 1.0])
    with cctx1:
        series_fmt = st.selectbox("Series format", ["BO1", "BO3", "BO5"], index=int(st.session_state.get("cs2_live_series_fmt_idx", 1)))
    with cctx2:
        contract_scope = st.selectbox("Contract priced on", ["Map winner (this map)", "Series winner"], index=int(st.session_state.get("cs2_live_contract_scope_idx", 0)))
    with cctx3:
        st.session_state.setdefault("cs2_live_maps_a_won", int(st.session_state.get("cs2_live_maps_a_won", 0)))
        maps_a_won = st.number_input("Maps A won", min_value=0, max_value=4, step=1, key="cs2_live_maps_a_won")
    with cctx4:
        st.session_state.setdefault("cs2_live_maps_b_won", int(st.session_state.get("cs2_live_maps_b_won", 0)))
        maps_b_won = st.number_input("Maps B won", min_value=0, max_value=4, step=1, key="cs2_live_maps_b_won")
        st.session_state.setdefault("cs2_prev_maps_a_won", int(st.session_state.get("cs2_live_maps_a_won", 0)))
        st.session_state.setdefault("cs2_prev_maps_b_won", int(st.session_state.get("cs2_live_maps_b_won", 0)))

    st.session_state["cs2_live_series_fmt_idx"] = ["BO1","BO3","BO5"].index(series_fmt)
    st.session_state["cs2_live_contract_scope_idx"] = ["Map winner (this map)","Series winner"].index(contract_scope)

    n_maps = int(series_fmt.replace("BO",""))
    # Hide/ignore map-score inputs when they don't apply
    if contract_scope == "Map winner (this map)" or n_maps == 1:
        maps_a_won = 0
        maps_b_won = 0

    if contract_scope == "Series winner" and n_maps > 1:
        current_map_num = int(maps_a_won) + int(maps_b_won) + 1
        st.caption(f"Series score: A {int(maps_a_won)} – {int(maps_b_won)} B (currently Map {current_map_num} of up to {n_maps}).")

    st.markdown("### Step 1 — Pre-match fair probability (from your model)")
    if contract_scope == "Series winner" and n_maps > 1:
        st.caption("Paste your **pre-match series win%** for Team A here (0–1). We'll infer the implied per-map prior for the in-map updater.")
        st.session_state.setdefault("cs2_live_p0_series", float(st.session_state.get("cs2_live_p0_series", 0.55)))
        p0_series = st.number_input("Pre-match fair series win% for Team A (0–1)", min_value=0.01, max_value=0.99,
                                    step=0.01, format="%.2f", key="cs2_live_p0_series")

        p0_map = invert_series_to_map_prob(float(p0_series), int(n_maps))
        st.info(f"Implied per-map prior from series (i.i.d approx): p_map ≈ {p0_map*100:.1f}%")
    else:
        st.caption("Paste your **pre-match map win%** for Team A here (0–1).")
        st.session_state.setdefault("cs2_live_p0_map", float(st.session_state.get("cs2_live_p0_map", 0.60)))
        p0_map = st.number_input("Pre-match fair win% for Team A (0–1)", min_value=0.01, max_value=0.99,
                                 step=0.01, format="%.2f", key="cs2_live_p0_map")
        p0_series = float(p0_map)  # placeholder for display

    st.markdown("### Step 2 — Live map inputs (update whenever you want)")
    if "cs2_live_rows" not in st.session_state:
        st.session_state["cs2_live_rows"] = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.session_state.setdefault("cs2_live_rounds_a", int(st.session_state.get("cs2_live_rounds_a", 0)))
        rounds_a = st.number_input("Rounds A", min_value=0, max_value=30, step=1, key="cs2_live_rounds_a")
    with c2:
        st.session_state.setdefault("cs2_live_rounds_b", int(st.session_state.get("cs2_live_rounds_b", 0)))
        rounds_b = st.number_input("Rounds B", min_value=0, max_value=30, step=1, key="cs2_live_rounds_b")



    # Optional: map/side context for CT/T bias (skip if unknown)
    MAP_OPTIONS = ["Average (no map)"] + list(CS2_MAP_CT_RATE.keys())
    cM1, cM2 = st.columns([1.3, 1.0])
    with cM1:
        st.session_state.setdefault("cs2_live_map_name", st.session_state.get("cs2_live_map_name", "Average (no map)"))
        map_sel = st.selectbox("Map (for CT/T bias)", MAP_OPTIONS, key="cs2_live_map_name")
    with cM2:
        st.session_state.setdefault("cs2_live_a_side", st.session_state.get("cs2_live_a_side", "Unknown"))
        a_side_sel = st.radio("Team A side now", options=["Unknown", "CT", "T"], horizontal=True, key="cs2_live_a_side")
    map_name = None if str(map_sel) == "Average (no map)" else str(map_sel)
    a_side = None if str(a_side_sel) == "Unknown" else str(a_side_sel)


    # Faster economy input (buy state) — avoids typing exact $ totals during buy time.
    BUY_STATES = ["Skip/Unknown", "Eco", "Light", "Full (fragile)", "Full", "Full+Save"]
    BUY_STATE_TO_ECON = {
        # Rough team-total proxies (MR12). Tune later if you want.
        "Skip/Unknown": 0.0,
        "Eco": 3500.0,
        "Light": 6500.0,
        "Full (fragile)": 9000.0,
        "Full": 10000.0,
        "Full+Save": 12000.0,
    }
    if "cs2_live_buy_a" not in st.session_state:
        st.session_state["cs2_live_buy_a"] = "Skip/Unknown"
    if "cs2_live_buy_b" not in st.session_state:
        st.session_state["cs2_live_buy_b"] = "Skip/Unknown"

    with c3:
        buy_a = st.selectbox("Team A buy state", BUY_STATES, key="cs2_live_buy_a")
    with c4:
        buy_b = st.selectbox("Team B buy state", BUY_STATES, key="cs2_live_buy_b")

    econ_a = float(BUY_STATE_TO_ECON.get(str(buy_a), 0.0))
    econ_b = float(BUY_STATE_TO_ECON.get(str(buy_b), 0.0))
    econ_missing = (str(buy_a) == "Skip/Unknown") or (str(buy_b) == "Skip/Unknown")
    econ_fragile = (str(buy_a) == "Full (fragile)") or (str(buy_b) == "Full (fragile)")

    # Keep numeric econ in session_state for downstream calcs / exports.
    st.session_state["cs2_live_econ_a"] = int(econ_a)
    st.session_state["cs2_live_econ_b"] = int(econ_b)

    st.caption(f"Econ proxy (from buy states): A=${econ_a:,.0f}, B=${econ_b:,.0f} — set Skip/Unknown to widen bands and avoid false precision.")

    colP1, colP2, colChaos = st.columns(3)
    with colP1:
        pistol_a = st.checkbox("A won most recent pistol?", value=bool(st.session_state.get("cs2_live_pistol_a", False)), key="cs2_live_pistol_a")
    with colP2:
        pistol_b = st.checkbox("B won most recent pistol?", value=bool(st.session_state.get("cs2_live_pistol_b", False)), key="cs2_live_pistol_b")
    with colChaos:
        chaos_boost = st.slider("Chaos widen (manual)", 0.00, 0.25, float(st.session_state.get("cs2_live_chaos", 0.00)), 0.01)
    st.session_state["cs2_live_chaos"] = float(chaos_boost)

    st.markdown("### Step 3 — Market bid/ask input")
    if contract_scope == "Series winner" and n_maps > 1:
        st.caption("Enter executable prices for **Team A to win the SERIES** (bid = sell YES, ask = buy YES).")
    else:
        st.caption("Enter executable prices for **Team A to win THIS MAP** (bid = sell YES, ask = buy YES).")

    st.markdown("#### Optional — Fetch bid/ask from an exchange API")

    # Persisted Kalshi URL + selected team market (so you paste/select once, then just refresh).
    colF1, colF2, colF3 = st.columns([1.05, 2.35, 1.10])
    with colF1:
        venue = st.selectbox("Venue", ["Manual", "Kalshi", "Polymarket"], index=0, key="cs2_mkt_fetch_venue")

    # NOTE: We deliberately keep Kalshi's "URL -> choose team market" flow separate from the generic ident field,
    # because Streamlit does not allow us to mutate a widget's key after it is instantiated.
    if venue == "Kalshi":
        with colF2:
            st.session_state.setdefault("cs2_kalshi_url", st.session_state.get("cs2_kalshi_url", ""))
            kalshi_url = st.text_input("Kalshi URL (paste once; event/game page)", key="cs2_kalshi_url")
        with colF3:
            load_markets = st.button("Load teams", key="cs2_kalshi_load", use_container_width=True)
            refresh_prices = st.button("Refresh bid/ask", key="cs2_kalshi_refresh", use_container_width=True)

        # Load markets (team contracts) for the event ticker derived from the URL.
        if load_markets:
            try:
                ev = _kalshi_parse_event_ticker(kalshi_url)
                markets = kalshi_list_markets_for_event(ev)
                st.session_state["cs2_kalshi_markets"] = markets

                # Choose a default if none set yet or if old one vanished.
                if "cs2_kalshi_market" not in st.session_state:
                    st.session_state["cs2_kalshi_market"] = markets[0]["ticker"]
                else:
                    cur = st.session_state.get("cs2_kalshi_market")
                    if cur not in {m["ticker"] for m in markets}:
                        st.session_state["cs2_kalshi_market"] = markets[0]["ticker"]

                st.success(f"Loaded {len(markets)} Kalshi team market(s).")
            except Exception as e:
                st.error(f"Kalshi load failed: {e}")

        markets = st.session_state.get("cs2_kalshi_markets", None)
        if markets:
            opts = [m["ticker"] for m in markets]
            labels = {m["ticker"]: f'{m["ticker"]} — {m.get("title","")}' for m in markets}

            st.session_state.setdefault("cs2_kalshi_market", opts[0])
            st.selectbox(
                "Kalshi team market (YES contract)",
                options=opts,
                format_func=lambda t: labels.get(t, t),
                key="cs2_kalshi_market",
            )

        # Refresh prices (fills Team A bid/ask inputs below)
        if refresh_prices:
            try:
                tkr = st.session_state.get("cs2_kalshi_market")
                if not tkr:
                    # If the user pasted a URL but didn't click "Load teams" yet, auto-load once.
                    if not st.session_state.get("cs2_kalshi_markets"):
                        kalshi_url2 = st.session_state.get("cs2_kalshi_url", "")
                        if kalshi_url2:
                            ev2 = _kalshi_parse_event_ticker(kalshi_url2)
                            markets2 = kalshi_list_markets_for_event(ev2)
                            st.session_state["cs2_kalshi_markets"] = markets2
                            # Safe here because the selectbox isn't created unless markets exist.
                            st.session_state["cs2_kalshi_market"] = markets2[0]["ticker"]
                            tkr = markets2[0]["ticker"]
                    if not tkr:
                        raise ValueError("Paste a Kalshi URL, click Load teams, then pick a team market.")
                bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)

                if bid_f is not None:
                    st.session_state["cs2_live_market_bid"] = float(bid_f)
                if ask_f is not None:
                    st.session_state["cs2_live_market_ask"] = float(ask_f)

                st.session_state["cs2_mkt_fetch_meta"] = meta
                st.success("Kalshi bid/ask updated.")
            except Exception as e:
                st.error(f"Kalshi refresh failed: {e}")

    else:
        with colF2:
            ident = st.text_input(
                "Ticker / token_id (or paste URL)",
                value=str(st.session_state.get("cs2_mkt_fetch_ident", "")),
                key="cs2_mkt_fetch_ident",
            )
        with colF3:
            do_fetch = st.button("Fetch bid/ask", key="cs2_mkt_fetch_btn", use_container_width=True)

        if do_fetch and venue != "Manual":
            try:
                if venue == "Polymarket":
                    bid_f, ask_f, meta = fetch_polymarket_bid_ask(ident)
                else:
                    # Shouldn't happen, but keep safe.
                    bid_f, ask_f, meta = None, None, {}

                if bid_f is not None:
                    st.session_state["cs2_live_market_bid"] = float(bid_f)
                if ask_f is not None:
                    st.session_state["cs2_live_market_ask"] = float(ask_f)

                st.session_state["cs2_mkt_fetch_meta"] = meta
                st.success(f"Fetched {venue} bid/ask.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    meta = st.session_state.get("cs2_mkt_fetch_meta")

    colBid, colAsk = st.columns(2)
    with colBid:
        st.session_state.setdefault("cs2_live_market_bid", float(st.session_state.get("cs2_live_market_bid", 0.48)))
        market_bid = st.number_input("Best bid (Sell) for Team A (0–1)", min_value=0.01, max_value=0.99,
                                     step=0.01, format="%.2f", key="cs2_live_market_bid")
    with colAsk:
        st.session_state.setdefault("cs2_live_market_ask", float(st.session_state.get("cs2_live_market_ask", 0.52)))
        market_ask = st.number_input("Best ask (Buy) for Team A (0–1)", min_value=0.01, max_value=0.99,
                                     step=0.01, format="%.2f", key="cs2_live_market_ask")

    bid = float(market_bid)
    ask = float(market_ask)
    if ask < bid:
        st.warning("Ask < bid (inverted). Using ask = bid for calculations.")
        ask = bid


    market_mid = 0.5 * (bid + ask)
    spread = ask - bid
    rel_spread = (spread / market_mid) if market_mid > 0 else 0.0

    st.markdown("### Model knobs (MVP)")
    colB1, colB2, colB3 = st.columns(3)
    with colB1:
        beta_score = st.slider("β score", 0.05, 0.60, float(st.session_state.get("beta_score", 0.22)), 0.01)
    with colB2:
        beta_econ = st.slider("β econ", 0.00, 0.20, float(st.session_state.get("beta_econ", 0.06)), 0.01)
    with colB3:
        beta_pistol = st.slider("β pistol", 0.00, 0.80, float(st.session_state.get("beta_pistol", 0.35)), 0.01)
    # Closeout lock knobs (reduce swinginess near map closeout)
    colL1, colL2, colL3 = st.columns(3)
    with colL1:
        beta_lock = st.slider("β lock", 0.00, 3.00, float(st.session_state.get("beta_lock", 0.90)), 0.05)
    with colL2:
        lock_start_offset = st.slider("Lock starts (win_target - N)", 1, 7, int(st.session_state.get("lock_start_offset", 3)), 1)
    with colL3:
        lock_ramp = st.slider("Lock ramp (rounds)", 1, 8, int(st.session_state.get("lock_ramp", 3)), 1)

    st.session_state["beta_score"] = float(beta_score)
    st.session_state["beta_econ"] = float(beta_econ)
    st.session_state["beta_pistol"] = float(beta_pistol)
    st.session_state["beta_lock"] = float(beta_lock)
    st.session_state["lock_start_offset"] = int(lock_start_offset)
    st.session_state["lock_ramp"] = int(lock_ramp)

    # ---- Compute fair probability (map) + K-based credible bands (map) ----
    
    # Dynamic win target / total rounds (regulation vs OT approximation)
    win_target = cs2_current_win_target(int(rounds_a), int(rounds_b))
    total_rounds = int(2 * win_target - 2)

    p_hat_map = estimate_inplay_prob(
        float(p0_map),
        int(rounds_a),
        int(rounds_b),
        float(econ_a),
        float(econ_b),
        pistol_a=bool(pistol_a),
        pistol_b=bool(pistol_b),
        beta_score=float(beta_score),
        beta_econ=float(beta_econ),
        beta_pistol=float(beta_pistol),
        map_name=map_name,
        a_side=a_side,
        pistol_decay=0.30,
        beta_side=0.85,
        beta_lock=float(beta_lock),
        lock_start_offset=int(lock_start_offset),
        lock_ramp=int(lock_ramp),
        win_target=int(win_target),
    )

    # Infer streak / reversal / gaps from score updates (no per-round input)
    stream = update_round_stream("cs2_inplay", int(rounds_a), int(rounds_b))

    kappa_map = compute_kappa_cs2(
        p0=float(p0_map),
        rounds_a=int(rounds_a),
        rounds_b=int(rounds_b),
        econ_missing=bool(econ_missing),
        econ_fragile=bool(econ_fragile),
        pistol_a=bool(pistol_a),
        pistol_b=bool(pistol_b),
        streak_len=int(stream.get("streak_len", 0)),
        streak_winner=stream.get("streak_winner", None),
        reversal=bool(stream.get("reversal", False)),
        gap_delta=int(stream.get("gap_delta", 0)),
        chaos_boost=float(chaos_boost),
        total_rounds=int(total_rounds),
    )
    calib = load_kappa_calibration()
    total_r = int(int(rounds_a) + int(rounds_b))
    is_ot = (int(rounds_a) >= int(win_target) - 1 and int(rounds_b) >= int(win_target) - 1)
    mult = get_kappa_multiplier(calib, "cs2", float(cs2_band_level), total_r, is_ot) if bool(st.session_state.get("cs2_use_calib", False)) else 1.0
    kappa_map = float(kappa_map) * float(cs2_k_scale) * float(mult)
    lo_map, hi_map = beta_credible_interval(float(p_hat_map), float(kappa_map), level=float(cs2_band_level))

    # Near-end continuity: soften map→series cliffs by "soft locking" map odds
    # as the score approaches the win target (especially helpful around 12-x).
    p_hat_map = cs2_soft_lock_map_prob(float(p_hat_map), int(rounds_a), int(rounds_b), int(win_target))
    lo_map = cs2_soft_lock_map_prob(float(lo_map), int(rounds_a), int(rounds_b), int(win_target))
    hi_map = cs2_soft_lock_map_prob(float(hi_map), int(rounds_a), int(rounds_b), int(win_target))


    # ---- If the market is SERIES, convert fair/map-bands to SERIES fair/bands ----
    if contract_scope == "Series winner" and n_maps > 1:
        p_hat = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(p_hat_map), float(p0_map))
        lo = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(lo_map), float(p0_map))
        hi = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(hi_map), float(p0_map))
        line_label = "Series fair p(A)"
    else:
        p_hat = float(p_hat_map)
        lo = float(lo_map)
        hi = float(hi_map)
        line_label = "Map fair p(A)"

    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        st.metric(line_label, f"{p_hat*100:.1f}%")
    with colM2:
        st.metric("Certainty band (K CI)", f"[{lo*100:.1f}%, {hi*100:.1f}%]")
    with colM3:
        st.metric("Market bid / ask", f"{bid*100:.1f}% / {ask*100:.1f}%")
    with colM4:
        st.metric("Spread (abs / rel)", f"{spread*100:.1f} pp / {rel_spread*100:.1f}%")

    # informational differences (not signals)
    colD1, colD2, colD3 = st.columns(3)
    with colD1:
        st.metric("Deviation (mid - fair)", f"{(market_mid - p_hat)*100:+.1f} pp")
    with colD2:
        st.metric("Fair - Ask (buy edge)", f"{(p_hat - ask)*100:+.1f} pp")
    with colD3:
        st.metric("Bid - Fair (sell edge)", f"{(bid - p_hat)*100:+.1f} pp")

    # Optional: show underlying map-fair too (helps sanity-check series math)
    with st.expander("Show underlying MAP fair (debug)"):
        st.write({
            "p0_map": float(p0_map),
            "p_hat_map": float(p_hat_map),
            "band_map": [float(lo_map), float(hi_map)],
            "series_fmt": series_fmt,
            "maps_a_won": int(maps_a_won),
            "maps_b_won": int(maps_b_won),
            "note": "Series conversion assumes future maps i.i.d with p_future = p0_map (MVP)."
        })

    colAdd, colClear, colExport = st.columns(3)
    with colAdd:
        if st.button("Add snapshot"):
            total_rounds_logged = int(rounds_a) + int(rounds_b)
            prev_total = st.session_state.get("cs2_inplay_last_total_rounds")
            gap_rounds = 0
            if prev_total is not None:
                expected_total = int(prev_total) + 1
                if total_rounds_logged > expected_total:
                    gap_rounds = int(total_rounds_logged - expected_total)
                elif total_rounds_logged < expected_total:
                    gap_rounds = int(expected_total - total_rounds_logged)
            st.session_state["cs2_inplay_last_total_rounds"] = int(total_rounds_logged)
            if gap_rounds > 0 and prev_total is not None:
                st.warning(f"Gap detected: expected total rounds {int(prev_total)+1}, got {total_rounds_logged} (gap {gap_rounds}).")
            map_index = int(st.session_state.get("cs2_inplay_map_index", 0))

            _snap_ts = datetime.now().isoformat(timespec="seconds")
            st.session_state["cs2_live_rows"].append({
                "t": len(st.session_state["cs2_live_rows"]),
                "snapshot_ts": _snap_ts,
                "series_fmt": series_fmt,
                "contract_scope": contract_scope,
                "maps_a_won": int(maps_a_won),
                "maps_b_won": int(maps_b_won),
                "rounds_a": int(rounds_a),
                "rounds_b": int(rounds_b),
                        "map_index": int(map_index),
                        "total_rounds": int(total_rounds_logged),
                        "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                        "gap_rounds": int(gap_rounds),
                        "gap_flag": bool(gap_rounds > 0),
                        "gap_reason": "",
                "map_index": int(map_index),
                "total_rounds": int(total_rounds_logged),
                "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                "gap_rounds": int(gap_rounds),
                "econ_a": float(econ_a),
                "econ_b": float(econ_b),
                "p0_series": float(p0_series) if 'p0_series' in locals() else float(p0_map),
                "p0_map": float(p0_map),
                "p_hat_map": float(p_hat_map),
                "p_hat": float(p_hat),
                "band_lo": float(lo),
                "band_hi": float(hi),
                "band_lo_map": float(lo_map),
                "band_hi_map": float(hi_map),
                "market_bid": float(bid),
                "market_ask": float(ask),
                "market_mid": float(market_mid),
                "spread": float(spread),
                "rel_spread": float(rel_spread),
                "dev_mid_pp": float((market_mid - p_hat)*100.0),
                "buy_edge_pp": float((p_hat - ask)*100.0),
                "sell_edge_pp": float((bid - p_hat)*100.0),
            })

            # Optional: persist this snapshot to CSV for K calibration
            if bool(st.session_state.get("cs2_inplay_persist", False)) and str(st.session_state.get("cs2_inplay_match_id", "")).strip():
                try:
                    persist_inplay_row({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "game": "CS2",
                        "match_id": str(st.session_state.get("cs2_inplay_match_id")).strip(),
                        "contract_scope": str(contract_scope),
                        "series_format": str(series_fmt),
                        "maps_a_won": int(maps_a_won),
                        "maps_b_won": int(maps_b_won),
                        "team_a": str(team_a),
                        "team_b": str(team_b),
                        "map_name": str(st.session_state.get("cs2_live_map_name", "")),
                        "a_side_now": str(st.session_state.get("cs2_live_a_side", "")),
                        "rounds_a": int(rounds_a),
                        "rounds_b": int(rounds_b),
                        "map_index": int(map_index),
                        "total_rounds": int(total_rounds_logged),
                        "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                        "gap_rounds": int(gap_rounds),
                        "gap_flag": bool(gap_rounds > 0) if prev_total is not None else False,
                        "gap_reason": "",
                        "buy_state_a": str(buy_a),
                        "buy_state_b": str(buy_b),
                        "econ_missing": bool(econ_missing),
                        "econ_fragile": bool(econ_fragile),
                        "pistol_a": bool(pistol_a),
                        "pistol_b": bool(pistol_b),
                        "gap_delta": int(stream.get("gap_delta", 0)) if 'stream' in locals() else 0,
                        "streak_winner": stream.get("streak_winner", "") if 'stream' in locals() else "",
                        "streak_len": int(stream.get("streak_len", 0)) if 'stream' in locals() else 0,
                        "reversal": bool(stream.get("reversal", False)) if 'stream' in locals() else False,
                        "n_tracked": int(stream.get("n_tracked", 0)) if 'stream' in locals() else 0,
                        "p0_map": float(p0_map),
                        "p_fair_map": float(p_hat_map),
                        "kappa_map": float(kappa_map) if 'kappa_map' in locals() else "",
                        "p_fair": float(p_hat),
                        "band_level": float(st.session_state.get("cs2_inplay_band_level", 0.80)),
                        "band_lo": float(lo),
                        "band_hi": float(hi),
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": float(market_mid),
                        "spread_abs": float(spread),
                        "spread_rel": float(rel_spread),
                        "notes": str(st.session_state.get("cs2_inplay_notes", "")),
                        "snapshot_idx": int(total_rounds_logged),
                        "half": ("1" if int(total_rounds_logged) <= 12 else ("2" if int(total_rounds_logged) <= 24 else "OT")),
                        "round_in_half": (int(total_rounds_logged) if int(total_rounds_logged) <= 12 else (int(total_rounds_logged)-12 if int(total_rounds_logged) <= 24 else int(total_rounds_logged)-24)),
                        "is_ot": bool(int(total_rounds_logged) > 24),
                        "round_in_map": int(total_rounds_logged),
                    })
                except Exception as e:
                    st.warning(f"Snapshot not persisted: {e}")
            # Incremental backtest update so In-Play Backtest tab chart updates without full run
            _row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "match_id": str(st.session_state.get("cs2_inplay_match_id", "")).strip(),
                "contract_scope": str(contract_scope),
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(market_mid),
                "spread_abs": float(spread),
                "p_fair": float(p_hat),
                "band_lo": float(lo),
                "band_hi": float(hi),
            }
            _run_inplay_incremental_and_refresh_session(_row)


def _cs2_start_next_half():
    # Half-time resets (pistol + economy context), keep score
    cur_side = str(st.session_state.get("cs2_live_a_side", "Unknown"))
    if cur_side == "CT":
        st.session_state["cs2_live_a_side"] = "T"
    elif cur_side == "T":
        st.session_state["cs2_live_a_side"] = "CT"
    # Set buy states to pistol default and clear "most recent pistol" flags
    st.session_state["cs2_live_buy_a"] = "Pistol"
    st.session_state["cs2_live_buy_b"] = "Pistol"
    st.session_state["cs2_live_pistol_a"] = False
    st.session_state["cs2_live_pistol_b"] = False
    # Keep chaos as-is
    st.session_state["cs2_live_chaos"] = float(st.session_state.get("cs2_live_chaos", 0.00))


def _cs2_start_next_map():

    # If operator is persisting, log the *completed* map winner before resetting for the next map.
    try:
        if bool(st.session_state.get("cs2_inplay_persist", False)):
            _mid = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
            if _mid:
                cur_map_index = int(st.session_state.get("cs2_inplay_map_index", 0))
                cur_map_name = str(st.session_state.get("cs2_live_map_name", ""))
                team_a = str(st.session_state.get("cs2_live_team_a", "")).strip()
                team_b = str(st.session_state.get("cs2_live_team_b", "")).strip()

                prev_a = int(st.session_state.get("cs2_prev_maps_a_won", 0))
                prev_b = int(st.session_state.get("cs2_prev_maps_b_won", 0))
                now_a = int(st.session_state.get("cs2_live_maps_a_won", 0))
                now_b = int(st.session_state.get("cs2_live_maps_b_won", 0))

                winner = ""
                if now_a == prev_a + 1 and now_b == prev_b:
                    winner = team_a
                elif now_b == prev_b + 1 and now_a == prev_a:
                    winner = team_b

                if winner and team_a and team_b:
                    persist_inplay_map_result(_mid, "CS2", cur_map_index, cur_map_name, team_a, team_b, winner)

                # Update prev counters for the next map
                st.session_state["cs2_prev_maps_a_won"] = now_a
                st.session_state["cs2_prev_maps_b_won"] = now_b
    except Exception:
        # Never block the UI on logging
        pass

    # Keep cs2_live_rows so the BO3 chart continues across maps
    # Reset per-map operator inputs
    st.session_state["cs2_live_rounds_a"] = 0
    st.session_state["cs2_live_rounds_b"] = 0
    st.session_state["cs2_live_buy_a"] = "Pistol"
    st.session_state["cs2_live_buy_b"] = "Pistol"
    st.session_state["cs2_live_pistol_a"] = False
    st.session_state["cs2_live_pistol_b"] = False
    st.session_state["cs2_live_chaos"] = 0.00

    # Map/side context
    st.session_state["cs2_live_map_name"] = "Average (no map)"
    st.session_state["cs2_live_a_side"] = "Unknown"

    # Advance map index + reset “missed rounds” tracking
    st.session_state["cs2_inplay_map_index"] = int(st.session_state.get("cs2_inplay_map_index", 0)) + 1
    st.session_state["cs2_inplay_last_total_rounds"] = None


with colClear:
    cA, cB, cC = st.columns(3)

    with cA:
        if st.button("Clear chart"):
            st.session_state["cs2_live_rows"] = []
    with cB:
        st.button("Start next half", key="cs2_next_half", on_click=_cs2_start_next_half)
    with cC:
        st.button("Start next map", key="cs2_next_map", on_click=_cs2_start_next_map)
    with colExport:
        if st.button("Export snapshots to Diagnostics"):
            st.session_state["cs2_live_export_df"] = pd.DataFrame(st.session_state["cs2_live_rows"])

    st.markdown("### Chart")
    if len(st.session_state["cs2_live_rows"]) > 0:
        chart_df = pd.DataFrame(st.session_state["cs2_live_rows"])

        # Optional: overlay probability-calibrated p_hat on the chart
        pcal = load_p_calibration_json(APP_DIR)
        show_pcal = st.checkbox("Show p_calibrated overlay", value=False, key="cs2_show_pcal")

        plot_df = chart_df[["t","p_hat","band_lo","band_hi","market_mid"]].copy()
        if show_pcal and pcal:
            plot_df["p_hat_cal"] = plot_df["p_hat"].apply(lambda x: apply_p_calibration(x, pcal, "cs2"))
        plot_df = plot_df.set_index("t")
        # Use Plotly so we can overlay backtest entry/exit markers on the same chart
        try:
            import plotly.graph_objects as go
            fig_cs2 = go.Figure()
            t_vals = plot_df.index.astype(int).tolist()
            for col in ["market_mid", "p_hat", "band_lo", "band_hi"]:
                if col in plot_df.columns:
                    fig_cs2.add_trace(go.Scatter(x=t_vals, y=plot_df[col].tolist(), name=col, mode="lines"))
            if show_pcal and pcal and "p_hat_cal" in plot_df.columns:
                fig_cs2.add_trace(go.Scatter(x=t_vals, y=plot_df["p_hat_cal"].tolist(), name="p_hat_cal", mode="lines"))
            # Overlay backtest entry/exit for current match when we have trades and snapshot_ts for alignment
            cs2_match_id = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
            bt_trades = st.session_state.get("inplay_bt_trades") or {}
            has_ts = "snapshot_ts" in chart_df.columns and chart_df["snapshot_ts"].notna().any()
            _strategy_styles_cs2 = {
                "S1_mr_to_entry_fair": {"color": "#22c55e", "sym_l": "triangle-up", "sym_s": "triangle-down"},
                "S2_mr_to_current_fair": {"color": "#3b82f6", "sym_l": "diamond", "sym_s": "diamond"},
                "S3_mr_inside_band": {"color": "#f97316", "sym_l": "square", "sym_s": "square"},
            }
            if cs2_match_id and has_ts:
                chart_ts = pd.to_datetime(chart_df["snapshot_ts"], errors="coerce")
                # Completed trades from CSV (no backtest run needed if already loaded)
                if bt_trades:
                    for sid, trades_df in bt_trades.items():
                        if trades_df is None or len(trades_df) == 0:
                            continue
                        match_trades = trades_df[trades_df["match_id"].astype(str) == cs2_match_id]
                        if len(match_trades) == 0:
                            continue
                        style = _strategy_styles_cs2.get(sid, {"color": "#6b7280", "sym_l": "triangle-up", "sym_s": "triangle-down"})
                        entry_ts = pd.to_datetime(match_trades["entry_ts"], errors="coerce")
                        exit_ts = pd.to_datetime(match_trades["exit_ts"], errors="coerce")
                        t_entry = []
                        y_entry = []
                        t_exit = []
                        y_exit = []
                        exit_reasons = []
                        exit_pnls = []
                        for _, tr in match_trades.iterrows():
                            et = pd.to_datetime(tr.get("entry_ts"), errors="coerce")
                            ex = pd.to_datetime(tr.get("exit_ts"), errors="coerce")
                            if pd.notna(et):
                                idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - et.value))
                                t_entry.append(chart_df["t"].iloc[idx])
                                y_entry.append(tr.get("entry_mid", np.nan))
                            if pd.notna(ex):
                                idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - ex.value))
                                t_exit.append(chart_df["t"].iloc[idx])
                                y_exit.append(tr.get("exit_px", np.nan))
                                exit_reasons.append(str(tr.get("exit_reason", "")))
                                exit_pnls.append(tr.get("pnl_$", np.nan))
                        if t_entry:
                            sides = match_trades["side"]
                            long_m = (sides == "LONG").tolist()
                            short_m = (sides == "SHORT").tolist()
                            n = len(t_entry)
                            if any(long_m[i] for i in range(min(n, len(long_m)))):
                                te = [t_entry[i] for i in range(n) if i < len(long_m) and long_m[i]]
                                ye = [y_entry[i] for i in range(n) if i < len(long_m) and long_m[i]]
                                if te:
                                    fig_cs2.add_trace(go.Scatter(x=te, y=ye, name=f"{sid} Entry LONG", mode="markers", marker=dict(symbol=style["sym_l"], size=12, color=style["color"])))
                            if any(short_m[i] for i in range(min(n, len(short_m)))):
                                te = [t_entry[i] for i in range(n) if i < len(short_m) and short_m[i]]
                                ye = [y_entry[i] for i in range(n) if i < len(short_m) and short_m[i]]
                                if te:
                                    fig_cs2.add_trace(go.Scatter(x=te, y=ye, name=f"{sid} Entry SHORT", mode="markers", marker=dict(symbol=style["sym_s"], size=12, color=style["color"])))
                        if t_exit:
                            customdata = np.column_stack((exit_reasons, exit_pnls))
                            fig_cs2.add_trace(go.Scatter(
                                x=t_exit, y=y_exit, name=f"{sid} Exit", mode="markers",
                                marker=dict(symbol="circle", size=10, color=style["color"], line=dict(width=1, color="white")),
                                customdata=customdata,
                                hovertemplate="t %{x}<br>exit_reason: %{customdata[0]}<br>pnl_$ %{customdata[1]:.2f}<extra></extra>",
                            ))
                # Open positions from state (show entry marker without running backtest)
                bt_state = st.session_state.get("inplay_bt_state")
                if bt_state is None:
                    try:
                        _state_path = PROJECT_ROOT / "logs" / "inplay_backtest_state.json"
                        if _state_path.exists():
                            bt_state = load_backtest_state(_state_path)
                    except Exception:
                        pass
                if bt_state:
                    for sid, s in (bt_state.get("strategies") or {}).items():
                        pos = s.get("position")
                        if not pos or str(pos.get("match_id")) != cs2_match_id:
                            continue
                        style = _strategy_styles_cs2.get(sid, {"color": "#6b7280", "sym_l": "triangle-up", "sym_s": "triangle-down"})
                        et = pd.to_datetime(pos.get("entry_ts"), errors="coerce")
                        if pd.notna(et):
                            idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - et.value))
                            t_open = chart_df["t"].iloc[idx]
                            y_open = pos.get("entry_mid", np.nan)
                            side = (pos.get("side") or "").upper()
                            if side == "LONG":
                                fig_cs2.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry LONG (open)", mode="markers", marker=dict(symbol=style["sym_l"], size=12, color=style["color"])))
                            elif side == "SHORT":
                                fig_cs2.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry SHORT (open)", mode="markers", marker=dict(symbol=style["sym_s"], size=12, color=style["color"])))
            fig_cs2.update_layout(title="Chart", xaxis_title="t (snapshot index)", yaxis_title="Price / Fair", height=420)
            st.plotly_chart(fig_cs2, use_container_width=True)
        except ImportError:
            st.line_chart(plot_df, use_container_width=True, height=420)
        except Exception:
            st.line_chart(plot_df, use_container_width=True, height=420)

        # Also show the raw table
        st.dataframe(chart_df, use_container_width=True)
    else:
        st.info("Add at least one snapshot to see the chart.")

    # --- Final result logging (needed for probability calibration + backtest match-end) ---
    st.markdown("### Log final result (for ML)")
    st.caption("Also used for **In-Play Backtest** match-end settlement: if a position is still open when the match ends, it closes at 1 (Team A wins) or 0 (Team B wins) using the winner you save here.")
    rcol1, rcol2 = st.columns([2.0, 1.0])
    with rcol1:
        _cs2_default_winner_idx = 0
        try:
            if int(rounds_a) >= 13 and int(rounds_a) > int(rounds_b):
                _cs2_default_winner_idx = 0
            elif int(rounds_b) >= 13 and int(rounds_b) > int(rounds_a):
                _cs2_default_winner_idx = 1
        except Exception:
            pass
        cs2_winner_sel = st.selectbox("Winner", [team_a, team_b], index=_cs2_default_winner_idx, key="cs2_inplay_winner_sel")
    with rcol2:
        if st.button("Save result", key="cs2_inplay_save_result"):
            _mid = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
            if not _mid:
                st.error("Set a Match ID above before saving a result.")
            else:
                persist_inplay_result(_mid, "CS2", str(team_a), str(team_b), str(cs2_winner_sel))
                st.success("Result saved to inplay_match_results_clean.csv")



with tabs[4]:
    st.header("Valorant In-Play Indicator (MVP)")
    st.caption("Same idea as CS2: fair probability line + certainty bands vs market bid/ask. "
               "Valorant MVP adds ult advantage + operator presence. Economy is coarse (Eco/Light/Full).")


    st.markdown("### K-bands + calibration logging (always visible)")
    l1, l2, l3, l4 = st.columns([1.35, 1.10, 1.05, 1.20])
    with l1:
        val_inplay_match_id = st.text_input("Match ID", key="val_inplay_match_id")
        if "val_inplay_map_index" not in st.session_state:
            st.session_state["val_inplay_map_index"] = 0
            st.session_state["val_inplay_last_total_rounds"] = None
    with l2:
        val_inplay_persist = st.checkbox("Persist snapshots + results", value=bool(st.session_state.get("val_inplay_persist", False)), key="val_inplay_persist")
    with l3:
        val_band_level = st.selectbox("Band level", options=[0.68, 0.75, 0.80, 0.90, 0.95], index=2,
                                     format_func=lambda x: f"{int(x*100)}%", key="val_inplay_band_level")
    with l4:
        val_k_scale = st.slider("K scale", 0.50, 2.00, float(st.session_state.get("val_inplay_k_scale", 1.00)), 0.05, key="val_inplay_k_scale")
    val_inplay_notes = st.text_input("Notes (optional)", key="val_inplay_notes")
    show_inplay_log_paths()

    # --- K calibration controls (optional) ---
    calib = load_kappa_calibration()
    vcal1, vcal2, vcal3 = st.columns([1.2, 1.0, 1.8])
    with vcal1:
        val_use_calib = st.checkbox("Use calibrated K multiplier", value=bool(st.session_state.get("val_use_calib", False)), key="val_use_calib")
    with vcal2:
        if st.button("Train calibration (clean logs)", key="val_train_calib_btn"):
            ok, out = run_kappa_trainer()
            if ok:
                st.success("Calibration trained.")
            else:
                st.error("Calibration training failed.")
            st.session_state["kappa_train_last_out"] = out
    with vcal3:
        if st.session_state.get("kappa_train_last_out"):
            st.caption(str(st.session_state.get("kappa_train_last_out"))[:240] + ("…" if len(str(st.session_state.get("kappa_train_last_out"))) > 240 else ""))
        elif calib:
            st.caption(f"Calibration loaded: {KAPPA_CALIB_PATH.name}")
        else:
            st.caption("No calibration loaded yet.")

    colA, colB = st.columns(2)
    with colA:
        val_team_a = st.text_input("Team A", value=str(st.session_state.get("val_team_a", "Team A")))
    with colB:
        val_team_b = st.text_input("Team B", value=str(st.session_state.get("val_team_b", "Team B")))
    st.session_state["val_team_a"] = val_team_a
    st.session_state["val_team_b"] = val_team_b


    # Optional: map name (useful for per-map priors / ML)
    # Widget with key="val_map_name" automatically updates st.session_state; do not assign manually
    val_map_name = st.text_input("Map name (optional)", value=str(st.session_state.get("val_map_name", "")), key="val_map_name")

    st.markdown("### Context — series format and what the contract represents")
    cctx1, cctx2, cctx3, cctx4 = st.columns([1.0, 1.5, 1.0, 1.0])
    with cctx1:
        val_series_fmt = st.selectbox("Series format", ["BO1", "BO3", "BO5"],
                                      index=int(st.session_state.get("val_series_fmt_idx", 1)),
                                      key="val_series_fmt")
    with cctx2:
        val_contract_scope = st.selectbox("Contract priced on", ["Map winner (this map)", "Series winner"],
                                          index=int(st.session_state.get("val_contract_scope_idx", 0)),
                                          key="val_contract_scope")
    with cctx3:
        val_maps_a_won = st.number_input("Maps A won", 0, 4, int(st.session_state.get("val_maps_a_won", 0)), 1, key="val_maps_a_won")
    with cctx4:
        val_maps_b_won = st.number_input("Maps B won", 0, 4, int(st.session_state.get("val_maps_b_won", 0)), 1, key="val_maps_b_won")
        st.session_state.setdefault("val_prev_maps_a_won", int(st.session_state.get("val_maps_a_won", 0)))
        st.session_state.setdefault("val_prev_maps_b_won", int(st.session_state.get("val_maps_b_won", 0)))

    st.session_state["val_series_fmt_idx"] = ["BO1","BO3","BO5"].index(val_series_fmt)
    st.session_state["val_contract_scope_idx"] = ["Map winner (this map)","Series winner"].index(val_contract_scope)

    val_n_maps = int(str(val_series_fmt).replace("BO",""))
    if val_contract_scope == "Map winner (this map)" or val_n_maps == 1:
        val_maps_a_won = 0
        val_maps_b_won = 0

    if val_contract_scope == "Series winner" and val_n_maps > 1:
        val_current_map_num = int(val_maps_a_won) + int(val_maps_b_won) + 1
        st.caption(f"Series score: A {int(val_maps_a_won)} – {int(val_maps_b_won)} B (currently Map {val_current_map_num} of up to {val_n_maps}).")

    st.markdown("### Step 1 — Pre-match fair probability (from your source/model)")
    if val_contract_scope == "Series winner" and val_n_maps > 1:
        p0_series = st.number_input("Pre-match fair series win% for Team A (0–1)",
                                    min_value=0.01, max_value=0.99,
                                    value=float(st.session_state.get("val_p0_series", 0.55)),
                                    step=0.01, format="%.2f", key="val_p0_series")
        p0_map = invert_series_to_map_prob(float(p0_series), int(val_n_maps))
        st.info(f"Implied per-map prior from series (i.i.d approx): p_map ≈ {p0_map*100:.1f}%")
    else:
        p0_map = st.number_input("Pre-match fair win% for Team A (0–1)", min_value=0.01, max_value=0.99,
                                 value=float(st.session_state.get("val_p0_map", 0.55)),
                                 step=0.01, format="%.2f", key="val_p0_map")
        p0_series = float(p0_map)

    st.markdown("### Step 2 — Live map inputs")
    if "val_rows" not in st.session_state:
        st.session_state["val_rows"] = []

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val_rounds_a = st.number_input("Rounds A", 0, 30, int(st.session_state.get("val_rounds_a", 0)), 1, key="val_rounds_a")
    with c2:
        val_rounds_b = st.number_input("Rounds B", 0, 30, int(st.session_state.get("val_rounds_b", 0)), 1, key="val_rounds_b")
    with c3:
        val_eco_a = st.selectbox("A buy state", ["Eco","Light","Full"],
                                 index=int(st.session_state.get("val_eco_a_idx", 1)), key="val_eco_a")
    with c4:
        val_eco_b = st.selectbox("B buy state", ["Eco","Light","Full"],
                                 index=int(st.session_state.get("val_eco_b_idx", 1)), key="val_eco_b")
    st.session_state["val_eco_a_idx"] = ["Eco","Light","Full"].index(val_eco_a)
    st.session_state["val_eco_b_idx"] = ["Eco","Light","Full"].index(val_eco_b)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        val_pistol_a = st.checkbox("A won most recent pistol?", value=bool(st.session_state.get("val_pistol_a", False)), key="val_pistol_a")
    with c6:
        val_pistol_b = st.checkbox("B won most recent pistol?", value=bool(st.session_state.get("val_pistol_b", False)), key="val_pistol_b")
    with c7:
        val_ults_diff = st.number_input("Ults online diff (A − B)", min_value=-6, max_value=6,
                                        value=int(st.session_state.get("val_ults_diff", 0)), step=1, key="val_ults_diff")
    with c8:
        val_op_diff = st.number_input("Op diff (A − B)", min_value=-3, max_value=3,
                                      value=int(st.session_state.get("val_op_diff", 0)), step=1, key="val_op_diff")

    colSide, colChaos, colCtx = st.columns([1.2, 1.0, 1.3])
    with colSide:
        side_sel = st.selectbox("A currently on", ["Unknown","Attack","Defense"],
                                index=int(st.session_state.get("val_side_idx", 0)), key="val_side")
    with colChaos:
        val_chaos = st.slider("Chaos widen (manual)", 0.00, 0.25, float(st.session_state.get("val_chaos", 0.00)), 0.01, key="val_chaos")
    with colCtx:
        limited_context = st.checkbox("Limited context mode (widen bands)", value=bool(st.session_state.get("val_limited_ctx", False)), key="val_limited_ctx")

    st.session_state["val_side_idx"] = ["Unknown","Attack","Defense"].index(side_sel)

    if side_sel == "Defense":
        a_on_def = True
    elif side_sel == "Attack":
        a_on_def = False
    else:
        a_on_def = None

    missing_context_widen = 0.05 if limited_context else 0.0
    if a_on_def is None:
        missing_context_widen += 0.02  # unknown side => widen

    st.markdown("### Step 3 — Market bid/ask input")
    if val_contract_scope == "Series winner" and val_n_maps > 1:
        st.caption("Enter executable prices for **Team A to win the SERIES** (bid=sell YES, ask=buy YES).\n"
                   "If you’re trading the map market instead, set Contract priced on = Map winner.")
    else:
        st.caption("Enter executable prices for **Team A to win THIS MAP** (bid=sell YES, ask=buy YES).")

    st.markdown("#### Optional — Fetch bid/ask from an exchange API")

    # Persisted Kalshi URL + selected team market (paste/select once, then just refresh).
    colF1, colF2, colF3 = st.columns([1.05, 2.35, 1.10])
    with colF1:
        val_venue = st.selectbox("Venue", ["Manual", "Kalshi", "Polymarket"], index=0, key="val_mkt_fetch_venue")

    if val_venue == "Kalshi":
        with colF2:
            st.session_state.setdefault("val_kalshi_url", st.session_state.get("val_kalshi_url", ""))
            val_kalshi_url = st.text_input("Kalshi URL (paste once; event/game page)", key="val_kalshi_url")
        with colF3:
            val_load_markets = st.button("Load teams", key="val_kalshi_load", use_container_width=True)
            val_refresh_prices = st.button("Refresh bid/ask", key="val_kalshi_refresh", use_container_width=True)

        if val_load_markets:
            try:
                ev = _kalshi_parse_event_ticker(val_kalshi_url)
                markets = kalshi_list_markets_for_event(ev)
                st.session_state["val_kalshi_markets"] = markets

                if "val_kalshi_market" not in st.session_state:
                    st.session_state["val_kalshi_market"] = markets[0]["ticker"]
                else:
                    cur = st.session_state.get("val_kalshi_market")
                    if cur not in {m["ticker"] for m in markets}:
                        st.session_state["val_kalshi_market"] = markets[0]["ticker"]

                st.success(f"Loaded {len(markets)} Kalshi team market(s).")
            except Exception as e:
                st.error(f"Kalshi load failed: {e}")

        markets = st.session_state.get("val_kalshi_markets", None)
        if markets:
            opts = [m["ticker"] for m in markets]
            labels = {m["ticker"]: f'{m["ticker"]} — {m.get("title","")}' for m in markets}

            st.session_state.setdefault("val_kalshi_market", opts[0])
            st.selectbox(
                "Kalshi team market (YES contract)",
                options=opts,
                format_func=lambda t: labels.get(t, t),
                key="val_kalshi_market",
            )

        if val_refresh_prices:
            try:
                tkr = st.session_state.get("val_kalshi_market")
                if not tkr:
                    if not st.session_state.get("val_kalshi_markets"):
                        kalshi_url2 = st.session_state.get("val_kalshi_url", "")
                        if kalshi_url2:
                            ev2 = _kalshi_parse_event_ticker(kalshi_url2)
                            markets2 = kalshi_list_markets_for_event(ev2)
                            st.session_state["val_kalshi_markets"] = markets2
                            st.session_state["val_kalshi_market"] = markets2[0]["ticker"]
                            tkr = markets2[0]["ticker"]
                    if not tkr:
                        raise ValueError("Paste a Kalshi URL, click Load teams, then pick a team market.")
                bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)

                if bid_f is not None:
                    st.session_state["val_market_bid"] = float(bid_f)
                    st.session_state["val_market_bid_input"] = float(bid_f)
                if ask_f is not None:
                    st.session_state["val_market_ask"] = float(ask_f)
                    st.session_state["val_market_ask_input"] = float(ask_f)

                st.session_state["val_mkt_fetch_meta"] = meta
                st.success("Kalshi bid/ask updated.")
            except Exception as e:
                st.error(f"Kalshi refresh failed: {e}")

    else:
        with colF2:
            val_ident = st.text_input(
                "Ticker / token_id (or paste URL)",
                value=str(st.session_state.get("val_mkt_fetch_ident", "")),
                key="val_mkt_fetch_ident",
            )
        with colF3:
            val_do_fetch = st.button("Fetch bid/ask", use_container_width=True, key="val_do_fetch")

        if val_do_fetch and val_venue != "Manual":
            try:
                if val_venue == "Kalshi":
                    bid_f, ask_f, meta = fetch_kalshi_bid_ask(val_ident)
                else:
                    bid_f, ask_f, meta = fetch_polymarket_bid_ask(val_ident)

                if bid_f is not None:
                    st.session_state["val_market_bid"] = float(bid_f)
                    st.session_state["val_market_bid_input"] = float(bid_f)
                if ask_f is not None:
                    st.session_state["val_market_ask"] = float(ask_f)
                    st.session_state["val_market_ask_input"] = float(ask_f)

                st.session_state["val_mkt_fetch_meta"] = meta
                st.success(f"Fetched {val_venue} bid/ask.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

    meta = st.session_state.get("val_mkt_fetch_meta")

    colBid, colAsk = st.columns(2)
    with colBid:
        val_bid = st.number_input("Best bid (Sell) for Team A (0–1)", min_value=0.01, max_value=0.99,
                                  value=float(st.session_state.get("val_market_bid", 0.48)),
                                  step=0.01, format="%.2f", key="val_market_bid_input")
    with colAsk:
        val_ask = st.number_input("Best ask (Buy) for Team A (0–1)", min_value=0.01, max_value=0.99,
                                  value=float(st.session_state.get("val_market_ask", 0.52)),
                                  step=0.01, format="%.2f", key="val_market_ask_input")

    bid = float(val_bid)
    ask = float(val_ask)
    if ask < bid:
        st.warning("Ask < bid (inverted). Using ask = bid for calculations.")
        ask = bid

    st.session_state["val_market_bid"] = float(val_bid)
    st.session_state["val_market_ask"] = float(val_ask)

    val_mid = 0.5 * (bid + ask)
    val_spread = ask - bid
    val_rel_spread = (val_spread / val_mid) if val_mid > 0 else 0.0

    st.markdown("### Model knobs (Valorant MVP)")
    kb1, kb2, kb3 = st.columns(3)
    with kb1:
        v_beta_score = st.slider("β score", 0.05, 0.50, float(st.session_state.get("v_beta_score", 0.18)), 0.01, key="v_beta_score")
    with kb2:
        v_beta_eco = st.slider("β eco (state)", 0.00, 0.60, float(st.session_state.get("v_beta_eco", 0.20)), 0.01, key="v_beta_eco")
    with kb3:
        v_beta_pistol = st.slider("β pistol", 0.00, 0.60, float(st.session_state.get("v_beta_pistol", 0.28)), 0.01, key="v_beta_pistol")

    kb4, kb5, kb6 = st.columns(3)
    with kb4:
        v_beta_ults = st.slider("β ults", 0.00, 0.15, float(st.session_state.get("v_beta_ults", 0.06)), 0.01, key="v_beta_ults")
    with kb5:
        v_beta_op = st.slider("β operator", 0.00, 0.20, float(st.session_state.get("v_beta_op", 0.08)), 0.01, key="v_beta_op")
    with kb6:
        v_beta_side = st.slider("β side", 0.00, 0.20, float(st.session_state.get("v_beta_side", 0.06)), 0.01, key="v_beta_side")

    # Closeout lock knobs (reduce swinginess near map closeout)
    kl1, kl2, kl3 = st.columns(3)
    with kl1:
        v_beta_lock = st.slider("β lock", 0.00, 3.00, float(st.session_state.get("v_beta_lock", 0.90)), 0.05, key="v_beta_lock")
    with kl2:
        v_lock_start_offset = st.slider("Lock starts (win_target - N)", 1, 7, int(st.session_state.get("v_lock_start_offset", 3)), 1, key="v_lock_start_offset")
    with kl3:
        v_lock_ramp = st.slider("Lock ramp (rounds)", 1, 8, int(st.session_state.get("v_lock_ramp", 3)), 1, key="v_lock_ramp")


    # ---- Compute fair probability (map) + K-based credible bands (map) ----
    p_hat_map = estimate_inplay_prob_valorant(
        p0=float(p0_map),
        rounds_a=int(val_rounds_a),
        rounds_b=int(val_rounds_b),
        eco_a_level=str(val_eco_a),
        eco_b_level=str(val_eco_b),
        pistol_a=bool(val_pistol_a),
        pistol_b=bool(val_pistol_b),
        ults_diff=int(val_ults_diff),
        op_diff=int(val_op_diff),
        a_on_defense=a_on_def,
        beta_score=float(v_beta_score),
        beta_eco=float(v_beta_eco),
        beta_pistol=float(v_beta_pistol),
        beta_ults=float(v_beta_ults),
        beta_op=float(v_beta_op),
        beta_side=float(v_beta_side),
        beta_lock=float(v_beta_lock),
        lock_start_offset=int(v_lock_start_offset),
        lock_ramp=int(v_lock_ramp),
        win_target=13,
    )

    stream = update_round_stream("val_inplay", int(val_rounds_a), int(val_rounds_b))

    kappa_map = compute_kappa_valorant(
        p0=float(p0_map),
        rounds_a=int(val_rounds_a),
        rounds_b=int(val_rounds_b),

        op_diff=int(val_op_diff),
        a_on_defense=a_on_def,
        limited_context=bool(limited_context),
        pistol_a=bool(val_pistol_a),
        pistol_b=bool(val_pistol_b),
        streak_len=int(stream.get("streak_len", 0)),
        streak_winner=stream.get("streak_winner", None),
        reversal=bool(stream.get("reversal", False)),
        gap_delta=int(stream.get("gap_delta", 0)),
        chaos_boost=float(val_chaos),
        total_rounds=24,
    )
    calib = load_kappa_calibration()
    total_r = int(int(val_rounds_a) + int(val_rounds_b))
    # Valorant OT: treat 12-12+ as OT-ish; keep simple
    is_ot = (int(val_rounds_a) >= 12 and int(val_rounds_b) >= 12)
    mult = get_kappa_multiplier(calib, "valorant", float(val_band_level), total_r, is_ot) if bool(st.session_state.get("val_use_calib", False)) else 1.0
    kappa_map = float(kappa_map) * float(val_k_scale) * float(mult)
    lo_map, hi_map = beta_credible_interval(float(p_hat_map), float(kappa_map), level=float(val_band_level))

    if val_contract_scope == "Series winner" and val_n_maps > 1:
        p_hat = series_win_prob_live(int(val_n_maps), int(val_maps_a_won), int(val_maps_b_won), float(p_hat_map), float(p0_map))
        lo = series_win_prob_live(int(val_n_maps), int(val_maps_a_won), int(val_maps_b_won), float(lo_map), float(p0_map))
        hi = series_win_prob_live(int(val_n_maps), int(val_maps_a_won), int(val_maps_b_won), float(hi_map), float(p0_map))
        line_label = "Series fair p(A)"
    else:
        p_hat = float(p_hat_map)
        lo = float(lo_map)
        hi = float(hi_map)
        line_label = "Map fair p(A)"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(line_label, f"{p_hat*100:.1f}%")
    with m2:
        st.metric("Certainty band (K CI)", f"[{lo*100:.1f}%, {hi*100:.1f}%]")
    with m3:
        st.metric("Market bid / ask", f"{bid*100:.1f}% / {ask*100:.1f}%")
    with m4:
        st.metric("Spread (abs / rel)", f"{val_spread*100:.1f} pp / {val_rel_spread*100:.1f}%")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Deviation (mid - fair)", f"{(val_mid - p_hat)*100:+.1f} pp")
    with d2:
        st.metric("Fair - Ask (buy edge)", f"{(p_hat - ask)*100:+.1f} pp")
    with d3:
        st.metric("Bid - Fair (sell edge)", f"{(bid - p_hat)*100:+.1f} pp")

    with st.expander("Show underlying MAP fair (debug)"):
        st.write({
            "p0_map": float(p0_map),
            "p_hat_map": float(p_hat_map),
            "band_map": [float(lo_map), float(hi_map)],
            "series_fmt": str(val_series_fmt),
            "maps_a_won": int(val_maps_a_won),
            "maps_b_won": int(val_maps_b_won),
            "eco_a": str(val_eco_a),
            "eco_b": str(val_eco_b),
            "ults_diff": int(val_ults_diff),
            "op_diff": int(val_op_diff),
            "side": str(side_sel),
            "note": "Series conversion assumes future maps i.i.d with p_future = p0_map (MVP)."
        })

    colAdd, colClear, colExport = st.columns(3)
    with colAdd:
        if st.button("Add snapshot", key="val_add_snap"):
            total_rounds_logged = int(val_rounds_a) + int(val_rounds_b)
            prev_total = st.session_state.get("val_inplay_last_total_rounds")
            gap_rounds = 0
            if prev_total is not None:
                expected_total = int(prev_total) + 1
                if total_rounds_logged > expected_total:
                    gap_rounds = int(total_rounds_logged - expected_total)
                elif total_rounds_logged < expected_total:
                    gap_rounds = int(expected_total - total_rounds_logged)
            st.session_state["val_inplay_last_total_rounds"] = int(total_rounds_logged)
            if gap_rounds > 0 and prev_total is not None:
                st.warning(f"Gap detected: expected total rounds {int(prev_total)+1}, got {total_rounds_logged} (gap {gap_rounds}).")
            map_index = int(st.session_state.get("val_inplay_map_index", 0))

            _snap_ts = datetime.now().isoformat(timespec="seconds")
            st.session_state["val_rows"].append({
                "t": len(st.session_state["val_rows"]),
                "snapshot_ts": _snap_ts,
                "series_fmt": str(val_series_fmt),
                "contract_scope": str(val_contract_scope),
                "maps_a_won": int(val_maps_a_won),
                "maps_b_won": int(val_maps_b_won),
                "rounds_a": int(val_rounds_a),
                "rounds_b": int(val_rounds_b),
                        "map_index": int(map_index),
                        "total_rounds": int(total_rounds_logged),
                        "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                        "gap_rounds": int(gap_rounds),
                        "gap_flag": bool(gap_rounds > 0),
                        "gap_reason": "",
                "map_index": int(map_index),
                "total_rounds": int(total_rounds_logged),
                "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                "gap_rounds": int(gap_rounds),
                "eco_a": str(val_eco_a),
                "eco_b": str(val_eco_b),
                "ults_diff": int(val_ults_diff),
                "op_diff": int(val_op_diff),
                "side": str(side_sel),
                "p0_series": float(p0_series),
                "p0_map": float(p0_map),
                "p_hat_map": float(p_hat_map),
                "p_hat": float(p_hat),
                "band_lo": float(lo),
                "band_hi": float(hi),
                "band_lo_map": float(lo_map),
                "band_hi_map": float(hi_map),
                "market_bid": float(bid),
                "market_ask": float(ask),
                "market_mid": float(val_mid),
                "spread": float(val_spread),
                "rel_spread": float(val_rel_spread),
                "dev_mid_pp": float((val_mid - p_hat)*100.0),
                "buy_edge_pp": float((p_hat - ask)*100.0),
                "sell_edge_pp": float((bid - p_hat)*100.0),
            })

            # Optional: persist this snapshot to CSV for K calibration
            if bool(st.session_state.get("val_inplay_persist", False)) and str(st.session_state.get("val_inplay_match_id", "")).strip():
                try:
                    persist_inplay_row({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "game": "VALORANT",
                        "match_id": str(st.session_state.get("val_inplay_match_id")).strip(),
                        "contract_scope": str(val_contract_scope),
                        "series_format": str(val_series_fmt),
                        "maps_a_won": int(val_maps_a_won),
                        "maps_b_won": int(val_maps_b_won),
                        "team_a": str(val_team_a),
                        "team_b": str(val_team_b),
                        "map_name": str(st.session_state.get("val_map_name", "")),
                        "a_side_now": str(side_sel),
                        "rounds_a": int(val_rounds_a),
                        "rounds_b": int(val_rounds_b),
                        "map_index": int(map_index),
                        "total_rounds": int(total_rounds_logged),
                        "prev_total_rounds": int(prev_total) if prev_total is not None else "",
                        "gap_rounds": int(gap_rounds),
                        "gap_flag": bool(gap_rounds > 0) if prev_total is not None else False,
                        "gap_reason": "",
                        "buy_state_a": str(val_eco_a),
                        "buy_state_b": str(val_eco_b),
                        "econ_missing": False,
                        "econ_fragile": False,
                        "pistol_a": bool(val_pistol_a),
                        "pistol_b": bool(val_pistol_b),
                        "gap_delta": int(stream.get("gap_delta", 0)) if 'stream' in locals() else 0,
                        "streak_winner": stream.get("streak_winner", "") if 'stream' in locals() else "",
                        "streak_len": int(stream.get("streak_len", 0)) if 'stream' in locals() else 0,
                        "reversal": bool(stream.get("reversal", False)) if 'stream' in locals() else False,
                        "n_tracked": int(stream.get("n_tracked", 0)) if 'stream' in locals() else 0,
                        "p0_map": float(p0_map),
                        "p_fair_map": float(p_hat_map),
                        "kappa_map": float(kappa_map) if 'kappa_map' in locals() else "",
                        "p_fair": float(p_hat),
                        "band_level": float(st.session_state.get("val_inplay_band_level", 0.80)),
                        "band_lo": float(lo),
                        "band_hi": float(hi),
                        "bid": float(bid),
                        "ask": float(ask),
                        "mid": float(val_mid),
                        "spread_abs": float(val_spread),
                        "spread_rel": float(val_rel_spread),
                        "notes": str(st.session_state.get("val_inplay_notes", "")),
                        "snapshot_idx": int(total_rounds_logged),
                        "half": ("1" if int(total_rounds_logged) <= 12 else ("2" if int(total_rounds_logged) <= 24 else "OT")),
                        "round_in_half": (int(total_rounds_logged) if int(total_rounds_logged) <= 12 else (int(total_rounds_logged)-12 if int(total_rounds_logged) <= 24 else int(total_rounds_logged)-24)),
                        "is_ot": bool(int(total_rounds_logged) > 24),
                        "round_in_map": int(total_rounds_logged),
                    })
                except Exception as e:
                    st.warning(f"Snapshot not persisted: {e}")
            # Incremental backtest update so In-Play Backtest tab chart updates without full run
            _row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "match_id": str(st.session_state.get("val_inplay_match_id", "")).strip(),
                "contract_scope": str(val_contract_scope),
                "bid": float(bid),
                "ask": float(ask),
                "mid": float(val_mid),
                "spread_abs": float(val_spread),
                "p_fair": float(p_hat),
                "band_lo": float(lo),
                "band_hi": float(hi),
            }
            _run_inplay_incremental_and_refresh_session(_row)

def _val_start_next_half():
    # Half-time resets (pistol + econ context), keep score
    cur_side = str(st.session_state.get("val_side", "Unknown"))
    if cur_side == "Attack":
        st.session_state["val_side"] = "Defense"
        st.session_state["val_side_idx"] = 2
    elif cur_side == "Defense":
        st.session_state["val_side"] = "Attack"
        st.session_state["val_side_idx"] = 1

    # Reset pistol context
    st.session_state["val_pistol_a"] = False
    st.session_state["val_pistol_b"] = False

    # Default buy states for half start
    st.session_state["val_eco_a"] = "Light"
    st.session_state["val_eco_b"] = "Light"
    st.session_state["val_eco_a_idx"] = 1
    st.session_state["val_eco_b_idx"] = 1

    # Reset half-sensitive context
    st.session_state["val_ults_diff"] = 0
    st.session_state["val_op_diff"] = 0

def _val_start_next_map():

    # If operator is persisting, log the *completed* map winner before resetting for the next map.
    try:
        if bool(st.session_state.get("val_inplay_persist", False)):
            _mid = str(st.session_state.get("val_inplay_match_id", "")).strip()
            if _mid:
                cur_map_index = int(st.session_state.get("val_inplay_map_index", 0))
                cur_map_name = str(st.session_state.get("val_map_name", ""))  # may be blank if not in UI
                team_a = str(st.session_state.get("val_team_a", "")).strip()
                team_b = str(st.session_state.get("val_team_b", "")).strip()

                prev_a = int(st.session_state.get("val_prev_maps_a_won", 0))
                prev_b = int(st.session_state.get("val_prev_maps_b_won", 0))
                now_a = int(st.session_state.get("val_maps_a_won", 0))
                now_b = int(st.session_state.get("val_maps_b_won", 0))

                winner = ""
                if now_a == prev_a + 1 and now_b == prev_b:
                    winner = team_a
                elif now_b == prev_b + 1 and now_a == prev_a:
                    winner = team_b

                if winner and team_a and team_b:
                    persist_inplay_map_result(_mid, "VALORANT", cur_map_index, cur_map_name, team_a, team_b, winner)

                st.session_state["val_prev_maps_a_won"] = now_a
                st.session_state["val_prev_maps_b_won"] = now_b
    except Exception:
        pass

    # Keep val_rows so the BO3 chart continues across maps

    # Reset per-map operator inputs
    st.session_state["val_rounds_a"] = 0
    st.session_state["val_rounds_b"] = 0
    st.session_state["val_pistol_a"] = False
    st.session_state["val_pistol_b"] = False
    st.session_state["val_eco_a"] = "Light"
    st.session_state["val_eco_b"] = "Light"
    st.session_state["val_eco_a_idx"] = 1
    st.session_state["val_eco_b_idx"] = 1
    st.session_state["val_ults_diff"] = 0
    st.session_state["val_op_diff"] = 0
    st.session_state["val_side"] = "Unknown"
    st.session_state["val_side_idx"] = 0
    st.session_state["val_chaos"] = 0.00

    st.session_state["val_inplay_map_index"] = int(st.session_state.get("val_inplay_map_index", 0)) + 1
    st.session_state["val_inplay_last_total_rounds"] = None


with colClear:
    cA, cB, cC = st.columns(3)
    with cA:
        if st.button("Clear chart", key="val_clear_chart"):
            st.session_state["val_rows"] = []
    with cB:
        st.button("Start next half", key="val_next_half", on_click=_val_start_next_half)
    with cC:
        st.button("Start next map", key="val_next_map", on_click=_val_start_next_map)
    with colExport:
        if st.button("Export snapshots to Diagnostics", key="val_export_snaps"):
            st.session_state["val_live_export_df"] = pd.DataFrame(st.session_state["val_rows"])

    st.markdown("### Chart")
    pcal = load_p_calibration_json(APP_DIR)
    show_pcal = st.checkbox("Show p_calibrated overlay", value=False, key="val_show_pcal",
                            help="Overlays calibrated win prob from p_calibration.json (for inspection; not used for decisions unless you wire it in).")

    if len(st.session_state["val_rows"]) > 0:
        chart_df = pd.DataFrame(st.session_state["val_rows"])
        plot_df = chart_df[["t","p_hat","band_lo","band_hi","market_mid"]].copy()
        if show_pcal and pcal:
            plot_df["p_hat_cal"] = plot_df["p_hat"].apply(lambda x: apply_p_calibration(x, pcal, "valorant"))
        try:
            import plotly.graph_objects as go
            fig_val = go.Figure()
            t_vals = plot_df["t"].astype(int).tolist()
            for col in ["market_mid", "p_hat", "band_lo", "band_hi"]:
                if col in plot_df.columns:
                    fig_val.add_trace(go.Scatter(x=t_vals, y=plot_df[col].tolist(), name=col, mode="lines"))
            if show_pcal and pcal and "p_hat_cal" in plot_df.columns:
                fig_val.add_trace(go.Scatter(x=t_vals, y=plot_df["p_hat_cal"].tolist(), name="p_hat_cal", mode="lines"))
            val_match_id = str(st.session_state.get("val_inplay_match_id", "")).strip()
            bt_trades = st.session_state.get("inplay_bt_trades") or {}
            has_ts = "snapshot_ts" in chart_df.columns and chart_df["snapshot_ts"].notna().any()
            _strategy_styles_val = {
                "S1_mr_to_entry_fair": {"color": "#22c55e", "sym_l": "triangle-up", "sym_s": "triangle-down"},
                "S2_mr_to_current_fair": {"color": "#3b82f6", "sym_l": "diamond", "sym_s": "diamond"},
                "S3_mr_inside_band": {"color": "#f97316", "sym_l": "square", "sym_s": "square"},
            }
            if val_match_id and has_ts:
                chart_ts = pd.to_datetime(chart_df["snapshot_ts"], errors="coerce")
                # Completed trades from CSV (no backtest run needed if already loaded)
                if bt_trades:
                    for sid, trades_df in bt_trades.items():
                        if trades_df is None or len(trades_df) == 0:
                            continue
                        match_trades = trades_df[trades_df["match_id"].astype(str) == val_match_id]
                        if len(match_trades) == 0:
                            continue
                        style = _strategy_styles_val.get(sid, {"color": "#6b7280", "sym_l": "triangle-up", "sym_s": "triangle-down"})
                        t_entry, y_entry, t_exit, y_exit = [], [], [], []
                        exit_reasons, exit_pnls = [], []
                        for _, tr in match_trades.iterrows():
                            et = pd.to_datetime(tr.get("entry_ts"), errors="coerce")
                            ex = pd.to_datetime(tr.get("exit_ts"), errors="coerce")
                            if pd.notna(et):
                                idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - et.value))
                                t_entry.append(chart_df["t"].iloc[idx])
                                y_entry.append(tr.get("entry_mid", np.nan))
                            if pd.notna(ex):
                                idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - ex.value))
                                t_exit.append(chart_df["t"].iloc[idx])
                                y_exit.append(tr.get("exit_px", np.nan))
                                exit_reasons.append(str(tr.get("exit_reason", "")))
                                exit_pnls.append(tr.get("pnl_$", np.nan))
                        if t_entry:
                            sides = match_trades["side"]
                            long_m = (sides == "LONG").tolist()
                            short_m = (sides == "SHORT").tolist()
                            n = len(t_entry)
                            if any(long_m[i] for i in range(min(n, len(long_m)))):
                                te = [t_entry[i] for i in range(n) if i < len(long_m) and long_m[i]]
                                ye = [y_entry[i] for i in range(n) if i < len(long_m) and long_m[i]]
                                if te:
                                    fig_val.add_trace(go.Scatter(x=te, y=ye, name=f"{sid} Entry LONG", mode="markers", marker=dict(symbol=style["sym_l"], size=12, color=style["color"])))
                            if any(short_m[i] for i in range(min(n, len(short_m)))):
                                te = [t_entry[i] for i in range(n) if i < len(short_m) and short_m[i]]
                                ye = [y_entry[i] for i in range(n) if i < len(short_m) and short_m[i]]
                                if te:
                                    fig_val.add_trace(go.Scatter(x=te, y=ye, name=f"{sid} Entry SHORT", mode="markers", marker=dict(symbol=style["sym_s"], size=12, color=style["color"])))
                        if t_exit:
                            customdata = np.column_stack((exit_reasons, exit_pnls))
                            fig_val.add_trace(go.Scatter(
                                x=t_exit, y=y_exit, name=f"{sid} Exit", mode="markers",
                                marker=dict(symbol="circle", size=10, color=style["color"], line=dict(width=1, color="white")),
                                customdata=customdata,
                                hovertemplate="t %{x}<br>exit_reason: %{customdata[0]}<br>pnl_$ %{customdata[1]:.2f}<extra></extra>",
                            ))
                # Open positions from state (show entry marker without running backtest)
                bt_state = st.session_state.get("inplay_bt_state")
                if bt_state is None:
                    try:
                        _state_path = PROJECT_ROOT / "logs" / "inplay_backtest_state.json"
                        if _state_path.exists():
                            bt_state = load_backtest_state(_state_path)
                    except Exception:
                        pass
                if bt_state:
                    for sid, s in (bt_state.get("strategies") or {}).items():
                        pos = s.get("position")
                        if not pos or str(pos.get("match_id")) != val_match_id:
                            continue
                        style = _strategy_styles_val.get(sid, {"color": "#6b7280", "sym_l": "triangle-up", "sym_s": "triangle-down"})
                        et = pd.to_datetime(pos.get("entry_ts"), errors="coerce")
                        if pd.notna(et):
                            idx = np.nanargmin(np.abs(chart_ts.astype(np.int64) - et.value))
                            t_open = chart_df["t"].iloc[idx]
                            y_open = pos.get("entry_mid", np.nan)
                            side = (pos.get("side") or "").upper()
                            if side == "LONG":
                                fig_val.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry LONG (open)", mode="markers", marker=dict(symbol=style["sym_l"], size=12, color=style["color"])))
                            elif side == "SHORT":
                                fig_val.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry SHORT (open)", mode="markers", marker=dict(symbol=style["sym_s"], size=12, color=style["color"])))
            fig_val.update_layout(title="Chart", xaxis_title="t (snapshot index)", yaxis_title="Price / Fair", height=420)
            st.plotly_chart(fig_val, use_container_width=True)
        except ImportError:
            st.line_chart(plot_df.set_index("t"), use_container_width=True, height=420)
        except Exception:
            st.line_chart(plot_df.set_index("t"), use_container_width=True, height=420)
        st.dataframe(chart_df, use_container_width=True)
    else:
        st.info("Add at least one snapshot to see the chart.")

    # --- Final result logging (needed for probability calibration + backtest match-end) ---
    st.markdown("### Log final result (for ML)")
    st.caption("Also used for **In-Play Backtest** match-end settlement: if a position is still open when the match ends, it closes at 1 (Team A wins) or 0 (Team B wins) using the winner you save here.")
    rcol1, rcol2 = st.columns([2.0, 1.0])
    with rcol1:
        _val_default_winner_idx = 0
        try:
            if int(val_rounds_a) >= 13 and int(val_rounds_a) > int(val_rounds_b):
                _val_default_winner_idx = 0
            elif int(val_rounds_b) >= 13 and int(val_rounds_b) > int(val_rounds_a):
                _val_default_winner_idx = 1
        except Exception:
            pass
        val_winner_sel = st.selectbox("Winner", [val_team_a, val_team_b], index=_val_default_winner_idx, key="val_inplay_winner_sel")
    with rcol2:
        if st.button("Save result", key="val_inplay_save_result"):
            _mid = str(st.session_state.get("val_inplay_match_id", "")).strip()
            if not _mid:
                st.error("Set a Match ID above before saving a result.")
            else:
                persist_inplay_result(_mid, "VALORANT", str(val_team_a), str(val_team_b), str(val_winner_sel))
                st.success("Result saved to inplay_match_results_clean.csv")




with tabs[5]:
    st.header("Calibration")
    st.caption("Dashboards for (1) Kappa band coverage vs market and (2) p_fair outcome calibration. "
               "Training runs write JSON reports to disk; this tab reads and visualizes them.")

    st.markdown("## 1) Kappa band calibration (containment vs market mid)")
    kc1, kc2, kc3 = st.columns([1.2, 1.0, 2.0])
    with kc1:
        if st.button("Run kappa calibration trainer", key="calib_run_kappa"):
            ok, out = run_kappa_trainer()
            st.session_state["calib_kappa_out"] = out
            if ok:
                st.success("Kappa calibration completed.")
            else:
                st.error("Kappa calibration failed.")
    with kc2:
        if st.button("Refresh kappa report", key="calib_refresh_kappa"):
            st.rerun()
    with kc3:
        if st.session_state.get("calib_kappa_out"):
            st.code(str(st.session_state.get("calib_kappa_out"))[:2000])

    kappa_report_path = PROJECT_ROOT / "config" / "kappa_calibration_report.json"
    if kappa_report_path.exists():
        try:
            krep = json.loads(kappa_report_path.read_text())
            st.caption(f"Kappa report updated: {krep.get('ran_at','unknown')}")
            games = krep.get('games', {})
            if games:
                for game_key, g in games.items():
                    st.subheader(f"{game_key.upper()} — Kappa bands")
                    st.write(f"Matches: **{g.get('match_id',{}).get('unique',0)}** | Rows: **{g.get('rows_usable',0)}**")
                    st.write("Buckets:", g.get('buckets',{}))
            with st.expander("Raw kappa calibration report JSON"):
                st.json(krep)
        except Exception as e:
            st.warning(f"Could not read kappa report: {e}")
    else:
        st.info("No kappa report found yet. Run the trainer above or from the live tabs.")

    st.markdown("## 2) Probability calibration (p_fair vs series outcomes)")
    pc1, pc2, pc3 = st.columns([1.2, 1.0, 2.0])
    with pc1:
        if st.button("Run probability calibration trainer", key="calib_run_prob"):
            ok, out = run_prob_trainer()
            st.session_state["calib_prob_out"] = out
            if ok:
                st.success("Probability calibration completed.")
            else:
                st.error("Probability calibration failed.")
    with pc2:
        if st.button("Refresh probability report", key="calib_refresh_prob"):
            st.rerun()
    with pc3:
        if st.session_state.get("calib_prob_out"):
            st.code(str(st.session_state.get("calib_prob_out"))[:2000])

    if P_CALIB_REPORT_PATH.exists():
        try:
            prep = json.loads(P_CALIB_REPORT_PATH.read_text())
            st.caption(f"Probability report updated: {prep.get('ran_at','unknown')}")
            pgames = prep.get('games', {})
            if pgames:
                for game_key, g in pgames.items():
                    st.subheader(f"{game_key.upper()} — Probability calibration")
                    lvl = g.get('level', {})
                    matches = int(g.get('matches',0))
                    target = int(lvl.get('target',0) or 0)
                    prog = float(lvl.get('progress',0.0) or 0.0)
                    st.write(f"Matches: **{matches}** | Rows: **{g.get('rows',0)}**")
                    if target > 0:
                        st.progress(min(max(prog,0.0),1.0), text=f"{matches}/{target} ({prog*100:.1f}%) — {lvl.get('name','')}")
                    mb = g.get('metrics_before', {})
                    ma = g.get('metrics_after', {})
                    cA,cB,cC = st.columns(3)
                    cA.metric("ECE", f"{mb.get('ece',0):.4f} → {ma.get('ece',0):.4f}")
                    cB.metric("Brier", f"{mb.get('brier',0):.4f} → {ma.get('brier',0):.4f}")
                    cC.metric("LogLoss", f"{mb.get('logloss',0):.4f} → {ma.get('logloss',0):.4f}")
            with st.expander("Raw probability calibration report JSON"):
                st.json(prep)
        except Exception as e:
            st.warning(f"Could not read probability report: {e}")
    else:
        st.info("No probability report found yet. Requires inplay_match_results_clean.csv with winners logged.")


# --------------------------
# In-Play Backtest (P/L Tracker) TAB — additive only
# --------------------------
with tabs[6]:
    st.header("In-Play Backtest (P/L Tracker)")
    st.caption(
        "Run the backtest script once to seed state; then **Add snapshot** on CS2/Valorant tabs updates incrementally (no full run). "
        "Results are written to logs; use **Run Backtest** to re-seed or re-sync."
    )
    with st.expander("How do trade markers get on the chart?", expanded=False):
        st.markdown(
            "Trade markers come from **Run Backtest** (full run) or **Add snapshot** (incremental). "
            "Click **Run Backtest** once to seed; each **Add snapshot** on CS2/Valorant updates state and appends new trades without re-running the full backtest. "
            "Select a **Strategy** and **Match ID** below to see the price series and entry/exit markers. "
            "**Match-end settlement:** If a position is still open when the match ends, it closes at 1 (Team A wins) or 0 (Team B wins) using the winner you log via **Winner** → **Save result** (inplay_match_results_clean.csv)."
        )

    default_inplay = str(INPLAY_LOG_PATH) if INPLAY_LOG_PATH else "logs/inplay_kappa_logs_clean.csv"
    default_config = str(PROJECT_ROOT / "configs" / "inplay_strategies.json")
    default_outdir = str(PROJECT_ROOT / "logs")

    inplay_path = st.text_input("In-play log CSV path", value=default_inplay, key="inplay_bt_inplay_path")
    config_path = st.text_input("Config path", value=default_config, key="inplay_bt_config_path")
    outdir_path = st.text_input("Output directory", value=default_outdir, key="inplay_bt_outdir")

    run_clicked = st.button("Run Backtest", key="inplay_bt_run")
    paths_key = (inplay_path.strip(), config_path.strip(), outdir_path.strip())

    if run_clicked:
        script_path = PROJECT_ROOT / "scripts" / "inplay_backtest_runner.py"
        if not script_path.exists():
            st.error(f"Backtest script not found: {script_path}")
        else:
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), "--inplay", inplay_path.strip(), "--config", config_path.strip(), "--outdir", outdir_path.strip()],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                st.success("Backtest completed.")
                st.session_state["inplay_bt_last_run"] = datetime.now().isoformat()
                st.session_state["inplay_bt_paths"] = paths_key
                st.session_state["inplay_bt_result_stdout"] = result.stdout
                # State will be loaded and repaired in the cached_paths block below
            except subprocess.CalledProcessError as e:
                st.error("Backtest script failed.")
                st.code(e.stderr or e.stdout or str(e))
            except Exception as e:
                st.error(str(e))

    # Load cached results only if paths match
    cached_paths = st.session_state.get("inplay_bt_paths")
    summary_loaded = None
    trades_by_strategy = {}
    config_loaded = None
    inplay_df_loaded = None

    if cached_paths == paths_key:
        summary_path = Path(outdir_path.strip()) / "inplay_backtest_summary.json"
        state_path = Path(outdir_path.strip()) / "inplay_backtest_state.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_loaded = json.load(f)
                st.session_state["inplay_bt_summary"] = summary_loaded
            except Exception:
                summary_loaded = st.session_state.get("inplay_bt_summary")
        else:
            summary_loaded = st.session_state.get("inplay_bt_summary")
        if state_path.exists():
            try:
                loaded_state = load_backtest_state(state_path)
                st.session_state["inplay_bt_state"] = loaded_state
            except Exception:
                pass
        for sid in (summary_loaded or {}).keys():
            trades_path = Path(outdir_path.strip()) / f"inplay_backtest_trades__{sid}.csv"
            if trades_path.exists():
                try:
                    trades_by_strategy[sid] = pd.read_csv(trades_path)
                except Exception:
                    pass
        st.session_state["inplay_bt_trades"] = trades_by_strategy
        # Repair state: clear any "open position" for which we have a completed trade for that match
        if trades_by_strategy and st.session_state.get("inplay_bt_state"):
            repaired = repair_state_from_trades(st.session_state["inplay_bt_state"], trades_by_strategy)
            if repaired != st.session_state["inplay_bt_state"]:
                try:
                    save_backtest_state(repaired, state_path)
                    st.session_state["inplay_bt_state"] = repaired
                except Exception:
                    st.session_state["inplay_bt_state"] = repaired

    if run_clicked and cached_paths == paths_key:
        st.session_state["inplay_bt_summary"] = summary_loaded
        st.session_state["inplay_bt_trades"] = trades_by_strategy

    # Only use cached summary/trades when paths match current inputs
    if cached_paths == paths_key:
        summary_loaded = st.session_state.get("inplay_bt_summary")
        trades_by_strategy = st.session_state.get("inplay_bt_trades") or {}
    else:
        summary_loaded = None
        trades_by_strategy = {}

    if cached_paths == paths_key and not summary_loaded and not trades_by_strategy:
        st.info("Run **Run Backtest** once to seed state and trades; then **Add snapshot** on CS2/Valorant tabs will update incrementally.")

    # Load config for strategy list and scope filter
    _cfg = (config_path or "").strip()
    config_path_resolved = Path(_cfg) if _cfg else PROJECT_ROOT / "configs" / "inplay_strategies.json"
    if not config_path_resolved.is_absolute():
        config_path_resolved = (PROJECT_ROOT / _cfg).resolve()
    if config_path_resolved.exists():
        try:
            with open(config_path_resolved, "r", encoding="utf-8") as f:
                config_loaded = json.load(f)
        except Exception:
            pass

    strategy_ids = [s["id"] for s in (config_loaded or {}).get("strategies", [])]
    if not strategy_ids and summary_loaded:
        strategy_ids = list(summary_loaded.keys())

    # KPI table
    if summary_loaded:
        st.subheader("Summary by strategy")
        rows = []
        for sid, kpis in summary_loaded.items():
            rows.append({
                "strategy_id": sid,
                "trades": kpis.get("trades", 0),
                "total_pnl_$": round(kpis.get("total_pnl_$", 0), 2),
                "end_bankroll": round(kpis.get("end_bankroll", 0), 2),
                "total_return_%": round(kpis.get("total_return_pct", 0) * 100, 2),
                "win_rate": round(kpis.get("win_rate", 0) * 100, 1),
                "avg_pnl_$": round(kpis.get("avg_pnl_$", 0), 2),
                "max_drawdown_%": round(abs(kpis.get("max_drawdown_pct", 0)) * 100, 2),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Load inplay CSV for match list and chart series (scope-filtered)
    _inp = (inplay_path or "").strip()
    inplay_path_resolved = Path(_inp) if _inp else Path(default_inplay)
    if not inplay_path_resolved.is_absolute():
        inplay_path_resolved = (PROJECT_ROOT / _inp).resolve()
    if inplay_path_resolved.exists() and config_loaded:
        try:
            inplay_df_loaded = pd.read_csv(inplay_path_resolved)
            scope_col = config_loaded.get("data", {}).get("scope_col", "contract_scope")
            allowlist = config_loaded.get("filters", {}).get("scope_allowlist") or []
            if allowlist and scope_col in inplay_df_loaded.columns:
                inplay_df_loaded = inplay_df_loaded[inplay_df_loaded[scope_col].astype(str).str.strip().isin(allowlist)]
            st.session_state["inplay_bt_inplay_df"] = inplay_df_loaded
        except Exception:
            inplay_df_loaded = st.session_state.get("inplay_bt_inplay_df")
    else:
        inplay_df_loaded = st.session_state.get("inplay_bt_inplay_df")

    stream_id_col = (config_loaded or {}).get("data", {}).get("stream_id_col", "match_id")
    ts_col = (config_loaded or {}).get("data", {}).get("timestamp_col", "timestamp")
    mid_col = (config_loaded or {}).get("data", {}).get("mid_col", "mid")
    fair_col = (config_loaded or {}).get("data", {}).get("fair_col", "p_fair")
    band_lo_col = (config_loaded or {}).get("data", {}).get("band_lo_col", "band_lo")
    band_hi_col = (config_loaded or {}).get("data", {}).get("band_hi_col", "band_hi")

    sel_strategy = st.selectbox("Strategy", strategy_ids or ["(none)"], key="inplay_bt_sel_strategy")
    match_ids = []
    if inplay_df_loaded is not None and stream_id_col in inplay_df_loaded.columns:
        match_ids = sorted(inplay_df_loaded[stream_id_col].dropna().astype(str).unique().tolist())
    sel_match = st.selectbox("Match ID", [""] + match_ids, key="inplay_bt_sel_match")

    # Strategy-specific colors and shapes for chart markers (differentiate strategies)
    _strategy_styles = {
        "S1_mr_to_entry_fair": {"color": "#22c55e", "symbol_long": "triangle-up", "symbol_short": "triangle-down"},
        "S2_mr_to_current_fair": {"color": "#3b82f6", "symbol_long": "diamond", "symbol_short": "diamond"},
        "S3_mr_inside_band": {"color": "#f97316", "symbol_long": "square", "symbol_short": "square"},
    }

    # Chart: time series + entry/exit markers
    if sel_strategy and sel_strategy != "(none)" and sel_match and inplay_df_loaded is not None:
        match_df = inplay_df_loaded[inplay_df_loaded[stream_id_col].astype(str) == sel_match].copy()
        match_df[ts_col] = pd.to_datetime(match_df[ts_col], errors="coerce")
        match_df = match_df.dropna(subset=[ts_col]).sort_values(ts_col)
        style = _strategy_styles.get(sel_strategy, {"color": "#6b7280", "symbol_long": "triangle-up", "symbol_short": "triangle-down"})

        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            if len(match_df):
                fig.add_trace(go.Scatter(x=match_df[ts_col], y=match_df[mid_col], name="mid", mode="lines"))
                if fair_col in match_df.columns:
                    fig.add_trace(go.Scatter(x=match_df[ts_col], y=match_df[fair_col], name="p_fair", mode="lines"))
                if band_lo_col in match_df.columns:
                    fig.add_trace(go.Scatter(x=match_df[ts_col], y=match_df[band_lo_col], name="band_lo", mode="lines"))
                if band_hi_col in match_df.columns:
                    fig.add_trace(go.Scatter(x=match_df[ts_col], y=match_df[band_hi_col], name="band_hi", mode="lines"))

            trades_df = trades_by_strategy.get(sel_strategy)
            if trades_df is not None and len(trades_df):
                match_trades = trades_df[trades_df["match_id"].astype(str) == sel_match]
                if len(match_trades):
                    entry_ts = pd.to_datetime(match_trades["entry_ts"], errors="coerce")
                    entry_mid = match_trades["entry_mid"]
                    sides = match_trades["side"]
                    long_mask = sides == "LONG"
                    short_mask = sides == "SHORT"
                    if long_mask.any():
                        fig.add_trace(
                            go.Scatter(
                                x=entry_ts[long_mask],
                                y=entry_mid[long_mask],
                                name=f"{sel_strategy} Entry LONG",
                                mode="markers",
                                marker=dict(symbol=style["symbol_long"], size=12, color=style["color"]),
                            )
                        )
                    if short_mask.any():
                        fig.add_trace(
                            go.Scatter(
                                x=entry_ts[short_mask],
                                y=entry_mid[short_mask],
                                name=f"{sel_strategy} Entry SHORT",
                                mode="markers",
                                marker=dict(symbol=style["symbol_short"], size=12, color=style["color"]),
                            )
                        )
                    exit_ts = pd.to_datetime(match_trades["exit_ts"], errors="coerce")
                    exit_px = match_trades["exit_px"]
                    pnl = match_trades["pnl_$"]
                    ret_pct = match_trades["ret_pct_account"]
                    exit_reason = match_trades["exit_reason"].astype(str) if "exit_reason" in match_trades.columns else np.full(len(match_trades), "")
                    fig.add_trace(
                        go.Scatter(
                            x=exit_ts,
                            y=exit_px,
                            name=f"{sel_strategy} Exit",
                            mode="markers",
                            marker=dict(symbol="circle", size=10, color=style["color"], line=dict(width=1, color="white")),
                            customdata=np.column_stack((pnl, ret_pct, exit_reason)),
                            hovertemplate="Exit %{x}<br>pnl_$ %{customdata[0]:.2f}<br>ret %{customdata[1]:.2%}<br>%{customdata[2]}<extra></extra>",
                        )
                    )
            fig.update_layout(title=f"Match {sel_match} — {sel_strategy}", xaxis_title="Time", yaxis_title="Price / Fair")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.warning("Install plotly to see the chart: pip install plotly")
        except Exception as e:
            st.warning(f"Chart error: {e}")
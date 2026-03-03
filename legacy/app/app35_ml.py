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
from datetime import datetime, timezone, timedelta
import csv
import time
from collections import deque
import copy
from typing import Optional  # 3.9-compatible Optional[...] for type hints

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Streamlit layout
st.set_page_config(layout="wide", page_title="Esports Fair Odds")

st.markdown("""
<style>
div[data-testid="stExpander"] > details > summary {
  background: #e6ff00 !important;
  color: #000 !important;
  border-radius: 10px !important;
  padding: 10px 14px !important;
  font-weight: 800 !important;
  border: 3px solid #fff200 !important;
  box-shadow: 0 0 14px rgba(230,255,0,0.75) !important;
}
div[data-testid="stExpander"] > details[open] > summary {
  position: sticky !important;
  top: 0 !important;
  z-index: 999 !important;
}
div[data-testid="stExpander"] > details > summary svg {
  color: #000 !important;
  filter: drop-shadow(0 0 4px rgba(0,0,0,0.4)) !important;
}
div[data-testid="stExpander"] > details[open] > summary + div {
  padding-top: 6px !important;
}
</style>
""", unsafe_allow_html=True)

def _inject_ui_polish_css():
    """Inject global CSS for dark theme / trading-cockpit polish. Cosmetic only; no logic."""
    st.markdown("""
<style>
/* Base app */
.stApp {
    background-color: #0e1117;
    color: #e6edf3;
}

/* Layout width + tighter spacing */
.block-container {
    padding-top: 0.6rem;
    padding-bottom: 0.8rem;
    max-width: 96%;
}

/* Headers */
h1, h2, h3 {
    color: #f0f6fc !important;
    letter-spacing: 0.2px;
    line-height: 1.2;
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}

/* Make markdown section titles stand out a bit more */
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h2 {
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 0.2rem;
}

/* Section card wrapper (use with _card_start / _card_end or single markdown) */
.app-card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    background: rgba(255,255,255,0.015);
    padding: 0.55rem 0.75rem 0.65rem 0.75rem;
    margin-bottom: 0.55rem;
}

/* CS2 main chart panel */
.cs2-chart-panel {
    border-radius: 14px;
    border: 1px solid rgba(77,163,255,0.35);
    background: radial-gradient(circle at top left, rgba(24,32,48,0.9), rgba(8,12,20,0.96));
    box-shadow:
        0 0 0 1px rgba(0,0,0,0.55),
        0 0 18px rgba(15, 76, 129, 0.45);
    padding: 0.65rem 0.9rem 0.75rem 0.9rem;
    margin: 0.6rem 0 0.9rem 0;
}

/* General containers / expanders */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    margin-bottom: 0.35rem;
    overflow: hidden;
}

/* Streamlit "elements" spacing */
div[data-testid="element-container"] {
    margin-bottom: 0.2rem;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.35rem 0.5rem;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.02);
    color: #e6edf3;
    transition: all 0.12s ease-in-out;
}
.stButton > button:hover {
    border-color: rgba(77,163,255,0.55);
    box-shadow: 0 0 0 1px rgba(77,163,255,0.15) inset;
}

/* Inputs / selects */
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    background-color: rgba(255,255,255,0.02) !important;
}

/* Sliders */
div[data-testid="stSlider"] {
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;
}

/* Checkbox / radio spacing */
div[data-testid="stCheckbox"],
div[role="radiogroup"] {
    margin-top: 0.1rem;
    margin-bottom: 0.1rem;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.06);
    background: #0b0f14;
}

/* Info / success / warning banners */
div[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Code blocks / debug text (if shown) */
pre, code {
    border-radius: 8px !important;
}

/* Horizontal rules */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 0.5rem 0 0.7rem 0;
}

/* Neon section divider */
.app-section-divider-wrap {
    margin: 0.45rem 0 0.55rem 0;
}

.app-section-divider-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: rgba(230, 237, 243, 0.92);
    margin-bottom: 0.2rem;
    letter-spacing: 0.2px;
}

.app-section-divider-line {
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        rgba(77,163,255,0.0) 0%,
        rgba(77,163,255,0.35) 10%,
        rgba(77,163,255,0.95) 35%,
        rgba(77,163,255,0.95) 65%,
        rgba(77,163,255,0.35) 90%,
        rgba(77,163,255,0.0) 100%
    );
    box-shadow:
        0 0 4px rgba(77,163,255,0.35),
        0 0 10px rgba(77,163,255,0.18);
    opacity: 0.95;
}

/* CS2 section divider (neon bar + title) */
.cs2-section-wrap {
    margin: 0.5rem 0 0.65rem 0;
}

.cs2-section-bar {
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        rgba(77,163,255,0.0) 0%,
        rgba(77,163,255,0.3) 8%,
        rgba(77,163,255,0.9) 40%,
        rgba(77,163,255,0.9) 60%,
        rgba(77,163,255,0.3) 92%,
        rgba(77,163,255,0.0) 100%
    );
    box-shadow: 0 0 6px rgba(77,163,255,0.3), 0 0 12px rgba(77,163,255,0.12);
    border: 1px solid rgba(77,163,255,0.15);
    opacity: 0.95;
}

.cs2-section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: rgba(240, 246, 252, 0.95);
    margin-bottom: 0.15rem;
    letter-spacing: 0.15px;
}

.cs2-section-subtle {
    font-size: 0.75rem;
    color: rgba(230, 237, 243, 0.65);
    margin-bottom: 0.25rem;
    letter-spacing: 0.1px;
}

/* Reusable section bar (neon-blue thin separator) */
.app-section-bar-wrap {
    margin: 0.4rem 0 0.5rem 0;
}
.app-section-bar-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: rgba(240, 246, 252, 0.9);
    margin-bottom: 0.15rem;
    letter-spacing: 0.15px;
}
.app-section-bar-line {
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        rgba(77,163,255,0.0) 0%,
        rgba(77,163,255,0.35) 12%,
        rgba(77,163,255,0.9) 45%,
        rgba(77,163,255,0.9) 55%,
        rgba(77,163,255,0.35) 88%,
        rgba(77,163,255,0.0) 100%
    );
    box-shadow: 0 0 4px rgba(77,163,255,0.3), 0 0 10px rgba(77,163,255,0.12);
    opacity: 0.95;
}

/* Optional: hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
/* Chart container prominence (trader workstation) */
div[data-testid="stVerticalBlock"] > div[data-testid="stPlotlyChart"] {
    margin: 0.5rem 0 0.8rem 0;
}
/* CS2 chart panel: ensure chart uses full width and feels dominant */
.cs2-chart-panel div[data-testid="stPlotlyChart"],
.cs2-chart-panel [data-testid="stPlotlyChart"] {
    min-height: 520px;
    width: 100% !important;
}

/* Section header (Instrument Panel, Chart, Market / Execution) */
.section-header-wrap {
    margin: 18px 0 10px 0;
}
.section-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #eaf3ff;
    letter-spacing: 0.2px;
    margin-bottom: 6px;
}
.section-header-glow {
    height: 2px;
    width: 100%;
    border-radius: 999px;
    background: linear-gradient(
        90deg,
        rgba(0,0,0,0) 0%,
        rgba(50,150,255,0.85) 12%,
        rgba(90,200,255,1.0) 50%,
        rgba(50,150,255,0.85) 88%,
        rgba(0,0,0,0) 100%
    );
    box-shadow:
        0 0 6px rgba(80,180,255,0.45),
        0 0 14px rgba(80,180,255,0.20);
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)


def render_section_bar(title: str):
    """Render a thin neon-blue section separator/header bar. Dark-theme compatible, soft blue glow. UI only."""
    st.markdown(
        f'<div class="app-section-bar-wrap">'
        f'<div class="app-section-bar-title">{_html_esc(title)}</div>'
        f'<div class="app-section-bar-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _card_start():
    """Emit opening wrapper for a section card. Cosmetic only; pair with _card_end() or wrap single markdown."""
    st.markdown("<div class=\"app-card\">", unsafe_allow_html=True)


def _card_end():
    """Emit closing wrapper for a section card. Cosmetic only."""
    st.markdown("</div>", unsafe_allow_html=True)


def _section_divider(label: Optional[str] = None):
    """Render a neon-blue section divider with optional label. Cosmetic only; no logic."""
    label_html = f'<div class="app-section-divider-label">{label}</div>' if label else ""
    st.markdown(
        f'<div class="app-section-divider-wrap">\n  {label_html}\n  <div class="app-section-divider-line"></div>\n</div>',
        unsafe_allow_html=True,
    )


def _html_esc(s: str) -> str:
    """Escape for safe use inside HTML. Cosmetic helper only."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def ui_section_divider(title: str, subtitle: Optional[str] = None):
    """Render a neon blue section divider with title and optional subtitle. Cosmetic only; no logic."""
    t = _html_esc(title)
    sub = f'<div class="cs2-section-subtle">{_html_esc(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="cs2-section-wrap"><div class="cs2-section-title">{t}</div>{sub}<div class="cs2-section-bar"></div></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, icon: str = ""):
    """Render a large section header with optional icon and glow. Use for Instrument Panel, Chart, Market / Execution, etc."""
    label = f"{icon} {title}".strip()
    label_esc = _html_esc(label)
    st.markdown(
        f'<div class="section-header-wrap"><div class="section-header-title">{label_esc}</div><div class="section-header-glow"></div></div>',
        unsafe_allow_html=True,
    )


_inject_ui_polish_css()

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
# Refactored modules (ensure project root on path so fair_odds can be imported)
# ==========================
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
from legacy.fair_odds.paths import (
    PROJECT_ROOT, APP_DIR, LOG_PATH, LOG_COLUMNS,
    INPLAY_LOG_PATH, INPLAY_RESULTS_PATH, INPLAY_MAP_RESULTS_PATH,
    INPLAY_LOG_COLUMNS, INPLAY_RESULTS_COLUMNS, INPLAY_MAP_RESULTS_COLUMNS,
    KAPPA_CALIB_PATH, P_CALIB_PATH, P_CALIB_REPORT_PATH,
)
from legacy.fair_odds.odds import (
    american_to_decimal, decimal_to_american, implied_prob_from_american,
    calculate_fair_odds_curve, calculate_fair_odds_from_p, logistic_mapping,
    color_ev, prob_to_fair_american, compute_bo2_probs, ev_pct_decimal,
    decide_bet, decide_bo2_3way,
)
from legacy.fair_odds.calibration import (
    load_kappa_calibration, load_p_calibration_json, load_p_calibration,
    apply_p_calibration, get_kappa_multiplier,
    run_kappa_trainer, run_prob_trainer,
)
from legacy.fair_odds.data import (
    load_cs2_teams, load_dota_teams, get_team_tier,
    normalize_name, gosu_name_from_slug, read_csv_tolerant, sniff_bad_csv,
    piecewise_recent_weights,
)
from legacy.fair_odds.logs import (
    migrate_log_schema, recompute_metrics_from_logs, load_persisted_logs,
    persist_log_row, init_metrics, update_metrics_binary, update_metrics_3way,
    log_row, export_logs_df,
    persist_inplay_row, persist_inplay_result, persist_inplay_map_result,
    persist_cs2_replay_snapshot, persist_cs2_ml_feature_snapshot, derive_cs2_ml_feature_row,
    show_inplay_log_paths,
)
from legacy.fair_odds.scrapers import scrape_cs2_matches, scrape_dota_matches_gosu_subprocess, fetch_dota_matches
from legacy.fair_odds.scoring import (
    calculate_score, base_points_from_tier, normalize_series_result_from_fields,
    series_score_modifier_5tier,
    SERIES_CLAMP, SERIES_WEIGHTS, SERIES_PCT_OF_BASE_CAP,
)
from legacy.fair_odds.backtest import (
    load_state as load_backtest_state,
    save_state as save_backtest_state,
    load_match_settlements,
    incremental_update as backtest_incremental_update,
    repair_state_from_trades,
)
from fair_odds.bo3_adapter import (
    fetch_bo3_live_matches,
    fetch_bo3_snapshot,
    normalize_bo3_snapshot_to_app,
)
from legacy.fair_odds.replay_bo3 import (
    load_bo3_jsonl,
    group_bo3_by_match,
    write_replay_frame_to_feed,
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

# Kalshi tickers can include a dot (e.g. KXCS2GAME-26FEB21FORZE.REXR)
_KALSHI_TICKER_RE = re.compile(r"^[A-Z0-9_.-]{3,80}$", re.IGNORECASE)


def _try_extract_kalshi_ticker(raw: str) -> str:
    """
    Best-effort extraction of a Kalshi market ticker from a pasted URL or raw input.
    If raw already looks like a ticker, returns it unchanged.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    # If it's already a ticker-like token (letters, digits, _, ., -), return it
    if _KALSHI_TICKER_RE.match(raw):
        return raw.upper() if raw.isascii() else raw
    try:
        u = urlparse(raw)
        path_parts = [p for p in (u.path or "").split("/") if p]
        # Common patterns: /markets/{TICKER} or /trade/{TICKER}
        for i, part in enumerate(path_parts[:-1]):
            if part.lower() in ("markets", "market", "trade", "event"):
                cand = path_parts[i + 1]
                if _KALSHI_TICKER_RE.match(cand):
                    return cand.upper() if cand.isascii() else cand
        # Fallback: last segment
        if path_parts:
            cand = path_parts[-1]
            if _KALSHI_TICKER_RE.match(cand):
                return cand.upper() if cand.isascii() else cand
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


def _cs2_kalshi_resolve_ticker() -> tuple[Optional[str], Optional[str]]:
    """
    Resolve CS2 Kalshi market ticker from session state (same fallback as Refresh bid/ask).
    Returns (ticker_str, error_message). ticker is valid for fetch_kalshi_bid_ask; error_message is for UI.
    """
    tkr = st.session_state.get("cs2_kalshi_market") or ""
    t = _try_extract_kalshi_ticker(tkr) if tkr else ""
    if t:
        return (t, None)
    markets = st.session_state.get("cs2_kalshi_markets") or []
    if markets and isinstance(markets[0], dict) and markets[0].get("ticker"):
        t = _try_extract_kalshi_ticker(markets[0]["ticker"])
        if t:
            st.session_state["cs2_kalshi_market"] = markets[0]["ticker"]
            return (t, None)
    url = (st.session_state.get("cs2_kalshi_url") or "").strip()
    if url:
        try:
            ev = _kalshi_parse_event_ticker(url)
            if ev:
                markets2 = kalshi_list_markets_for_event(ev)
                st.session_state["cs2_kalshi_markets"] = markets2
                if markets2 and markets2[0].get("ticker"):
                    st.session_state["cs2_kalshi_market"] = markets2[0]["ticker"]
                    t = _try_extract_kalshi_ticker(markets2[0]["ticker"])
                    if t:
                        return (t, None)
        except Exception:
            pass
    return (None, "Load teams and select a market, or paste a Kalshi URL and click Load teams.")


# BO3 MARKET DELAY ALIGN: buffer and helper for delayed market logging (align snapshots with delayed BO3 feed)
def _cs2_market_delay_push(bid: float, ask: float) -> None:
    """Append current market snapshot to the delay buffer. Call after each Kalshi fetch."""
    if "cs2_market_delay_buffer" not in st.session_state:
        st.session_state["cs2_market_delay_buffer"] = deque(maxlen=5000)
    mid = 0.5 * (float(bid) + float(ask))
    st.session_state["cs2_market_delay_buffer"].append({
        "ts_epoch": time.time(),
        "bid": float(bid),
        "ask": float(ask),
        "mid": float(mid),
    })


def _cs2_market_delayed_snapshot(delay_sec: float, fallback_bid: float, fallback_ask: float) -> tuple:
    """Return (bid, ask, mid, ts_epoch_or_None, buffer_hit). target_ts = now - delay_sec; use nearest buffered entry at or before target_ts; else fallback and buffer_hit=False."""
    buf = st.session_state.get("cs2_market_delay_buffer")
    if not buf or len(buf) == 0:
        mid_fb = 0.5 * (float(fallback_bid) + float(fallback_ask))
        return (float(fallback_bid), float(fallback_ask), float(mid_fb), None, False)
    target_ts = time.time() - max(0.0, float(delay_sec))
    best = None
    best_ts = None
    for entry in reversed(buf):
        ts = entry.get("ts_epoch")
        if ts is None:
            continue
        if ts <= target_ts:
            if best is None or (ts > best_ts):
                best = entry
                best_ts = ts
    if best is None:
        mid_fb = 0.5 * (float(fallback_bid) + float(fallback_ask))
        return (float(fallback_bid), float(fallback_ask), float(mid_fb), None, False)
    bid_d = best.get("bid", fallback_bid)
    ask_d = best.get("ask", fallback_ask)
    mid_d = best.get("mid", 0.5 * (bid_d + ask_d))
    return (float(bid_d), float(ask_d), float(mid_d), best_ts, True)
# BO3 MARKET DELAY ALIGN


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


@st.cache_data(ttl=10, show_spinner=False)
def _load_bo3_replay_jsonl(path_str: str) -> tuple:
    """Load BO3 JSONL and group by match_id. Returns (entries_list, by_match_dict)."""
    from pathlib import Path
    entries = load_bo3_jsonl(Path(path_str))
    by_match = group_bo3_by_match(entries)
    return (entries, by_match)


def _replay_skip_duplicate_frames(entries: list, idx: int) -> int:
    """Advance idx past consecutive duplicate payloads (json.dumps sort_keys=True). Bounded; returns idx in [0, len(entries)-1]."""
    if not entries or idx < 0 or idx >= len(entries):
        return min(max(0, idx), len(entries) - 1) if entries else 0
    max_idx = len(entries) - 1
    n = 0
    while n < len(entries) and idx < max_idx:
        cur = entries[idx].get("payload") or {}
        prev = entries[idx - 1].get("payload") or {}
        if json.dumps(cur, sort_keys=True) != json.dumps(prev, sort_keys=True):
            break
        idx += 1
        n += 1
    return min(idx, max_idx)


def _replay_match_display_label(entries_for_match: list) -> str:
    """Get 'TeamOne vs TeamTwo' from first frame payload; fallback to match_slug or 'Match {id}'."""
    if not entries_for_match:
        return "Unknown"
    payload = entries_for_match[0].get("payload") or {}
    t1 = (payload.get("team_one") or {}).get("name") or (payload.get("team_one") or {}).get("clan_name")
    t2 = (payload.get("team_two") or {}).get("name") or (payload.get("team_two") or {}).get("clan_name")
    if t1 and t2:
        return f"{t1} vs {t2}"
    slug = (payload.get("match_fixture") or {}).get("match_slug")
    if slug:
        return slug.replace("-", " ")
    return f"Match {entries_for_match[0].get('match_id', '?')}"


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


def _load_inplay_trades_from_disk(outdir: Path, strategy_ids: Optional[list] = None) -> dict:
    """Load all in-play backtest trade CSVs from outdir. Returns {strategy_id: DataFrame}.
    If strategy_ids is None, discovers from inplay_backtest_trades__*.csv filenames."""
    outdir = Path(outdir)
    if strategy_ids is None:
        strategy_ids = []
        for fp in outdir.glob("inplay_backtest_trades__*.csv"):
            try:
                sid = fp.stem.replace("inplay_backtest_trades__", "")
                if sid:
                    strategy_ids.append(sid)
            except Exception:
                pass
    trades = {}
    for sid in strategy_ids:
        tp = outdir / f"inplay_backtest_trades__{sid}.csv"
        if tp.exists():
            try:
                trades[sid] = pd.read_csv(tp)
            except Exception:
                pass
    return trades


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
    strategy_ids = [s["id"] if isinstance(s, dict) else s for s in (config.get("strategies") or [])]
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
    # Reload all strategy trades from CSV so chart always has full data (avoids entry "disappearing" when trade closes)
    trades = _load_inplay_trades_from_disk(outdir, strategy_ids or None)
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


# BO3 STATE BANDS CONSISTENCY: single canonical fair path for p_hat and round-state bands
def _compute_cs2_live_fair_for_state(
    p0_map: float,
    rounds_a: int,
    rounds_b: int,
    econ_a: float,
    econ_b: float,
    pistol_a: bool,
    pistol_b: bool,
    beta_score: float,
    beta_econ: float,
    beta_pistol: float,
    map_name,
    a_side,
    beta_lock: float,
    lock_start_offset: int,
    lock_ramp: int,
    contract_scope: str,
    n_maps: int,
    maps_a_won: int,
    maps_b_won: int,
) -> tuple:
    """Compute final fair probability for a given CS2 live state (same path as p_hat).
    Returns (p_final, p_hat_map_raw). p_hat_map_raw is before soft lock (for kappa/credible interval).
    On failure returns (None, None). Does NOT compute kappa or call update_round_stream."""
    try:
        ra, rb = int(rounds_a), int(rounds_b)
        win_target = cs2_current_win_target(ra, rb)
        p_hat_map_raw = estimate_inplay_prob(
            float(p0_map), ra, rb, float(econ_a), float(econ_b),
            pistol_a=pistol_a, pistol_b=pistol_b,
            beta_score=float(beta_score), beta_econ=float(beta_econ), beta_pistol=float(beta_pistol),
            map_name=map_name, a_side=a_side, pistol_decay=0.30, beta_side=0.85,
            beta_lock=float(beta_lock), lock_start_offset=int(lock_start_offset), lock_ramp=int(lock_ramp),
            win_target=int(win_target),
        )
        p_hat_map = cs2_soft_lock_map_prob(float(p_hat_map_raw), ra, rb, int(win_target))
        if contract_scope == "Series winner" and int(n_maps) > 1:
            p_final = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(p_hat_map), float(p0_map))
        else:
            p_final = float(p_hat_map)
        return (p_final, float(p_hat_map_raw))
    except Exception:
        return (None, None)
# BO3 STATE BANDS CONSISTENCY

# BO3 STATE BANDS: asymmetric rail — distance-weighted toward map-win / map-loss anchors
# Max single-side corridor width so rails don't explode
RAIL_MAX_HALF_WIDTH = 0.40


def _compute_cs2_round_state_bands(
    p0_map: float,
    rounds_a: int,
    rounds_b: int,
    econ_a: float,
    econ_b: float,
    pistol_a: bool,
    pistol_b: bool,
    beta_score: float,
    beta_econ: float,
    beta_pistol: float,
    map_name,
    a_side,
    beta_lock: float,
    lock_start_offset: int,
    lock_ramp: int,
    contract_scope: str,
    n_maps: int,
    maps_a_won: int,
    maps_b_won: int,
    rail_mode: str = "ANCHOR_PULLED",
    rail_pull_scale: float = 0.7,
    rail_pull_cap: float = 0.6,
) -> tuple:
    """Compute asymmetric round-state rail: p_state_center, p_if_next_round_win, p_if_next_round_loss.
    Win/loss transitions are distance-weighted toward map-win (1.0) and map-loss (0.0) anchors when rail_mode==ANCHOR_PULLED.
    Returns (p_if_a, p_if_b, band_lo, band_hi, rail_debug_dict). On failure rail_debug is empty."""
    ra, rb = int(rounds_a), int(rounds_b)
    # Raw counterfactuals from fair(ra,rb), fair(ra+1,rb), fair(ra,rb+1)
    p_center, _ = _compute_cs2_live_fair_for_state(
        float(p0_map), ra, rb, float(econ_a), float(econ_b),
        pistol_a, pistol_b, float(beta_score), float(beta_econ), float(beta_pistol),
        map_name, a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
        str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
    )
    p_if_a_raw, _ = _compute_cs2_live_fair_for_state(
        float(p0_map), ra + 1, rb, float(econ_a), float(econ_b),
        pistol_a, pistol_b, float(beta_score), float(beta_econ), float(beta_pistol),
        map_name, a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
        str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
    )
    p_if_b_raw, _ = _compute_cs2_live_fair_for_state(
        float(p0_map), ra, rb + 1, float(econ_a), float(econ_b),
        pistol_a, pistol_b, float(beta_score), float(beta_econ), float(beta_pistol),
        map_name, a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
        str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
    )
    if p_if_a_raw is None or p_if_b_raw is None:
        return (None, None, None, None, {})
    if p_center is None:
        p_center = 0.5 * (float(p_if_a_raw) + float(p_if_b_raw))

    win_target = cs2_current_win_target(ra, rb)
    steps_to_a_win = max(0, win_target - (ra + 1))
    steps_to_b_win = max(0, win_target - (rb + 1))

    pull_win_used = 0.0
    pull_loss_used = 0.0
    mode_used = str(rail_mode) if rail_mode in ("ANCHOR_PULLED", "RAW_COUNTERFACTUAL") else "ANCHOR_PULLED"

    if mode_used == "RAW_COUNTERFACTUAL":
        p_if_a = float(p_if_a_raw)
        p_if_b = float(p_if_b_raw)
        p_if_next_round_win = p_if_a
        p_if_next_round_loss = p_if_b
    else:
        pull_win = 1.0 / (1.0 + float(steps_to_a_win))
        pull_loss = 1.0 / (1.0 + float(steps_to_b_win))
        scale = float(rail_pull_scale) if rail_pull_scale is not None else 0.7
        cap = float(rail_pull_cap) if rail_pull_cap is not None else 0.6
        pull_win = min(pull_win * scale, cap)
        pull_loss = min(pull_loss * scale, cap)
        pull_win_used = pull_win
        pull_loss_used = pull_loss
        p_if_a = float(p_if_a_raw)
        p_if_b = float(p_if_b_raw)
        p_if_next_round_win = float(p_if_a) + (1.0 - float(p_if_a)) * pull_win
        p_if_next_round_loss = float(p_if_b) * (1.0 - pull_loss)
        p_if_next_round_win = float(np.clip(p_if_next_round_win, 0.0, 1.0))
        p_if_next_round_loss = float(np.clip(p_if_next_round_loss, 0.0, 1.0))

    # Ensure ordering and cap corridor widths (unchanged)
    p_center_f = float(p_center)
    upper_width = min(p_if_next_round_win - p_center_f, RAIL_MAX_HALF_WIDTH)
    lower_width = min(p_center_f - p_if_next_round_loss, RAIL_MAX_HALF_WIDTH)
    p_if_next_round_win = p_center_f + upper_width
    p_if_next_round_loss = p_center_f - lower_width
    p_if_next_round_win = float(np.clip(p_if_next_round_win, 0.0, 1.0))
    p_if_next_round_loss = float(np.clip(p_if_next_round_loss, 0.0, 1.0))
    if p_if_next_round_loss > p_center_f:
        p_if_next_round_loss = p_center_f
    if p_if_next_round_win < p_center_f:
        p_if_next_round_win = p_center_f
    band_lo = p_if_next_round_loss
    band_hi = p_if_next_round_win
    lower_w = p_center_f - band_lo
    upper_w = band_hi - p_center_f
    asym_ratio = (upper_w / (lower_w + 1e-9)) if (lower_w + 1e-9) else None
    rail_debug = {
        "rail_p_state_center": p_center_f,
        "rail_p_if_next_round_win": band_hi,
        "rail_p_if_next_round_loss": band_lo,
        "rail_upper_width": upper_w,
        "rail_lower_width": lower_w,
        "rail_asymmetry_ratio": float(asym_ratio) if asym_ratio is not None else None,
        "current_rounds_a": ra,
        "current_rounds_b": rb,
        "band_if_a_round_raw": float(p_if_a_raw),
        "band_if_b_round_raw": float(p_if_b_raw),
        "rail_pull_win_used": float(pull_win_used),
        "rail_pull_loss_used": float(pull_loss_used),
        "rail_steps_to_a_win": int(steps_to_a_win),
        "rail_steps_to_b_win": int(steps_to_b_win),
        "rail_mode_used": mode_used,
    }
    return (p_if_a, p_if_b, band_lo, band_hi, rail_debug)
# BO3 STATE BANDS


# Live CS2 round-state rails v2: nonlinear, score/econ/side-aware (logistic model)
# Tuning: conservative weights to reduce overreaction / corridor jumpiness (same formulas, smaller scale)
RAIL_V2_K_SCORE = 1.8
RAIL_V2_K_ECON = 0.22
RAIL_V2_K_LOADOUT = 0.18
RAIL_V2_K_SIDE = 0.12
RAIL_V2_K_SERIES = 0.25
RAIL_V2_SERIES_ASYMMETRY_STRENGTH = 0.06
RAIL_V2_CLIP_LO = 0.01
RAIL_V2_CLIP_HI = 0.99
# Within-round smoothing: blend new rails with previous tick when round unchanged (no smooth on transition)
RAIL_SMOOTHING_ALPHA = 0.70
# Canonical rails: asymmetry wrapper around canonical endpoints (bias and distance multipliers)
RAIL_ASYM_BIAS_CLIP = 0.25
RAIL_ASYM_MULT_SPREAD = 0.12


def _sigmoid(x: float) -> float:
    """Bounded sigmoid (used only for shape bias, not for rail endpoints)."""
    x = float(np.clip(x, -20.0, 20.0))
    return 1.0 / (1.0 + np.exp(-x))


def _compute_cs2_round_state_rails_v2(
    rounds_a: int,
    rounds_b: int,
    win_target: int,
    map_name: Optional[str] = None,
    team_a_side: Optional[str] = None,
    econ_a: Optional[float] = None,
    econ_b: Optional[float] = None,
    loadout_a: Optional[float] = None,
    loadout_b: Optional[float] = None,
    series_maps_won_a: Optional[int] = None,
    series_maps_won_b: Optional[int] = None,
    best_of: int = 3,
    p0_map: Optional[float] = None,
    pistol_a: Optional[bool] = None,
    pistol_b: Optional[bool] = None,
    beta_score: Optional[float] = None,
    beta_econ: Optional[float] = None,
    beta_pistol: Optional[float] = None,
    beta_lock: Optional[float] = None,
    lock_start_offset: Optional[int] = None,
    lock_ramp: Optional[int] = None,
    contract_scope: Optional[str] = None,
    n_maps: Optional[int] = None,
    maps_a_won: Optional[int] = None,
    maps_b_won: Optional[int] = None,
    canonical_econ_a: Optional[float] = None,
    canonical_econ_b: Optional[float] = None,
    rail_mode: Optional[str] = None,
    rail_pull_scale: Optional[float] = None,
    rail_pull_cap: Optional[float] = None,
) -> dict:
    """Round-state rails v2 canonical: corridor from SAME probability engine as p_hat; v2 terms only shape/asymmetry.
    Canonical anchor and endpoints from _compute_cs2_live_fair_for_state (series-aware when contract is Series winner).
    Bounded bias from econ/loadout/side/series/score adjusts corridor shape around canonical. Fallback: _compute_cs2_round_state_bands.
    """
    ra, rb = int(rounds_a), int(rounds_b)
    wt = max(1, int(win_target))
    clip_lo, clip_hi = RAIL_V2_CLIP_LO, RAIL_V2_CLIP_HI

    # --- Canonical probability engine (same path as live p_hat) ---
    canonical_anchor = None
    canonical_if_a = None
    canonical_if_b = None
    rail_anchor_source = "canonical_live_fair"
    rail_series_context_used = False
    if (
        p0_map is not None and pistol_a is not None and pistol_b is not None
        and beta_score is not None and beta_econ is not None and beta_pistol is not None
        and beta_lock is not None and lock_start_offset is not None and lock_ramp is not None
        and contract_scope is not None and n_maps is not None
        and maps_a_won is not None and maps_b_won is not None
    ):
        cecon_a = float(canonical_econ_a) if canonical_econ_a is not None else (float(econ_a) if econ_a is not None else 0.0)
        cecon_b = float(canonical_econ_b) if canonical_econ_b is not None else (float(econ_b) if econ_b is not None else 0.0)
        try:
            canonical_anchor, _ = _compute_cs2_live_fair_for_state(
                float(p0_map), ra, rb, cecon_a, cecon_b,
                bool(pistol_a), bool(pistol_b), float(beta_score), float(beta_econ), float(beta_pistol),
                map_name, team_a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
                str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
            )
            canonical_if_a, _ = _compute_cs2_live_fair_for_state(
                float(p0_map), ra + 1, rb, cecon_a, cecon_b,
                bool(pistol_a), bool(pistol_b), float(beta_score), float(beta_econ), float(beta_pistol),
                map_name, team_a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
                str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
            )
            canonical_if_b, _ = _compute_cs2_live_fair_for_state(
                float(p0_map), ra, rb + 1, cecon_a, cecon_b,
                bool(pistol_a), bool(pistol_b), float(beta_score), float(beta_econ), float(beta_pistol),
                map_name, team_a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
                str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
            )
            if contract_scope == "Series winner" and int(n_maps) > 1:
                rail_series_context_used = True
        except Exception:
            canonical_anchor = canonical_if_a = canonical_if_b = None

    # Terminal override: at map point (ra+1 or rb+1 >= win_target), set canonical endpoint to the exact map
    # terminal line (state_bound_lower / state_bound_upper). Use win_target from same path as fair (cs2_current_win_target).
    rail_terminal_override_a_used = False
    rail_terminal_override_b_used = False
    rail_terminal_if_a = canonical_if_a
    rail_terminal_if_b = canonical_if_b
    rail_maps_a_won_used = None
    rail_maps_b_won_used = None
    wt_rail = cs2_current_win_target(ra, rb)
    if canonical_anchor is not None and canonical_if_a is not None and canonical_if_b is not None:
        _n = int(n_maps) if n_maps is not None else 3
        _ma = int(maps_a_won) if maps_a_won is not None else 0
        _mb = int(maps_b_won) if maps_b_won is not None else 0
        _p0 = float(p0_map) if p0_map is not None else 0.5
        _scope = str(contract_scope or "")
        rail_maps_a_won_used = _ma
        rail_maps_b_won_used = _mb
        # (rb + 1) >= win_target => B wins this map => canonical_if_b = state_bound_lower (series fair "B wins map")
        if (rb + 1) >= wt_rail:
            rail_terminal_override_b_used = True
            if _scope == "Series winner" and _n > 1:
                rail_terminal_if_b = series_win_prob_live(_n, _ma, _mb + 1, _p0, _p0)
            else:
                rail_terminal_if_b = 0.0
            canonical_if_b = rail_terminal_if_b
        # (ra + 1) >= win_target => A wins this map => canonical_if_a = state_bound_upper (series fair "A wins map")
        if (ra + 1) >= wt_rail:
            rail_terminal_override_a_used = True
            if _scope == "Series winner" and _n > 1:
                rail_terminal_if_a = series_win_prob_live(_n, _ma + 1, _mb, _p0, _p0)
            else:
                rail_terminal_if_a = 1.0
            canonical_if_a = rail_terminal_if_a

    if canonical_anchor is None or canonical_if_a is None or canonical_if_b is None:
        # Fallback: old canonical round-state bands (same engine, different shape)
        _rail_mode = (rail_mode or "ANCHOR_PULLED") if rail_mode in ("ANCHOR_PULLED", "RAW_COUNTERFACTUAL") else "ANCHOR_PULLED"
        _pull_scale = float(rail_pull_scale) if rail_pull_scale is not None else 0.7
        _pull_cap = float(rail_pull_cap) if rail_pull_cap is not None else 0.6
        fallback = _compute_cs2_round_state_bands(
            float(p0_map or 0.5), ra, rb,
            float(canonical_econ_a or econ_a or 0.0), float(canonical_econ_b or econ_b or 0.0),
            bool(pistol_a or False), bool(pistol_b or False),
            float(beta_score or 0.22), float(beta_econ or 0.06), float(beta_pistol or 0.35),
            map_name, team_a_side, float(beta_lock or 0.9), int(lock_start_offset or 3), int(lock_ramp or 3),
            str(contract_scope or "Map winner (this map)"), int(n_maps or 3), int(maps_a_won or 0), int(maps_b_won or 0),
            rail_mode=_rail_mode, rail_pull_scale=_pull_scale, rail_pull_cap=_pull_cap,
        )
        p_if_a_fb, p_if_b_fb, band_lo_fb, band_hi_fb, debug_fb = fallback
        anchor_fb = None
        if isinstance(debug_fb, dict) and debug_fb.get("rail_p_state_center") is not None:
            anchor_fb = float(debug_fb["rail_p_state_center"])
        if anchor_fb is None and p_if_a_fb is not None and p_if_b_fb is not None:
            anchor_fb = 0.5 * (float(p_if_a_fb) + float(p_if_b_fb))
        if anchor_fb is None:
            anchor_fb = 0.5
        # Dead-path envelope diagnostics (no behavior change)
        _env_fb = _compute_cs2_branch_endpoint_envelope_debug(
            endpoint_a_base=float(p_if_a_fb) if p_if_a_fb is not None else 0.5,
            endpoint_b_base=float(p_if_b_fb) if p_if_b_fb is not None else 0.5,
            rounds_a=ra, rounds_b=rb,
            best_of=int(best_of) if best_of is not None else 3,
            maps_a_won=int(maps_a_won or 0), maps_b_won=int(maps_b_won or 0),
            win_target=wt,
        )
        return {
            "anchor": float(np.clip(anchor_fb, clip_lo, clip_hi)),
            "p_if_a_wins": p_if_a_fb,
            "p_if_b_wins": p_if_b_fb,
            "band_lo": band_lo_fb,
            "band_hi": band_hi_fb,
            "band_if_a_round": p_if_a_fb,
            "band_if_b_round": p_if_b_fb,
            "rail_model_version": "round_rail_v2_canonical",
            "rail_anchor_source": "fallback_bands",
            "rail_anchor_canonical": canonical_anchor,
            "rail_if_a_canonical": canonical_if_a,
            "rail_if_b_canonical": canonical_if_b,
            "rail_if_a_adjusted": p_if_a_fb,
            "rail_if_b_adjusted": p_if_b_fb,
            "rail_asymmetry_bias": 0.0,
            "rail_asymmetry_mult_a": 1.0,
            "rail_asymmetry_mult_b": 1.0,
            "rail_series_context_used": rail_series_context_used,
            "rail_score_term": None,
            "rail_econ_term": None,
            "rail_loadout_term": None,
            "rail_side_term": None,
            "rail_series_term": None,
            "rail_series_maps_won_a": int(series_maps_won_a or 0),
            "rail_series_maps_won_b": int(series_maps_won_b or 0),
            "rail_endpoint_env_valid": _env_fb.get("env_valid"),
            "rail_endpoint_a_base_dbg": _env_fb.get("endpoint_a_base"),
            "rail_endpoint_b_base_dbg": _env_fb.get("endpoint_b_base"),
            "rail_endpoint_env_base_span_abs": _env_fb.get("endpoint_base_span_abs"),
            "rail_endpoint_env_fraction_k": _env_fb.get("endpoint_env_fraction_k"),
            "rail_endpoint_env_round_leverage": _env_fb.get("endpoint_env_round_leverage"),
            "rail_endpoint_env_series_leverage": _env_fb.get("endpoint_env_series_leverage"),
            "rail_endpoint_env_half_width": _env_fb.get("endpoint_env_half_width"),
            "rail_endpoint_a_env_low": _env_fb.get("endpoint_a_env_low"),
            "rail_endpoint_a_env_high": _env_fb.get("endpoint_a_env_high"),
            "rail_endpoint_b_env_low": _env_fb.get("endpoint_b_env_low"),
            "rail_endpoint_b_env_high": _env_fb.get("endpoint_b_env_high"),
            "rail_endpoint_a_env_center": _env_fb.get("endpoint_a_env_center"),
            "rail_endpoint_b_env_center": _env_fb.get("endpoint_b_env_center"),
            "rail_endpoint_env_map_index": _env_fb.get("endpoint_env_map_index"),
            "rail_endpoint_env_best_of": _env_fb.get("endpoint_env_best_of"),
            "rail_endpoint_env_win_target": _env_fb.get("endpoint_env_win_target"),
            "band_if_a_round_raw": debug_fb.get("band_if_a_round_raw"),
            "band_if_b_round_raw": debug_fb.get("band_if_b_round_raw"),
            "rail_pull_win_used": debug_fb.get("rail_pull_win_used"),
            "rail_pull_loss_used": debug_fb.get("rail_pull_loss_used"),
            "rail_steps_to_a_win": debug_fb.get("rail_steps_to_a_win"),
            "rail_steps_to_b_win": debug_fb.get("rail_steps_to_b_win"),
            "rail_mode_used": debug_fb.get("rail_mode_used"),
            "rail_terminal_override_a_used": False,
            "rail_terminal_override_b_used": False,
            "rail_terminal_if_a": None,
            "rail_terminal_if_b": None,
            "rail_ra": ra,
            "rail_rb": rb,
            "win_target": wt,
            "rail_win_target_used": wt,
            "rail_maps_a_won_used": int(maps_a_won) if maps_a_won is not None else None,
            "rail_maps_b_won_used": int(maps_b_won) if maps_b_won is not None else None,
        }

    canonical_anchor = float(np.clip(canonical_anchor, clip_lo, clip_hi))
    canonical_if_a = float(np.clip(canonical_if_a, clip_lo, clip_hi))
    canonical_if_b = float(np.clip(canonical_if_b, clip_lo, clip_hi))

    # --- V2 shape terms (bounded bias only; no sigmoid endpoints) ---
    def score_signal(ra_val: int, rb_val: int) -> float:
        diff = (ra_val - rb_val) / float(wt)
        return float(np.clip(diff, -2.0, 2.0)) * RAIL_V2_K_SCORE

    econ_a_f = float(econ_a) if econ_a is not None else 0.0
    econ_b_f = float(econ_b) if econ_b is not None else 0.0
    econ_sum = econ_a_f + econ_b_f + 1e-6
    econ_adv = float(np.clip((econ_a_f - econ_b_f) / econ_sum, -1.0, 1.0))
    rail_econ_term = RAIL_V2_K_ECON * econ_adv

    load_a = float(loadout_a) if loadout_a is not None else 0.0
    load_b = float(loadout_b) if loadout_b is not None else 0.0
    load_sum = load_a + load_b + 1e-6
    loadout_adv = float(np.clip((load_a - load_b) / load_sum, -1.0, 1.0))
    rail_loadout_term = RAIL_V2_K_LOADOUT * loadout_adv

    rail_side_term = 0.0
    if map_name and team_a_side and str(team_a_side).strip().upper() in ("CT", "T"):
        try:
            ct_rate = float(CS2_MAP_CT_RATE.get(str(map_name).strip(), CS2_CT_RATE_AVG))
            if str(team_a_side).strip().upper() == "CT":
                rail_side_term = RAIL_V2_K_SIDE * (ct_rate - 0.5)
            else:
                rail_side_term = RAIL_V2_K_SIDE * (0.5 - ct_rate)
            rail_side_term = float(np.clip(rail_side_term, -0.15, 0.15))
        except (TypeError, ValueError, KeyError):
            pass

    ma = int(series_maps_won_a) if series_maps_won_a is not None else 0
    mb = int(series_maps_won_b) if series_maps_won_b is not None else 0
    bo = max(1, int(best_of))
    series_lead = float(np.clip((ma - mb) / float(bo), -1.0, 1.0))
    rail_series_term = RAIL_V2_K_SERIES * series_lead

    raw_bias = rail_econ_term + rail_loadout_term + rail_side_term + rail_series_term + float(score_signal(ra, rb))
    rail_asymmetry_bias = float(np.clip(raw_bias, -1.0, 1.0)) * RAIL_ASYM_BIAS_CLIP

    # --- Asymmetry multipliers: positive bias = expand A (upper) side, contract B (lower) side ---
    mult_a = 1.0 + rail_asymmetry_bias
    mult_b = 1.0 - rail_asymmetry_bias
    mult_a = float(np.clip(mult_a, 1.0 - RAIL_ASYM_MULT_SPREAD, 1.0 + RAIL_ASYM_MULT_SPREAD))
    mult_b = float(np.clip(mult_b, 1.0 - RAIL_ASYM_MULT_SPREAD, 1.0 + RAIL_ASYM_MULT_SPREAD))

    d_up = canonical_if_a - canonical_anchor
    d_dn = canonical_anchor - canonical_if_b
    adjusted_d_up = d_up * mult_a
    adjusted_d_dn = d_dn * mult_b
    if rail_terminal_override_a_used:
        adj_if_a = float(np.clip(rail_terminal_if_a, 0.0, 1.0))  # preserve exact 0/1 for map terminal
    else:
        adj_if_a = canonical_anchor + adjusted_d_up
        adj_if_a = float(np.clip(adj_if_a, clip_lo, clip_hi))
    if rail_terminal_override_b_used:
        adj_if_b = float(np.clip(rail_terminal_if_b, 0.0, 1.0))  # preserve exact 0/1 for map terminal
    else:
        adj_if_b = canonical_anchor - adjusted_d_dn
        adj_if_b = float(np.clip(adj_if_b, clip_lo, clip_hi))

    # Shaped endpoints (pre-envelope): these are the final candidates before intraround envelope drift
    shaped_pre_envelope_a = adj_if_a
    shaped_pre_envelope_b = adj_if_b

    # Dead-path envelope diagnostics; also used to bound intraround drift
    _env = _compute_cs2_branch_endpoint_envelope_debug(
        endpoint_a_base=adj_if_a,
        endpoint_b_base=adj_if_b,
        rounds_a=ra,
        rounds_b=rb,
        best_of=bo,
        maps_a_won=ma,
        maps_b_won=mb,
        win_target=wt,
    )

    # Bounded drifted endpoints: shaped clipped to envelope so visible rails can move intraround within envelope only
    a_lo = _env.get("endpoint_a_env_low")
    a_hi = _env.get("endpoint_a_env_high")
    b_lo = _env.get("endpoint_b_env_low")
    b_hi = _env.get("endpoint_b_env_high")
    env_valid = _env.get("env_valid") is True and a_lo is not None and a_hi is not None and b_lo is not None and b_hi is not None
    if env_valid:
        try:
            a_lo_f = float(a_lo)
            a_hi_f = float(a_hi)
            b_lo_f = float(b_lo)
            b_hi_f = float(b_hi)
            if a_lo_f <= a_hi_f and b_lo_f <= b_hi_f:
                drifted_a = float(np.clip(adj_if_a, a_lo_f, a_hi_f))
                drifted_b = float(np.clip(adj_if_b, b_lo_f, b_hi_f))
                envelope_applied = True
                envelope_fallback_reason = "ok"
            else:
                drifted_a = adj_if_a
                drifted_b = adj_if_b
                envelope_applied = False
                envelope_fallback_reason = "invalid_bounds"
        except (TypeError, ValueError):
            drifted_a = adj_if_a
            drifted_b = adj_if_b
            envelope_applied = False
            envelope_fallback_reason = "invalid_bounds"
    else:
        drifted_a = adj_if_a
        drifted_b = adj_if_b
        envelope_applied = False
        envelope_fallback_reason = "missing_bounds" if not _env.get("env_valid") else "invalid_bounds"

    # Visible round-state rails use drifted endpoints; preserve ordering and anchor clamps
    band_lo = min(drifted_a, drifted_b)
    band_hi = max(drifted_a, drifted_b)
    if band_lo > canonical_anchor:
        band_lo = min(band_lo, canonical_anchor)
    if band_hi < canonical_anchor:
        band_hi = max(band_hi, canonical_anchor)
    anchor = canonical_anchor

    return {
        "anchor": anchor,
        "p_if_a_wins": drifted_a,
        "p_if_b_wins": drifted_b,
        "band_lo": band_lo,
        "band_hi": band_hi,
        "band_if_a_round": drifted_a,
        "band_if_b_round": drifted_b,
        "rail_model_version": "round_rail_v2_canonical",
        "rail_anchor_source": rail_anchor_source,
        "rail_anchor_canonical": canonical_anchor,
        "rail_if_a_canonical": canonical_if_a,
        "rail_if_b_canonical": canonical_if_b,
        "rail_if_a_adjusted": adj_if_a,
        "rail_if_b_adjusted": adj_if_b,
        "rail_endpoint_a_canonical": canonical_if_a,
        "rail_endpoint_b_canonical": canonical_if_b,
        "rail_endpoint_a_shaped_pre_envelope": shaped_pre_envelope_a,
        "rail_endpoint_b_shaped_pre_envelope": shaped_pre_envelope_b,
        "rail_endpoint_a_drifted": drifted_a,
        "rail_endpoint_b_drifted": drifted_b,
        "rail_endpoint_a_envelope_lo": a_lo if env_valid else None,
        "rail_endpoint_a_envelope_hi": a_hi if env_valid else None,
        "rail_endpoint_b_envelope_lo": b_lo if env_valid else None,
        "rail_endpoint_b_envelope_hi": b_hi if env_valid else None,
        "rail_endpoint_envelope_applied": envelope_applied,
        "rail_endpoint_envelope_fallback_reason": envelope_fallback_reason,
        "rail_asymmetry_bias": rail_asymmetry_bias,
        "rail_asymmetry_mult_a": mult_a,
        "rail_asymmetry_mult_b": mult_b,
        "rail_series_context_used": rail_series_context_used,
        "rail_score_term": float(score_signal(ra, rb)),
        "rail_econ_term": rail_econ_term,
        "rail_loadout_term": rail_loadout_term,
        "rail_side_term": rail_side_term,
        "rail_series_term": rail_series_term,
        "rail_series_maps_won_a": ma,
        "rail_series_maps_won_b": mb,
        "rail_endpoint_env_valid": _env.get("env_valid"),
        "rail_endpoint_a_base_dbg": _env.get("endpoint_a_base"),
        "rail_endpoint_b_base_dbg": _env.get("endpoint_b_base"),
        "rail_endpoint_env_base_span_abs": _env.get("endpoint_base_span_abs"),
        "rail_endpoint_env_fraction_k": _env.get("endpoint_env_fraction_k"),
        "rail_endpoint_env_round_leverage": _env.get("endpoint_env_round_leverage"),
        "rail_endpoint_env_series_leverage": _env.get("endpoint_env_series_leverage"),
        "rail_endpoint_env_half_width": _env.get("endpoint_env_half_width"),
        "rail_endpoint_a_env_low": _env.get("endpoint_a_env_low"),
        "rail_endpoint_a_env_high": _env.get("endpoint_a_env_high"),
        "rail_endpoint_b_env_low": _env.get("endpoint_b_env_low"),
        "rail_endpoint_b_env_high": _env.get("endpoint_b_env_high"),
        "rail_endpoint_a_env_center": _env.get("endpoint_a_env_center"),
        "rail_endpoint_b_env_center": _env.get("endpoint_b_env_center"),
        "rail_endpoint_env_map_index": _env.get("endpoint_env_map_index"),
        "rail_endpoint_env_best_of": _env.get("endpoint_env_best_of"),
        "rail_endpoint_env_win_target": _env.get("endpoint_env_win_target"),
        "rail_terminal_override_a_used": rail_terminal_override_a_used,
        "rail_terminal_override_b_used": rail_terminal_override_b_used,
        "rail_terminal_if_a": rail_terminal_if_a,
        "rail_terminal_if_b": rail_terminal_if_b,
        "rail_ra": ra,
        "rail_rb": rb,
        "win_target": wt_rail,
        "rail_win_target": wt,
        "rail_win_target_used": wt_rail,
        "rail_maps_a_won_used": rail_maps_a_won_used,
        "rail_maps_b_won_used": rail_maps_b_won_used,
    }


# BO3 MIDROUND V1 — round-state latching and intraround adjustment (alive/bomb/HP/time)
def _build_cs2_round_key(map_name, rounds_a, rounds_b, maps_a_won, maps_b_won, contract_scope, n_maps, game_number=None) -> str:
    """Stable key for current round state; only changes when score/map/series changes. Optional game_number for map-transition robustness."""
    base = f"{map_name}|{int(rounds_a)}|{int(rounds_b)}|{int(maps_a_won)}|{int(maps_b_won)}|{contract_scope}|{int(n_maps)}"
    if game_number is not None:
        try:
            base = f"{base}|{int(game_number)}"
        except (TypeError, ValueError):
            pass
    return base


# Phase 3: pending round-start latch — max ticks to wait before finalizing with best available
CS2_ROUND_START_LATCH_PENDING_MAX_ATTEMPTS = 3


def _cs2_round_start_context_stable(current_round_key: str, pending_round_key: str, ss: dict) -> tuple:
    """Return (stable: bool, reason: str). Used to decide when to finalize round-start latches."""
    if current_round_key != pending_round_key:
        return False, "key_mismatch"
    alive_a = ss.get("cs2_live_team_a_alive_count")
    alive_b = ss.get("cs2_live_team_b_alive_count")
    econ_a = ss.get("cs2_live_econ_a")
    econ_b = ss.get("cs2_live_econ_b")
    loadout_a = ss.get("cs2_live_team_a_loadout_est_total")
    loadout_b = ss.get("cs2_live_team_b_loadout_est_total")
    phase = ss.get("cs2_live_round_phase")
    try:
        phase_ok = phase is None or (isinstance(phase, str) and str(phase).strip().upper() in ("FREEZE", "WARMUP", "BUY", "IN_PROGRESS", ""))
    except Exception:
        phase_ok = True
    alive_ok = alive_a is not None and alive_b is not None and int(alive_a) >= 3 and int(alive_b) >= 3
    econ_ok = econ_a is not None and econ_b is not None
    loadout_ok = loadout_a is not None and loadout_b is not None
    if phase_ok and alive_ok and econ_ok and loadout_ok:
        return True, "phase_ok_alive_ok"
    return False, "retry_waiting"


# BO3 MIDROUND V1 — default V1 weights (conservative, easy to tune)
ALIVE_DIFF_WEIGHT_PER_PLAYER = 0.035
HP_DIFF_WEIGHT_PER_100HP = 0.010
BOMB_PLANTED_WEIGHT = 0.060
TIME_SCALE_MIN = 0.35
TIME_SCALE_MAX = 1.00
MAX_ABS_MID_ADJ = 0.18
# MIDROUND V2 — armor, loadout, reliability (conservative)
ARMOR_DIFF_WEIGHT_PER_100 = 0.008
LOADOUT_DIFF_WEIGHT_PER_1000 = 0.012
RELIABILITY_MIN_MULT = 0.35
# Series-endpoint mixture: temperature for q_intra = sigmoid(intra_score_raw / temp)
INTRA_Q_SIGMOID_TEMP = 1.0

# Round-start context offset: max absolute offset (baseline context only)
ROUND_CONTEXT_OFFSET_MAX = 0.04


def _compute_cs2_round_context_offset(
    p_base_frozen: float,
    band_lo_frozen: Optional[float],
    band_hi_frozen: Optional[float],
    map_name: Optional[str],
    team_a_side: Optional[str],
    team_b_side: Optional[str],
    round_start_econ_a: Optional[float],
    round_start_econ_b: Optional[float],
    round_start_loadout_a: Optional[float],
    round_start_loadout_b: Optional[float],
    round_start_armor_a: Optional[float] = None,
    round_start_armor_b: Optional[float] = None,
) -> tuple[float, dict]:
    """Compute a small round-start context offset from latched values only. Returns (offset, debug_dict).
    Offset is zero-safe when fields are missing; total offset is hard-clipped to ROUND_CONTEXT_OFFSET_MAX.
    """
    debug = {
        "round_ctx_econ_component": 0.0,
        "round_ctx_loadout_component": 0.0,
        "round_ctx_armor_component": 0.0,
        "round_ctx_side_component": 0.0,
    }
    total = 0.0

    # Econ: advantage (A - B) / (A + B) scaled modestly
    if round_start_econ_a is not None and round_start_econ_b is not None:
        try:
            a, b = float(round_start_econ_a), float(round_start_econ_b)
            s = a + b + 1e-6
            adv = (a - b) / s
            comp = float(np.clip(adv * 0.015, -0.015, 0.015))
            debug["round_ctx_econ_component"] = comp
            total += comp
        except (TypeError, ValueError):
            pass

    # Loadout: same idea
    if round_start_loadout_a is not None and round_start_loadout_b is not None:
        try:
            a, b = float(round_start_loadout_a), float(round_start_loadout_b)
            s = a + b + 1e-6
            adv = (a - b) / s
            comp = float(np.clip(adv * 0.012, -0.012, 0.012))
            debug["round_ctx_loadout_component"] = comp
            total += comp
        except (TypeError, ValueError):
            pass

    # Armor: optional, small effect
    if round_start_armor_a is not None and round_start_armor_b is not None:
        try:
            a, b = float(round_start_armor_a), float(round_start_armor_b)
            s = a + b + 1e-6
            adv = (a - b) / s
            comp = float(np.clip(adv * 0.008, -0.008, 0.008))
            debug["round_ctx_armor_component"] = comp
            total += comp
        except (TypeError, ValueError):
            pass

    # Side/map: Team A side vs map CT rate (small bias)
    if map_name and team_a_side and str(team_a_side).strip().upper() in ("CT", "T"):
        try:
            ct_rate = CS2_MAP_CT_RATE.get(str(map_name).strip(), CS2_CT_RATE_AVG)
            ct_rate = float(ct_rate)
            # Team A is CT -> bias toward A when map is CT-favored
            side_a = str(team_a_side).strip().upper()
            if side_a == "CT":
                comp = float(np.clip((ct_rate - 0.5) * 0.02, -0.01, 0.01))
            else:
                comp = float(np.clip((0.5 - ct_rate) * 0.02, -0.01, 0.01))
            debug["round_ctx_side_component"] = comp
            total += comp
        except (TypeError, ValueError, KeyError):
            pass

    offset = float(np.clip(total, -ROUND_CONTEXT_OFFSET_MAX, ROUND_CONTEXT_OFFSET_MAX))
    return (offset, debug)


def _compute_cs2_branch_endpoint_envelope_debug(
    endpoint_a_base: float,
    endpoint_b_base: float,
    rounds_a: int,
    rounds_b: int,
    best_of: int,
    maps_a_won: int,
    maps_b_won: int,
    win_target: int = 13,
) -> dict:
    """Pure helper: compute per-branch endpoint envelope diagnostics from base endpoints,
    round score, and series state. Dead-path utility for testing; no integration into live pipeline.
    Returns a debug dict with env_valid=False and None fields on invalid inputs.
    """
    def _clip01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    invalid = {
        "env_valid": False,
        "endpoint_a_base": None,
        "endpoint_b_base": None,
        "endpoint_base_span_abs": None,
        "endpoint_env_fraction_k": None,
        "endpoint_env_round_leverage": None,
        "endpoint_env_series_leverage": None,
        "endpoint_env_half_width": None,
        "endpoint_a_env_low": None,
        "endpoint_a_env_high": None,
        "endpoint_b_env_low": None,
        "endpoint_b_env_high": None,
        "endpoint_a_env_center": None,
        "endpoint_b_env_center": None,
        "endpoint_env_map_index": None,
        "endpoint_env_best_of": None,
        "endpoint_env_win_target": None,
    }
    try:
        ea = float(endpoint_a_base)
        eb = float(endpoint_b_base)
        ra = int(rounds_a)
        rb = int(rounds_b)
        bo = int(best_of)
        ma = int(maps_a_won)
        mb = int(maps_b_won)
        wt = int(win_target)
    except (TypeError, ValueError):
        return invalid
    if wt <= 0 or ra < 0 or rb < 0 or ma < 0 or mb < 0 or bo not in (3, 5):
        return invalid

    # 1) Round leverage L_round in [0,1]
    leader = max(ra, rb)
    P = leader / float(wt)
    d = abs(ra - rb)
    C = 1.0 - min(d / 8.0, 1.0)
    L_round = _clip01(0.6 * P + 0.4 * C)

    # 2) Series leverage L_series in [0,1]
    map_idx = ma + mb + 1
    if bo == 3:
        series_leverage_map = {1: 0.45, 2: 0.65, 3: 1.00}
    elif bo == 5:
        series_leverage_map = {1: 0.30, 2: 0.45, 3: 0.65, 4: 0.85, 5: 1.00}
    else:
        series_leverage_map = {}
    L_series = series_leverage_map.get(map_idx, 0.60)

    # 3) Envelope fraction k
    k_raw = 0.08 + 0.14 * (0.75 * L_round + 0.25 * L_series)
    k = float(np.clip(k_raw, 0.08, 0.22))

    # 4) Base span and envelope half-width
    base_span = abs(ea - eb)
    env_half_width = base_span * k

    # 5) Branch-oriented envelope bounds (do NOT sort A/B together)
    a_env_low = _clip01(ea - env_half_width)
    a_env_high = _clip01(ea + env_half_width)
    b_env_low = _clip01(eb - env_half_width)
    b_env_high = _clip01(eb + env_half_width)

    # 6) Centers
    a_env_center = 0.5 * (a_env_low + a_env_high)
    b_env_center = 0.5 * (b_env_low + b_env_high)

    return {
        "env_valid": True,
        "endpoint_a_base": ea,
        "endpoint_b_base": eb,
        "endpoint_base_span_abs": base_span,
        "endpoint_env_fraction_k": k,
        "endpoint_env_round_leverage": L_round,
        "endpoint_env_series_leverage": L_series,
        "endpoint_env_half_width": env_half_width,
        "endpoint_a_env_low": a_env_low,
        "endpoint_a_env_high": a_env_high,
        "endpoint_b_env_low": b_env_low,
        "endpoint_b_env_high": b_env_high,
        "endpoint_a_env_center": a_env_center,
        "endpoint_b_env_center": b_env_center,
        "endpoint_env_map_index": map_idx,
        "endpoint_env_best_of": bo,
        "endpoint_env_win_target": wt,
    }


if False:  # Optional disabled self-test: uncomment to inspect outputs
    _env = _compute_cs2_branch_endpoint_envelope_debug(0.55, 0.45, 6, 5, 3, 0, 0, 13)
    print("BO3 map1 6-5:", _env)
    _env2 = _compute_cs2_branch_endpoint_envelope_debug(0.72, 0.28, 12, 10, 5, 1, 0, 13)
    print("BO5 map2 12-10:", _env2)
    _inv = _compute_cs2_branch_endpoint_envelope_debug(0.5, 0.5, -1, 0, 3, 0, 0, 13)
    print("Invalid:", _inv.get("env_valid"), _inv.get("endpoint_a_base"))


def _compute_cs2_branch_econ_consequence_debug(
    team_a_alive_count: float,
    team_b_alive_count: float,
    team_a_alive_loadout_est_total: float,
    team_b_alive_loadout_est_total: float,
    team_a_alive_cash_total: float,
    team_b_alive_cash_total: float,
) -> dict:
    """Pure helper: branch econ consequence quality (A-favored vs B-favored) from normalized alive resource fields.
    Debug-only; no loss-bonus simulation. Returns econ_valid=False and None fields on invalid input.
    """
    invalid = {
        "econ_valid": False,
        "endpoint_econ_d_alive_survival": None,
        "endpoint_econ_d_alive_loadout": None,
        "endpoint_econ_d_alive_cash": None,
        "endpoint_econ_score_a": None,
        "endpoint_econ_score_b": None,
    }
    try:
        n_a = float(team_a_alive_count)
        n_b = float(team_b_alive_count)
        load_a = float(team_a_alive_loadout_est_total)
        load_b = float(team_b_alive_loadout_est_total)
        cash_a = float(team_a_alive_cash_total)
        cash_b = float(team_b_alive_cash_total)
    except (TypeError, ValueError):
        return invalid
    if not (np.isfinite(n_a) and np.isfinite(n_b) and np.isfinite(load_a) and np.isfinite(load_b)
            and np.isfinite(cash_a) and np.isfinite(cash_b)):
        return invalid

    d_alive_survival = (n_a - n_b) / 5.0
    d_alive_survival = float(np.clip(d_alive_survival, -1.0, 1.0))

    loadout_sum = load_a + load_b
    if loadout_sum >= 1e-9:
        d_alive_loadout = (load_a - load_b) / loadout_sum
    else:
        d_alive_loadout = 0.0
    d_alive_loadout = float(np.clip(d_alive_loadout, -1.0, 1.0))

    cash_sum = cash_a + cash_b
    if cash_sum >= 1e-9:
        d_alive_cash = (cash_a - cash_b) / cash_sum
    else:
        d_alive_cash = 0.0
    d_alive_cash = float(np.clip(d_alive_cash, -1.0, 1.0))

    econ_score_a = 0.45 * d_alive_survival + 0.35 * d_alive_loadout + 0.20 * d_alive_cash
    econ_score_a = float(np.clip(econ_score_a, -1.0, 1.0))
    econ_score_b = -econ_score_a

    return {
        "econ_valid": True,
        "endpoint_econ_d_alive_survival": d_alive_survival,
        "endpoint_econ_d_alive_loadout": d_alive_loadout,
        "endpoint_econ_d_alive_cash": d_alive_cash,
        "endpoint_econ_score_a": econ_score_a,
        "endpoint_econ_score_b": econ_score_b,
    }


def _compute_cs2_branch_endpoint_position_debug(
    a_env_low: float,
    a_env_high: float,
    b_env_low: float,
    b_env_high: float,
    team_a_alive_count: float,
    team_b_alive_count: float,
    team_a_alive_loadout_total: float,
    team_b_alive_loadout_total: float,
    team_a_armor_alive_total: float,
    team_b_armor_alive_total: float,
    branch_temp: float = 0.40,
) -> dict:
    """Pure helper: branch-quality position inside endpoint envelopes from intraround microstate.
    Dead-path diagnostics only; do not use for p_hat. Returns pos_valid=False and None fields on invalid input.
    """
    EPS = 1e-9
    invalid = {
        "pos_valid": False,
        "branch_temp_used": None,
        "endpoint_pos_d_alive": None,
        "endpoint_pos_d_loadout": None,
        "endpoint_pos_d_armor": None,
        "endpoint_pos_score_a": None,
        "endpoint_pos_score_b": None,
        "endpoint_pos_a_quality_pos": None,
        "endpoint_pos_b_quality_pos": None,
        "endpoint_pos_a_active_dbg": None,
        "endpoint_pos_b_active_dbg": None,
        "endpoint_pos_a_active_within_env": None,
        "endpoint_pos_b_active_within_env": None,
        "endpoint_pos_loadout_sum": None,
        "endpoint_pos_armor_sum": None,
    }
    try:
        a_lo = float(a_env_low)
        a_hi = float(a_env_high)
        b_lo = float(b_env_low)
        b_hi = float(b_env_high)
        n_a = float(team_a_alive_count)
        n_b = float(team_b_alive_count)
        load_a = float(team_a_alive_loadout_total)
        load_b = float(team_b_alive_loadout_total)
        arm_a = float(team_a_armor_alive_total)
        arm_b = float(team_b_armor_alive_total)
        temp = float(branch_temp)
    except (TypeError, ValueError):
        return invalid
    if not (np.isfinite(a_lo) and np.isfinite(a_hi) and np.isfinite(b_lo) and np.isfinite(b_hi)
            and np.isfinite(n_a) and np.isfinite(n_b) and np.isfinite(load_a) and np.isfinite(load_b)
            and np.isfinite(arm_a) and np.isfinite(arm_b) and np.isfinite(temp)):
        return invalid
    temp = max(float(temp), 1e-6)

    # Normalized microstate diffs
    d_alive = (n_a - n_b) / 5.0
    d_alive = float(np.clip(d_alive, -1.0, 1.0))

    loadout_sum = load_a + load_b
    if loadout_sum > EPS:
        d_loadout = (load_a - load_b) / loadout_sum
    else:
        d_loadout = 0.0
    d_loadout = float(np.clip(d_loadout, -1.0, 1.0))

    armor_sum = arm_a + arm_b
    if armor_sum > EPS:
        d_armor = (arm_a - arm_b) / armor_sum
    else:
        d_armor = 0.0
    d_armor = float(np.clip(d_armor, -1.0, 1.0))

    # Branch-quality score (50/30/20)
    s_A = 0.50 * d_alive + 0.30 * d_loadout + 0.20 * d_armor
    s_B = -s_A
    s_A = float(np.clip(s_A, -5.0, 5.0))
    s_B = float(np.clip(s_B, -5.0, 5.0))

    # Map to [0,1] via sigmoid
    a_quality_pos = _sigmoid_stable(s_A, temp)
    b_quality_pos = _sigmoid_stable(s_B, temp)

    # Active endpoints inside envelope (linear interpolation)
    a_active_dbg = a_lo + (a_hi - a_lo) * a_quality_pos
    b_active_dbg = b_lo + (b_hi - b_lo) * b_quality_pos

    a_active_within_env = bool((a_lo - 1e-9) <= a_active_dbg <= (a_hi + 1e-9))
    b_active_within_env = bool((b_lo - 1e-9) <= b_active_dbg <= (b_hi + 1e-9))

    return {
        "pos_valid": True,
        "branch_temp_used": temp,
        "endpoint_pos_d_alive": d_alive,
        "endpoint_pos_d_loadout": d_loadout,
        "endpoint_pos_d_armor": d_armor,
        "endpoint_pos_score_a": s_A,
        "endpoint_pos_score_b": s_B,
        "endpoint_pos_a_quality_pos": a_quality_pos,
        "endpoint_pos_b_quality_pos": b_quality_pos,
        "endpoint_pos_a_active_dbg": a_active_dbg,
        "endpoint_pos_b_active_dbg": b_active_dbg,
        "endpoint_pos_a_active_within_env": a_active_within_env,
        "endpoint_pos_b_active_within_env": b_active_within_env,
        "endpoint_pos_loadout_sum": loadout_sum,
        "endpoint_pos_armor_sum": armor_sum,
    }


def _debug_run_cs2_endpoint_model_scenarios() -> None:
    """Print-only debug harness for CS2 endpoint envelope + position helpers. No session state or Streamlit.
    Run manually by enabling the 'if False' block that calls this."""
    scenarios = [
        {
            "name": "1. Neutral early BO3 map1",
            "endpoint_a_base": 0.57,
            "endpoint_b_base": 0.43,
            "rounds_a": 2,
            "rounds_b": 1,
            "best_of": 3,
            "maps_a_won": 0,
            "maps_b_won": 0,
            "a_alive": 5,
            "b_alive": 5,
            "a_loadout": 22000.0,
            "b_loadout": 22000.0,
            "a_armor": 500.0,
            "b_armor": 500.0,
            "q_intra_assumed": 0.50,
        },
        {
            "name": "2. Mid-round A advantage (alive + loadout + armor)",
            "endpoint_a_base": 0.57,
            "endpoint_b_base": 0.43,
            "rounds_a": 2,
            "rounds_b": 1,
            "best_of": 3,
            "maps_a_won": 0,
            "maps_b_won": 0,
            "a_alive": 5,
            "b_alive": 3,
            "a_loadout": 24000.0,
            "b_loadout": 14000.0,
            "a_armor": 450.0,
            "b_armor": 220.0,
            "q_intra_assumed": 0.65,
        },
        {
            "name": "3. Alive equal, A gear-trash",
            "endpoint_a_base": 0.57,
            "endpoint_b_base": 0.43,
            "rounds_a": 2,
            "rounds_b": 1,
            "best_of": 3,
            "maps_a_won": 0,
            "maps_b_won": 0,
            "a_alive": 3,
            "b_alive": 3,
            "a_loadout": 2500.0,
            "b_loadout": 15000.0,
            "a_armor": 50.0,
            "b_armor": 250.0,
            "q_intra_assumed": 0.50,
        },
        {
            "name": "4. Armor-only advantage check",
            "endpoint_a_base": 0.57,
            "endpoint_b_base": 0.43,
            "rounds_a": 2,
            "rounds_b": 1,
            "best_of": 3,
            "maps_a_won": 0,
            "maps_b_won": 0,
            "a_alive": 4,
            "b_alive": 4,
            "a_loadout": 16000.0,
            "b_loadout": 16000.0,
            "a_armor": 380.0,
            "b_armor": 80.0,
            "q_intra_assumed": 0.50,
        },
        {
            "name": "5. Close late BO3 decider (11-11, map3)",
            "endpoint_a_base": 0.62,
            "endpoint_b_base": 0.38,
            "rounds_a": 11,
            "rounds_b": 11,
            "best_of": 3,
            "maps_a_won": 1,
            "maps_b_won": 1,
            "a_alive": 5,
            "b_alive": 2,
            "a_loadout": 26000.0,
            "b_loadout": 9000.0,
            "a_armor": 500.0,
            "b_armor": 120.0,
            "q_intra_assumed": 0.80,
        },
        {
            "name": "6. High-leverage but ugly A win path (1v1 low gear)",
            "endpoint_a_base": 0.62,
            "endpoint_b_base": 0.38,
            "rounds_a": 11,
            "rounds_b": 11,
            "best_of": 3,
            "maps_a_won": 1,
            "maps_b_won": 1,
            "a_alive": 1,
            "b_alive": 1,
            "a_loadout": 3500.0,
            "b_loadout": 2500.0,
            "a_armor": 0.0,
            "b_armor": 0.0,
            "q_intra_assumed": 0.60,
        },
        {
            "name": "7. BO5 late-map leverage check (map4)",
            "endpoint_a_base": 0.55,
            "endpoint_b_base": 0.45,
            "rounds_a": 10,
            "rounds_b": 10,
            "best_of": 5,
            "maps_a_won": 2,
            "maps_b_won": 1,
            "a_alive": 4,
            "b_alive": 3,
            "a_loadout": 20000.0,
            "b_loadout": 15000.0,
            "a_armor": 300.0,
            "b_armor": 180.0,
            "q_intra_assumed": 0.60,
        },
        {
            "name": "8. Invalid input smoke test (best_of=7)",
            "endpoint_a_base": 0.55,
            "endpoint_b_base": 0.45,
            "rounds_a": 5,
            "rounds_b": 5,
            "best_of": 7,
            "maps_a_won": 0,
            "maps_b_won": 0,
            "a_alive": 5,
            "b_alive": 5,
            "a_loadout": 20000.0,
            "b_loadout": 20000.0,
            "a_armor": 300.0,
            "b_armor": 300.0,
            "q_intra_assumed": 0.50,
        },
    ]
    win_target = 13
    for s in scenarios:
        name = s.get("name", "?")
        q_intra = float(s.get("q_intra_assumed", 0.50))
        env = _compute_cs2_branch_endpoint_envelope_debug(
            endpoint_a_base=s["endpoint_a_base"],
            endpoint_b_base=s["endpoint_b_base"],
            rounds_a=s["rounds_a"],
            rounds_b=s["rounds_b"],
            best_of=s["best_of"],
            maps_a_won=s["maps_a_won"],
            maps_b_won=s["maps_b_won"],
            win_target=win_target,
        )
        a_lo = env.get("endpoint_a_env_low") if env.get("env_valid") else None
        a_hi = env.get("endpoint_a_env_high") if env.get("env_valid") else None
        b_lo = env.get("endpoint_b_env_low") if env.get("env_valid") else None
        b_hi = env.get("endpoint_b_env_high") if env.get("env_valid") else None
        pos = None
        if a_lo is not None and a_hi is not None and b_lo is not None and b_hi is not None:
            pos = _compute_cs2_branch_endpoint_position_debug(
                a_env_low=a_lo,
                a_env_high=a_hi,
                b_env_low=b_lo,
                b_env_high=b_hi,
                team_a_alive_count=float(s["a_alive"]),
                team_b_alive_count=float(s["b_alive"]),
                team_a_alive_loadout_total=float(s["a_loadout"]),
                team_b_alive_loadout_total=float(s["b_loadout"]),
                team_a_armor_alive_total=float(s["a_armor"]),
                team_b_armor_alive_total=float(s["b_armor"]),
                branch_temp=0.40,
            )
        else:
            pos = {"pos_valid": False}
        p_hat_dynamic_dbg = None
        if env.get("env_valid") and pos.get("pos_valid"):
            aa = pos.get("endpoint_pos_a_active_dbg")
            bb = pos.get("endpoint_pos_b_active_dbg")
            if aa is not None and bb is not None:
                p_hat_dynamic_dbg = q_intra * aa + (1.0 - q_intra) * bb
        print("=" * 60)
        print(name)
        print("-" * 40)
        print("  Envelope: valid=%s" % env.get("env_valid", False))
        if env.get("env_valid"):
            print("    base_span=%.4f  k=%.4f  L_round=%.4f  L_series=%.4f" % (
                env.get("endpoint_base_span_abs", 0) or 0,
                env.get("endpoint_env_fraction_k", 0) or 0,
                env.get("endpoint_env_round_leverage", 0) or 0,
                env.get("endpoint_env_series_leverage", 0) or 0,
            ))
            print("    A env [%.4f, %.4f]  B env [%.4f, %.4f]" % (
                env.get("endpoint_a_env_low", 0) or 0,
                env.get("endpoint_a_env_high", 0) or 0,
                env.get("endpoint_b_env_low", 0) or 0,
                env.get("endpoint_b_env_high", 0) or 0,
            ))
        else:
            print("    (invalid envelope)")
        print("  Position: valid=%s" % pos.get("pos_valid", False))
        if pos.get("pos_valid"):
            print("    d_alive=%.4f  d_loadout=%.4f  d_armor=%.4f" % (
                pos.get("endpoint_pos_d_alive", 0) or 0,
                pos.get("endpoint_pos_d_loadout", 0) or 0,
                pos.get("endpoint_pos_d_armor", 0) or 0,
            ))
            print("    s_A=%.4f  s_B=%.4f" % (
                pos.get("endpoint_pos_score_a", 0) or 0,
                pos.get("endpoint_pos_score_b", 0) or 0,
            ))
            print("    a_quality_pos=%.4f  b_quality_pos=%.4f" % (
                pos.get("endpoint_pos_a_quality_pos", 0) or 0,
                pos.get("endpoint_pos_b_quality_pos", 0) or 0,
            ))
            print("    a_active_dbg=%.4f  b_active_dbg=%.4f" % (
                pos.get("endpoint_pos_a_active_dbg", 0) or 0,
                pos.get("endpoint_pos_b_active_dbg", 0) or 0,
            ))
            print("    a_within_env=%s  b_within_env=%s" % (
                pos.get("endpoint_pos_a_active_within_env", False),
                pos.get("endpoint_pos_b_active_within_env", False),
            ))
        else:
            print("    (invalid position)")
        print("  q_intra_assumed=%.2f" % q_intra)
        if p_hat_dynamic_dbg is not None:
            print("  p_hat_dynamic_dbg=%.4f" % p_hat_dynamic_dbg)
        print("")
    print("=" * 60)
    print("debug scenarios done")


def _compute_cs2_phat_intra_score_debug(
    team_a_alive_count,
    team_b_alive_count,
    team_a_hp_alive_total,
    team_b_hp_alive_total,
    team_a_alive_loadout_est_total,
    team_b_alive_loadout_est_total,
    bomb_planted,
    round_phase,
    round_time_remaining_s,
    team_a_side,
    q_temp: float = 1.0,
    reliability_mult: float = 1.0,
) -> dict:
    """Pure helper: structural q_intra candidate from BO3-normalized fields for diagnostics only.
    Alive/HP/Loadout, bomb, clock, bomb×clock interaction, alive-quality scaling. No session state."""
    _none = {
        "phat_intra_valid": False,
        "phat_intra_side_known": None,
        "phat_intra_clock_known": None,
        "phat_intra_bomb_planted_bool": None,
        "phat_intra_round_phase_norm": None,
        "phat_intra_phase_mult": None,
        "phat_intra_urgency": None,
        "phat_intra_d_alive": None,
        "phat_intra_d_hp": None,
        "phat_intra_d_loadout": None,
        "phat_intra_bomb_term_signed": None,
        "phat_intra_clock_term_signed": None,
        "phat_intra_bomb_clock_combo_signed": None,
        "phat_intra_loadout_presence": None,
        "phat_intra_w_alive_eff": None,
        "phat_intra_score_alive": None,
        "phat_intra_score_hp": None,
        "phat_intra_score_loadout": None,
        "phat_intra_score_bomb": None,
        "phat_intra_score_clock": None,
        "phat_intra_score_bomb_clock_combo": None,
        "phat_intra_score_raw_pre_reliability": None,
        "phat_intra_score_raw_post_reliability": None,
        "phat_intra_q_temp_used": None,
        "phat_intra_q_intra_round_win_a_dbg": None,
    }
    try:
        n_a = float(team_a_alive_count) if team_a_alive_count is not None else float("nan")
        n_b = float(team_b_alive_count) if team_b_alive_count is not None else float("nan")
        hp_a = float(team_a_hp_alive_total) if team_a_hp_alive_total is not None else float("nan")
        hp_b = float(team_b_hp_alive_total) if team_b_hp_alive_total is not None else float("nan")
        load_a = float(team_a_alive_loadout_est_total) if team_a_alive_loadout_est_total is not None else float("nan")
        load_b = float(team_b_alive_loadout_est_total) if team_b_alive_loadout_est_total is not None else float("nan")
        temp = float(q_temp) if q_temp is not None else 1.0
        rel_mult = float(reliability_mult) if reliability_mult is not None else 1.0
    except (TypeError, ValueError):
        return _none
    if not (np.isfinite(n_a) and np.isfinite(n_b)):
        return _none
    temp = max(temp, 1e-6)

    # B) Normalized diffs (clip to [-1, 1])
    d_alive = (n_a - n_b) / 5.0
    d_alive = float(np.clip(d_alive, -1.0, 1.0))

    hp_sum = (hp_a + hp_b) if (np.isfinite(hp_a) and np.isfinite(hp_b)) else 0.0
    if hp_sum >= 1e-9:
        d_hp = (hp_a - hp_b) / hp_sum
    else:
        d_hp = 0.0
    d_hp = float(np.clip(d_hp, -1.0, 1.0))

    _load_sum_raw = (load_a + load_b) if (np.isfinite(load_a) and np.isfinite(load_b)) else 0.0
    if _load_sum_raw >= 1e-9:
        d_loadout = (load_a - load_b) / _load_sum_raw
    else:
        d_loadout = 0.0
    d_loadout = float(np.clip(d_loadout, -1.0, 1.0))

    # C) Side normalization
    side_str = str(team_a_side).strip().upper() if team_a_side is not None else ""
    if side_str in ("T", "CT"):
        side_known = True
    else:
        side_known = False
        side_str = ""

    # D) Bomb term (signed toward Team A)
    bomb_bool = bool(bomb_planted) if bomb_planted is not None else False
    if not bomb_bool:
        bomb_term_signed = 0.0
    elif side_str == "T":
        bomb_term_signed = 1.0
    elif side_str == "CT":
        bomb_term_signed = -1.0
    else:
        bomb_term_signed = 0.0

    # E) Clock / phase term
    round_phase_norm = str(round_phase).lower() if round_phase is not None else ""
    phase_is_freeze_buy = "freeze" in round_phase_norm or "buy" in round_phase_norm
    phase_is_post_end = "end" in round_phase_norm or "post" in round_phase_norm
    phase_mult = 0.0 if (phase_is_freeze_buy or phase_is_post_end) else 1.0

    t_rem = float(round_time_remaining_s) if round_time_remaining_s is not None and np.isfinite(float(round_time_remaining_s)) else None
    if t_rem is not None:
        clock_known = True
        time_clamped = float(np.clip(t_rem, 0.0, 40.0))
        urgency = 1.0 - (time_clamped / 40.0)
    else:
        clock_known = False
        urgency = 0.0

    if bomb_bool:
        if side_str == "T":
            clock_term_signed = urgency
        elif side_str == "CT":
            clock_term_signed = -urgency
        else:
            clock_term_signed = 0.0
    else:
        if side_str == "CT":
            clock_term_signed = urgency
        elif side_str == "T":
            clock_term_signed = -urgency
        else:
            clock_term_signed = 0.0

    bomb_term_signed *= phase_mult
    clock_term_signed *= phase_mult

    # F) Alive-quality scaling
    _load_sum = max(_load_sum_raw, 1e-9) if np.isfinite(_load_sum_raw) else 1e-9
    loadout_presence = float(np.clip(_load_sum / 30000.0, 0.0, 1.0))
    w_alive_base = 0.15
    w_alive_bonus = 0.15
    w_alive_eff = w_alive_base + w_alive_bonus * loadout_presence

    # G) Component weights
    w_hp = 0.20
    w_loadout = 0.20
    w_bomb = 0.10
    w_clock = 0.05
    w_bomb_clock_combo = 0.15

    bomb_clock_combo_signed = bomb_term_signed * abs(clock_term_signed)

    # H) Component scores
    score_alive = w_alive_eff * d_alive
    score_hp = w_hp * d_hp
    score_loadout = w_loadout * d_loadout
    score_bomb = w_bomb * bomb_term_signed
    score_clock = w_clock * clock_term_signed
    score_bomb_clock_combo = w_bomb_clock_combo * bomb_clock_combo_signed

    # I) Raw score + q
    raw_pre = score_alive + score_hp + score_loadout + score_bomb + score_clock + score_bomb_clock_combo
    raw_post = raw_pre * rel_mult
    q_dbg = _sigmoid_stable(raw_post, temp)

    return {
        "phat_intra_valid": True,
        "phat_intra_side_known": side_known,
        "phat_intra_clock_known": clock_known,
        "phat_intra_bomb_planted_bool": bomb_bool,
        "phat_intra_round_phase_norm": round_phase_norm,
        "phat_intra_phase_mult": phase_mult,
        "phat_intra_urgency": urgency,
        "phat_intra_d_alive": d_alive,
        "phat_intra_d_hp": d_hp,
        "phat_intra_d_loadout": d_loadout,
        "phat_intra_bomb_term_signed": bomb_term_signed,
        "phat_intra_clock_term_signed": clock_term_signed,
        "phat_intra_bomb_clock_combo_signed": bomb_clock_combo_signed,
        "phat_intra_loadout_presence": loadout_presence,
        "phat_intra_w_alive_eff": w_alive_eff,
        "phat_intra_score_alive": score_alive,
        "phat_intra_score_hp": score_hp,
        "phat_intra_score_loadout": score_loadout,
        "phat_intra_score_bomb": score_bomb,
        "phat_intra_score_clock": score_clock,
        "phat_intra_score_bomb_clock_combo": score_bomb_clock_combo,
        "phat_intra_score_raw_pre_reliability": raw_pre,
        "phat_intra_score_raw_post_reliability": raw_post,
        "phat_intra_q_temp_used": temp,
        "phat_intra_q_intra_round_win_a_dbg": q_dbg,
    }


def _compute_cs2_midround_features(
    team_a_alive_count=None,
    team_b_alive_count=None,
    team_a_hp_alive_total=None,
    team_b_hp_alive_total=None,
    bomb_planted=None,
    round_time_remaining_s=None,
    round_phase=None,
    a_side=None,
    team_a_armor_alive_total=None,
    team_b_armor_alive_total=None,
    team_a_alive_loadout_total=None,
    team_b_alive_loadout_total=None,
    live_source=None,
    grid_used_reduced_features=None,
    grid_completeness_score=None,
    grid_staleness_seconds=None,
    grid_has_players=None,
    grid_has_clock=None,
) -> dict:
    """Build intraround features from BO3/GRID session state. feature_ok=False if alive missing. V2: armor, loadout, reliability_mult."""
    out = {
        "alive_diff": 0,
        "hp_diff_alive": 0.0,
        "bomb_planted": 0,
        "time_remaining_s": 0.0,
        "time_progress": 0.5,
        "is_live_round_phase": True,
        "feature_ok": False,
        "armor_diff_alive": 0.0,
        "loadout_diff_alive": 0.0,
        "reliability_mult": 1.0,
        "feature_has_armor": False,
        "feature_has_alive_loadout": False,
    }
    try:
        a_alive = int(team_a_alive_count) if team_a_alive_count is not None else None
        b_alive = int(team_b_alive_count) if team_b_alive_count is not None else None
        if a_alive is None or b_alive is None:
            return out
        out["alive_diff"] = a_alive - b_alive
        hp_a = float(team_a_hp_alive_total) if team_a_hp_alive_total is not None else 0.0
        hp_b = float(team_b_hp_alive_total) if team_b_hp_alive_total is not None else 0.0
        out["hp_diff_alive"] = hp_a - hp_b
        out["hp_a_total"] = hp_a
        out["hp_b_total"] = hp_b
        out["bomb_planted"] = 1 if bomb_planted else 0
        t_rem = float(round_time_remaining_s) if round_time_remaining_s is not None else None
        if t_rem is not None and t_rem >= 0:
            out["time_remaining_s"] = t_rem
            # MR12 round ~115s; progress 0=start, 1=late
            out["time_progress"] = float(np.clip(1.0 - (t_rem / 115.0), 0.0, 1.0))
        else:
            out["time_progress"] = 0.5
        out["is_live_round_phase"] = str(round_phase).lower() not in ("ended", "freezetime", "warmup") if round_phase else True

        # V2: armor diff (alive only); fallback 0
        armor_a = float(team_a_armor_alive_total) if team_a_armor_alive_total is not None else 0.0
        armor_b = float(team_b_armor_alive_total) if team_b_armor_alive_total is not None else 0.0
        out["armor_diff_alive"] = armor_a - armor_b
        out["feature_has_armor"] = (team_a_armor_alive_total is not None or team_b_armor_alive_total is not None)

        # V2: loadout diff (alive only); fallback 0
        load_a = float(team_a_alive_loadout_total) if team_a_alive_loadout_total is not None else 0.0
        load_b = float(team_b_alive_loadout_total) if team_b_alive_loadout_total is not None else 0.0
        out["loadout_diff_alive"] = load_a - load_b
        out["load_a_total"] = load_a
        out["load_b_total"] = load_b
        out["feature_has_alive_loadout"] = (team_a_alive_loadout_total is not None or team_b_alive_loadout_total is not None)

        # V2: reliability_mult from GRID debug keys
        if str(live_source or "").strip() != "GRID":
            out["reliability_mult"] = 1.0
        elif grid_used_reduced_features is True:
            out["reliability_mult"] = RELIABILITY_MIN_MULT
        else:
            mult = 1.0
            if grid_completeness_score is not None and float(grid_completeness_score) < 0.80:
                mult *= 0.85
            if grid_staleness_seconds is not None:
                s = int(grid_staleness_seconds)
                if s > 30:
                    mult *= 0.60
                elif s > 15:
                    mult *= 0.80
            if grid_has_players is False:
                mult *= 0.70
            if grid_has_clock is False:
                mult *= 0.90
            out["reliability_mult"] = float(np.clip(mult, RELIABILITY_MIN_MULT, 1.0))

        out["feature_ok"] = True
    except Exception:
        pass
    return out


def _sigmoid_stable(x: float, temp: float = 1.0) -> float:
    """Numerically stable sigmoid; output in (0, 1). q = 1 / (1 + exp(-x/temp))."""
    if temp is None or temp <= 0:
        temp = 1.0
    z = float(x) / float(temp)
    if z >= 20.0:
        return 1.0
    if z <= -20.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-z)))


if False:  # OFF-by-default: set to True to run CS2 endpoint envelope+position scenarios
    _debug_run_cs2_endpoint_model_scenarios()


def _compute_cs2_intra_score(features: dict, settings: Optional[dict] = None) -> dict:
    """
    Compute intra-round round-win score (latent, unbounded) from same V2 components as mid-round adj.
    Used for series-endpoint mixture: q_intra = sigmoid(intra_score_raw), p_hat = q*endpoint_A + (1-q)*endpoint_B.
    Returns dict: intra_score_alive, intra_score_hp, intra_score_bomb, intra_score_armor, intra_score_loadout,
    intra_score_post_reliability, intra_score_time_scale, intra_score_raw (unbounded).
    """
    if settings is None:
        settings = {}
    w_alive = float(settings.get("alive_weight", ALIVE_DIFF_WEIGHT_PER_PLAYER))
    w_hp = float(settings.get("hp_weight", HP_DIFF_WEIGHT_PER_100HP))
    w_bomb = float(settings.get("bomb_weight", BOMB_PLANTED_WEIGHT))
    w_armor = float(settings.get("armor_weight", ARMOR_DIFF_WEIGHT_PER_100))
    w_loadout = float(settings.get("loadout_weight", LOADOUT_DIFF_WEIGHT_PER_1000))
    t_min = float(settings.get("time_scale_min", TIME_SCALE_MIN))
    t_max = float(settings.get("time_scale_max", TIME_SCALE_MAX))
    rel_min = float(settings.get("reliability_min_mult", RELIABILITY_MIN_MULT))
    a_side = str(settings.get("a_side", "") or "").strip().upper()

    intra_score_alive = float(features.get("alive_diff", 0)) * w_alive
    intra_score_hp = (float(features.get("hp_diff_alive", 0)) / 100.0) * w_hp
    bomb = features.get("bomb_planted", 0)
    if bomb:
        intra_score_bomb = w_bomb if a_side == "T" else -w_bomb
    else:
        intra_score_bomb = 0.0
    intra_score_armor = (float(features.get("armor_diff_alive", 0)) / 100.0) * w_armor
    intra_score_loadout = (float(features.get("loadout_diff_alive", 0)) / 1000.0) * w_loadout
    reliability_mult = float(np.clip(features.get("reliability_mult", 1.0), rel_min, 1.0))

    raw_pre_rel = intra_score_alive + intra_score_hp + intra_score_bomb + intra_score_armor + intra_score_loadout
    intra_score_post_reliability = raw_pre_rel * reliability_mult
    time_progress = float(np.clip(features.get("time_progress", 0.5), 0.0, 1.0))
    intra_score_time_scale = t_min + (t_max - t_min) * time_progress
    intra_score_raw = intra_score_post_reliability * intra_score_time_scale

    return {
        "intra_score_alive": intra_score_alive,
        "intra_score_hp": intra_score_hp,
        "intra_score_bomb": intra_score_bomb,
        "intra_score_armor": intra_score_armor,
        "intra_score_loadout": intra_score_loadout,
        "intra_score_post_reliability": intra_score_post_reliability,
        "intra_score_time_scale": intra_score_time_scale,
        "intra_score_raw": intra_score_raw,
    }


def _apply_cs2_midround_adjustment_v1(
    p_base: float,
    band_lo: float,
    band_hi: float,
    features: dict,
    settings: Optional[dict] = None,
) -> dict:
    """Apply conservative mid-round adjustment; clamp to corridor. BO3 MIDROUND V1."""
    eps = 1e-6
    if settings is None:
        settings = {}
    w_alive = float(settings.get("alive_weight", ALIVE_DIFF_WEIGHT_PER_PLAYER))
    w_hp = float(settings.get("hp_weight", HP_DIFF_WEIGHT_PER_100HP))
    w_bomb = float(settings.get("bomb_weight", BOMB_PLANTED_WEIGHT))
    t_min = float(settings.get("time_scale_min", TIME_SCALE_MIN))
    t_max = float(settings.get("time_scale_max", TIME_SCALE_MAX))
    max_adj = float(settings.get("max_abs_mid_adj", MAX_ABS_MID_ADJ))
    a_side = str(settings.get("a_side", "") or "").strip().upper()

    adj_alive = float(features.get("alive_diff", 0)) * w_alive
    adj_hp = (float(features.get("hp_diff_alive", 0)) / 100.0) * w_hp
    bomb = features.get("bomb_planted", 0)
    if bomb:
        # Bomb helps T, hurts CT. Team A is a_side; + for T, - for CT
        if a_side == "T":
            adj_bomb = w_bomb
        else:
            adj_bomb = -w_bomb
    else:
        adj_bomb = 0.0
    time_progress = float(np.clip(features.get("time_progress", 0.5), 0.0, 1.0))
    time_scale = t_min + (t_max - t_min) * time_progress
    mid_adj_total = (adj_alive + adj_hp + adj_bomb) * time_scale
    mid_adj_total = float(np.clip(mid_adj_total, -max_adj, max_adj))
    p_mid_raw = p_base + mid_adj_total
    if band_hi <= band_lo or band_lo is None or band_hi is None:
        p_mid_clamped = p_base
        mid_clamped_hit = False
        mid_clamp_distance = 0.0
    else:
        lo_bound = band_lo + eps
        hi_bound = band_hi - eps
        p_mid_clamped = float(np.clip(p_mid_raw, lo_bound, hi_bound))
        mid_clamped_hit = abs(p_mid_raw - p_mid_clamped) > 1e-9
        mid_clamp_distance = p_mid_raw - p_mid_clamped
    return {
        "p_mid_raw": p_mid_raw,
        "p_mid_clamped": p_mid_clamped,
        "mid_adj_total": mid_adj_total,
        "mid_adj_alive": adj_alive,
        "mid_adj_bomb": adj_bomb,
        "mid_adj_hp": adj_hp,
        "mid_time_scale": time_scale,
        "mid_clamped_hit": mid_clamped_hit,
        "mid_clamp_distance": mid_clamp_distance,
    }


def _apply_cs2_midround_adjustment_v2(
    p_base: float,
    band_lo: float,
    band_hi: float,
    features: dict,
    settings: Optional[dict] = None,
) -> dict:
    """Apply mid-round adjustment V2: V1 terms + armor + loadout; reliability_mult on intraround total (before time scale + clip). Same return shape as V1 plus V2 debug keys."""
    eps = 1e-6
    if settings is None:
        settings = {}
    w_alive = float(settings.get("alive_weight", ALIVE_DIFF_WEIGHT_PER_PLAYER))
    w_hp = float(settings.get("hp_weight", HP_DIFF_WEIGHT_PER_100HP))
    w_bomb = float(settings.get("bomb_weight", BOMB_PLANTED_WEIGHT))
    w_armor = float(settings.get("armor_weight", ARMOR_DIFF_WEIGHT_PER_100))
    w_loadout = float(settings.get("loadout_weight", LOADOUT_DIFF_WEIGHT_PER_1000))
    t_min = float(settings.get("time_scale_min", TIME_SCALE_MIN))
    t_max = float(settings.get("time_scale_max", TIME_SCALE_MAX))
    max_adj = float(settings.get("max_abs_mid_adj", MAX_ABS_MID_ADJ))
    a_side = str(settings.get("a_side", "") or "").strip().upper()
    rel_min = float(settings.get("reliability_min_mult", RELIABILITY_MIN_MULT))

    adj_alive = float(features.get("alive_diff", 0)) * w_alive
    adj_hp = (float(features.get("hp_diff_alive", 0)) / 100.0) * w_hp
    bomb = features.get("bomb_planted", 0)
    if bomb:
        if a_side == "T":
            adj_bomb = w_bomb
        else:
            adj_bomb = -w_bomb
    else:
        adj_bomb = 0.0
    adj_armor = (float(features.get("armor_diff_alive", 0)) / 100.0) * w_armor
    adj_loadout = (float(features.get("loadout_diff_alive", 0)) / 1000.0) * w_loadout
    reliability_mult = float(np.clip(features.get("reliability_mult", 1.0), rel_min, 1.0))

    raw_total_pre_rel = adj_alive + adj_hp + adj_bomb + adj_armor + adj_loadout
    raw_total_post_rel = raw_total_pre_rel * reliability_mult

    time_progress = float(np.clip(features.get("time_progress", 0.5), 0.0, 1.0))
    time_scale = t_min + (t_max - t_min) * time_progress
    mid_adj_total = raw_total_post_rel * time_scale
    mid_adj_total = float(np.clip(mid_adj_total, -max_adj, max_adj))
    p_mid_raw = p_base + mid_adj_total
    if band_hi <= band_lo or band_lo is None or band_hi is None:
        p_mid_clamped = p_base
        mid_clamped_hit = False
        mid_clamp_distance = 0.0
    else:
        lo_bound = band_lo + eps
        hi_bound = band_hi - eps
        p_mid_clamped = float(np.clip(p_mid_raw, lo_bound, hi_bound))
        mid_clamped_hit = abs(p_mid_raw - p_mid_clamped) > 1e-9
        mid_clamp_distance = p_mid_raw - p_mid_clamped
    return {
        "p_mid_raw": p_mid_raw,
        "p_mid_clamped": p_mid_clamped,
        "mid_adj_total": mid_adj_total,
        "mid_adj_alive": adj_alive,
        "mid_adj_bomb": adj_bomb,
        "mid_adj_hp": adj_hp,
        "mid_adj_armor": adj_armor,
        "mid_adj_loadout": adj_loadout,
        "mid_time_scale": time_scale,
        "mid_clamped_hit": mid_clamped_hit,
        "mid_clamp_distance": mid_clamp_distance,
        "mid_reliability_mult": reliability_mult,
        "mid_feature_has_armor": bool(features.get("feature_has_armor", False)),
        "mid_feature_has_alive_loadout": bool(features.get("feature_has_alive_loadout", False)),
        "mid_raw_total_pre_reliability": raw_total_pre_rel,
        "mid_raw_total_post_reliability": raw_total_post_rel,
    }


def _apply_cs2_midround_adjustment_v2_mixture(
    p_if_a: float,
    p_if_b: float,
    features: dict,
    settings: Optional[dict] = None,
) -> dict:
    """
    Mid-round V2 mixture: q_intra from alive/hp/loadout/bomb/time; p_mid = p_if_b + q*(p_if_a - p_if_b); clamp to rails.
    No live cash in intraround q. Returns q, raw_score, component scores, temp, urgency, p_mid_raw, p_mid_clamped, clamp_hit.
    """
    eps = 1e-6
    if settings is None:
        settings = {}
    temp = float(settings.get("mixture_temp", 0.8))
    w_alive = float(settings.get("alive_weight", ALIVE_DIFF_WEIGHT_PER_PLAYER))
    w_hp = float(settings.get("hp_weight", HP_DIFF_WEIGHT_PER_100HP))
    w_bomb = float(settings.get("bomb_weight", BOMB_PLANTED_WEIGHT))
    w_loadout = float(settings.get("loadout_weight", LOADOUT_DIFF_WEIGHT_PER_1000))
    a_side = str(settings.get("a_side", "") or "").strip().upper()

    # alive_diff normalized by 5 (players)
    alive_diff = float(features.get("alive_diff", 0))
    score_alive = (alive_diff / 5.0) * w_alive if abs(w_alive) > 0 else 0.0

    # hp_diff normalized by (hp_a+hp_b) or per-100
    hp_diff = float(features.get("hp_diff_alive", 0))
    hp_a = float(features.get("hp_a_total", 0))
    hp_b = float(features.get("hp_b_total", 0))
    hp_sum = hp_a + hp_b
    if hp_sum > 0:
        score_hp = (hp_diff / hp_sum) * 100.0 * w_hp
    else:
        score_hp = (hp_diff / 100.0) * w_hp

    # loadout_diff normalized by (load_a+load_b) or use raw diff per 1000
    load_diff = float(features.get("loadout_diff_alive", 0))
    load_a = float(features.get("load_a_total", 0))
    load_b = float(features.get("load_b_total", 0))
    load_sum = load_a + load_b
    if load_sum > 0:
        score_loadout = (load_diff / load_sum) * 1000.0 * w_loadout
    else:
        score_loadout = (load_diff / 1000.0) * w_loadout

    # bomb term signed by a_side (bomb favors T, hurts CT)
    bomb = features.get("bomb_planted", 0)
    if bomb:
        score_bomb = w_bomb if a_side == "T" else -w_bomb
    else:
        score_bomb = 0.0

    # time_progress-based urgency with nonzero floor: 0.15 + 0.85*time_progress
    time_progress = float(np.clip(features.get("time_progress", 0.5), 0.0, 1.0))
    urgency = 0.15 + 0.85 * time_progress

    raw_score = (score_alive + score_hp + score_loadout + score_bomb) * urgency
    q = _sigmoid_stable(raw_score, temp)

    p_if_a_f = float(p_if_a)
    p_if_b_f = float(p_if_b)
    p_mid_raw = p_if_b_f + q * (p_if_a_f - p_if_b_f)

    lo_ep = min(p_if_a_f, p_if_b_f) + eps
    hi_ep = max(p_if_a_f, p_if_b_f) - eps
    p_mid_clamped = float(np.clip(p_mid_raw, lo_ep, hi_ep))
    clamp_hit = abs(p_mid_raw - p_mid_clamped) > 1e-9

    return {
        "q": q,
        "raw_score": raw_score,
        "score_alive": score_alive,
        "score_hp": score_hp,
        "score_loadout": score_loadout,
        "score_bomb": score_bomb,
        "temp": temp,
        "urgency": urgency,
        "p_mid_raw": p_mid_raw,
        "p_mid_clamped": p_mid_clamped,
        "clamp_hit": clamp_hit,
        "mid_clamped_hit": clamp_hit,
    }


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


def _soft_damp_into_round_band(
    raw_p: float,
    p_base: float,
    band_lo: float,
    band_hi: float,
    *,
    eps: float = 1e-6,
    gamma: float = 1.25,
    min_damp: float = 0.05,
) -> dict:
    """
    Soft edge damping: asymptotic approach to round rails instead of hard clip.
    Movement toward a rail decelerates as p_hat approaches that rail.
    """
    raw_p = float(raw_p)
    p_base = float(p_base)
    band_lo = float(band_lo)
    band_hi = float(band_hi)
    delta_raw = raw_p - p_base
    room_up = max((band_hi - eps) - p_base, 0.0)
    room_dn = max(p_base - (band_lo + eps), 0.0)
    span = max((band_hi - band_lo) - 2.0 * eps, eps)

    if abs(delta_raw) <= 1e-9:
        damp = 1.0
        delta_soft = 0.0
        room_frac = 1.0
        direction = "flat"
    elif delta_raw > 0:
        room_frac = float(np.clip(room_up / span, 0.0, 1.0))
        damp = max(min_damp, room_frac ** gamma)
        delta_soft = delta_raw * damp
        direction = "up"
    else:
        room_frac = float(np.clip(room_dn / span, 0.0, 1.0))
        damp = max(min_damp, room_frac ** gamma)
        delta_soft = delta_raw * damp
        direction = "down"

    p_soft = p_base + delta_soft
    p_soft = float(np.clip(p_soft, band_lo + eps, band_hi - eps))
    edge_soft_hit = abs(delta_soft - delta_raw) > 1e-9

    return {
        "p_soft": p_soft,
        "delta_raw": delta_raw,
        "delta_soft": delta_soft,
        "edge_damp_factor": damp,
        "edge_room_frac": room_frac,
        "edge_direction": direction,
        "edge_soft_hit": edge_soft_hit,
    }


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
    """
    Live series win probability.
    maps_a_won/maps_b_won are maps already won BEFORE the current map resolves.
    p_current_map is the live probability Team A wins the CURRENT map.
    p_future_map is used for any remaining future maps after the current one.
    Resolves directly to 1.0/0.0 on the deciding map branch.
    """
    bo_n = int(n_maps)
    target = bo_n // 2 + 1
    mwA = int(max(0, maps_a_won))
    mwB = int(max(0, maps_b_won))
    pf = float(p_future_map) if p_future_map is not None else float(p_current_map)

    pc = float(np.clip(p_current_map, 1e-6, 1 - 1e-6))
    pf = float(np.clip(pf, 1e-6, 1 - 1e-6))

    if mwA >= target:
        return 1.0
    if mwB >= target:
        return 0.0

    if mwA + 1 >= target:
        p_if_win_current = 1.0
    else:
        p_if_win_current = series_prob_needed(target - (mwA + 1), target - mwB, pf)

    if mwB + 1 >= target:
        p_if_lose_current = 0.0
    else:
        p_if_lose_current = series_prob_needed(target - mwA, target - (mwB + 1), pf)

    return float(pc * p_if_win_current + (1.0 - pc) * p_if_lose_current)

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

    # Apply pending Match ID from Activate (before widget is instantiated; Streamlit forbids modifying widget key after creation)
    if "cs2_bo3_pending_match_id" in st.session_state and st.session_state["cs2_bo3_pending_match_id"] is not None:
        st.session_state["cs2_inplay_match_id"] = str(st.session_state["cs2_bo3_pending_match_id"])
        del st.session_state["cs2_bo3_pending_match_id"]

    # --- Live CS2 source: BO3.gg or GRID (docs/GRID_CS2_NORMALIZED_FEATURE_CONTRACT.md) ---
    # Do not assign to st.session_state["cs2_live_source"] — the radio owns that key.
    CS2_LIVE_SOURCE_BO3 = "BO3.gg"
    CS2_LIVE_SOURCE_GRID = "GRID"
    _live_src_default = st.session_state.get("cs2_live_source", CS2_LIVE_SOURCE_BO3)
    cs2_live_source = st.radio(
        "Live data source",
        [CS2_LIVE_SOURCE_BO3, CS2_LIVE_SOURCE_GRID],
        index=0 if _live_src_default == CS2_LIVE_SOURCE_BO3 else 1,
        key="cs2_live_source",
        horizontal=True,
        help="BO3.gg: feed from logs/bo3_live_feed.json when auto is on. GRID: feed from adapters/grid_probe normalized V2 file.",
    )

    # BO3 Source: LIVE (poller) or REPLAY (from bo3_pulls.jsonl) — only when Live data source is BO3
    if cs2_live_source == CS2_LIVE_SOURCE_BO3:
        _bo3_src_default = st.session_state.get("cs2_bo3_source_mode", "LIVE (poller feed file)")
        _bo3_src_opts = ["LIVE (poller feed file)", "REPLAY (from bo3_pulls.jsonl)"]
        if _bo3_src_default not in _bo3_src_opts:
            _bo3_src_default = "LIVE (poller feed file)"
        cs2_bo3_source_mode = st.selectbox(
            "BO3 Source",
            options=_bo3_src_opts,
            index=_bo3_src_opts.index(_bo3_src_default),
            key="cs2_bo3_source_mode",
            help="LIVE: poller writes logs/bo3_live_feed.json. REPLAY: replay recorded snapshots from JSONL into the same feed file.",
        )
    else:
        cs2_bo3_source_mode = st.session_state.get("cs2_bo3_source_mode", "LIVE (poller feed file)")

    GRID_FEED_FILE = PROJECT_ROOT / "adapters" / "grid_probe" / "raw_grid_series_state_normalized_preview.json"
    GRID_PULLS_JSONL = PROJECT_ROOT / "logs" / "grid_pulls.jsonl"
    GRID_COMPLETENESS_MIN = 0.5
    GRID_STALENESS_MAX_SEC = 180

    # --- BO3.gg auto data pull / replay: overwrite session_state from feed when active or replay ---
    BO3_FEED_FILE = PROJECT_ROOT / "logs" / "bo3_live_feed.json"
    BO3_CONTROL_FILE = PROJECT_ROOT / "logs" / "bo3_live_control.json"
    BO3_FEED_STALE_SEC = 15
    _bo3_feed_active = (
        cs2_live_source == CS2_LIVE_SOURCE_BO3
        and (
            st.session_state.get("cs2_bo3_auto_active")
            or st.session_state.get("cs2_bo3_source_mode") == "REPLAY (from bo3_pulls.jsonl)"
        )
        and BO3_FEED_FILE.exists()
    )

    # When REPLAY: init state, load JSONL, advance idx if playing, write current frame to feed (so feed read below sees it)
    _replay_wrote_frame = False
    if cs2_live_source == CS2_LIVE_SOURCE_BO3 and cs2_bo3_source_mode == "REPLAY (from bo3_pulls.jsonl)":
        if "cs2_bo3_replay_idx" not in st.session_state:
            st.session_state["cs2_bo3_replay_idx"] = 0
        if "cs2_bo3_replay_playing" not in st.session_state:
            st.session_state["cs2_bo3_replay_playing"] = False
        if "cs2_bo3_replay_speed" not in st.session_state:
            st.session_state["cs2_bo3_replay_speed"] = 1.0
        replay_path = st.session_state.get("cs2_bo3_replay_path") or str(PROJECT_ROOT / "logs" / "bo3_pulls.jsonl")
        try:
            entries_list, by_match = _load_bo3_replay_jsonl(replay_path)
        except Exception:
            entries_list = []
            by_match = {}
        match_id = st.session_state.get("cs2_bo3_replay_match_id")
        entries = list(by_match.get(match_id, [])) if match_id is not None else []
        if st.session_state.get("cs2_bo3_replay_playing") and entries:
            idx = st.session_state.get("cs2_bo3_replay_idx", 0)
            idx = min(idx + 1, max(0, len(entries) - 1))
            idx = _replay_skip_duplicate_frames(entries, idx)
            st.session_state["cs2_bo3_replay_idx"] = idx
            if len(entries) > 0 and idx >= len(entries) - 1:
                st.session_state["cs2_bo3_replay_playing"] = False
        idx = st.session_state.get("cs2_bo3_replay_idx", 0)
        if match_id is not None and entries and 0 <= idx < len(entries):
            frame = entries[idx]
            team_a_is_team_one = st.session_state.get("cs2_bo3_replay_team_a_is_team_one", True)
            write_replay_frame_to_feed(BO3_FEED_FILE, frame.get("payload") or {}, team_a_is_team_one, error=None)
            _replay_wrote_frame = True
    if _replay_wrote_frame:
        _bo3_feed_active = True
        st.session_state["cs2_auto_add_snapshot_this_run"] = True

    if _bo3_feed_active:
        try:
            with open(BO3_FEED_FILE, "r", encoding="utf-8") as f:
                feed = json.load(f)
            payload = feed.get("payload") or {}
            feed_error = feed.get("error")
            snapshot_status = feed.get("snapshot_status")
            st.session_state["cs2_bo3_feed_error"] = feed_error
            st.session_state["cs2_bo3_feed_snapshot_status"] = snapshot_status
            # If feed contains full raw snapshot (team_one/team_two), normalize it and compute economy from it
            # (cash + equipment value per team; latch at round start is applied below)
            if isinstance(payload, dict) and payload and ("team_one" in payload or "team_two" in payload):
                team_a_is_team_one = bool(st.session_state.get("cs2_bo3_live_team_a_is_team_one", feed.get("team_a_is_team_one", True)))
                normalized = normalize_bo3_snapshot_to_app(
                    payload, team_a_is_team_one, valid_map_keys=list(CS2_MAP_CT_RATE.keys())
                )
            else:
                team_a_is_team_one = True
                normalized = payload if isinstance(payload, dict) else {}
            # Apply normalized state whenever non-empty (canonical session_state keys shared with GRID/calculator)
            if normalized:
                if not st.session_state.get("cs2_live_team_lock", False):
                    st.session_state["cs2_live_team_a"] = str(normalized.get("team_a", st.session_state.get("cs2_live_team_a", "")))
                    st.session_state["cs2_live_team_b"] = str(normalized.get("team_b", st.session_state.get("cs2_live_team_b", "")))
                else:
                    st.session_state["cs2_feed_team_a"] = normalized.get("team_a", "")
                    st.session_state["cs2_feed_team_b"] = normalized.get("team_b", "")
                # Detect maps_won change before applying rounds/map/side so we can reset for next map when a map just ended
                _prev_ma = int(st.session_state.get("cs2_prev_maps_a_won", 0))
                _prev_mb = int(st.session_state.get("cs2_prev_maps_b_won", 0))
                _now_ma = int(normalized.get("maps_a_won", 0))
                _now_mb = int(normalized.get("maps_b_won", 0))
                # If feed sends rounds 0-0 (new map) but series still 0-0, infer series from previous map winner (avoids series stuck at 0-0 after map 1)
                _prev_ra = int(st.session_state.get("cs2_prev_rounds_a", 0))
                _prev_rb = int(st.session_state.get("cs2_prev_rounds_b", 0))
                _ra = int(normalized.get("rounds_a", 0))
                _rb = int(normalized.get("rounds_b", 0))
                if (_ra, _rb) == (0, 0) and (_prev_ra + _prev_rb) > 0 and (_now_ma, _now_mb) == (0, 0):
                    if _prev_ra > _prev_rb:
                        _now_ma, _now_mb = _prev_ma + 1, _prev_mb
                    elif _prev_rb > _prev_ra:
                        _now_ma, _now_mb = _prev_ma, _prev_mb + 1
                # Never overwrite session series score when feed reports 0-0 (so manual edits e.g. 1-1 persist)
                _cur_ma = int(st.session_state.get("cs2_live_maps_a_won", 0))
                _cur_mb = int(st.session_state.get("cs2_live_maps_b_won", 0))
                if (_now_ma, _now_mb) == (0, 0):
                    _now_ma, _now_mb = _cur_ma, _cur_mb  # keep existing score; never apply feed 0-0
                _maps_just_changed = (_now_ma, _now_mb) != (_prev_ma, _prev_mb)
                if _maps_just_changed:
                    # Persist map result before resetting (use current session state for winner / map name)
                    _winner = ""
                    if _now_ma == _prev_ma + 1 and _now_mb == _prev_mb:
                        _winner = str(st.session_state.get("cs2_live_team_a", "")).strip()
                    elif _now_mb == _prev_mb + 1 and _now_ma == _prev_ma:
                        _winner = str(st.session_state.get("cs2_live_team_b", "")).strip()
                    _mid = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
                    if _winner and _mid and st.session_state.get("cs2_inplay_persist", False):
                        _cur_map_idx = int(st.session_state.get("cs2_inplay_map_index", 0))
                        _cur_map_name = str(st.session_state.get("cs2_live_map_name", ""))
                        _ta = str(st.session_state.get("cs2_live_team_a", "")).strip()
                        _tb = str(st.session_state.get("cs2_live_team_b", "")).strip()
                        if _ta and _tb:
                            persist_inplay_map_result(_mid, "CS2", _cur_map_idx, _cur_map_name, _ta, _tb, _winner)
                    st.session_state["cs2_prev_maps_a_won"] = _now_ma
                    st.session_state["cs2_prev_maps_b_won"] = _now_mb
                # Apply series score only when feed sent non-0-0 (above we kept _cur when feed was 0-0)
                st.session_state["cs2_live_maps_a_won"] = _now_ma
                st.session_state["cs2_live_maps_b_won"] = _now_mb
                _game_ended = bool(normalized.get("game_ended", False))
                _match_ended = bool(normalized.get("match_ended", False))
                # Use 0-0 and defaults when map just ended, or when feed says game/match ended (between maps)
                if _maps_just_changed or _game_ended or _match_ended:
                    st.session_state["cs2_live_rounds_a"] = 0
                    st.session_state["cs2_live_rounds_b"] = 0
                    st.session_state["cs2_live_map_name"] = "Average (no map)"
                    st.session_state["cs2_live_a_side"] = "Unknown"
                    if _maps_just_changed:
                        st.session_state["cs2_live_buy_a"] = "Pistol"
                        st.session_state["cs2_live_buy_b"] = "Pistol"
                        st.session_state["cs2_live_pistol_a"] = False
                        st.session_state["cs2_live_pistol_b"] = False
                        st.session_state["cs2_live_chaos"] = 0.00
                        st.session_state["cs2_inplay_map_index"] = int(st.session_state.get("cs2_inplay_map_index", 0)) + 1
                        st.session_state["cs2_inplay_last_total_rounds"] = None
                        st.session_state["cs2_prev_rounds_a"] = 0
                        st.session_state["cs2_prev_rounds_b"] = 0
                else:
                    st.session_state["cs2_live_rounds_a"] = int(normalized.get("rounds_a", 0))
                    st.session_state["cs2_live_rounds_b"] = int(normalized.get("rounds_b", 0))
                    st.session_state["cs2_live_map_name"] = str(normalized.get("map_name", "Average (no map)"))
                    st.session_state["cs2_live_a_side"] = str(normalized.get("a_side", "Unknown"))
                    # Infer "won last pistol" from round 1 or round 13 when not in feed (round 1 = first pistol, round 13 = second half pistol)
                    ra = int(normalized.get("rounds_a", 0))
                    rb = int(normalized.get("rounds_b", 0))
                    prev_ra = int(st.session_state.get("cs2_prev_rounds_a", 0))
                    prev_rb = int(st.session_state.get("cs2_prev_rounds_b", 0))
                    total = ra + rb
                    if total == 1:
                        st.session_state["cs2_live_pistol_a"] = ra == 1
                        st.session_state["cs2_live_pistol_b"] = rb == 1
                    elif (prev_ra + prev_rb) == 12 and total == 13:
                        st.session_state["cs2_live_pistol_a"] = ra == prev_ra + 1
                        st.session_state["cs2_live_pistol_b"] = rb == prev_rb + 1
                    st.session_state["cs2_prev_rounds_a"] = ra
                    st.session_state["cs2_prev_rounds_b"] = rb
                st.session_state["cs2_live_game_ended"] = bool(normalized.get("game_ended", False))
                st.session_state["cs2_live_match_ended"] = bool(normalized.get("match_ended", False))
                st.session_state["cs2_live_match_status"] = str(normalized.get("match_status", "") or "").strip().lower()
                # BO3 canonical V2: team_a_side / team_b_side (normalized CT/T) from payload when available
                if isinstance(payload, dict) and (payload.get("team_one") or payload.get("team_two")):
                    def _norm_side_bo3(s):
                        if s is None or not str(s).strip():
                            return "Unknown"
                        t = str(s).strip().upper()
                        if t == "TERRORIST" or t == "T":
                            return "T"
                        if t in ("CT", "COUNTER-TERRORIST", "COUNTER TERRORIST"):
                            return "CT"
                        return "Unknown"
                    t1 = payload.get("team_one") or {}
                    t2 = payload.get("team_two") or {}
                    s1 = _norm_side_bo3(t1.get("side") if isinstance(t1, dict) else None)
                    s2 = _norm_side_bo3(t2.get("side") if isinstance(t2, dict) else None)
                    if team_a_is_team_one:
                        st.session_state["cs2_live_team_a_side"] = s1
                        st.session_state["cs2_live_team_b_side"] = s2
                    else:
                        st.session_state["cs2_live_team_a_side"] = s2
                        st.session_state["cs2_live_team_b_side"] = s1
                if normalized.get("series_fmt") == "BO3":
                    st.session_state["cs2_live_series_fmt_idx"] = 1
                elif normalized.get("series_fmt") == "BO5":
                    st.session_state["cs2_live_series_fmt_idx"] = 2
                elif normalized.get("series_fmt") == "BO1":
                    st.session_state["cs2_live_series_fmt_idx"] = 0
                if normalized.get("round_number") is not None:
                    try:
                        st.session_state["cs2_live_round_number"] = int(normalized["round_number"])
                    except (TypeError, ValueError):
                        pass
                if normalized.get("game_number") is not None:
                    try:
                        st.session_state["cs2_live_game_number"] = int(normalized["game_number"])
                    except (TypeError, ValueError):
                        pass
                # BO3 ECON — prefer derived total_resources (cash + loadout) when available, else balance+equipment
                # BO3 LOADOUT VALUE FIX — integrate: use total_resources for app econ so p_hat/bands use full economy
                use_total_resources = (
                    normalized.get("loadout_derived_valid")
                    and normalized.get("team_a_total_resources") is not None
                    and normalized.get("team_b_total_resources") is not None
                )
                if use_total_resources:
                    try:
                        feed_econ_a = int(float(normalized["team_a_total_resources"]))
                        feed_econ_b = int(float(normalized["team_b_total_resources"]))
                        st.session_state["cs2_live_econ_a"] = feed_econ_a
                        st.session_state["cs2_live_econ_b"] = feed_econ_b
                        st.session_state["cs2_live_econ_is_total_resources"] = True
                    except (TypeError, ValueError):
                        use_total_resources = False
                if not use_total_resources and "team_a_econ" in normalized and "team_b_econ" in normalized:
                    try:
                        feed_econ_a = int(float(normalized["team_a_econ"]))
                        feed_econ_b = int(float(normalized["team_b_econ"]))
                        st.session_state["cs2_live_econ_a"] = feed_econ_a
                        st.session_state["cs2_live_econ_b"] = feed_econ_b
                        st.session_state["cs2_live_econ_is_total_resources"] = False
                    except (TypeError, ValueError):
                        pass
                # BO3 ECON LATCH — only refresh latched econ when round_number changes (locked at beginning of round)
                # BO3 LOADOUT VALUE FIX — latch total_resources when available so model uses cash+loadout
                try:
                    current_round = normalized.get("round_number")
                    if current_round is not None:
                        current_round = int(current_round)
                    else:
                        current_round = None
                except (TypeError, ValueError):
                    current_round = None
                feed_a = feed_b = None
                if current_round is not None:
                    if use_total_resources:
                        try:
                            feed_a = float(normalized["team_a_total_resources"])
                            feed_b = float(normalized["team_b_total_resources"])
                        except (TypeError, ValueError):
                            pass
                    if feed_a is None and feed_b is None and "team_a_econ" in normalized and "team_b_econ" in normalized:
                        try:
                            feed_a = float(normalized["team_a_econ"])
                            feed_b = float(normalized["team_b_econ"])
                        except (TypeError, ValueError):
                            pass
                    if feed_a is not None and feed_b is not None:
                        latched_round = st.session_state.get("cs2_live_latched_round_number")
                        if latched_round is None or current_round != latched_round:
                            st.session_state["cs2_live_latched_round_number"] = current_round
                            st.session_state["cs2_live_latched_econ_a"] = feed_a
                            st.session_state["cs2_live_latched_econ_b"] = feed_b
                            st.session_state["cs2_live_econ_is_total_resources"] = bool(use_total_resources)
                # BO3 state hygiene: phase / bomb / clock every tick when present (decoupled from loadout validity)
                _bp = normalized.get("bomb_planted")
                if _bp is None and isinstance(payload, dict) and "is_bomb_planted" in payload:
                    _bp = payload.get("is_bomb_planted")
                if _bp is not None:
                    st.session_state["cs2_live_bomb_planted"] = bool(_bp)
                tr = normalized.get("round_time_remaining_s")
                if tr is None and isinstance(payload, dict):
                    tr = payload.get("round_time_remaining")
                if tr is not None and isinstance(tr, (int, float)):
                    tr = float(tr) / 1000.0 if float(tr) > 120 else float(tr)
                    st.session_state["cs2_live_round_time_remaining_s"] = tr
                _phase = normalized.get("round_phase")
                if _phase is None and isinstance(payload, dict):
                    _phase = payload.get("round_phase")
                if _phase is not None:
                    st.session_state["cs2_live_round_phase"] = _phase
                # BO3 LOADOUT VALUE FIX — compute and store cash/loadout/total resources (only when snapshot valid)
                if normalized.get("loadout_derived_valid"):
                    try:
                        st.session_state["cs2_live_team_a_cash_total"] = float(normalized.get("team_a_cash_total", 0))
                        st.session_state["cs2_live_team_b_cash_total"] = float(normalized.get("team_b_cash_total", 0))
                        # Loadout: prefer team equipment_value from payload when present, else normalized derived
                        t1 = (payload.get("team_one") or {}) if isinstance(payload, dict) else {}
                        t2 = (payload.get("team_two") or {}) if isinstance(payload, dict) else {}
                        loadout_a = float(t1.get("equipment_value")) if isinstance(t1.get("equipment_value"), (int, float)) else None
                        loadout_b = float(t2.get("equipment_value")) if isinstance(t2.get("equipment_value"), (int, float)) else None
                        if loadout_a is None:
                            loadout_a = float(normalized.get("team_a_loadout_est_total", 0))
                        if loadout_b is None:
                            loadout_b = float(normalized.get("team_b_loadout_est_total", 0))
                        st.session_state["cs2_live_team_a_loadout_est_total"] = loadout_a
                        st.session_state["cs2_live_team_b_loadout_est_total"] = loadout_b
                        cash_a = float(normalized.get("team_a_cash_total", 0))
                        cash_b = float(normalized.get("team_b_cash_total", 0))
                        st.session_state["cs2_live_team_a_total_resources"] = float(normalized.get("team_a_total_resources", cash_a + loadout_a))
                        st.session_state["cs2_live_team_b_total_resources"] = float(normalized.get("team_b_total_resources", cash_b + loadout_b))
                        st.session_state["cs2_live_team_a_alive_cash_total"] = float(normalized.get("team_a_alive_cash_total", 0))
                        st.session_state["cs2_live_team_b_alive_cash_total"] = float(normalized.get("team_b_alive_cash_total", 0))
                        st.session_state["cs2_live_team_a_alive_loadout_est_total"] = float(normalized.get("team_a_alive_loadout_est_total", 0))
                        st.session_state["cs2_live_team_b_alive_loadout_est_total"] = float(normalized.get("team_b_alive_loadout_est_total", 0))
                        st.session_state["cs2_live_team_a_alive_total_resources"] = float(normalized.get("team_a_alive_total_resources", 0))
                        st.session_state["cs2_live_team_b_alive_total_resources"] = float(normalized.get("team_b_alive_total_resources", 0))
                        st.session_state["cs2_live_team_a_alive_count"] = int(normalized.get("team_a_alive_count", 0))
                        st.session_state["cs2_live_team_b_alive_count"] = int(normalized.get("team_b_alive_count", 0))
                        st.session_state["cs2_live_team_a_hp_alive_total"] = float(normalized.get("team_a_hp_alive_total", 0))
                        st.session_state["cs2_live_team_b_hp_alive_total"] = float(normalized.get("team_b_hp_alive_total", 0))
                        # BO3 canonical V2: armor and alive_loadout from payload player_states (alive-only) so mid_feature_has_armor / mid_feature_has_alive_loadout = True
                        if isinstance(payload, dict) and (payload.get("team_one") or payload.get("team_two")):
                            ps1 = t1.get("player_states") if isinstance(t1.get("player_states"), list) else []
                            ps2 = t2.get("player_states") if isinstance(t2.get("player_states"), list) else []
                            def _sum_alive(players, key, default=0):
                                total = 0
                                for p in players:
                                    if not isinstance(p, dict) or p.get("is_alive") is not True:
                                        continue
                                    v = p.get(key)
                                    total += int(v) if isinstance(v, (int, float)) and v is not None else default
                                return total
                            arm1 = _sum_alive(ps1, "armor")
                            arm2 = _sum_alive(ps2, "armor")
                            load1 = _sum_alive(ps1, "equipment_value")
                            load2 = _sum_alive(ps2, "equipment_value")
                            if team_a_is_team_one:
                                st.session_state["cs2_live_team_a_armor_alive_total"] = arm1
                                st.session_state["cs2_live_team_b_armor_alive_total"] = arm2
                                st.session_state["cs2_live_team_a_alive_loadout_total"] = load1
                                st.session_state["cs2_live_team_b_alive_loadout_total"] = load2
                            else:
                                st.session_state["cs2_live_team_a_armor_alive_total"] = arm2
                                st.session_state["cs2_live_team_b_armor_alive_total"] = arm1
                                st.session_state["cs2_live_team_a_alive_loadout_total"] = load2
                                st.session_state["cs2_live_team_b_alive_loadout_total"] = load1
                        st.session_state["cs2_live_intraround_state_valid"] = True
                        st.session_state["cs2_live_intraround_state_source"] = "bo3_player_states"
                    except (TypeError, ValueError):
                        pass
                    # BO3 LOADOUT VALUE FIX — unknown-item tracking
                    for item in normalized.get("unknown_items") or []:
                        if item and str(item).strip():
                            if "cs2_bo3_unknown_items_seen" not in st.session_state:
                                st.session_state["cs2_bo3_unknown_items_seen"] = set()
                            st.session_state["cs2_bo3_unknown_items_seen"].add(str(item).strip())
                else:
                    # Intraround microstate invalid: clear so stale values do not leak across round/map transitions
                    st.session_state["cs2_live_team_a_alive_count"] = None
                    st.session_state["cs2_live_team_b_alive_count"] = None
                    st.session_state["cs2_live_team_a_hp_alive_total"] = None
                    st.session_state["cs2_live_team_b_hp_alive_total"] = None
                    st.session_state["cs2_live_team_a_armor_alive_total"] = None
                    st.session_state["cs2_live_team_b_armor_alive_total"] = None
                    st.session_state["cs2_live_team_a_alive_loadout_total"] = None
                    st.session_state["cs2_live_team_b_alive_loadout_total"] = None
                    st.session_state["cs2_live_intraround_state_valid"] = False
                    st.session_state["cs2_live_intraround_state_source"] = "stale_cleared"
                # CS2 raw payload debug stream: append one row per BO3 live tick (display-only; rolling 50)
                _buf = st.session_state.get("cs2_live_raw_payload_debug_rows") or []
                _idx = len(_buf)
                try:
                    _norm_copy = copy.deepcopy(normalized) if normalized else None
                    _raw_copy = copy.deepcopy(payload) if isinstance(payload, dict) and payload else None
                except Exception:
                    _norm_copy = None
                    _raw_copy = None
                _ss = st.session_state
                _buf.append({
                    "ts_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "debug_index": _idx,
                    "cs2_live_game_number": _ss.get("cs2_live_game_number"),
                    "cs2_live_round_number": _ss.get("cs2_live_round_number"),
                    "cs2_live_round_phase": _ss.get("cs2_live_round_phase"),
                    "cs2_live_round_time_remaining_s": _ss.get("cs2_live_round_time_remaining_s"),
                    "cs2_live_bomb_planted": _ss.get("cs2_live_bomb_planted"),
                    "cs2_live_team_a_alive_count": _ss.get("cs2_live_team_a_alive_count"),
                    "cs2_live_team_b_alive_count": _ss.get("cs2_live_team_b_alive_count"),
                    "cs2_live_team_a_alive_loadout_total": _ss.get("cs2_live_team_a_alive_loadout_total"),
                    "cs2_live_team_b_alive_loadout_total": _ss.get("cs2_live_team_b_alive_loadout_total"),
                    "cs2_live_team_a_armor_alive_total": _ss.get("cs2_live_team_a_armor_alive_total"),
                    "cs2_live_team_b_armor_alive_total": _ss.get("cs2_live_team_b_armor_alive_total"),
                    "cs2_live_intraround_state_valid": _ss.get("cs2_live_intraround_state_valid"),
                    "cs2_live_intraround_state_source": _ss.get("cs2_live_intraround_state_source"),
                    "normalized_payload": _norm_copy,
                    "raw_payload": _raw_copy,
                })
                if len(_buf) > 50:
                    _buf = _buf[-50:]
                st.session_state["cs2_live_raw_payload_debug_rows"] = _buf
        except Exception:
            pass
    else:
        st.session_state["cs2_bo3_feed_error"] = None
        st.session_state["cs2_bo3_feed_snapshot_status"] = None

    # --- GRID auto-pull tick: when active, run Pull (API + normalize + Kalshi + snapshot) every run (10s) ---
    if cs2_live_source == CS2_LIVE_SOURCE_GRID and st.session_state.get("cs2_grid_auto_pull_active") and st.session_state.get("cs2_grid_selected_series_id"):
        _selected = st.session_state["cs2_grid_selected_series_id"]
        try:
            from adapters.grid_probe import grid_graphql_client
            from adapters.grid_probe import grid_queries
            from adapters.grid_probe.grid_normalize_series_state_probe import _normalize_series_state
            api_key = grid_graphql_client.load_api_key(PROJECT_ROOT / ".env")
            _grid_record_path = GRID_PULLS_JSONL if st.session_state.get("cs2_grid_record_jsonl", True) else None
            state_resp = grid_graphql_client.post_graphql(
                grid_graphql_client.SERIES_STATE_GRAPHQL_URL,
                grid_queries.QUERY_SERIES_STATE_RICH.strip(),
                variables={"id": _selected},
                api_key=api_key,
                record_jsonl_path=_grid_record_path,
                record_label="series_state_poll",
            )
            if not state_resp.get("errors"):
                data = state_resp.get("data") or {}
                ss = data.get("seriesState")
                if isinstance(ss, dict):
                    normalized = _normalize_series_state(ss)
                    with open(GRID_FEED_FILE, "w", encoding="utf-8") as f:
                        json.dump(normalized, f, indent=2, ensure_ascii=False)
                    st.session_state["cs2_grid_last_error"] = None
                    st.session_state["cs2_economy_source_preserve"] = st.session_state.get("cs2_economy_source")
                    _mkt_keys = ("cs2_mkt_fetch_venue", "cs2_kalshi_url", "cs2_kalshi_markets", "cs2_kalshi_market",
                                "cs2_live_market_bid", "cs2_live_market_ask", "cs2_mkt_fetch_meta", "cs2_mkt_fetch_ident")
                    st.session_state["cs2_mkt_preserve"] = {k: st.session_state.get(k) for k in _mkt_keys if k in st.session_state}
                    _list = st.session_state.get("cs2_grid_live_series_list") or []
                    _sel = st.session_state.get("cs2_grid_selected_series_id")
                    _r = next((x for x in _list if str(x.get("series_id")) == str(_sel)), None) if _sel else None
                    if _r and _r.get("team_id_name_pairs"):
                        st.session_state["cs2_grid_team_id_name_pairs"] = _r["team_id_name_pairs"]
                        st.session_state["cs2_grid_team_id_name_pairs_series_id"] = _sel
                    _chart_keys = ("cs2_show_pcal", "cs2_show_round_state_bands", "cs2_show_state_bounds", "cs2_show_kappa_bands", "cs2_chart_window")
                    st.session_state["cs2_chart_preserve"] = {k: st.session_state.get(k) for k in _chart_keys if k in st.session_state}
                    if st.session_state.get("cs2_mkt_fetch_venue") == "Kalshi":
                        tkr, _ = _cs2_kalshi_resolve_ticker()
                        if tkr:
                            try:
                                bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                                if bid_f is not None:
                                    st.session_state["cs2_live_market_bid"] = max(0.0, min(1.0, float(bid_f)))
                                if ask_f is not None:
                                    st.session_state["cs2_live_market_ask"] = max(0.0, min(1.0, float(ask_f)))
                                if meta is not None:
                                    st.session_state["cs2_mkt_fetch_meta"] = meta
                                if bid_f is not None and ask_f is not None:
                                    _cs2_market_delay_push(float(bid_f), float(ask_f))
                                st.session_state["cs2_auto_add_snapshot_this_run"] = True
                                st.session_state["cs2_mkt_bid_ask_fetched_this_run"] = True
                            except Exception:
                                pass
                    else:
                        st.session_state["cs2_auto_add_snapshot_this_run"] = True
                else:
                    st.session_state["cs2_grid_last_error"] = "No seriesState in response"
            else:
                st.session_state["cs2_grid_last_error"] = str(state_resp.get("errors"))
        except Exception as e:
            st.session_state["cs2_grid_last_error"] = str(e)

    # --- GRID live source: overwrite session_state from normalized V2 when selected ---
    grid_used_reduced_features = False
    grid_valid = None
    grid_completeness_score = None
    grid_staleness_seconds = None
    grid_has_players = None
    grid_has_clock = None
    if cs2_live_source == CS2_LIVE_SOURCE_GRID and GRID_FEED_FILE.exists():
        try:
            with open(GRID_FEED_FILE, "r", encoding="utf-8") as f:
                grid_data = json.load(f)
            v2 = grid_data.get("v2") if isinstance(grid_data, dict) else {}
            if not isinstance(v2, dict):
                v2 = {}
            grid_valid = v2.get("valid")
            grid_completeness_score = v2.get("completeness_score")
            grid_staleness_seconds = v2.get("staleness_seconds")
            grid_has_players = v2.get("has_players")
            grid_has_clock = v2.get("has_clock")
            st.session_state["cs2_grid_valid"] = grid_valid
            st.session_state["cs2_grid_completeness_score"] = grid_completeness_score
            st.session_state["cs2_grid_staleness_seconds"] = grid_staleness_seconds
            st.session_state["cs2_grid_has_players"] = grid_has_players
            st.session_state["cs2_grid_has_clock"] = grid_has_clock
            # Gating: fall back to base_frozen / reduced features if invalid, low completeness, or stale
            use_reduced = (
                grid_valid is False
                or (grid_completeness_score is not None and float(grid_completeness_score) < GRID_COMPLETENESS_MIN)
                or (grid_staleness_seconds is not None and int(grid_staleness_seconds) > GRID_STALENESS_MAX_SEC)
            )
            grid_used_reduced_features = use_reduced
            st.session_state["cs2_grid_used_reduced_features"] = use_reduced
            st.session_state["cs2_grid_last_apply_result"] = "reduced" if use_reduced else "applied"
            if use_reduced:
                # Do not overwrite session_state with GRID mid-round; leave rails/econ as-is or from previous
                pass
            else:
                # Map GRID V2 -> app internal keys (fallback-safe; never crash)
                def _g(k, default=None):
                    return v2.get(k) if v2.get(k) is not None else default
                # Match ID from GRID series ID (for Kalshi sync)
                _series_id = _g("series_id") or st.session_state.get("cs2_grid_selected_series_id")
                if _series_id is not None:
                    st.session_state["cs2_inplay_match_id"] = str(_series_id)
                st.session_state["cs2_live_rounds_a"] = int(_g("rounds_a", 0))
                st.session_state["cs2_live_rounds_b"] = int(_g("rounds_b", 0))
                st.session_state["cs2_live_map_name"] = str(_g("map_name", "Average (no map)"))
                _maps_a = _g("maps_a_won")
                _maps_b = _g("maps_b_won")
                _cur_ma = int(st.session_state.get("cs2_live_maps_a_won", 0))
                _cur_mb = int(st.session_state.get("cs2_live_maps_b_won", 0))
                # Never overwrite session series when feed reports 0-0 (same as BO3: preserve manual edit)
                if _maps_a is not None and _maps_b is not None:
                    _now_ma, _now_mb = int(_maps_a), int(_maps_b)
                    if (_now_ma, _now_mb) == (0, 0):
                        _now_ma, _now_mb = _cur_ma, _cur_mb  # keep existing; never apply feed 0-0
                    st.session_state["cs2_live_maps_a_won"] = _now_ma
                    st.session_state["cs2_live_maps_b_won"] = _now_mb
                elif _maps_a is not None and int(_maps_a) != 0:
                    st.session_state["cs2_live_maps_a_won"] = int(_maps_a)
                elif _maps_b is not None and int(_maps_b) != 0:
                    st.session_state["cs2_live_maps_b_won"] = int(_maps_b)
                # Team names for display: prefer Central id->name (from Fetch list or preserved), then v2 name, then id
                _id_a = _g("team_a_id")
                _id_b = _g("team_b_id")
                _name_a = _g("team_a_name")
                _name_b = _g("team_b_name")
                _live_list = st.session_state.get("cs2_grid_live_series_list") or []
                _sel_id = st.session_state.get("cs2_grid_selected_series_id")
                _row = next((r for r in _live_list if str(r.get("series_id")) == str(_sel_id)), None) if _sel_id else None
                _pairs = None
                if _row and _row.get("team_id_name_pairs"):
                    _pairs = _row["team_id_name_pairs"]
                    st.session_state["cs2_grid_team_id_name_pairs"] = _pairs
                    st.session_state["cs2_grid_team_id_name_pairs_series_id"] = _sel_id
                else:
                    _preserved_series_id = st.session_state.get("cs2_grid_team_id_name_pairs_series_id")
                    if _sel_id and _preserved_series_id == _sel_id:
                        _pairs = st.session_state.get("cs2_grid_team_id_name_pairs") or []
                    else:
                        _pairs = []
                _id_to_name = {p["id"]: p["name"] for p in _pairs if isinstance(p, dict) and p.get("id")}
                if _id_to_name:
                    if _name_a is None and _id_a is not None:
                        _name_a = _id_to_name.get(str(_id_a))
                    if _name_b is None and _id_b is not None:
                        _name_b = _id_to_name.get(str(_id_b))
                _name_a = _name_a or _id_a
                _name_b = _name_b or _id_b
                st.session_state["cs2_grid_raw_team_a_name"] = str(_name_a) if _name_a is not None else None
                st.session_state["cs2_grid_raw_team_b_name"] = str(_name_b) if _name_b is not None else None
                # Team A / B assignment: from "Team A is" radio (sync with Kalshi); default first = Team A
                _choice = st.session_state.get("cs2_grid_team_a_choice")
                _raw_a = st.session_state.get("cs2_grid_raw_team_a_name")
                st.session_state["cs2_grid_team_a_is_first"] = (_choice == _raw_a) if (_raw_a and _choice is not None) else True
                if _name_a is not None:
                    st.session_state["cs2_live_team_a"] = str(_name_a)
                if _name_b is not None:
                    st.session_state["cs2_live_team_b"] = str(_name_b)
                # Team A side (CT/T) for display and CT/T bias: GRID returns e.g. "counter-terrorists" / "terrorists"
                _side_a = _g("team_a_side")
                _side_b = _g("team_b_side")
                def _norm_side(s):
                    if s is None or not str(s).strip():
                        return "Unknown"
                    t = str(s).strip().lower().replace("-", "")
                    if t in ("counterterrorists", "ct"):
                        return "CT"
                    if t in ("terrorists", "t"):
                        return "T"
                    return "Unknown"
                _team_a_is_first = st.session_state.get("cs2_grid_team_a_is_first", True)
                _side_for_app_a = _side_a if _team_a_is_first else _side_b
                st.session_state["cs2_live_a_side"] = _norm_side(_side_for_app_a)
                ma = _g("team_a_money")
                mb = _g("team_b_money")
                la = _g("team_a_loadout_value")
                lb = _g("team_b_loadout_value")
                if ma is not None:
                    st.session_state["cs2_live_team_a_cash_total"] = float(ma)
                if mb is not None:
                    st.session_state["cs2_live_team_b_cash_total"] = float(mb)
                if la is not None:
                    st.session_state["cs2_live_team_a_loadout_est_total"] = float(la)
                if lb is not None:
                    st.session_state["cs2_live_team_b_loadout_est_total"] = float(lb)
                total_a = (float(ma) + float(la)) if (ma is not None and la is not None) else (float(ma) if ma is not None else (float(la) if la is not None else None))
                total_b = (float(mb) + float(lb)) if (mb is not None and lb is not None) else (float(mb) if mb is not None else (float(lb) if lb is not None else None))
                if total_a is not None and total_b is not None:
                    st.session_state["cs2_live_econ_a"] = int(total_a)
                    st.session_state["cs2_live_econ_b"] = int(total_b)
                    st.session_state["cs2_live_team_a_total_resources"] = float(total_a)
                    st.session_state["cs2_live_team_b_total_resources"] = float(total_b)
                    st.session_state["cs2_live_econ_is_total_resources"] = True
                    st.session_state["cs2_live_latched_econ_a"] = float(total_a)
                    st.session_state["cs2_live_latched_econ_b"] = float(total_b)
                    st.session_state["cs2_live_latched_round_number"] = int(_g("rounds_a", 0)) + int(_g("rounds_b", 0))
                st.session_state["cs2_live_team_a_alive_count"] = _g("alive_count_a")
                st.session_state["cs2_live_team_b_alive_count"] = _g("alive_count_b")
                st.session_state["cs2_live_team_a_hp_alive_total"] = _g("hp_total_a")
                st.session_state["cs2_live_team_b_hp_alive_total"] = _g("hp_total_b")
                # V2.1 aggregates: armor and alive loadout from GRID v2 (fallback 0)
                try:
                    st.session_state["cs2_live_team_a_armor_alive_total"] = int(_g("armor_total_a") or 0)
                    st.session_state["cs2_live_team_b_armor_alive_total"] = int(_g("armor_total_b") or 0)
                except (TypeError, ValueError):
                    st.session_state["cs2_live_team_a_armor_alive_total"] = 0
                    st.session_state["cs2_live_team_b_armor_alive_total"] = 0
                try:
                    pa = v2.get("players_a") or []
                    pb = v2.get("players_b") or []
                    load_a = sum(int(p.get("loadout_value") or 0) for p in pa if p.get("alive") is True)
                    load_b = sum(int(p.get("loadout_value") or 0) for p in pb if p.get("alive") is True)
                    st.session_state["cs2_live_team_a_alive_loadout_total"] = load_a
                    st.session_state["cs2_live_team_b_alive_loadout_total"] = load_b
                except (TypeError, ValueError, AttributeError):
                    st.session_state["cs2_live_team_a_alive_loadout_total"] = 0
                    st.session_state["cs2_live_team_b_alive_loadout_total"] = 0
                st.session_state["cs2_live_bomb_planted"] = False
                st.session_state["cs2_live_round_time_remaining_s"] = _g("clock_seconds")
                st.session_state["cs2_live_round_phase"] = _g("clock_type")
                # Most recent pistol winner (from GRID segments: round 1 or round 13; else infer from round 1/13)
                _pa = _g("pistol_a_won")
                _pb = _g("pistol_b_won")
                if _pa is not None:
                    st.session_state["cs2_live_pistol_a"] = bool(_pa)
                if _pb is not None:
                    st.session_state["cs2_live_pistol_b"] = bool(_pb)
                if _pa is None and _pb is None:
                    ra = int(_g("rounds_a") or 0)
                    rb = int(_g("rounds_b") or 0)
                    prev_ra = int(st.session_state.get("cs2_prev_rounds_a", 0))
                    prev_rb = int(st.session_state.get("cs2_prev_rounds_b", 0))
                    total = ra + rb
                    if total == 1:
                        st.session_state["cs2_live_pistol_a"] = ra == 1
                        st.session_state["cs2_live_pistol_b"] = rb == 1
                    elif (prev_ra + prev_rb) == 12 and total == 13:
                        st.session_state["cs2_live_pistol_a"] = ra == prev_ra + 1
                        st.session_state["cs2_live_pistol_b"] = rb == prev_rb + 1
                    st.session_state["cs2_prev_rounds_a"] = ra
                    st.session_state["cs2_prev_rounds_b"] = rb
                else:
                    ra = int(_g("rounds_a") or 0)
                    rb = int(_g("rounds_b") or 0)
                    st.session_state["cs2_prev_rounds_a"] = ra
                    st.session_state["cs2_prev_rounds_b"] = rb
                # If user chose "Team A is [second GRID team]", swap all A/B so app matches Kalshi order
                if st.session_state.get("cs2_grid_team_a_is_first") is False:
                    _swap = (
                        ("cs2_live_team_a", "cs2_live_team_b"),
                        ("cs2_live_rounds_a", "cs2_live_rounds_b"),
                        ("cs2_live_team_a_cash_total", "cs2_live_team_b_cash_total"),
                        ("cs2_live_team_a_loadout_est_total", "cs2_live_team_b_loadout_est_total"),
                        ("cs2_live_team_a_total_resources", "cs2_live_team_b_total_resources"),
                        ("cs2_live_econ_a", "cs2_live_econ_b"),
                        ("cs2_live_latched_econ_a", "cs2_live_latched_econ_b"),
                        ("cs2_live_team_a_alive_count", "cs2_live_team_b_alive_count"),
                        ("cs2_live_team_a_hp_alive_total", "cs2_live_team_b_hp_alive_total"),
                        ("cs2_live_team_a_armor_alive_total", "cs2_live_team_b_armor_alive_total"),
                        ("cs2_live_team_a_alive_loadout_total", "cs2_live_team_b_alive_loadout_total"),
                        ("cs2_live_pistol_a", "cs2_live_pistol_b"),
                        ("cs2_live_maps_a_won", "cs2_live_maps_b_won"),
                    )
                    for _ka, _kb in _swap:
                        _va = st.session_state.get(_ka)
                        _vb = st.session_state.get(_kb)
                        st.session_state[_ka] = _vb
                        st.session_state[_kb] = _va
                # When maps_won changes (map just finished): series score already updated from feed;
                # reset round/map/side to defaults for next map until new data fills in; persist map result if enabled.
                prev_a = int(st.session_state.get("cs2_prev_maps_a_won", 0))
                prev_b = int(st.session_state.get("cs2_prev_maps_b_won", 0))
                now_a = int(st.session_state.get("cs2_live_maps_a_won", 0))
                now_b = int(st.session_state.get("cs2_live_maps_b_won", 0))
                if (now_a, now_b) != (prev_a, prev_b):
                    winner = ""
                    if now_a == prev_a + 1 and now_b == prev_b:
                        winner = str(st.session_state.get("cs2_live_team_a", "")).strip()
                    elif now_b == prev_b + 1 and now_a == prev_a:
                        winner = str(st.session_state.get("cs2_live_team_b", "")).strip()
                    _mid = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
                    if winner and _mid and st.session_state.get("cs2_inplay_persist", False):
                        cur_map_index = int(st.session_state.get("cs2_inplay_map_index", 0))
                        cur_map_name = str(st.session_state.get("cs2_live_map_name", ""))
                        team_a = str(st.session_state.get("cs2_live_team_a", "")).strip()
                        team_b = str(st.session_state.get("cs2_live_team_b", "")).strip()
                        if team_a and team_b:
                            persist_inplay_map_result(_mid, "CS2", cur_map_index, cur_map_name, team_a, team_b, winner)
                    # Reset mid-map state for next map until new data fills in
                    st.session_state["cs2_live_rounds_a"] = 0
                    st.session_state["cs2_live_rounds_b"] = 0
                    st.session_state["cs2_live_map_name"] = "Average (no map)"
                    st.session_state["cs2_live_a_side"] = "Unknown"
                    st.session_state["cs2_live_buy_a"] = "Pistol"
                    st.session_state["cs2_live_buy_b"] = "Pistol"
                    st.session_state["cs2_live_pistol_a"] = False
                    st.session_state["cs2_live_pistol_b"] = False
                    st.session_state["cs2_live_chaos"] = 0.00
                    st.session_state["cs2_inplay_map_index"] = int(st.session_state.get("cs2_inplay_map_index", 0)) + 1
                    st.session_state["cs2_inplay_last_total_rounds"] = None
                    st.session_state["cs2_prev_rounds_a"] = 0
                    st.session_state["cs2_prev_rounds_b"] = 0
                    st.session_state["cs2_prev_maps_a_won"] = now_a
                    st.session_state["cs2_prev_maps_b_won"] = now_b
        except Exception as _grid_err:
            grid_used_reduced_features = True
            st.session_state["cs2_grid_used_reduced_features"] = True
            st.session_state["cs2_grid_valid"] = None
            st.session_state["cs2_grid_completeness_score"] = None
            st.session_state["cs2_grid_staleness_seconds"] = None
            st.session_state["cs2_grid_has_players"] = None
            st.session_state["cs2_grid_has_clock"] = None
            st.session_state["cs2_grid_last_error"] = str(_grid_err)
    else:
        st.session_state["cs2_grid_used_reduced_features"] = None
        st.session_state["cs2_grid_valid"] = None
        st.session_state["cs2_grid_completeness_score"] = None
        st.session_state["cs2_grid_staleness_seconds"] = None
        st.session_state["cs2_grid_has_players"] = None
        st.session_state["cs2_grid_has_clock"] = None

    # When auto is on (BO3 or GRID): timer triggers full script rerun (BO3: 5s; GRID: 10s)
    if st.session_state.get("cs2_bo3_auto_active"):
        st_autorefresh(interval=5000, limit=None, key="cs2_bo3_autorefresh")
        # On each run (including 5s tick): if Kalshi, fetch bid/ask; then request one chart snapshot
        if st.session_state.get("cs2_mkt_fetch_venue") == "Kalshi":
            tkr, _ = _cs2_kalshi_resolve_ticker()
            if tkr:
                try:
                    bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                    _b = float(bid_f) if bid_f is not None else 0.0
                    _a = float(ask_f) if ask_f is not None else 1.0
                    st.session_state["cs2_live_market_bid"] = max(0.0, min(1.0, _b))
                    st.session_state["cs2_live_market_ask"] = max(0.0, min(1.0, _a))
                    if meta is not None:
                        st.session_state["cs2_mkt_fetch_meta"] = meta
                    # BO3 MARKET DELAY ALIGN: push to buffer on each Kalshi fetch (skip when no orders)
                    if bid_f is not None and ask_f is not None:
                        _cs2_market_delay_push(float(bid_f), float(ask_f))
                    # BO3 MARKET DELAY ALIGN
                except Exception:
                    pass
        # When auto is on, add one snapshot every run so incremental backtest runs and chart shows trades (even when not on Kalshi)
        st.session_state["cs2_auto_add_snapshot_this_run"] = True
    elif cs2_live_source == CS2_LIVE_SOURCE_GRID and st.session_state.get("cs2_grid_auto_pull_active"):
        st_autorefresh(interval=10000, limit=None, key="cs2_grid_autorefresh")

    # --- Auto data pull (BO3.gg): match selection, activate/deactivate, 5s refresh ---
    ui_section_divider("Match selection & auto data pull", "BO3.gg")
    st.markdown("#### Match selection & auto data pull (BO3.gg)")
    st.caption("When **Live data source** is BO3.gg: load matches below, pick one, choose Team A, then **Activate auto data pull**. Auto refresh runs every 5 s. Use **Deactivate** to stop.")
    with st.expander("Auto data pull (BO3.gg) — match selection & activate", expanded=True):
        bo3_active = st.session_state.get("cs2_bo3_auto_active", False)
        if cs2_live_source == CS2_LIVE_SOURCE_BO3 and cs2_bo3_source_mode == "REPLAY (from bo3_pulls.jsonl)":
            # REPLAY mode: no poller activation; replay controls only
            default_replay_path = str(PROJECT_ROOT / "logs" / "bo3_pulls.jsonl")
            replay_path = st.text_input(
                "Replay file path",
                value=st.session_state.get("cs2_bo3_replay_path") or default_replay_path,
                key="cs2_bo3_replay_path",
            )
            if st.button("Reload JSONL", key="cs2_bo3_replay_reload_jsonl", help="Clear cache and re-read the JSONL file (e.g. when bo3_pulls.jsonl is growing)."):
                try:
                    _load_bo3_replay_jsonl.clear()
                except Exception:
                    pass
                st.rerun()
            try:
                _replay_entries, _replay_by_match = _load_bo3_replay_jsonl(replay_path or default_replay_path)
            except Exception:
                _replay_entries = []
                _replay_by_match = {}
            if not _replay_by_match:
                st.warning("No BO3 snapshots found in JSONL (or file missing/empty). Use LIVE mode or record pulls first.")
            else:
                match_ids = sorted(_replay_by_match.keys())
                opts = [
                    f"{_replay_match_display_label(_replay_by_match[mid])} ({len(_replay_by_match[mid])} frames)"
                    for mid in match_ids
                ]
                sel_idx = st.selectbox("Match", range(len(opts)), format_func=lambda i: opts[i], key="cs2_bo3_replay_match_sel_idx")
                if sel_idx is not None and 0 <= sel_idx < len(match_ids):
                    current_match_id = match_ids[int(sel_idx)]
                    prev_match_id = st.session_state.get("cs2_bo3_replay_prev_match_id")
                    if prev_match_id != current_match_id:
                        st.session_state["cs2_bo3_replay_idx"] = 0
                        st.session_state["cs2_bo3_replay_playing"] = False
                        st.session_state["cs2_bo3_replay_prev_match_id"] = current_match_id
                    st.session_state["cs2_bo3_replay_match_id"] = current_match_id
                _replay_entries_for_match = list(_replay_by_match.get(st.session_state.get("cs2_bo3_replay_match_id"), []))
                st.checkbox(
                    "Team A is team one", value=st.session_state.get("cs2_bo3_replay_team_a_is_team_one", True), key="cs2_bo3_replay_team_a_is_team_one"
                )
                speed_opts = [0.5, 1.0, 2.0, 5.0, 10.0]
                _speed_val = st.session_state.get("cs2_bo3_replay_speed", 1.0)
                _speed_idx = speed_opts.index(_speed_val) if _speed_val in speed_opts else 1
                st.session_state["cs2_bo3_replay_speed"] = st.selectbox(
                    "Speed", options=speed_opts, index=_speed_idx, format_func=lambda x: f"{x}x", key="cs2_bo3_replay_speed_sb"
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Step", key="cs2_bo3_replay_step"):
                        idx = st.session_state.get("cs2_bo3_replay_idx", 0)
                        idx = min(idx + 1, max(0, len(_replay_entries_for_match) - 1))
                        idx = _replay_skip_duplicate_frames(_replay_entries_for_match, idx)
                        st.session_state["cs2_bo3_replay_idx"] = idx
                        st.rerun()
                with col2:
                    if st.button("Play" if not st.session_state.get("cs2_bo3_replay_playing") else "Pause", key="cs2_bo3_replay_play_pause"):
                        st.session_state["cs2_bo3_replay_playing"] = not st.session_state.get("cs2_bo3_replay_playing", False)
                        st.rerun()
                with col3:
                    if st.button("Restart", key="cs2_bo3_replay_restart"):
                        st.session_state["cs2_bo3_replay_idx"] = 0
                        st.session_state["cs2_bo3_replay_playing"] = False
                        st.rerun()
                idx = st.session_state.get("cs2_bo3_replay_idx", 0)
                if _replay_entries_for_match and 0 <= idx < len(_replay_entries_for_match):
                    frame = _replay_entries_for_match[idx]
                    payload_keys_count = len((frame.get("payload") or {}).keys())
                    st.caption(f"Frame {idx + 1} / {len(_replay_entries_for_match)} | ts_utc: {frame.get('ts_utc', '—')} | payload keys: {payload_keys_count}")
                if st.session_state.get("cs2_bo3_replay_playing"):
                    interval_ms = max(250, int(5000 / st.session_state.get("cs2_bo3_replay_speed", 1.0)))
                    _replay_mid = st.session_state.get("cs2_bo3_replay_match_id")
                    _replay_speed = st.session_state.get("cs2_bo3_replay_speed", 1.0)
                    st_autorefresh(interval=interval_ms, limit=None, key=f"bo3_replay_autorefresh_{_replay_mid}_{_replay_speed}")
        else:
            cs2_bo3_record_jsonl = st.checkbox("Record BO3 pulls (JSONL) for replay", value=True, key="cs2_bo3_record_jsonl", help="Append each poll to logs/bo3_pulls.jsonl for later replay without market data.")
            if bo3_active:
                st.success("Auto data pull ON (every 5 s). Live scores and sides are updating from BO3.gg.")
            # Read feed file for current status so UI matches logs/bo3_live_feed.json
                feed_err = st.session_state.get("cs2_bo3_feed_error")
                snap_status = st.session_state.get("cs2_bo3_feed_snapshot_status")
                if BO3_FEED_FILE.exists():
                    try:
                        with open(BO3_FEED_FILE, "r", encoding="utf-8") as f:
                            _feed = json.load(f)
                        snap_status = _feed.get("snapshot_status", snap_status)
                        feed_err = _feed.get("error", feed_err)
                    except Exception:
                        pass
                # Poller writes "poller_starting" once at start before first fetch; treat as transient
                if feed_err == "poller_starting":
                    st.info("**Polling starting…** Data and (when Kalshi) market/chart update every 5 s automatically.")
                else:
                    is_empty = snap_status == "empty" or (feed_err and feed_err != "inactive")
                    if is_empty:
                        if feed_err and ("empty" in str(feed_err).lower() or "snapshot" in str(feed_err).lower()):
                            st.info("**Last poll: no snapshot** — match may not be live yet, or BO3.gg returned no data. Rounds/map/economy will update when live. In `logs/bo3_live_feed.json`, `snapshot_status` = empty means not live.")
                        else:
                            st.info(f"**Last poll: no snapshot** — {feed_err or 'snapshot_status: empty'}. Match may not be live yet.")
                    else:
                        st.caption("Last poll: live data received. Rounds, map, side, and economy are updating from the feed.")
                # Diagnostics: what the app read from the feed (for evaluating auto data pull)
                with st.expander("Diagnostics", expanded=False):
                    feed_path = str(BO3_FEED_FILE)
                    feed_exists = BO3_FEED_FILE.exists()
                    st.text(f"Feed file path: {feed_path}")
                    st.text(f"File exists: {feed_exists}")
                    if feed_exists:
                        try:
                            mtime = BO3_FEED_FILE.stat().st_mtime
                            st.text(f"Last modified: {datetime.fromtimestamp(mtime).isoformat()} (UTC)")
                        except Exception:
                            st.text("Last modified: (could not read)")
                        try:
                            with open(BO3_FEED_FILE, "r", encoding="utf-8") as _f:
                                _d = json.load(_f)
                            payload_keys = list((_d.get("payload") or {}).keys())
                            st.text(f"Payload keys: {', '.join(payload_keys) if payload_keys else 'empty'}")
                            st.text(f"snapshot_status: {_d.get('snapshot_status', '—')}")
                            st.text(f"error: {_d.get('error', '—')}")
                        except Exception as e:
                            st.text(f"Read error: {e}")
                    else:
                        st.text("Payload keys: (file missing)")
                        st.text("snapshot_status: —")
                        st.text("error: —")
                    # BO3 LOADOUT VALUE FIX — unknown BO3 items seen (from weapon/loadout parsing)
                    _unknown = st.session_state.get("cs2_bo3_unknown_items_seen") or set()
                    st.text(f"Unknown BO3 items seen: {len(_unknown)}")
                    # BO3 LOADOUT VALUE FIX
                if st.session_state.get("cs2_mkt_fetch_venue") == "Kalshi":
                    if st.button("Refresh Kalshi + Add snapshot", key="cs2_bo3_refresh_kalshi_snap", help="Fetch Kalshi bid/ask and add one chart point"):
                        tkr, err = _cs2_kalshi_resolve_ticker()
                        if err or not tkr:
                            st.error(err or "Load teams and select a market, or paste a Kalshi URL and click Load teams.")
                        else:
                            try:
                                bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                                _b = float(bid_f) if bid_f is not None else 0.0
                                _a = float(ask_f) if ask_f is not None else 1.0
                                st.session_state["cs2_live_market_bid"] = max(0.0, min(1.0, _b))
                                st.session_state["cs2_live_market_ask"] = max(0.0, min(1.0, _a))
                                if meta is not None:
                                    st.session_state["cs2_mkt_fetch_meta"] = meta
                                # BO3 MARKET DELAY ALIGN
                                if bid_f is not None and ask_f is not None:
                                    _cs2_market_delay_push(float(bid_f), float(ask_f))
                                # BO3 MARKET DELAY ALIGN
                                st.session_state["cs2_auto_add_snapshot_this_run"] = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Kalshi + snapshot failed: {e}")
                if st.button("Deactivate auto data pull", key="cs2_bo3_deactivate"):
                    try:
                        BO3_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
                        with open(BO3_CONTROL_FILE, "w", encoding="utf-8") as f:
                            json.dump({"active": False}, f)
                    except Exception:
                        pass
                    st.session_state["cs2_bo3_auto_active"] = False
                    st.rerun()
                st.caption("With auto on: score, status, and (when Kalshi) market + chart update every 5 s. Use **Refresh Kalshi + Add snapshot** above for an immediate update.")
            else:
                bo3_matches = st.session_state.get("cs2_bo3_live_matches") or []
                if st.button("Load live matches", key="cs2_bo3_load_matches"):
                    try:
                        matches = fetch_bo3_live_matches()
                        st.session_state["cs2_bo3_live_matches"] = matches
                        st.session_state["cs2_bo3_load_error"] = None
                    except Exception as e:
                        st.session_state["cs2_bo3_live_matches"] = []
                        st.session_state["cs2_bo3_load_error"] = str(e)
                    st.rerun()
                if st.session_state.get("cs2_bo3_load_error"):
                    st.error(st.session_state["cs2_bo3_load_error"])
                if not bo3_matches:
                    st.caption("Click **Load live matches** to fetch current BO3.gg live matches.")
                else:
                    opts = [f"{m.get('team1_name', '?')} vs {m.get('team2_name', '?')} (id: {m.get('id', '')})" for m in bo3_matches]
                    sel_idx = st.selectbox(
                        "Select match",
                        range(len(opts)),
                        format_func=lambda i: opts[i],
                        key="cs2_bo3_match_idx",
                    )
                    if sel_idx is not None and 0 <= sel_idx < len(bo3_matches):
                        m = bo3_matches[int(sel_idx)]
                        team1_name = m.get("team1_name") or "Team 1"
                        team2_name = m.get("team2_name") or "Team 2"
                        team_a_choice = st.radio(
                            "Team A is",
                            options=[team1_name, team2_name],
                            index=0,
                            key="cs2_bo3_team_a_choice",
                            horizontal=True,
                        )
                        if team_a_choice is None:
                            team_a_choice = team1_name
                        team_a_is_team_one = team_a_choice == team1_name
                        if st.button("Activate auto data pull", key="cs2_bo3_activate"):
                            match_id = m.get("id") or m.get("match_id")
                            if not match_id:
                                st.error("No match id.")
                            else:
                                try:
                                    BO3_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
                                    with open(BO3_CONTROL_FILE, "w", encoding="utf-8") as f:
                                        json.dump({
                                            "active": True,
                                            "match_id": int(match_id),
                                            "team_a_is_team_one": team_a_is_team_one,
                                            "record_enabled": st.session_state.get("cs2_bo3_record_jsonl", True),
                                            "record_path": str(PROJECT_ROOT / "logs" / "bo3_pulls.jsonl"),
                                        }, f)
                                    # Write initial feed so file exists and app can show team names until poller overwrites
                                    team_a_name = team1_name if team_a_is_team_one else team2_name
                                    team_b_name = team2_name if team_a_is_team_one else team1_name
                                    initial_payload = {
                                        "team_a": team_a_name,
                                        "team_b": team_b_name,
                                        "rounds_a": 0,
                                        "rounds_b": 0,
                                        "maps_a_won": 0,
                                        "maps_b_won": 0,
                                        "map_name": "Average (no map)",
                                        "a_side": "Unknown",
                                        "series_fmt": "BO3",
                                    }
                                    BO3_FEED_FILE = PROJECT_ROOT / "logs" / "bo3_live_feed.json"
                                    BO3_FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
                                    with open(BO3_FEED_FILE, "w", encoding="utf-8") as f:
                                        json.dump({"timestamp": time.time(), "payload": initial_payload}, f, indent=2)
                                    st.session_state["cs2_bo3_auto_active"] = True
                                    st.session_state["cs2_bo3_pending_match_id"] = str(match_id)
                                    # Set initial live fields so they show after rerun even if poller overwrites feed with empty payload
                                    st.session_state["cs2_live_team_a"] = team_a_name
                                    st.session_state["cs2_live_team_b"] = team_b_name
                                    st.session_state["cs2_live_rounds_a"] = 0
                                    st.session_state["cs2_live_rounds_b"] = 0
                                    st.session_state["cs2_live_maps_a_won"] = 0
                                    st.session_state["cs2_live_maps_b_won"] = 0
                                    st.session_state["cs2_live_map_name"] = "Average (no map)"
                                    st.session_state["cs2_live_a_side"] = "Unknown"
                                    st.session_state["cs2_live_series_fmt_idx"] = 1
                                    st.session_state["cs2_live_econ_a"] = 0
                                    st.session_state["cs2_live_econ_b"] = 0
                                    # Start poller subprocess (runs in background).
                                    # Run as script (bo3.gg is a folder name, not package bo3.gg for -m).
                                    try:
                                        env = os.environ.copy()
                                        env["PYTHONPATH"] = str(PROJECT_ROOT)
                                        poller_script = PROJECT_ROOT / "bo3.gg" / "poller.py"
                                        subprocess.Popen(
                                            [sys.executable, str(poller_script)],
                                            cwd=str(PROJECT_ROOT),
                                            env=env,
                                            stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                        )
                                    except Exception:
                                        pass
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

    # --- GRID data: refresh & status (when Live data source is GRID) ---
    ui_section_divider("GRID data", "Refresh & status")
    st.markdown("#### GRID data — refresh & status")
    st.caption("When **Live data source** is GRID: data is read from the normalized preview file. **Activate auto pull** to refresh every 10 s (Pull + Kalshi + snapshot), or use **Pull** and **Apply** below.")
    cs2_grid_record_jsonl = st.checkbox("Record GRID pulls (JSONL) for replay", value=True, key="cs2_grid_record_jsonl", help="Append each GraphQL request/response to logs/grid_pulls.jsonl for later replay.")
    with st.expander("GRID data — auto pull", expanded=True):
        if cs2_live_source != CS2_LIVE_SOURCE_GRID:
            st.caption("Select **GRID** as Live data source above to use these controls.")
        else:
            _grid_auto_pull_active = st.session_state.get("cs2_grid_auto_pull_active", False)
            if _grid_auto_pull_active:
                if st.button("Deactivate auto pull", key="cs2_grid_deactivate_auto_pull", help="Stop auto pull (Pull + Kalshi + snapshot every 10 s)."):
                    st.session_state["cs2_grid_auto_pull_active"] = False
                    st.rerun()
                st.caption("Auto pull is on: Pull GRID state + Kalshi + snapshot every 10 s.")
            else:
                if st.button("Activate auto pull", key="cs2_grid_activate_auto_pull", help="Pull GRID state + Kalshi + snapshot every 10 s. Select a series below first."):
                    _sel = st.session_state.get("cs2_grid_selected_series_id")
                    if not _sel:
                        st.warning("Select a series first (Fetch GRID Live Series, then choose one).")
                    else:
                        st.session_state["cs2_grid_auto_pull_active"] = True
                        st.rerun()
            st.markdown("**Last read status** (from current session):")
            _gv = st.session_state.get("cs2_grid_valid")
            _gc = st.session_state.get("cs2_grid_completeness_score")
            _gs = st.session_state.get("cs2_grid_staleness_seconds")
            _gp = st.session_state.get("cs2_grid_has_players")
            _gcl = st.session_state.get("cs2_grid_has_clock")
            _gr = st.session_state.get("cs2_grid_used_reduced_features")
            st.text(f"valid: {_gv}  |  completeness_score: {_gc}  |  staleness_seconds: {_gs}")
            st.text(f"has_players: {_gp}  |  has_clock: {_gcl}  |  used_reduced_features: {_gr}")

            # --- GRID Live Series panel (manual: fetch -> select -> pull -> apply) ---
            st.markdown("---")
            st.markdown("#### GRID Live Series")
            if "cs2_grid_live_series_list" not in st.session_state:
                st.session_state["cs2_grid_live_series_list"] = []
            _fetch_filter_opts = ["Live only", "All (no live filter)"]
            _fetch_filter_key = st.session_state.get("cs2_grid_fetch_filter", "Live only")
            _fetch_filter_idx = _fetch_filter_opts.index(_fetch_filter_key) if _fetch_filter_key in _fetch_filter_opts else 0
            st.selectbox(
                "Fetch from API",
                options=_fetch_filter_opts,
                index=_fetch_filter_idx,
                key="cs2_grid_fetch_filter",
                help="Live only = Central live filter (may miss ETO/upcoming). All = series with start time within ±5 hours of now, ordered by start time.",
            )
            if st.button("Fetch GRID Live Series", key="cs2_grid_fetch_series", help="Call Central allSeries; filter and limit depend on 'Fetch from API'."):
                try:
                    from adapters.grid_probe import grid_graphql_client
                    from adapters.grid_probe import grid_queries
                    api_key = grid_graphql_client.load_api_key(PROJECT_ROOT / ".env")
                    _use_live_filter = st.session_state.get("cs2_grid_fetch_filter", "Live only") == "Live only"
                    if _use_live_filter:
                        _filter = {"live": {"games": {}}}
                    else:
                        # All: restrict to start time within +/- 5 hours of now so we don't get matches weeks out
                        _now = datetime.now(timezone.utc)
                        _start = (_now - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                        _end = (_now + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                        _filter = {"startTimeScheduled": {"gte": _start, "lte": _end}}
                    central_vars = {
                        "orderBy": "StartTimeScheduled",
                        "orderDirection": "DESC",
                        "first": 50,
                        "filter": _filter,
                    }
                    _grid_record_path = GRID_PULLS_JSONL if st.session_state.get("cs2_grid_record_jsonl", True) else None
                    central = grid_graphql_client.post_graphql(
                        grid_graphql_client.CENTRAL_DATA_GRAPHQL_URL,
                        grid_queries.QUERY_FIND_CS2_SERIES.strip(),
                        variables=central_vars,
                        api_key=api_key,
                        record_jsonl_path=_grid_record_path,
                        record_label="central_find_series",
                    )
                    if central.get("errors"):
                        st.session_state["cs2_grid_last_error"] = str(central.get("errors"))
                        st.error("Central GraphQL errors: " + str(central.get("errors"))[:300])
                    else:
                        data = central.get("data") or {}
                        all_series = data.get("allSeries") or {}
                        edges = all_series.get("edges") or []
                        rows = []
                        for edge in edges:
                            node = edge.get("node") if isinstance(edge, dict) else {}
                            if not isinstance(node, dict):
                                continue
                            sid = node.get("id")
                            if sid is None:
                                continue
                            title = node.get("title")
                            name = None
                            if isinstance(title, dict):
                                name = title.get("name") or title.get("nameShortened") or str(title)
                            elif title is not None:
                                name = str(title)
                            teams_raw = node.get("teams")
                            team_names = []
                            team_id_name_pairs = []
                            if isinstance(teams_raw, list):
                                for t in teams_raw:
                                    if isinstance(t, dict):
                                        # TeamParticipant has baseInfo: Team! (Central API); Team has id, name, nameShortened
                                        team_obj = t.get("baseInfo") if isinstance(t.get("baseInfo"), dict) else t.get("team") or t
                                        if isinstance(team_obj, dict):
                                            team_name = (team_obj.get("name") or team_obj.get("nameShortened") or "").strip()
                                            if team_name:
                                                team_names.append(team_name)
                                            tid = team_obj.get("id")
                                            if tid is not None:
                                                team_id_name_pairs.append({
                                                    "id": str(tid),
                                                    "name": team_name or (team_obj.get("nameShortened") or "").strip() or str(tid),
                                                })
                            teams_display = " vs ".join(team_names) if team_names else None
                            series_type = node.get("type")
                            title_short = None
                            if isinstance(title, dict):
                                title_short = title.get("nameShortened") or title.get("name")
                            if title_short is None and name:
                                title_short = name
                            rows.append({
                                "series_id": str(sid),
                                "title": title,
                                "title_short": title_short,
                                "name": name,
                                "teams": team_names or None,
                                "teams_display": teams_display,
                                "team_id_name_pairs": team_id_name_pairs,
                                "map": None,
                                "updated_at": node.get("updatedAt"),
                                "startTimeScheduled": node.get("startTimeScheduled"),
                                "valid": None,
                                "series_type": series_type,
                            })
                        st.session_state["cs2_grid_live_series_list"] = rows
                        st.session_state["cs2_grid_last_error"] = None
                        st.session_state["cs2_economy_source_preserve"] = st.session_state.get("cs2_economy_source")
                        _mkt_keys = ("cs2_mkt_fetch_venue", "cs2_kalshi_url", "cs2_kalshi_markets", "cs2_kalshi_market",
                                    "cs2_live_market_bid", "cs2_live_market_ask", "cs2_mkt_fetch_meta", "cs2_mkt_fetch_ident")
                        st.session_state["cs2_mkt_preserve"] = {k: st.session_state.get(k) for k in _mkt_keys if k in st.session_state}
                        _list = st.session_state.get("cs2_grid_live_series_list") or []
                        _sel = st.session_state.get("cs2_grid_selected_series_id")
                        _r = next((x for x in _list if str(x.get("series_id")) == str(_sel)), None) if _sel else None
                        if _r and _r.get("team_id_name_pairs"):
                            st.session_state["cs2_grid_team_id_name_pairs"] = _r["team_id_name_pairs"]
                            st.session_state["cs2_grid_team_id_name_pairs_series_id"] = _sel
                        _chart_keys = ("cs2_show_pcal", "cs2_show_round_state_bands", "cs2_show_state_bounds", "cs2_show_kappa_bands", "cs2_chart_window")
                        st.session_state["cs2_chart_preserve"] = {k: st.session_state.get(k) for k in _chart_keys if k in st.session_state}
                        st.success(f"Fetched {len(rows)} series. Select one and pull state.")
                        st.rerun()
                except Exception as e:
                    st.session_state["cs2_grid_last_error"] = str(e)
                    st.error(f"Fetch failed: {e}")

            _list = st.session_state.get("cs2_grid_live_series_list") or []
            if _list:
                # Local status label (UI-only; 10 min recent threshold; does not affect gating)
                _GRID_RECENT_MINUTES = 10
                def _parse_iso(s):
                    if s is None:
                        return None
                    try:
                        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt
                    except (ValueError, TypeError):
                        return None
                def _local_status_label(r):
                    t = (r.get("series_type") or "").upper()
                    if t == "LOOPFEED":
                        return "Loopfeed"
                    now = datetime.now(timezone.utc)
                    start_dt = _parse_iso(r.get("startTimeScheduled"))
                    updated_dt = _parse_iso(r.get("updatedAt"))
                    if start_dt and start_dt > now:
                        return "Upcoming"
                    if start_dt and start_dt <= now:
                        if updated_dt:
                            delta_min = (now - updated_dt).total_seconds() / 60.0
                            if delta_min <= _GRID_RECENT_MINUTES:
                                return "Likely Live"
                        return "Recent/Stale"
                    return "—"
                for _r in _list:
                    _r["_status_label"] = _local_status_label(_r)
                include_loopfeed = st.checkbox(
                    "Include LOOPFEED rows",
                    value=False,
                    key="cs2_grid_include_loopfeed",
                    help="When unchecked, LOOPFEED series are hidden from the list and dropdown.",
                )
                _status_opts = ["Live only", "Live + Upcoming", "All"]
                _status_key = st.session_state.get("cs2_grid_series_status_filter", "Live only")
                _status_idx = _status_opts.index(_status_key) if _status_key in _status_opts else 0
                _status_filter = st.selectbox(
                    "Show series",
                    options=_status_opts,
                    index=_status_idx,
                    key="cs2_grid_series_status_filter",
                    help="Filter the fetched list: Live only = show all returned; Live + Upcoming = series type LIVE or UPCOMING (ETO); All = show everything.",
                )
                def _status_ok(r):
                    t = (r.get("series_type") or "").upper()
                    if _status_filter == "Live only":
                        # Trust GRID live filter: list is already from allSeries(live: { games: {} }), show all
                        return True
                    if _status_filter == "Live + Upcoming":
                        return t in ("LIVE", "UPCOMING")
                    return True
                _filtered = [r for r in _list if _status_ok(r)]
                if not include_loopfeed:
                    _filtered = [r for r in _filtered if (r.get("series_type") or "").upper() != "LOOPFEED"]
                # Sort by start time (latest first) so ETO/upcoming and live appear in clear order
                def _sort_key(r):
                    dt = _parse_iso(r.get("startTimeScheduled")) if r.get("startTimeScheduled") else None
                    return (dt is None, -(dt.timestamp() if dt else 0))
                _filtered = sorted(_filtered, key=_sort_key)
                _display = []
                for _r in _filtered:
                    _display.append({
                        "series_id": _r.get("series_id"),
                        "title_short": _r.get("title_short") or _r.get("name"),
                        "type": _r.get("series_type"),
                        "teams": _r.get("teams_display"),
                        "startTimeScheduled": _r.get("startTimeScheduled"),
                        "updatedAt": _r.get("updated_at"),
                        "status": _r.get("_status_label"),
                    })
                _df = pd.DataFrame(_display)
                _cols = [c for c in ["series_id", "title_short", "type", "teams", "startTimeScheduled", "updatedAt", "status"] if c in _df.columns]
                if _cols and len(_df) > 0:
                    st.dataframe(_df[_cols].head(60), use_container_width=True, hide_index=True)
                _ids = [r.get("series_id") for r in _filtered if r.get("series_id")]
                _cur = st.session_state.get("cs2_grid_selected_series_id")
                if _cur not in _ids and _ids:
                    st.session_state["cs2_grid_selected_series_id"] = _ids[0]
                if _ids:
                    def _series_label(sid):
                        r = next((r for r in _list if r.get("series_id") == sid), None)
                        if not r:
                            return sid
                        return r.get("teams_display") or r.get("name") or sid
                    st.selectbox(
                        "Selected GRID series",
                        options=_ids,
                        format_func=_series_label,
                        index=_ids.index(st.session_state["cs2_grid_selected_series_id"]) if st.session_state.get("cs2_grid_selected_series_id") in _ids else 0,
                        key="cs2_grid_selected_series_id",
                    )
                else:
                    st.caption("No series match the current filters. Try **All** or check **Include LOOPFEED rows**.")
            _selected = st.session_state.get("cs2_grid_selected_series_id") if _list else None
            if not _list:
                st.caption("Click **Fetch GRID Live Series** to load live series from Central.")

            _pull_col, _apply_col = st.columns(2)
            with _pull_col:
                if st.button("Pull GRID Series State (One Shot)", key="cs2_grid_pull_one_shot", help="Fetch seriesState for selected series, normalize, save to preview file."):
                    if not _selected:
                        st.warning("Select a series first (fetch list, then choose).")
                    else:
                        try:
                            from adapters.grid_probe import grid_graphql_client
                            from adapters.grid_probe import grid_queries
                            from adapters.grid_probe.grid_normalize_series_state_probe import _normalize_series_state
                            api_key = grid_graphql_client.load_api_key(PROJECT_ROOT / ".env")
                            _grid_record_path = GRID_PULLS_JSONL if st.session_state.get("cs2_grid_record_jsonl", True) else None
                            state_resp = grid_graphql_client.post_graphql(
                                grid_graphql_client.SERIES_STATE_GRAPHQL_URL,
                                grid_queries.QUERY_SERIES_STATE_RICH.strip(),
                                variables={"id": _selected},
                                api_key=api_key,
                                record_jsonl_path=_grid_record_path,
                                record_label="series_state_poll",
                            )
                            if state_resp.get("errors"):
                                st.session_state["cs2_grid_last_error"] = str(state_resp.get("errors"))
                                st.error("Series State errors: " + (str(state_resp.get("errors"))[:300]))
                            else:
                                data = state_resp.get("data") or {}
                                ss = data.get("seriesState")
                                if not isinstance(ss, dict):
                                    st.session_state["cs2_grid_last_error"] = "No seriesState in response"
                                    st.error("No seriesState in response.")
                                else:
                                    normalized = _normalize_series_state(ss)
                                    with open(GRID_FEED_FILE, "w", encoding="utf-8") as f:
                                        json.dump(normalized, f, indent=2, ensure_ascii=False)
                                    v2 = normalized.get("v2") or {}
                                    _valid = v2.get("valid")
                                    _comp = v2.get("completeness_score")
                                    _stal = v2.get("staleness_seconds")
                                    st.session_state["cs2_grid_last_error"] = None
                                    st.session_state["cs2_economy_source_preserve"] = st.session_state.get("cs2_economy_source")
                                    _mkt_keys = ("cs2_mkt_fetch_venue", "cs2_kalshi_url", "cs2_kalshi_markets", "cs2_kalshi_market",
                                                "cs2_live_market_bid", "cs2_live_market_ask", "cs2_mkt_fetch_meta", "cs2_mkt_fetch_ident")
                                    st.session_state["cs2_mkt_preserve"] = {k: st.session_state.get(k) for k in _mkt_keys if k in st.session_state}
                                    _list = st.session_state.get("cs2_grid_live_series_list") or []
                                    _sel = st.session_state.get("cs2_grid_selected_series_id")
                                    _r = next((x for x in _list if str(x.get("series_id")) == str(_sel)), None) if _sel else None
                                    if _r and _r.get("team_id_name_pairs"):
                                        st.session_state["cs2_grid_team_id_name_pairs"] = _r["team_id_name_pairs"]
                                        st.session_state["cs2_grid_team_id_name_pairs_series_id"] = _sel
                                    _chart_keys = ("cs2_show_pcal", "cs2_show_round_state_bands", "cs2_show_state_bounds", "cs2_show_kappa_bands", "cs2_chart_window")
                                    st.session_state["cs2_chart_preserve"] = {k: st.session_state.get(k) for k in _chart_keys if k in st.session_state}
                                    # Refresh Kalshi bid/ask + request chart snapshot (align with future auto-pull: GRID → Kalshi → snapshot)
                                    if st.session_state.get("cs2_mkt_fetch_venue") == "Kalshi":
                                        tkr, _ = _cs2_kalshi_resolve_ticker()
                                        if tkr:
                                            try:
                                                bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                                                if bid_f is not None:
                                                    st.session_state["cs2_live_market_bid"] = max(0.0, min(1.0, float(bid_f)))
                                                if ask_f is not None:
                                                    st.session_state["cs2_live_market_ask"] = max(0.0, min(1.0, float(ask_f)))
                                                if meta is not None:
                                                    st.session_state["cs2_mkt_fetch_meta"] = meta
                                                if bid_f is not None and ask_f is not None:
                                                    _cs2_market_delay_push(float(bid_f), float(ask_f))
                                                st.session_state["cs2_auto_add_snapshot_this_run"] = True
                                                st.session_state["cs2_mkt_bid_ask_fetched_this_run"] = True
                                            except Exception:
                                                pass
                                    st.success(f"Pull OK. valid={_valid} completeness_score={_comp} staleness_seconds={_stal}")
                                    st.rerun()
                        except Exception as e:
                            st.session_state["cs2_grid_last_error"] = str(e)
                            st.error(f"Pull failed: {e}")
            with _apply_col:
                if st.button("Apply GRID Snapshot to Live State", key="cs2_grid_apply_snapshot", help="Re-read preview file and apply GRID-to-session mapping (same as on load)."):
                    st.session_state["cs2_grid_pending_apply"] = True
                    st.session_state["cs2_economy_source_preserve"] = st.session_state.get("cs2_economy_source")
                    _mkt_keys = ("cs2_mkt_fetch_venue", "cs2_kalshi_url", "cs2_kalshi_markets", "cs2_kalshi_market",
                                "cs2_live_market_bid", "cs2_live_market_ask", "cs2_mkt_fetch_meta", "cs2_mkt_fetch_ident")
                    st.session_state["cs2_mkt_preserve"] = {k: st.session_state.get(k) for k in _mkt_keys if k in st.session_state}
                    _list = st.session_state.get("cs2_grid_live_series_list") or []
                    _sel = st.session_state.get("cs2_grid_selected_series_id")
                    _r = next((x for x in _list if str(x.get("series_id")) == str(_sel)), None) if _sel else None
                    if _r and _r.get("team_id_name_pairs"):
                        st.session_state["cs2_grid_team_id_name_pairs"] = _r["team_id_name_pairs"]
                        st.session_state["cs2_grid_team_id_name_pairs_series_id"] = _sel
                    _chart_keys = ("cs2_show_pcal", "cs2_show_round_state_bands", "cs2_show_state_bounds", "cs2_show_kappa_bands", "cs2_chart_window")
                    st.session_state["cs2_chart_preserve"] = {k: st.session_state.get(k) for k in _chart_keys if k in st.session_state}
                    st.rerun()

            _last_apply = st.session_state.get("cs2_grid_last_apply_result")
            if _last_apply:
                st.caption(f"Last apply result: **{_last_apply}** (applied = full mapping; reduced = fallback used).")

            # Team A / Team B assignment (sync with Kalshi: pick which GRID team is Team A)
            _raw_a = st.session_state.get("cs2_grid_raw_team_a_name")
            _raw_b = st.session_state.get("cs2_grid_raw_team_b_name")
            if _raw_a and _raw_b:
                _team_a_opts = [_raw_a, _raw_b]
                _cur = st.session_state.get("cs2_grid_team_a_choice")
                _idx = _team_a_opts.index(_cur) if _cur in _team_a_opts else 0
                st.radio(
                    "Team A is (match Kalshi order)",
                    options=_team_a_opts,
                    index=_idx,
                    key="cs2_grid_team_a_choice",
                    horizontal=True,
                    help="Choose which team is Team A so odds and logs align with Kalshi. Then click Apply GRID Snapshot to apply.",
                )

            with st.expander("GRID Live Status / Debug", expanded=False):
                st.text(f"cs2_live_source: {st.session_state.get('cs2_live_source')}")
                st.text(f"cs2_grid_selected_series_id: {st.session_state.get('cs2_grid_selected_series_id')}")
                st.text(f"grid_valid: {st.session_state.get('cs2_grid_valid')}")
                st.text(f"grid_completeness_score: {st.session_state.get('cs2_grid_completeness_score')}")
                st.text(f"grid_staleness_seconds: {st.session_state.get('cs2_grid_staleness_seconds')}")
                st.text(f"grid_has_players: {st.session_state.get('cs2_grid_has_players')}")
                st.text(f"grid_has_clock: {st.session_state.get('cs2_grid_has_clock')}")
                st.text(f"grid_used_reduced_features: {st.session_state.get('cs2_grid_used_reduced_features')}")
                _rounds = st.session_state.get("cs2_live_rounds_a"), st.session_state.get("cs2_live_rounds_b")
                st.text(f"Latest applied rounds (A/B): {_rounds}")
                st.text(f"Latest applied map_name: {st.session_state.get('cs2_live_map_name')}")
                st.text(f"Latest applied team names: A={st.session_state.get('cs2_live_team_a')} B={st.session_state.get('cs2_live_team_b')}")
                _err = st.session_state.get("cs2_grid_last_error")
                st.text(f"Last GRID error: {_err if _err else '—'}")

    # --- Instrument Panel (trading summary at a glance; values from previous run) ---
    section_header("Instrument Panel")
    _ta = st.session_state.get("cs2_instrument_team_a") or "—"
    _tb = st.session_state.get("cs2_instrument_team_b") or "—"
    _ra = st.session_state.get("cs2_instrument_rounds_a")
    _rb = st.session_state.get("cs2_instrument_rounds_b")
    _map = st.session_state.get("cs2_instrument_map_name") or "—"
    _mid = st.session_state.get("cs2_instrument_market_mid")
    _ph = st.session_state.get("cs2_instrument_p_hat")
    _blo = st.session_state.get("cs2_instrument_band_state_lo")
    _bhi = st.session_state.get("cs2_instrument_band_state_hi")
    _slo = st.session_state.get("cs2_instrument_state_bound_lower")
    _shi = st.session_state.get("cs2_instrument_state_bound_upper")
    _edge = st.session_state.get("cs2_instrument_edge_pp")
    _spr = st.session_state.get("cs2_instrument_spread_pp")
    _src = st.session_state.get("cs2_instrument_live_source") or "—"
    _rail = st.session_state.get("cs2_instrument_rail_mode") or "—"
    _qsrc = st.session_state.get("cs2_instrument_q_intra_source") or "—"
    ip1, ip2, ip3, ip4, ip5 = st.columns(5)
    with ip1:
        st.metric("Match", f"{_ta} vs {_tb}", f"Map: {_map}")
    with ip2:
        if _ra is not None and _rb is not None:
            st.metric("Score (Rounds A–B)", f"{_ra} – {_rb}", f"Mid: {_mid*100:.1f}%" if _mid is not None else "—")
        else:
            st.metric("Score (Rounds A–B)", "—", f"Mid: {_mid*100:.1f}%" if _mid is not None else "—")
    with ip3:
        _ph_str = f"{_ph*100:.1f}%" if _ph is not None else "—"
        _band_str = f"[{_blo*100:.1f}%, {_bhi*100:.1f}%]" if (_blo is not None and _bhi is not None) else "—"
        st.metric("p_hat", _ph_str, f"Band: {_band_str}")
    with ip4:
        _bounds_str = f"[{_slo*100:.1f}%, {_shi*100:.1f}%]" if (_slo is not None and _shi is not None) else "—"
        _edge_str = f"Edge: {_edge:+.1f} pp" if _edge is not None else "—"
        st.metric("State bounds", _bounds_str, _edge_str)
    with ip5:
        st.metric("Spread", f"{_spr:.1f} pp" if _spr is not None else "—", f"Source: {_src} · {_rail}")
    st.caption(f"Live source: {_src}  |  Rail: {_rail}  |  q_intra source: {_qsrc}")

    # --- Advanced / Calibration / Tuning (collapsed by default) ---
    with st.expander("Advanced / Calibration / Tuning", expanded=False):
        # K-bands + calibration logging (match ID, persist, band level, K scale)
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
        # K calibration (optional)
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
        # Model knobs (MVP)
        ui_section_divider("Model knobs", "MVP")
        st.markdown("### Model knobs (MVP)")
        colB1, colB2, colB3 = st.columns(3)
        with colB1:
            beta_score = st.slider("β score", 0.05, 0.60, float(st.session_state.get("beta_score", 0.22)), 0.01)
        with colB2:
            beta_econ = st.slider("β econ", 0.00, 0.20, float(st.session_state.get("beta_econ", 0.06)), 0.01)
        with colB3:
            beta_pistol = st.slider("β pistol", 0.00, 0.80, float(st.session_state.get("beta_pistol", 0.35)), 0.01)
        colL1, colL2, colL3 = st.columns(3)
        with colL1:
            beta_lock = st.slider("β lock", 0.00, 3.00, float(st.session_state.get("beta_lock", 0.90)), 0.05)
        with colL2:
            lock_start_offset = st.slider("Lock starts (win_target - N)", 1, 7, int(st.session_state.get("lock_start_offset", 3)), 1)
        with colL3:
            lock_ramp = st.slider("Lock ramp (rounds)", 1, 8, int(st.session_state.get("lock_ramp", 3)), 1)
        cs2_midround_v1_enabled = st.checkbox(
            "Enable Mid-Round V1 (alive/bomb/HP)",
            value=bool(st.session_state.get("cs2_midround_v1_enabled", True)),
            key="cs2_midround_v1_enabled",
            help="Move p_hat within frozen round-state corridor using alive count, HP, and bomb. Rails update only when score changes.",
        )
        _midround_mode_opts = ["OFF", "V1 (additive)", "V2 (mixture)"]
        _midround_mode_default = st.session_state.get("cs2_midround_mode", "V2 (mixture)")
        if _midround_mode_default not in _midround_mode_opts:
            _midround_mode_default = "V2 (mixture)"
        cs2_midround_mode = st.selectbox(
            "Mid-Round Mode",
            options=_midround_mode_opts,
            index=_midround_mode_opts.index(_midround_mode_default),
            key="cs2_midround_mode",
            help="OFF = no intraround adjustment. V1 = additive corridor. V2 = mixture q*(p_if_a) + (1-q)*(p_if_b) from features.",
        )
        _bounds_mode_opts = ["Normal (bounded)", "No bounds (diagnostic)", "Map-only bounds", "Round-only bounds"]
        st.session_state.setdefault("cs2_bounds_mode", "Normal (bounded)")
        st.selectbox(
            "CS2 Bounds Mode",
            options=_bounds_mode_opts,
            index=_bounds_mode_opts.index(st.session_state.get("cs2_bounds_mode", "Normal (bounded)")) if st.session_state.get("cs2_bounds_mode") in _bounds_mode_opts else 0,
            key="cs2_bounds_mode",
            help="Normal = round corridor then map-state clamp. No bounds = raw p_hat. Map-only / Round-only = one clamp only.",
        )
        st.session_state["beta_score"] = float(beta_score)
        st.session_state["beta_econ"] = float(beta_econ)
        st.session_state["beta_pistol"] = float(beta_pistol)
        st.session_state["beta_lock"] = float(beta_lock)
        st.session_state["lock_start_offset"] = int(lock_start_offset)
        st.session_state["lock_ramp"] = int(lock_ramp)

    # Team names (typed input so logs reflect exactly what you enter)
    # (We still load the teams list in case you want to use it elsewhere, but we don't force a dropdown here.)
    with st.expander("Context + Live map inputs", expanded=False):
        df_cs2_live = load_cs2_teams()

        colA, colB = st.columns(2)
        with colA:
            st.session_state.setdefault("cs2_live_team_a", st.session_state.get("cs2_live_team_a", ""))
            team_a = st.text_input("Team A", value=st.session_state.get("cs2_live_team_a", ""), key="cs2_live_team_a")
        with colB:
            st.session_state.setdefault("cs2_live_team_b", st.session_state.get("cs2_live_team_b", ""))
            team_b = st.text_input("Team B", value=st.session_state.get("cs2_live_team_b", ""), key="cs2_live_team_b")

        st.checkbox("BO3: Team A is team one", key="cs2_bo3_live_team_a_is_team_one", value=st.session_state.get("cs2_bo3_live_team_a_is_team_one", True))
        st.checkbox("Lock Team A/B (don't overwrite from BO3)", key="cs2_live_team_lock", value=st.session_state.get("cs2_live_team_lock", False))

        # Back-compat aliases (some downstream code expects these names)
        team_a_live = team_a
        team_b_live = team_b

        ui_section_divider("Context", "Series format and contract")
        st.markdown("### Context — series format and what the contract represents")
        # Keyed widgets prevent Streamlit rerun snapback (index= without key reinitializes from index on every rerun).
        _series_opts = ["BO1", "BO3", "BO5"]
        _scope_opts = ["Map winner (this map)", "Series winner"]
        if "cs2_live_series_fmt" not in st.session_state:
            _idx = int(st.session_state.get("cs2_live_series_fmt_idx", 1))
            st.session_state["cs2_live_series_fmt"] = _series_opts[_idx] if 0 <= _idx < len(_series_opts) else "BO3"
        if st.session_state.get("cs2_live_series_fmt") not in _series_opts:
            st.session_state["cs2_live_series_fmt"] = "BO3"
        if "cs2_live_contract_scope" not in st.session_state:
            _idx = int(st.session_state.get("cs2_live_contract_scope_idx", 0))
            st.session_state["cs2_live_contract_scope"] = _scope_opts[_idx] if 0 <= _idx < len(_scope_opts) else "Map winner (this map)"
        if st.session_state.get("cs2_live_contract_scope") not in _scope_opts:
            st.session_state["cs2_live_contract_scope"] = "Map winner (this map)"
        cctx1, cctx2, cctx3, cctx4 = st.columns([1.0, 1.5, 1.0, 1.0])
        with cctx1:
            series_fmt = st.selectbox("Series format", _series_opts, key="cs2_live_series_fmt")
        with cctx2:
            contract_scope = st.selectbox("Contract priced on", _scope_opts, key="cs2_live_contract_scope")
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

        # Mode-switch sync: when switching contract scope, copy/sync pre-match prior so odds don't look reset
        _prev_scope = st.session_state.get("cs2_live_prev_contract_scope")
        if _prev_scope != contract_scope and n_maps > 1:
            if _prev_scope == "Map winner (this map)" and contract_scope == "Series winner":
                _p_map = st.session_state.get("cs2_live_p0_map")
                if _p_map is not None:
                    try:
                        _p_map_f = float(_p_map)
                        if np.isfinite(_p_map_f):
                            target = _bestof_target(n_maps)
                            _p_series = series_prob_needed(target, target, _p_map_f)
                            _p_series = float(np.clip(_p_series, 0.01, 0.99))
                            st.session_state["cs2_live_p0_series"] = _p_series
                    except (TypeError, ValueError):
                        pass
            elif _prev_scope == "Series winner" and contract_scope == "Map winner (this map)":
                _p_series = st.session_state.get("cs2_live_p0_series")
                if _p_series is not None:
                    try:
                        _p_series_f = float(_p_series)
                        if np.isfinite(_p_series_f):
                            _p_map = invert_series_to_map_prob(_p_series_f, n_maps)
                            _p_map = float(np.clip(_p_map, 0.01, 0.99))
                            st.session_state["cs2_live_p0_map"] = _p_map
                    except (TypeError, ValueError):
                        pass
        st.session_state["cs2_live_prev_contract_scope"] = contract_scope

        ui_section_divider("Step 1", "Pre-match fair probability (from your model)")
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

        st.checkbox("Lock config (start logging snapshots)", key="cs2_live_config_locked")
        prev_locked = st.session_state.get("cs2_live_config_locked_prev", False)
        cur_locked = bool(st.session_state.get("cs2_live_config_locked", False))
        st.session_state["cs2_live_config_locked_prev"] = cur_locked
        if cur_locked and not prev_locked:
            st.session_state["cs2_live_rows"] = []
            st.session_state.pop("cs2_inplay_last_total_rounds", None)

        ui_section_divider("Step 2", "Live map inputs")
        st.markdown("### Step 2 — Live map inputs (update whenever you want)")
        if "cs2_live_rows" not in st.session_state:
            st.session_state["cs2_live_rows"] = []
        if "cs2_live_raw_payload_debug_rows" not in st.session_state:
            st.session_state["cs2_live_raw_payload_debug_rows"] = []

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

        # BO3 ECON INTEGRATION — helper for team total money from player list (balance only; safe parsing)
        def _compute_team_total_money_from_players(players):
            if not players or not isinstance(players, list):
                return 0.0
            total = 0.0
            for p in players:
                if not isinstance(p, dict):
                    continue
                b = p.get("balance")
                try:
                    total += float(b if b is not None else 0)
                except (TypeError, ValueError):
                    pass
            return total

        # BO3 ECON INTEGRATION — economy source: Manual (legacy), BO3 feed, or GRID feed
        ECONOMY_SOURCE_LEGACY = "Manual Buy-State (legacy)"
        ECONOMY_SOURCE_BO3 = "Auto from BO3 feed (team total money)"
        ECONOMY_SOURCE_GRID = "Auto from GRID feed (money + loadout)"
        _econ_opts = [ECONOMY_SOURCE_LEGACY, ECONOMY_SOURCE_BO3, ECONOMY_SOURCE_GRID]
        # Restore economy source after GRID Pull/Apply rerun (so selection stays constant)
        if "cs2_economy_source_preserve" in st.session_state:
            _restore = st.session_state.pop("cs2_economy_source_preserve", None)
            if _restore in _econ_opts:
                st.session_state["cs2_economy_source"] = _restore
        _econ_idx = 0 if st.session_state.get("cs2_economy_source", ECONOMY_SOURCE_LEGACY) == ECONOMY_SOURCE_LEGACY else (1 if st.session_state.get("cs2_economy_source") == ECONOMY_SOURCE_BO3 else 2)
        economy_source = st.radio(
            "Economy source",
            _econ_opts,
            index=min(_econ_idx, 2),
            key="cs2_economy_source",
            horizontal=True,
            help="Manual: use buy-state dropdowns. BO3: team total from BO3 feed. GRID: money+loadout from GRID V2 when live source is GRID.",
        )
        # Faster economy input (buy state) — legacy path; dropdowns always visible for fallback + comparison
        BUY_STATES = ["Skip/Unknown", "Eco", "Light", "Full (fragile)", "Full", "Full+Save"]
        BUY_STATE_TO_ECON = {
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
        manual_econ_a = float(BUY_STATE_TO_ECON.get(str(buy_a), 0.0))
        manual_econ_b = float(BUY_STATE_TO_ECON.get(str(buy_b), 0.0))

        # BO3 ECON LATCH — choose econ_a/econ_b: Auto uses latched (round-frozen) values when available, else fall back to manual
        econ_feed_a_raw = None
        econ_feed_b_raw = None
        econ_latched_a = None
        econ_latched_b = None
        econ_latched_round_number = None
        if economy_source == ECONOMY_SOURCE_BO3:
            # BO3 ECON LATCH — prefer latched econ (frozen per round) so mid-round swings don't move probability
            latched_a = st.session_state.get("cs2_live_latched_econ_a")
            latched_b = st.session_state.get("cs2_live_latched_econ_b")
            latched_round = st.session_state.get("cs2_live_latched_round_number")
            feed_a_raw = st.session_state.get("cs2_live_econ_a")
            feed_b_raw = st.session_state.get("cs2_live_econ_b")
            if latched_a is not None and latched_b is not None:
                try:
                    econ_a = float(latched_a)
                    econ_b = float(latched_b)
                    # BO3 LOADOUT VALUE FIX — caption reflects cash+loadout when integrated
                    econ_source_used = "bo3_total_resources_latched" if st.session_state.get("cs2_live_econ_is_total_resources") else "bo3_balance_latched"
                    team_a_money_total = econ_a
                    team_b_money_total = econ_b
                    econ_latched_a = econ_a
                    econ_latched_b = econ_b
                    econ_latched_round_number = latched_round
                    econ_feed_a_raw = float(feed_a_raw) if feed_a_raw is not None else None
                    econ_feed_b_raw = float(feed_b_raw) if feed_b_raw is not None else None
                except (TypeError, ValueError):
                    econ_a = manual_econ_a
                    econ_b = manual_econ_b
                    econ_source_used = "manual"
                    team_a_money_total = manual_econ_a
                    team_b_money_total = manual_econ_b
            elif feed_a_raw is not None and feed_b_raw is not None:
                # BO3 ECON LATCH — latched missing/invalid: fall back to current feed then manual
                try:
                    econ_a = float(feed_a_raw)
                    econ_b = float(feed_b_raw)
                    econ_source_used = "bo3_total_resources" if st.session_state.get("cs2_live_econ_is_total_resources") else "bo3_balance"
                    team_a_money_total = econ_a
                    team_b_money_total = econ_b
                    econ_feed_a_raw = econ_a
                    econ_feed_b_raw = econ_b
                except (TypeError, ValueError):
                    econ_a = manual_econ_a
                    econ_b = manual_econ_b
                    econ_source_used = "manual"
                    team_a_money_total = manual_econ_a
                    team_b_money_total = manual_econ_b
            else:
                econ_a = manual_econ_a
                econ_b = manual_econ_b
                econ_source_used = "manual"
                team_a_money_total = manual_econ_a
                team_b_money_total = manual_econ_b
        elif economy_source == ECONOMY_SOURCE_GRID:
            # GRID economy: use money+loadout from GRID V2 (already applied to session_state when live source is GRID)
            feed_a_raw = st.session_state.get("cs2_live_econ_a")
            feed_b_raw = st.session_state.get("cs2_live_econ_b")
            if feed_a_raw is not None and feed_b_raw is not None:
                try:
                    econ_a = float(feed_a_raw)
                    econ_b = float(feed_b_raw)
                    econ_source_used = "grid_total_resources"
                    team_a_money_total = econ_a
                    team_b_money_total = econ_b
                    econ_feed_a_raw = econ_a
                    econ_feed_b_raw = econ_b
                except (TypeError, ValueError):
                    econ_a = manual_econ_a
                    econ_b = manual_econ_b
                    econ_source_used = "manual"
                    team_a_money_total = manual_econ_a
                    team_b_money_total = manual_econ_b
                    econ_feed_a_raw = None
                    econ_feed_b_raw = None
            else:
                econ_a = manual_econ_a
                econ_b = manual_econ_b
                econ_source_used = "manual"
                team_a_money_total = manual_econ_a
                team_b_money_total = manual_econ_b
                econ_feed_a_raw = feed_a_raw
                econ_feed_b_raw = feed_b_raw
        else:
            econ_a = manual_econ_a
            econ_b = manual_econ_b
            econ_source_used = "manual"
            team_a_money_total = econ_a
            team_b_money_total = econ_b
            st.session_state["cs2_live_econ_a"] = int(econ_a)
            st.session_state["cs2_live_econ_b"] = int(econ_b)

        econ_missing = (str(buy_a) == "Skip/Unknown") or (str(buy_b) == "Skip/Unknown")
        econ_fragile = (str(buy_a) == "Full (fragile)") or (str(buy_b) == "Full (fragile)")
        # BO3 ECON LATCH — caption: show latched or feed econ when Auto (BO3 or GRID)
        if economy_source == ECONOMY_SOURCE_BO3 and econ_source_used != "manual":
            st.caption(f"Econ from BO3 ({econ_source_used}): A=${econ_a:,.0f}, B=${econ_b:,.0f} — switch to Manual to use buy-state dropdown.")
        elif economy_source == ECONOMY_SOURCE_GRID and econ_source_used != "manual":
            st.caption(f"Econ from GRID ({econ_source_used}): A=${econ_a:,.0f}, B=${econ_b:,.0f} — switch to Manual to use buy-state dropdown.")
        else:
            st.caption(f"Econ proxy (from buy states): A=${econ_a:,.0f}, B=${econ_b:,.0f} — set Skip/Unknown to widen bands and avoid false precision.")
        # BO3 LOADOUT VALUE FIX — show derived cash / loadout / total resources when available
        _ca = st.session_state.get("cs2_live_team_a_cash_total")
        _cb = st.session_state.get("cs2_live_team_b_cash_total")
        _la = st.session_state.get("cs2_live_team_a_loadout_est_total")
        _lb = st.session_state.get("cs2_live_team_b_loadout_est_total")
        _ra = st.session_state.get("cs2_live_team_a_total_resources")
        _rb = st.session_state.get("cs2_live_team_b_total_resources")
        if _ca is not None and _cb is not None:
            _cash = f"Cash A=${float(_ca):,.0f}, B=${float(_cb):,.0f}"
            _load = ""
            if _la is not None and _lb is not None:
                _load = f" | Loadout est A=${float(_la):,.0f}, B=${float(_lb):,.0f}"
            _res = ""
            if _ra is not None and _rb is not None:
                _res = f" | Total resources A=${float(_ra):,.0f}, B=${float(_rb):,.0f}"
            st.caption(f"BO3 derived: {_cash}{_load}{_res}")
        # BO3 LOADOUT VALUE FIX

        colP1, colP2, colChaos = st.columns(3)
        with colP1:
            pistol_a = st.checkbox("A won most recent pistol?", value=bool(st.session_state.get("cs2_live_pistol_a", False)), key="cs2_live_pistol_a")
        with colP2:
            pistol_b = st.checkbox("B won most recent pistol?", value=bool(st.session_state.get("cs2_live_pistol_b", False)), key="cs2_live_pistol_b")
        with colChaos:
            chaos_boost = st.slider("Chaos widen (manual)", 0.00, 0.25, float(st.session_state.get("cs2_live_chaos", 0.00)), 0.01)
        st.session_state["cs2_live_chaos"] = float(chaos_boost)

    section_header("Market / Execution")
    if contract_scope == "Series winner" and n_maps > 1:
        st.caption("Enter executable prices for **Team A to win the SERIES** (bid = sell YES, ask = buy YES).")
    else:
        st.caption("Enter executable prices for **Team A to win THIS MAP** (bid = sell YES, ask = buy YES).")

    st.markdown("#### Optional — Fetch bid/ask from an exchange API")

    # Restore Kalshi/market state after GRID-triggered rerun (Pull, Apply, Fetch) so link, team, odds stay.
    # Skip restoring bid/ask when we just fetched them this run (auto-pull or Pull) so Kalshi odds stay fresh.
    _CS2_MKT_PRESERVE_KEYS = (
        "cs2_mkt_fetch_venue", "cs2_kalshi_url", "cs2_kalshi_markets", "cs2_kalshi_market",
        "cs2_live_market_bid", "cs2_live_market_ask", "cs2_mkt_fetch_meta", "cs2_mkt_fetch_ident",
    )
    _skip_bid_ask_restore = st.session_state.pop("cs2_mkt_bid_ask_fetched_this_run", False)
    if "cs2_mkt_preserve" in st.session_state:
        _pres = st.session_state.pop("cs2_mkt_preserve", None)
        if isinstance(_pres, dict):
            for _k in _CS2_MKT_PRESERVE_KEYS:
                if _k not in _pres:
                    continue
                if _skip_bid_ask_restore and _k in ("cs2_live_market_bid", "cs2_live_market_ask"):
                    continue
                st.session_state[_k] = _pres[_k]

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
            refresh_and_snap = st.button("Refresh Kalshi + Add snapshot", key="cs2_kalshi_refresh_and_snap", use_container_width=True)

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
            tkr, err = _cs2_kalshi_resolve_ticker()
            if err or not tkr:
                st.error(err or "Load teams and select a market, or paste a Kalshi URL and click Load teams.")
            else:
                try:
                    bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                    # No bid -> 0, no ask -> 1 (represents no available orders); clamp to [0,1] for widget
                    _bid = float(bid_f) if bid_f is not None else 0.0
                    _ask = float(ask_f) if ask_f is not None else 1.0
                    st.session_state["cs2_live_market_bid"] = max(0.0, min(1.0, _bid))
                    st.session_state["cs2_live_market_ask"] = max(0.0, min(1.0, _ask))
                    st.session_state["cs2_mkt_fetch_meta"] = meta
                    # BO3 MARKET DELAY ALIGN
                    if bid_f is not None and ask_f is not None:
                        _cs2_market_delay_push(float(bid_f), float(ask_f))
                    # BO3 MARKET DELAY ALIGN
                    st.success("Kalshi bid/ask updated." + (" (No bid/ask: shown as 0 / 1.)" if bid_f is None or ask_f is None else ""))
                except Exception as e:
                    st.error(f"Kalshi refresh failed: {e}")

        # Refresh Kalshi + add one chart snapshot (one click: fetch bid/ask then rerun to add snapshot)
        if refresh_and_snap:
            tkr, err = _cs2_kalshi_resolve_ticker()
            if err or not tkr:
                st.error(err or "Load teams and select a market, or paste a Kalshi URL and click Load teams.")
            else:
                try:
                    bid_f, ask_f, meta = fetch_kalshi_bid_ask(tkr)
                    st.session_state["cs2_live_market_bid"] = float(bid_f) if bid_f is not None else 0.0
                    st.session_state["cs2_live_market_ask"] = float(ask_f) if ask_f is not None else 1.0
                    if meta is not None:
                        st.session_state["cs2_mkt_fetch_meta"] = meta
                    # BO3 MARKET DELAY ALIGN
                    if bid_f is not None and ask_f is not None:
                        _cs2_market_delay_push(float(bid_f), float(ask_f))
                    # BO3 MARKET DELAY ALIGN
                    st.session_state["cs2_auto_add_snapshot_this_run"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Kalshi + snapshot failed: {e}")

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

    # Allow 0 and 1 so "no bid" = 0 and "no ask" = 1 (no available orders) don't error
    colBid, colAsk = st.columns(2)
    with colBid:
        st.session_state.setdefault("cs2_live_market_bid", float(st.session_state.get("cs2_live_market_bid", 0.48)))
        market_bid = st.number_input("Best bid (Sell) for Team A (0–1)", min_value=0.0, max_value=1.0,
                                     step=0.01, format="%.2f", key="cs2_live_market_bid",
                                     help="Use 0 when there are no bids (no one willing to buy Team A).")
    with colAsk:
        st.session_state.setdefault("cs2_live_market_ask", float(st.session_state.get("cs2_live_market_ask", 0.52)))
        market_ask = st.number_input("Best ask (Buy) for Team A (0–1)", min_value=0.0, max_value=1.0,
                                     step=0.01, format="%.2f", key="cs2_live_market_ask",
                                     help="Use 1 when there are no asks (no one willing to sell Team A).")

    # BO3 MARKET DELAY ALIGN: delay market used for snapshot logging (align with delayed BO3 feed)
    st.number_input(
        "Market logging delay (sec)", min_value=0, max_value=600, value=120,
        step=10, key="cs2_market_logging_delay_sec", help="Use market from this many seconds ago when logging snapshots (0 = live)."
    )
    # BO3 MARKET DELAY ALIGN

    bid = float(market_bid)
    ask = float(market_ask)
    if ask < bid:
        st.warning("Ask < bid (inverted). Using ask = bid for calculations.")
        ask = bid
    if bid == 0.0 or ask == 1.0:
        st.caption("Bid=0 or Ask=1 means no available orders on that side (Kalshi returned no bid/ask).")

    market_mid = 0.5 * (bid + ask)
    spread = ask - bid
    rel_spread = (spread / market_mid) if market_mid > 0 else 0.0

    # ---- Compute fair probability (map) + K-based credible bands (map) ----
    # BO3 STATE BANDS CONSISTENCY: single canonical fair path via helper
    (p_hat, p_hat_map_raw) = _compute_cs2_live_fair_for_state(
        float(p0_map), int(rounds_a), int(rounds_b), float(econ_a), float(econ_b),
        bool(pistol_a), bool(pistol_b), float(beta_score), float(beta_econ), float(beta_pistol),
        map_name, a_side, float(beta_lock), int(lock_start_offset), int(lock_ramp),
        str(contract_scope), int(n_maps), int(maps_a_won), int(maps_b_won),
    )
    if p_hat is None or p_hat_map_raw is None:
        # Fallback: existing path (do not crash)
        win_target = cs2_current_win_target(int(rounds_a), int(rounds_b))
        total_rounds = int(2 * win_target - 2)
        p_hat_map = estimate_inplay_prob(
            float(p0_map), int(rounds_a), int(rounds_b), float(econ_a), float(econ_b),
            pistol_a=bool(pistol_a), pistol_b=bool(pistol_b),
            beta_score=float(beta_score), beta_econ=float(beta_econ), beta_pistol=float(beta_pistol),
            map_name=map_name, a_side=a_side, pistol_decay=0.30, beta_side=0.85,
            beta_lock=float(beta_lock), lock_start_offset=int(lock_start_offset), lock_ramp=int(lock_ramp),
            win_target=int(win_target),
        )
        stream = update_round_stream("cs2_inplay", int(rounds_a), int(rounds_b))
        kappa_map = compute_kappa_cs2(
            p0=float(p0_map), rounds_a=int(rounds_a), rounds_b=int(rounds_b),
            econ_missing=bool(econ_missing), econ_fragile=bool(econ_fragile),
            pistol_a=bool(pistol_a), pistol_b=bool(pistol_b),
            streak_len=int(stream.get("streak_len", 0)), streak_winner=stream.get("streak_winner", None),
            reversal=bool(stream.get("reversal", False)), gap_delta=int(stream.get("gap_delta", 0)),
            chaos_boost=float(chaos_boost), total_rounds=int(2 * win_target - 2),
        )
        calib = load_kappa_calibration()
        total_r = int(int(rounds_a) + int(rounds_b))
        is_ot = (int(rounds_a) >= int(win_target) - 1 and int(rounds_b) >= int(win_target) - 1)
        mult = get_kappa_multiplier(calib, "cs2", float(cs2_band_level), total_r, is_ot) if bool(st.session_state.get("cs2_use_calib", False)) else 1.0
        kappa_map = float(kappa_map) * float(cs2_k_scale) * float(mult)
        lo_map, hi_map = beta_credible_interval(float(p_hat_map), float(kappa_map), level=float(cs2_band_level))
        p_hat_map = cs2_soft_lock_map_prob(float(p_hat_map), int(rounds_a), int(rounds_b), int(win_target))
        lo_map = cs2_soft_lock_map_prob(float(lo_map), int(rounds_a), int(rounds_b), int(win_target))
        hi_map = cs2_soft_lock_map_prob(float(hi_map), int(rounds_a), int(rounds_b), int(win_target))
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
    else:
        win_target = cs2_current_win_target(int(rounds_a), int(rounds_b))
        total_rounds = int(2 * win_target - 2)
        stream = update_round_stream("cs2_inplay", int(rounds_a), int(rounds_b))
        kappa_map = compute_kappa_cs2(
            p0=float(p0_map), rounds_a=int(rounds_a), rounds_b=int(rounds_b),
            econ_missing=bool(econ_missing), econ_fragile=bool(econ_fragile),
            pistol_a=bool(pistol_a), pistol_b=bool(pistol_b),
            streak_len=int(stream.get("streak_len", 0)), streak_winner=stream.get("streak_winner", None),
            reversal=bool(stream.get("reversal", False)), gap_delta=int(stream.get("gap_delta", 0)),
            chaos_boost=float(chaos_boost), total_rounds=int(total_rounds),
        )
        calib = load_kappa_calibration()
        total_r = int(int(rounds_a) + int(rounds_b))
        is_ot = (int(rounds_a) >= int(win_target) - 1 and int(rounds_b) >= int(win_target) - 1)
        mult = get_kappa_multiplier(calib, "cs2", float(cs2_band_level), total_r, is_ot) if bool(st.session_state.get("cs2_use_calib", False)) else 1.0
        kappa_map = float(kappa_map) * float(cs2_k_scale) * float(mult)
        lo_map, hi_map = beta_credible_interval(float(p_hat_map_raw), float(kappa_map), level=float(cs2_band_level))
        p_hat_map = cs2_soft_lock_map_prob(float(p_hat_map_raw), int(rounds_a), int(rounds_b), int(win_target))
        lo_map = cs2_soft_lock_map_prob(float(lo_map), int(rounds_a), int(rounds_b), int(win_target))
        hi_map = cs2_soft_lock_map_prob(float(hi_map), int(rounds_a), int(rounds_b), int(win_target))
        if contract_scope == "Series winner" and n_maps > 1:
            lo = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(lo_map), float(p0_map))
            hi = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(hi_map), float(p0_map))
            line_label = "Series fair p(A)"
        else:
            lo = float(lo_map)
            hi = float(hi_map)
            line_label = "Map fair p(A)"
    # BO3 STATE BANDS CONSISTENCY

    # BO3 MIDROUND V1 — detect round transition BEFORE rail computation so new-round rails use new-round latched context
    game_number_for_key = st.session_state.get("cs2_live_game_number")
    current_round_key = _build_cs2_round_key(
        map_name, int(rounds_a), int(rounds_b), int(maps_a_won), int(maps_b_won), str(contract_scope), int(n_maps),
        game_number=game_number_for_key,
    )
    prev_round_key = st.session_state.get("cs2_midround_round_key")
    st.session_state.pop("cs2_midround_transition_tick", None)
    st.session_state["cs2_live_round_finalization_snap_occurred"] = False
    st.session_state["cs2_round_snap_applied_this_tick"] = False
    st.session_state["cs2_rail_round_end_latched"] = False
    st.session_state["cs2_rail_resolved_to_prev_endpoint"] = False
    use_snap_this_tick = False
    round_transition_detected = False
    round_transition_prev_key = None
    round_transition_new_key = None
    round_transition_winner = None
    round_transition_resolved_p = None
    round_transition_resolution_source = "none"
    transition_this_tick = False
    p_hat_pre_snap = float(p_hat)
    used_prev_frozen_for_snap = False
    latched_new_round_context = False

    if current_round_key != prev_round_key:
        transition_this_tick = True
        round_transition_detected = True
        round_transition_prev_key = prev_round_key
        round_transition_new_key = current_round_key
        prev_band_if_a = st.session_state.get("cs2_midround_band_if_a_round_frozen")
        prev_band_if_b = st.session_state.get("cs2_midround_band_if_b_round_frozen")
        prev_rounds_a = st.session_state.get("cs2_midround_prev_rounds_a")
        prev_rounds_b = st.session_state.get("cs2_midround_prev_rounds_b")
        prev_frozen_lo = st.session_state.get("cs2_midround_band_state_lo_frozen")
        prev_frozen_hi = st.session_state.get("cs2_midround_band_state_hi_frozen")
        prev_p_base = st.session_state.get("cs2_midround_p_base_frozen")
        st.session_state["cs2_live_latched_band_lo"] = float(prev_frozen_lo) if prev_frozen_lo is not None else None
        st.session_state["cs2_live_latched_band_hi"] = float(prev_frozen_hi) if prev_frozen_hi is not None else None
        st.session_state["cs2_live_latched_p_base"] = float(prev_p_base) if prev_p_base is not None else None
        st.session_state["cs2_live_latched_round_key"] = prev_round_key
        delta_a = (int(rounds_a) - prev_rounds_a) if prev_rounds_a is not None else None
        delta_b = (int(rounds_b) - prev_rounds_b) if prev_rounds_b is not None else None
        p_hat_pre_snap = float(p_hat)
        if delta_a == 1 and delta_b == 0 and prev_band_if_a is not None:
            round_transition_winner = "A"
            round_transition_resolved_p = float(prev_band_if_a)
            round_transition_resolution_source = "prev_band_if_a"
            p_hat = float(prev_band_if_a)
            st.session_state["cs2_midround_transition_tick"] = True
            st.session_state["cs2_live_round_finalization_snap_occurred"] = True
            st.session_state["cs2_rail_round_end_latched"] = True
            st.session_state["cs2_rail_resolved_to_prev_endpoint"] = True
            use_snap_this_tick = True
            used_prev_frozen_for_snap = True
            st.session_state["cs2_round_snap_used"] = True
            st.session_state["cs2_round_snap_target_side"] = "A"
            st.session_state["cs2_round_snap_value"] = float(prev_band_if_a)
            st.session_state["cs2_round_snap_prev_freeze_key"] = prev_round_key
        elif delta_a == 0 and delta_b == 1 and prev_band_if_b is not None:
            round_transition_winner = "B"
            round_transition_resolved_p = float(prev_band_if_b)
            round_transition_resolution_source = "prev_band_if_b"
            p_hat = float(prev_band_if_b)
            st.session_state["cs2_midround_transition_tick"] = True
            st.session_state["cs2_live_round_finalization_snap_occurred"] = True
            st.session_state["cs2_rail_round_end_latched"] = True
            st.session_state["cs2_rail_resolved_to_prev_endpoint"] = True
            use_snap_this_tick = True
            used_prev_frozen_for_snap = True
            st.session_state["cs2_round_snap_used"] = True
            st.session_state["cs2_round_snap_target_side"] = "B"
            st.session_state["cs2_round_snap_value"] = float(prev_band_if_b)
            st.session_state["cs2_round_snap_prev_freeze_key"] = prev_round_key
        else:
            round_transition_winner = None
            st.session_state["cs2_round_snap_used"] = False
            st.session_state["cs2_round_snap_target_side"] = None
            st.session_state["cs2_round_snap_value"] = None
            st.session_state["cs2_round_snap_prev_freeze_key"] = prev_round_key
        # Phase 3: set pending latch instead of immediately finalizing round-start context (avoid mixed-phase on score-flip tick)
        st.session_state["cs2_round_start_latch_pending"] = True
        st.session_state["cs2_round_start_latch_pending_round_key"] = current_round_key
        st.session_state["cs2_round_start_latch_pending_since_ts"] = time.time()
        st.session_state["cs2_round_start_latch_pending_attempts"] = 0
        st.session_state["cs2_midround_round_key"] = current_round_key
        st.session_state["cs2_midround_prev_rounds_a"] = int(rounds_a)
        st.session_state["cs2_midround_prev_rounds_b"] = int(rounds_b)
        latched_new_round_context = False

    # Phase 3: resolve pending round-start latch (every tick when pending is set)
    if st.session_state.get("cs2_round_start_latch_pending"):
        _pending_key = st.session_state.get("cs2_round_start_latch_pending_round_key")
        _attempts = int(st.session_state.get("cs2_round_start_latch_pending_attempts", 0))
        _do_finalize = False
        _latch_status = "pending"
        _latch_reason = "retry_waiting"
        if current_round_key != _pending_key:
            _do_finalize = True
            _latch_status = "latched_fallback"
            _latch_reason = "key_mismatch"
        else:
            _stable, _reason = _cs2_round_start_context_stable(current_round_key, _pending_key, st.session_state)
            if _stable:
                _do_finalize = True
                _latch_status = "latched_stable"
                _latch_reason = _reason if _reason != "key_mismatch" else "same_tick_stable"
            elif _attempts >= CS2_ROUND_START_LATCH_PENDING_MAX_ATTEMPTS:
                _do_finalize = True
                _latch_status = "latched_fallback"
                _latch_reason = "timeout_fallback"
            else:
                st.session_state["cs2_round_start_latch_pending_attempts"] = _attempts + 1
        if _do_finalize:
            st.session_state["round_start_team_a_econ_total"] = st.session_state.get("cs2_live_econ_a")
            st.session_state["round_start_team_b_econ_total"] = st.session_state.get("cs2_live_econ_b")
            st.session_state["round_start_team_a_loadout_total"] = st.session_state.get("cs2_live_team_a_loadout_est_total")
            st.session_state["round_start_team_b_loadout_total"] = st.session_state.get("cs2_live_team_b_loadout_est_total")
            st.session_state["round_start_team_a_armor_total"] = st.session_state.get("cs2_live_team_a_armor_alive_total")
            st.session_state["round_start_team_b_armor_total"] = st.session_state.get("cs2_live_team_b_armor_alive_total")
            st.session_state["round_start_team_a_side"] = st.session_state.get("cs2_live_team_a_side")
            st.session_state["round_start_team_b_side"] = st.session_state.get("cs2_live_team_b_side")
            st.session_state["round_start_map_name"] = st.session_state.get("cs2_live_map_name")
            st.session_state["round_start_game_number"] = st.session_state.get("cs2_live_game_number")
            st.session_state["cs2_round_start_latch_pending"] = False
            st.session_state["cs2_round_start_latch_pending_round_key"] = None
            st.session_state["cs2_round_start_latch_pending_since_ts"] = None
            st.session_state["cs2_round_start_latch_pending_attempts"] = None
            latched_new_round_context = True
            st.session_state["round_start_latch_status"] = _latch_status
            st.session_state["round_start_latch_reason"] = _latch_reason
            st.session_state["round_start_latch_finalized_tick_ts"] = time.time()
        else:
            st.session_state["round_start_latch_status"] = _latch_status
            st.session_state["round_start_latch_reason"] = _latch_reason
            st.session_state["round_start_latch_finalized_tick_ts"] = None
    else:
        st.session_state["round_start_latch_status"] = "idle"
        st.session_state["round_start_latch_reason"] = None
        st.session_state["round_start_latch_finalized_tick_ts"] = None

    # Latch debug fields (additive; written every tick for inspection)
    st.session_state["round_start_latch_pending"] = bool(st.session_state.get("cs2_round_start_latch_pending"))
    st.session_state["round_start_latch_pending_round_key"] = st.session_state.get("cs2_round_start_latch_pending_round_key")
    st.session_state["round_start_latch_attempts"] = st.session_state.get("cs2_round_start_latch_pending_attempts")

    # BO3 STATE BANDS: round-state endpoint bands (if A wins round / if B wins round)
    # MAP RESOLVED: when one team has reached win_target, map is over — force resolved 0/1 so chart doesn't oscillate
    win_target = cs2_current_win_target(int(rounds_a), int(rounds_b))
    map_over_a = int(rounds_a) >= int(win_target) and int(rounds_a) > int(rounds_b)
    map_over_b = int(rounds_b) >= int(win_target) and int(rounds_b) > int(rounds_a)
    if map_over_a or map_over_b:
        p_map_resolved = 1.0 if map_over_a else 0.0
        p_hat_map = float(p_map_resolved)
        if contract_scope == "Series winner" and int(n_maps) > 1:
            p_hat = series_win_prob_live(int(n_maps), int(maps_a_won), int(maps_b_won), float(p_hat_map), float(p0_map))
        else:
            p_hat = float(p_hat_map)
        lo = hi = float(p_hat)
        lo_map = hi_map = float(p_hat_map)
        band_if_a_round = band_if_b_round = band_state_lo = band_state_hi = float(p_hat)
        st.session_state["cs2_rail_debug"] = {}
    else:
        # Round-state rails v2: nonlinear score/econ/side/loadout model (corridor bounds for mid-round V2)
        # Rail inputs use round-start latched values when available to avoid mid-round drift
        current_round_key_rail = _build_cs2_round_key(
            map_name, int(rounds_a), int(rounds_b), int(maps_a_won), int(maps_b_won), str(contract_scope), int(n_maps),
            game_number=game_number_for_key,
        )
        prev_round_key_rail = st.session_state.get("cs2_midround_round_key")
        econ_a_rail = st.session_state.get("round_start_team_a_econ_total")
        if econ_a_rail is None:
            econ_a_rail = float(econ_a) if econ_a is not None else None
        econ_b_rail = st.session_state.get("round_start_team_b_econ_total")
        if econ_b_rail is None:
            econ_b_rail = float(econ_b) if econ_b is not None else None
        loadout_a_rail = st.session_state.get("round_start_team_a_loadout_total")
        if loadout_a_rail is None:
            loadout_a_rail = st.session_state.get("cs2_live_team_a_loadout_est_total")
        loadout_b_rail = st.session_state.get("round_start_team_b_loadout_total")
        if loadout_b_rail is None:
            loadout_b_rail = st.session_state.get("cs2_live_team_b_loadout_est_total")
        rail_inputs_latched = (
            st.session_state.get("round_start_team_a_econ_total") is not None
            and st.session_state.get("round_start_team_b_econ_total") is not None
        )
        # Canonical rail anchor/endpoints use round-start latched econ (and side) so they stay stable within a round
        canonical_econ_a_rail = float(econ_a_rail) if econ_a_rail is not None else (float(econ_a) if econ_a is not None else None)
        canonical_econ_b_rail = float(econ_b_rail) if econ_b_rail is not None else (float(econ_b) if econ_b is not None else None)
        team_a_side_rail = st.session_state.get("round_start_team_a_side") if st.session_state.get("round_start_team_a_side") is not None else a_side
        rail_canonical_econ_source = "latched_round_start" if (econ_a_rail is not None and econ_b_rail is not None) else "live_fallback"
        v2_rails = _compute_cs2_round_state_rails_v2(
            int(rounds_a), int(rounds_b), int(win_target),
            map_name=map_name, team_a_side=team_a_side_rail,
            econ_a=float(econ_a_rail) if econ_a_rail is not None else None,
            econ_b=float(econ_b_rail) if econ_b_rail is not None else None,
            loadout_a=float(loadout_a_rail) if loadout_a_rail is not None else None,
            loadout_b=float(loadout_b_rail) if loadout_b_rail is not None else None,
            series_maps_won_a=int(maps_a_won) if maps_a_won is not None else None,
            series_maps_won_b=int(maps_b_won) if maps_b_won is not None else None,
            best_of=int(n_maps) if n_maps is not None else 3,
            p0_map=float(p0_map) if p0_map is not None else None,
            pistol_a=bool(pistol_a), pistol_b=bool(pistol_b),
            beta_score=float(beta_score), beta_econ=float(beta_econ), beta_pistol=float(beta_pistol),
            beta_lock=float(beta_lock), lock_start_offset=int(lock_start_offset), lock_ramp=int(lock_ramp),
            contract_scope=str(contract_scope), n_maps=int(n_maps) if n_maps is not None else None,
            maps_a_won=int(maps_a_won) if maps_a_won is not None else None,
            maps_b_won=int(maps_b_won) if maps_b_won is not None else None,
            canonical_econ_a=canonical_econ_a_rail,
            canonical_econ_b=canonical_econ_b_rail,
            rail_mode=st.session_state.get("cs2_rail_mode", "ANCHOR_PULLED"),
            rail_pull_scale=st.session_state.get("cs2_rail_pull_scale", 0.7),
            rail_pull_cap=st.session_state.get("cs2_rail_pull_cap", 0.6),
        )
        band_if_a_round = v2_rails.get("band_if_a_round")
        band_if_b_round = v2_rails.get("band_if_b_round")
        band_state_lo = v2_rails.get("band_lo")
        band_state_hi = v2_rails.get("band_hi")
        anchor_raw = v2_rails.get("anchor")
        # Terminal override flags from v2_rails for smoothing bypass (exact terminal geometry)
        term_override_a = bool(v2_rails.get("rail_terminal_override_a_used", False))
        term_override_b = bool(v2_rails.get("rail_terminal_override_b_used", False))
        term_if_a = v2_rails.get("rail_terminal_if_a")
        term_if_b = v2_rails.get("rail_terminal_if_b")
        band_if_a_pre_smooth = float(band_if_a_round) if band_if_a_round is not None else None
        band_if_b_pre_smooth = float(band_if_b_round) if band_if_b_round is not None else None
        rail_used_smoothing = False
        rail_smoothing_alpha_used = None
        if current_round_key_rail == prev_round_key_rail and prev_round_key_rail is not None:
            prev_anchor = st.session_state.get("cs2_rail_prev_anchor")
            prev_lo = st.session_state.get("cs2_rail_prev_band_lo")
            prev_hi = st.session_state.get("cs2_rail_prev_band_hi")
            prev_if_a = st.session_state.get("cs2_rail_prev_band_if_a")
            prev_if_b = st.session_state.get("cs2_rail_prev_band_if_b")
            if prev_anchor is not None and prev_lo is not None and prev_hi is not None:
                alpha = RAIL_SMOOTHING_ALPHA
                anchor_sm = alpha * float(prev_anchor) + (1.0 - alpha) * float(anchor_raw)
                # Per-branch smoothing: skip for terminal-overridden branches so exact terminal geometry is preserved
                if term_override_a:
                    band_if_a_round = band_if_a_round  # keep current (exact terminal)
                else:
                    band_if_a_round = alpha * float(prev_if_a) + (1.0 - alpha) * float(band_if_a_round) if prev_if_a is not None else band_if_a_round
                if term_override_b:
                    band_if_b_round = band_if_b_round  # keep current (exact terminal)
                else:
                    band_if_b_round = alpha * float(prev_if_b) + (1.0 - alpha) * float(band_if_b_round) if prev_if_b is not None else band_if_b_round
                # Rebuild band_lo/band_hi from final per-branch endpoints (do not use smoothed prev_lo/prev_hi)
                if band_if_a_round is not None and band_if_b_round is not None:
                    band_state_lo = min(float(band_if_a_round), float(band_if_b_round))
                    band_state_hi = max(float(band_if_a_round), float(band_if_b_round))
                v2_rails["anchor"] = anchor_sm
                v2_rails["band_lo"] = band_state_lo
                v2_rails["band_hi"] = band_state_hi
                v2_rails["band_if_a_round"] = band_if_a_round
                v2_rails["band_if_b_round"] = band_if_b_round
                v2_rails["p_if_a_wins"] = band_if_a_round
                v2_rails["p_if_b_wins"] = band_if_b_round
                rail_used_smoothing = True
                rail_smoothing_alpha_used = float(alpha)
        # When not smoothing, still rebuild band_lo/band_hi from branch endpoints so visible rails match endpoints
        if band_if_a_round is not None and band_if_b_round is not None:
            band_state_lo = min(float(band_if_a_round), float(band_if_b_round))
            band_state_hi = max(float(band_if_a_round), float(band_if_b_round))
            v2_rails["band_lo"] = band_state_lo
            v2_rails["band_hi"] = band_state_hi
        # Debug: terminal smoothing bypass and pre/post smooth values
        v2_rails["rail_terminal_smoothing_bypass_a"] = rail_used_smoothing and term_override_a
        v2_rails["rail_terminal_smoothing_bypass_b"] = rail_used_smoothing and term_override_b
        v2_rails["rail_band_if_a_round_pre_smooth"] = band_if_a_pre_smooth
        v2_rails["rail_band_if_b_round_pre_smooth"] = band_if_b_pre_smooth
        v2_rails["rail_band_if_a_round_post_smooth"] = float(band_if_a_round) if band_if_a_round is not None else None
        v2_rails["rail_band_if_b_round_post_smooth"] = float(band_if_b_round) if band_if_b_round is not None else None
        _exact_a = (term_if_a is not None and band_if_a_round is not None and abs(float(band_if_a_round) - float(term_if_a)) < 1e-9) if term_override_a else True
        _exact_b = (term_if_b is not None and band_if_b_round is not None and abs(float(band_if_b_round) - float(term_if_b)) < 1e-9) if term_override_b else True
        v2_rails["rail_terminal_endpoint_exact_preserved"] = _exact_a and _exact_b
        st.session_state["cs2_rail_prev_anchor"] = float(v2_rails.get("anchor", anchor_raw)) if (v2_rails.get("anchor") is not None or anchor_raw is not None) else None
        st.session_state["cs2_rail_prev_band_lo"] = float(band_state_lo) if band_state_lo is not None else None
        st.session_state["cs2_rail_prev_band_hi"] = float(band_state_hi) if band_state_hi is not None else None
        st.session_state["cs2_rail_prev_band_if_a"] = float(band_if_a_round) if band_if_a_round is not None else None
        st.session_state["cs2_rail_prev_band_if_b"] = float(band_if_b_round) if band_if_b_round is not None else None
        v2_rails["rail_inputs_latched"] = rail_inputs_latched
        v2_rails["rail_used_smoothing"] = rail_used_smoothing
        v2_rails["rail_smoothing_alpha"] = rail_smoothing_alpha_used
        v2_rails["rail_input_econ_a"] = float(econ_a_rail) if econ_a_rail is not None else None
        v2_rails["rail_input_econ_b"] = float(econ_b_rail) if econ_b_rail is not None else None
        v2_rails["rail_input_loadout_a"] = float(loadout_a_rail) if loadout_a_rail is not None else None
        v2_rails["rail_input_loadout_b"] = float(loadout_b_rail) if loadout_b_rail is not None else None
        v2_rails["rail_input_side_latched"] = team_a_side_rail
        v2_rails["rail_canonical_econ_source"] = rail_canonical_econ_source
        st.session_state["cs2_rail_debug"] = dict(v2_rails) if isinstance(v2_rails, dict) else {}
        _rd = st.session_state["cs2_rail_debug"]
        if isinstance(_rd, dict):
            _rd["round_start_latch_pending"] = bool(st.session_state.get("cs2_round_start_latch_pending"))
            _rd["round_start_latch_pending_round_key"] = st.session_state.get("cs2_round_start_latch_pending_round_key")
            _rd["round_start_latch_attempts"] = st.session_state.get("cs2_round_start_latch_pending_attempts")
            _rd["round_start_latch_status"] = st.session_state.get("round_start_latch_status")
            _rd["round_start_latch_reason"] = st.session_state.get("round_start_latch_reason")
            _rd["round_start_latch_finalized_tick_ts"] = st.session_state.get("round_start_latch_finalized_tick_ts")
    if band_state_lo is not None and band_state_hi is not None:
        st.session_state["cs2_live_band_if_a_round"] = band_if_a_round
        st.session_state["cs2_live_band_if_b_round"] = band_if_b_round
        st.session_state["cs2_live_band_lo"] = band_state_lo
        st.session_state["cs2_live_band_hi"] = band_state_hi
    # Round-endpoint counterfactual trace diagnostics (no math changes)
    _ra = int(rounds_a)
    _rb = int(rounds_b)
    st.session_state["cs2_rail_base_rounds_a"] = _ra
    st.session_state["cs2_rail_base_rounds_b"] = _rb
    st.session_state["cs2_rail_base_maps_a_won"] = int(maps_a_won) if maps_a_won is not None else None
    st.session_state["cs2_rail_base_maps_b_won"] = int(maps_b_won) if maps_b_won is not None else None
    st.session_state["cs2_rail_base_game_number"] = game_number_for_key
    st.session_state["cs2_rail_base_series_fmt"] = f"BO{int(n_maps)}" if n_maps is not None else "BO3"
    st.session_state["cs2_rail_base_side_a"] = str(a_side) if a_side is not None else None
    st.session_state["cs2_rail_base_side_b"] = "CT" if str(a_side or "").strip().upper() == "T" else ("T" if str(a_side or "").strip().upper() == "CT" else None)
    st.session_state["cs2_rail_cf_a_rounds_a"] = _ra + 1
    st.session_state["cs2_rail_cf_a_rounds_b"] = _rb
    st.session_state["cs2_rail_cf_b_rounds_a"] = _ra
    st.session_state["cs2_rail_cf_b_rounds_b"] = _rb + 1
    st.session_state["cs2_rail_cf_a_maps_a_won"] = int(maps_a_won) if maps_a_won is not None else None
    st.session_state["cs2_rail_cf_a_maps_b_won"] = int(maps_b_won) if maps_b_won is not None else None
    st.session_state["cs2_rail_cf_b_maps_a_won"] = int(maps_a_won) if maps_a_won is not None else None
    st.session_state["cs2_rail_cf_b_maps_b_won"] = int(maps_b_won) if maps_b_won is not None else None
    st.session_state["cs2_rail_endpoint_if_a_round"] = float(band_if_a_round) if band_if_a_round is not None else None
    st.session_state["cs2_rail_endpoint_if_b_round"] = float(band_if_b_round) if band_if_b_round is not None else None
    _span = None
    if band_if_a_round is not None and band_if_b_round is not None:
        _span = float(abs(float(band_if_a_round) - float(band_if_b_round)))
    st.session_state["cs2_rail_endpoint_span_abs"] = _span
    st.session_state["cs2_rail_endpoint_degenerate"] = bool(_span is not None and _span < 1e-4)
    st.session_state["cs2_rail_map_winprob_if_a_round"] = None
    st.session_state["cs2_rail_map_winprob_if_b_round"] = None
    st.session_state["cs2_rail_series_prob_if_a_round"] = float(band_if_a_round) if band_if_a_round is not None else None
    st.session_state["cs2_rail_series_prob_if_b_round"] = float(band_if_b_round) if band_if_b_round is not None else None
    # Dead-path envelope diagnostics (from v2_rails / cs2_rail_debug)
    _ed = st.session_state.get("cs2_rail_debug") or {}
    st.session_state["cs2_rail_endpoint_env_valid"] = bool(_ed.get("rail_endpoint_env_valid", False))
    st.session_state["cs2_rail_endpoint_a_base_dbg"] = float(_ed["rail_endpoint_a_base_dbg"]) if _ed.get("rail_endpoint_a_base_dbg") is not None else None
    st.session_state["cs2_rail_endpoint_b_base_dbg"] = float(_ed["rail_endpoint_b_base_dbg"]) if _ed.get("rail_endpoint_b_base_dbg") is not None else None
    st.session_state["cs2_rail_endpoint_env_base_span_abs"] = float(_ed["rail_endpoint_env_base_span_abs"]) if _ed.get("rail_endpoint_env_base_span_abs") is not None else None
    st.session_state["cs2_rail_endpoint_env_fraction_k"] = float(_ed["rail_endpoint_env_fraction_k"]) if _ed.get("rail_endpoint_env_fraction_k") is not None else None
    st.session_state["cs2_rail_endpoint_env_round_leverage"] = float(_ed["rail_endpoint_env_round_leverage"]) if _ed.get("rail_endpoint_env_round_leverage") is not None else None
    st.session_state["cs2_rail_endpoint_env_series_leverage"] = float(_ed["rail_endpoint_env_series_leverage"]) if _ed.get("rail_endpoint_env_series_leverage") is not None else None
    st.session_state["cs2_rail_endpoint_env_half_width"] = float(_ed["rail_endpoint_env_half_width"]) if _ed.get("rail_endpoint_env_half_width") is not None else None
    st.session_state["cs2_rail_endpoint_a_env_low"] = float(_ed["rail_endpoint_a_env_low"]) if _ed.get("rail_endpoint_a_env_low") is not None else None
    st.session_state["cs2_rail_endpoint_a_env_high"] = float(_ed["rail_endpoint_a_env_high"]) if _ed.get("rail_endpoint_a_env_high") is not None else None
    st.session_state["cs2_rail_endpoint_b_env_low"] = float(_ed["rail_endpoint_b_env_low"]) if _ed.get("rail_endpoint_b_env_low") is not None else None
    st.session_state["cs2_rail_endpoint_b_env_high"] = float(_ed["rail_endpoint_b_env_high"]) if _ed.get("rail_endpoint_b_env_high") is not None else None
    st.session_state["cs2_rail_endpoint_a_env_center"] = float(_ed["rail_endpoint_a_env_center"]) if _ed.get("rail_endpoint_a_env_center") is not None else None
    st.session_state["cs2_rail_endpoint_b_env_center"] = float(_ed["rail_endpoint_b_env_center"]) if _ed.get("rail_endpoint_b_env_center") is not None else None
    st.session_state["cs2_rail_endpoint_env_map_index"] = int(_ed["rail_endpoint_env_map_index"]) if _ed.get("rail_endpoint_env_map_index") is not None else None
    st.session_state["cs2_rail_endpoint_env_best_of"] = int(_ed["rail_endpoint_env_best_of"]) if _ed.get("rail_endpoint_env_best_of") is not None else None
    st.session_state["cs2_rail_endpoint_env_win_target"] = int(_ed["rail_endpoint_env_win_target"]) if _ed.get("rail_endpoint_env_win_target") is not None else None
    # Canonical debug driver bundle (shared by endpoint/phat debug helpers; do not leave stale values)
    _drv_na = st.session_state.get("cs2_live_team_a_alive_count")
    _drv_nb = st.session_state.get("cs2_live_team_b_alive_count")
    _drv_hp_a = st.session_state.get("cs2_live_team_a_hp_alive_total")
    _drv_hp_b = st.session_state.get("cs2_live_team_b_hp_alive_total")
    _drv_load_a = st.session_state.get("cs2_live_team_a_alive_loadout_est_total") if st.session_state.get("cs2_live_team_a_alive_loadout_est_total") is not None else st.session_state.get("cs2_live_team_a_alive_loadout_total")
    _drv_load_b = st.session_state.get("cs2_live_team_b_alive_loadout_est_total") if st.session_state.get("cs2_live_team_b_alive_loadout_est_total") is not None else st.session_state.get("cs2_live_team_b_alive_loadout_total")
    _drv_cash_a = st.session_state.get("cs2_live_team_a_alive_cash_total")
    _drv_cash_b = st.session_state.get("cs2_live_team_b_alive_cash_total")
    _drv_bomb = st.session_state.get("cs2_live_bomb_planted")
    _drv_phase = st.session_state.get("cs2_live_round_phase")
    _drv_clock = st.session_state.get("cs2_live_round_time_remaining_s")
    _drv_side = st.session_state.get("cs2_live_team_a_side") if st.session_state.get("cs2_live_team_a_side") is not None else st.session_state.get("cs2_round_start_team_a_side")
    st.session_state["cs2_drv_alive_count_a"] = _drv_na
    st.session_state["cs2_drv_alive_count_b"] = _drv_nb
    st.session_state["cs2_drv_hp_alive_total_a"] = _drv_hp_a
    st.session_state["cs2_drv_hp_alive_total_b"] = _drv_hp_b
    st.session_state["cs2_drv_alive_loadout_est_total_a"] = _drv_load_a
    st.session_state["cs2_drv_alive_loadout_est_total_b"] = _drv_load_b
    st.session_state["cs2_drv_alive_cash_total_a"] = _drv_cash_a
    st.session_state["cs2_drv_alive_cash_total_b"] = _drv_cash_b
    st.session_state["cs2_drv_bomb_planted"] = _drv_bomb
    st.session_state["cs2_drv_round_phase"] = _drv_phase
    st.session_state["cs2_drv_round_time_remaining_s"] = _drv_clock
    st.session_state["cs2_drv_team_a_side"] = _drv_side
    st.session_state["cs2_drv_source_contract"] = "live_normalized_bundle"
    # valid_microstate: alive counts + hp totals + alive loadout est totals all non-None
    st.session_state["cs2_drv_valid_microstate"] = (
        _drv_na is not None and _drv_nb is not None
        and _drv_hp_a is not None and _drv_hp_b is not None
        and _drv_load_a is not None and _drv_load_b is not None
    )
    # valid_roundstate: bomb/phase/side present (clock may be None and still valid)
    st.session_state["cs2_drv_valid_roundstate"] = (
        _drv_bomb is not None and _drv_phase is not None and _drv_side is not None
    )
    # Dead-path branch econ consequence (reads from cs2_drv_* only)
    _econ_na = st.session_state.get("cs2_drv_alive_count_a")
    _econ_nb = st.session_state.get("cs2_drv_alive_count_b")
    _econ_load_a = st.session_state.get("cs2_drv_alive_loadout_est_total_a")
    _econ_load_b = st.session_state.get("cs2_drv_alive_loadout_est_total_b")
    _econ_cash_a = st.session_state.get("cs2_drv_alive_cash_total_a")
    _econ_cash_b = st.session_state.get("cs2_drv_alive_cash_total_b")
    _econ = _compute_cs2_branch_econ_consequence_debug(
        float(_econ_na) if _econ_na is not None else 0.0,
        float(_econ_nb) if _econ_nb is not None else 0.0,
        float(_econ_load_a) if _econ_load_a is not None else 0.0,
        float(_econ_load_b) if _econ_load_b is not None else 0.0,
        float(_econ_cash_a) if _econ_cash_a is not None else 0.0,
        float(_econ_cash_b) if _econ_cash_b is not None else 0.0,
    )
    if _econ.get("econ_valid"):
        st.session_state["cs2_endpoint_econ_valid"] = True
        st.session_state["cs2_endpoint_econ_d_alive_survival"] = float(_econ["endpoint_econ_d_alive_survival"]) if _econ.get("endpoint_econ_d_alive_survival") is not None else None
        st.session_state["cs2_endpoint_econ_d_alive_loadout"] = float(_econ["endpoint_econ_d_alive_loadout"]) if _econ.get("endpoint_econ_d_alive_loadout") is not None else None
        st.session_state["cs2_endpoint_econ_d_alive_cash"] = float(_econ["endpoint_econ_d_alive_cash"]) if _econ.get("endpoint_econ_d_alive_cash") is not None else None
        st.session_state["cs2_endpoint_econ_score_a"] = float(_econ["endpoint_econ_score_a"]) if _econ.get("endpoint_econ_score_a") is not None else None
        st.session_state["cs2_endpoint_econ_score_b"] = float(_econ["endpoint_econ_score_b"]) if _econ.get("endpoint_econ_score_b") is not None else None
    else:
        st.session_state["cs2_endpoint_econ_valid"] = False
        st.session_state["cs2_endpoint_econ_d_alive_survival"] = None
        st.session_state["cs2_endpoint_econ_d_alive_loadout"] = None
        st.session_state["cs2_endpoint_econ_d_alive_cash"] = None
        st.session_state["cs2_endpoint_econ_score_a"] = None
        st.session_state["cs2_endpoint_econ_score_b"] = None
    # Dead-path PHat intraround debug (reads from cs2_drv_* only)
    _pi = _compute_cs2_phat_intra_score_debug(
        team_a_alive_count=st.session_state.get("cs2_drv_alive_count_a"),
        team_b_alive_count=st.session_state.get("cs2_drv_alive_count_b"),
        team_a_hp_alive_total=st.session_state.get("cs2_drv_hp_alive_total_a"),
        team_b_hp_alive_total=st.session_state.get("cs2_drv_hp_alive_total_b"),
        team_a_alive_loadout_est_total=st.session_state.get("cs2_drv_alive_loadout_est_total_a"),
        team_b_alive_loadout_est_total=st.session_state.get("cs2_drv_alive_loadout_est_total_b"),
        bomb_planted=st.session_state.get("cs2_drv_bomb_planted"),
        round_phase=st.session_state.get("cs2_drv_round_phase"),
        round_time_remaining_s=st.session_state.get("cs2_drv_round_time_remaining_s"),
        team_a_side=st.session_state.get("cs2_drv_team_a_side"),
        q_temp=1.0,
        reliability_mult=1.0,
    )
    for _k, _v in _pi.items():
        st.session_state["cs2_" + _k] = _v
    # Dead-path branch endpoint position diagnostics (no behavior change)
    _a_lo = st.session_state.get("cs2_rail_endpoint_a_env_low")
    _a_hi = st.session_state.get("cs2_rail_endpoint_a_env_high")
    _b_lo = st.session_state.get("cs2_rail_endpoint_b_env_low")
    _b_hi = st.session_state.get("cs2_rail_endpoint_b_env_high")
    _env_ok = bool(st.session_state.get("cs2_rail_endpoint_env_valid", False))
    if _env_ok and _a_lo is not None and _a_hi is not None and _b_lo is not None and _b_hi is not None:
        _pos = _compute_cs2_branch_endpoint_position_debug(
            a_env_low=float(_a_lo),
            a_env_high=float(_a_hi),
            b_env_low=float(_b_lo),
            b_env_high=float(_b_hi),
            team_a_alive_count=float(st.session_state.get("cs2_live_team_a_alive_count") or 0),
            team_b_alive_count=float(st.session_state.get("cs2_live_team_b_alive_count") or 0),
            team_a_alive_loadout_total=float(st.session_state.get("cs2_live_team_a_alive_loadout_total") or 0),
            team_b_alive_loadout_total=float(st.session_state.get("cs2_live_team_b_alive_loadout_total") or 0),
            team_a_armor_alive_total=float(st.session_state.get("cs2_live_team_a_armor_alive_total") or 0),
            team_b_armor_alive_total=float(st.session_state.get("cs2_live_team_b_armor_alive_total") or 0),
            branch_temp=0.30,
        )
        st.session_state["cs2_endpoint_pos_valid"] = bool(_pos.get("pos_valid", False))
        st.session_state["cs2_endpoint_pos_branch_temp_used"] = float(_pos["branch_temp_used"]) if _pos.get("branch_temp_used") is not None else None
        st.session_state["cs2_endpoint_pos_d_alive"] = float(_pos["endpoint_pos_d_alive"]) if _pos.get("endpoint_pos_d_alive") is not None else None
        st.session_state["cs2_endpoint_pos_d_loadout"] = float(_pos["endpoint_pos_d_loadout"]) if _pos.get("endpoint_pos_d_loadout") is not None else None
        st.session_state["cs2_endpoint_pos_d_armor"] = float(_pos["endpoint_pos_d_armor"]) if _pos.get("endpoint_pos_d_armor") is not None else None
        st.session_state["cs2_endpoint_pos_score_a"] = float(_pos["endpoint_pos_score_a"]) if _pos.get("endpoint_pos_score_a") is not None else None
        st.session_state["cs2_endpoint_pos_score_b"] = float(_pos["endpoint_pos_score_b"]) if _pos.get("endpoint_pos_score_b") is not None else None
        st.session_state["cs2_endpoint_pos_a_quality_pos"] = float(_pos["endpoint_pos_a_quality_pos"]) if _pos.get("endpoint_pos_a_quality_pos") is not None else None
        st.session_state["cs2_endpoint_pos_b_quality_pos"] = float(_pos["endpoint_pos_b_quality_pos"]) if _pos.get("endpoint_pos_b_quality_pos") is not None else None
        st.session_state["cs2_endpoint_pos_a_active_dbg"] = float(_pos["endpoint_pos_a_active_dbg"]) if _pos.get("endpoint_pos_a_active_dbg") is not None else None
        st.session_state["cs2_endpoint_pos_b_active_dbg"] = float(_pos["endpoint_pos_b_active_dbg"]) if _pos.get("endpoint_pos_b_active_dbg") is not None else None
        st.session_state["cs2_endpoint_pos_a_active_within_env"] = bool(_pos.get("endpoint_pos_a_active_within_env", False))
        st.session_state["cs2_endpoint_pos_b_active_within_env"] = bool(_pos.get("endpoint_pos_b_active_within_env", False))
        # Dynamic endpoints (debug): use econ consequence only; never armor. If econ invalid, mark position invalid.
        _branch_src = st.session_state.get("cs2_branch_endpoint_source_mode", "Frozen endpoints (current)")
        _econ_ok = bool(st.session_state.get("cs2_endpoint_econ_valid", False))
        _econ_score_a = st.session_state.get("cs2_endpoint_econ_score_a")
        if _branch_src == "Dynamic endpoints (debug)":
            if _econ_ok and _econ_score_a is not None and np.isfinite(float(_econ_score_a)):
                _d_alive = _pos.get("endpoint_pos_d_alive")
                _d_loadout = _pos.get("endpoint_pos_d_loadout")
                _temp = float(_pos.get("branch_temp_used") or 0.30)
                if _d_alive is not None and _d_loadout is not None and np.isfinite(_d_alive) and np.isfinite(_d_loadout):
                    _s_a = 0.50 * float(_d_alive) + 0.30 * float(_d_loadout) + 0.20 * float(_econ_score_a)
                    _s_a = float(np.clip(_s_a, -5.0, 5.0))
                    _s_b = -_s_a
                    _a_q = _sigmoid_stable(_s_a, _temp)
                    _b_q = _sigmoid_stable(_s_b, _temp)
                    _a_lo_f = float(_a_lo)
                    _a_hi_f = float(_a_hi)
                    _b_lo_f = float(_b_lo)
                    _b_hi_f = float(_b_hi)
                    st.session_state["cs2_endpoint_pos_score_a"] = _s_a
                    st.session_state["cs2_endpoint_pos_score_b"] = _s_b
                    st.session_state["cs2_endpoint_pos_a_quality_pos"] = _a_q
                    st.session_state["cs2_endpoint_pos_b_quality_pos"] = _b_q
                    st.session_state["cs2_endpoint_pos_a_active_dbg"] = _a_lo_f + (_a_hi_f - _a_lo_f) * _a_q
                    st.session_state["cs2_endpoint_pos_b_active_dbg"] = _b_lo_f + (_b_hi_f - _b_lo_f) * _b_q
                    st.session_state["cs2_endpoint_pos_a_active_within_env"] = bool((_a_lo_f - 1e-9) <= st.session_state["cs2_endpoint_pos_a_active_dbg"] <= (_a_hi_f + 1e-9))
                    st.session_state["cs2_endpoint_pos_b_active_within_env"] = bool((_b_lo_f - 1e-9) <= st.session_state["cs2_endpoint_pos_b_active_dbg"] <= (_b_hi_f + 1e-9))
                st.session_state["cs2_endpoint_pos_third_driver_source"] = "econ_debug"
            else:
                # Econ invalid: do not fallback to armor; mark dynamic endpoint position debug invalid
                st.session_state["cs2_endpoint_pos_valid"] = False
                st.session_state["cs2_endpoint_pos_score_a"] = None
                st.session_state["cs2_endpoint_pos_score_b"] = None
                st.session_state["cs2_endpoint_pos_a_quality_pos"] = None
                st.session_state["cs2_endpoint_pos_b_quality_pos"] = None
                st.session_state["cs2_endpoint_pos_a_active_dbg"] = None
                st.session_state["cs2_endpoint_pos_b_active_dbg"] = None
                st.session_state["cs2_endpoint_pos_a_active_within_env"] = False
                st.session_state["cs2_endpoint_pos_b_active_within_env"] = False
                st.session_state["cs2_endpoint_pos_third_driver_source"] = "econ_missing_invalid"
        else:
            st.session_state["cs2_endpoint_pos_third_driver_source"] = "legacy_non_dynamic"
    else:
        st.session_state["cs2_endpoint_pos_valid"] = False
        st.session_state["cs2_endpoint_pos_branch_temp_used"] = None
        st.session_state["cs2_endpoint_pos_d_alive"] = None
        st.session_state["cs2_endpoint_pos_d_loadout"] = None
        st.session_state["cs2_endpoint_pos_d_armor"] = None
        st.session_state["cs2_endpoint_pos_score_a"] = None
        st.session_state["cs2_endpoint_pos_score_b"] = None
        st.session_state["cs2_endpoint_pos_a_quality_pos"] = None
        st.session_state["cs2_endpoint_pos_b_quality_pos"] = None
        st.session_state["cs2_endpoint_pos_a_active_dbg"] = None
        st.session_state["cs2_endpoint_pos_b_active_dbg"] = None
        st.session_state["cs2_endpoint_pos_a_active_within_env"] = False
        st.session_state["cs2_endpoint_pos_b_active_within_env"] = False
        st.session_state["cs2_endpoint_pos_third_driver_source"] = "legacy_non_dynamic"
    # BO3 STATE BANDS

    # Freeze rails ONLY after rail computation, and only on transition ticks (rails were computed from newly latched context)
    if transition_this_tick and band_state_lo is not None and band_state_hi is not None:
        st.session_state["cs2_midround_band_state_lo_frozen"] = float(band_state_lo)
        st.session_state["cs2_midround_band_state_hi_frozen"] = float(band_state_hi)
        st.session_state["cs2_midround_band_if_a_round_frozen"] = float(band_if_a_round) if band_if_a_round is not None else None
        st.session_state["cs2_midround_band_if_b_round_frozen"] = float(band_if_b_round) if band_if_b_round is not None else None
        st.session_state["cs2_midround_p_base_frozen"] = float(p_hat)
        if isinstance(st.session_state.get("cs2_rail_debug"), dict):
            st.session_state["cs2_rail_debug"]["round_transition_p_base_frozen_source"] = "post_snap_p_hat"
            st.session_state["cs2_rail_debug"]["round_transition_p_base_frozen_value"] = float(p_hat)
        st.session_state["cs2_midround_round_start_ts"] = time.time()
        st.session_state["cs2_round_freeze_game_number"] = st.session_state.get("cs2_live_game_number")
        st.session_state["cs2_round_freeze_rounds_a"] = int(rounds_a)
        st.session_state["cs2_round_freeze_rounds_b"] = int(rounds_b)
        st.session_state["cs2_round_freeze_phase_at_capture"] = st.session_state.get("cs2_live_round_phase")
        st.session_state["cs2_round_freeze_band_if_a"] = float(band_if_a_round) if band_if_a_round is not None else None
        st.session_state["cs2_round_freeze_band_if_b"] = float(band_if_b_round) if band_if_b_round is not None else None
        st.session_state["cs2_round_freeze_tick_ts"] = time.time()
        st.session_state["cs2_midround_latched_team_a_alive_count"] = st.session_state.get("cs2_live_team_a_alive_count")
        st.session_state["cs2_midround_latched_team_b_alive_count"] = st.session_state.get("cs2_live_team_b_alive_count")
    if not transition_this_tick:
        st.session_state["cs2_midround_prev_rounds_a"] = int(rounds_a)
        st.session_state["cs2_midround_prev_rounds_b"] = int(rounds_b)
        st.session_state["round_transition_detected"] = False
        st.session_state["round_transition_prev_key"] = None
        st.session_state["round_transition_new_key"] = None
        st.session_state["round_transition_winner"] = None
        st.session_state["round_transition_resolved_p"] = None
        st.session_state["round_transition_resolution_source"] = "none"
        st.session_state["cs2_round_snap_used"] = False
        st.session_state["cs2_round_snap_target_side"] = None
        st.session_state["cs2_round_snap_value"] = None
        st.session_state["cs2_round_snap_prev_freeze_key"] = None
        st.session_state["cs2_round_transition_tick_detected"] = False
        st.session_state["cs2_round_transition_used_prev_frozen_for_snap"] = False
        st.session_state["cs2_round_transition_latched_new_round_context"] = False
        st.session_state["cs2_round_transition_rail_computed_after_latch"] = False
        st.session_state["cs2_round_transition_game_number"] = None
        st.session_state["cs2_round_transition_prev_round_key"] = None
        st.session_state["cs2_round_transition_new_round_key"] = None
    st.session_state["round_transition_detected"] = round_transition_detected
    st.session_state["round_transition_prev_key"] = round_transition_prev_key
    st.session_state["round_transition_new_key"] = round_transition_new_key
    st.session_state["round_transition_winner"] = round_transition_winner
    st.session_state["round_transition_resolved_p"] = round_transition_resolved_p
    st.session_state["round_transition_resolution_source"] = round_transition_resolution_source
    if transition_this_tick:
        st.session_state["cs2_round_transition_tick_detected"] = True
        st.session_state["cs2_round_transition_used_prev_frozen_for_snap"] = used_prev_frozen_for_snap
        st.session_state["cs2_round_transition_latched_new_round_context"] = latched_new_round_context
        st.session_state["cs2_round_transition_rail_computed_after_latch"] = True
        st.session_state["cs2_round_transition_game_number"] = game_number_for_key
        st.session_state["cs2_round_transition_prev_round_key"] = round_transition_prev_key
        st.session_state["cs2_round_transition_new_round_key"] = round_transition_new_key

    st.session_state["cs2_round_freeze_active"] = (st.session_state.get("cs2_midround_round_key") is not None)
    st.session_state["cs2_round_freeze_key"] = st.session_state.get("cs2_midround_round_key")

    p_base_frozen = st.session_state.get("cs2_midround_p_base_frozen")
    frozen_lo = st.session_state.get("cs2_midround_band_state_lo_frozen")
    frozen_hi = st.session_state.get("cs2_midround_band_state_hi_frozen")
    if p_base_frozen is None:
        p_base_frozen = float(p_hat)
    if frozen_lo is not None and frozen_hi is not None:
        band_state_lo = float(frozen_lo)
        band_state_hi = float(frozen_hi)

    # Round-start context offset: small latched baseline shift (econ/loadout/armor/side/map)
    round_context_offset = 0.0
    round_ctx_debug = {}
    p_round_context = float(p_base_frozen) if p_base_frozen is not None else float(p_hat)
    if p_base_frozen is not None and frozen_lo is not None and frozen_hi is not None:
        round_context_offset, round_ctx_debug = _compute_cs2_round_context_offset(
            float(p_base_frozen), frozen_lo, frozen_hi,
            st.session_state.get("round_start_map_name"),
            st.session_state.get("round_start_team_a_side"),
            st.session_state.get("round_start_team_b_side"),
            st.session_state.get("round_start_team_a_econ_total"),
            st.session_state.get("round_start_team_b_econ_total"),
            st.session_state.get("round_start_team_a_loadout_total"),
            st.session_state.get("round_start_team_b_loadout_total"),
            st.session_state.get("round_start_team_a_armor_total"),
            st.session_state.get("round_start_team_b_armor_total"),
        )
        p_round_context = float(np.clip(float(p_base_frozen) + round_context_offset, 0.0, 1.0))
    st.session_state["cs2_round_context_offset"] = round_context_offset
    st.session_state["cs2_round_context_debug"] = round_ctx_debug
    st.session_state["cs2_p_round_context"] = p_round_context

    midround_result = None
    p_hat_final_source = "base_frozen"
    p_hat_pre_bounds_source = float(p_hat)
    # Series-endpoint mixture: semantic endpoints + q_intra from intraround microstate (structural p_hat in [0,1] by construction)
    p_series_if_a_wins_next_round = None
    p_series_if_b_wins_next_round = None
    q_intra_round_win_a = None
    intra_score_raw = None
    intra_score_alive = None
    intra_score_hp = None
    intra_score_bomb = None
    intra_score_armor = None
    intra_score_loadout = None
    intra_score_post_reliability = None
    intra_score_time_scale = None
    p_hat_structural_mixture = None
    p_hat_between_endpoints = None
    endpoint_span_abs = None
    branch_endpoint_source_used = "n/a"
    branch_endpoint_source_reason = "unknown"
    branch_endpoint_selected_a = None
    branch_endpoint_selected_b = None
    branch_endpoint_selected_span_abs = None

    # Use snap only on the single tick where we detected round end (use_snap_this_tick); then resume mid-round
    if use_snap_this_tick:
        p_hat_final_source = "round_transition_resolved"
        st.session_state.pop("cs2_midround_transition_tick", None)
        st.session_state["cs2_round_snap_applied_this_tick"] = True
    elif cs2_midround_mode != "OFF" and frozen_lo is not None and frozen_hi is not None and frozen_hi > frozen_lo:
        features = _compute_cs2_midround_features(
            team_a_alive_count=st.session_state.get("cs2_live_team_a_alive_count"),
            team_b_alive_count=st.session_state.get("cs2_live_team_b_alive_count"),
            team_a_hp_alive_total=st.session_state.get("cs2_live_team_a_hp_alive_total"),
            team_b_hp_alive_total=st.session_state.get("cs2_live_team_b_hp_alive_total"),
            bomb_planted=st.session_state.get("cs2_live_bomb_planted"),
            round_time_remaining_s=st.session_state.get("cs2_live_round_time_remaining_s"),
            round_phase=st.session_state.get("cs2_live_round_phase"),
            a_side=a_side,
            team_a_armor_alive_total=st.session_state.get("cs2_live_team_a_armor_alive_total"),
            team_b_armor_alive_total=st.session_state.get("cs2_live_team_b_armor_alive_total"),
            team_a_alive_loadout_total=st.session_state.get("cs2_live_team_a_alive_loadout_total"),
            team_b_alive_loadout_total=st.session_state.get("cs2_live_team_b_alive_loadout_total"),
            live_source=st.session_state.get("cs2_live_source"),
            grid_used_reduced_features=st.session_state.get("cs2_grid_used_reduced_features"),
            grid_completeness_score=st.session_state.get("cs2_grid_completeness_score"),
            grid_staleness_seconds=st.session_state.get("cs2_grid_staleness_seconds"),
            grid_has_players=st.session_state.get("cs2_grid_has_players"),
            grid_has_clock=st.session_state.get("cs2_grid_has_clock"),
        )
        if not features.get("feature_ok"):
            p_hat = float(p_round_context)
            p_hat_pre_bounds_source = float(p_hat)
        elif cs2_midround_mode == "V2 (mixture)":
            frozen_a = st.session_state.get("cs2_midround_band_if_a_round_frozen")
            frozen_b = st.session_state.get("cs2_midround_band_if_b_round_frozen")
            if frozen_a is not None and frozen_b is not None:
                midround_result = _apply_cs2_midround_adjustment_v2_mixture(
                    float(frozen_a), float(frozen_b), features, settings={"a_side": a_side},
                )
                st.session_state["cs2_midround_last_result"] = midround_result
                st.session_state["cs2_midround_last_features"] = features
                p_hat = float(midround_result["p_mid_clamped"])
                p_hat_final_source = "midround_v2_mixture"
                p_hat_pre_bounds_source = float(midround_result.get("p_mid_raw", p_hat))
            else:
                p_hat = float(p_round_context)
                p_hat_pre_bounds_source = float(p_hat)
        elif cs2_midround_mode == "V1 (additive)":
            midround_result = _apply_cs2_midround_adjustment_v1(
                float(p_round_context), float(frozen_lo), float(frozen_hi), features, settings={"a_side": a_side},
            )
            st.session_state["cs2_midround_last_result"] = midround_result
            st.session_state["cs2_midround_last_features"] = features
            p_hat = float(midround_result["p_mid_clamped"])
            p_hat_final_source = "midround_v1"
            p_hat_pre_bounds_source = float(midround_result.get("p_mid_raw", p_hat))
        else:
            p_hat = float(p_round_context)
            p_hat_pre_bounds_source = float(p_hat)
    else:
        p_hat = float(p_round_context)
        p_hat_pre_bounds_source = float(p_hat)
    st.session_state["cs2_midround_p_hat_final_source"] = p_hat_final_source
    if midround_result is not None:
        st.session_state["cs2_midround_last_result"] = midround_result
    # BO3 MIDROUND V1

    # Raw p_hat (pre-bounds): structural mixture when used, else mid-round raw or p_hat
    p_hat_pre_bounds = float(p_hat_pre_bounds_source)

    # State bounds (map resolution): compute here so bounds-mode can apply map clamp
    # Use session state for series score so bounds always reflect current series (0-1, 1-0, etc.) not stale widget state
    _maps_for_bounds_a = int(st.session_state.get("cs2_live_maps_a_won", 0))
    _maps_for_bounds_b = int(st.session_state.get("cs2_live_maps_b_won", 0))
    state_bound_upper, state_bound_lower = None, None
    if contract_scope == "Series winner" and int(n_maps) > 1:
        target = _bestof_target(int(n_maps))
        if _maps_for_bounds_a < target and _maps_for_bounds_b < target:
            try:
                state_bound_upper = series_win_prob_live(int(n_maps), _maps_for_bounds_a + 1, _maps_for_bounds_b, float(p0_map), float(p0_map))
                state_bound_lower = series_win_prob_live(int(n_maps), _maps_for_bounds_a, _maps_for_bounds_b + 1, float(p0_map), float(p0_map))
            except Exception:
                pass

    # Round anchor (base before mid-round adjustment) — used for soft edge damping direction
    p_hat_round_anchor_before_mid = float(p_round_context)

    # Bounds mode: conditional round clamp and map clamp (all calculations unchanged; only clamp application is conditional)
    bounds_mode = str(st.session_state.get("cs2_bounds_mode", "Normal (bounded)"))
    apply_round_clamp = bounds_mode in ("Normal (bounded)", "Round-only bounds")
    apply_map_clamp = bounds_mode in ("Normal (bounded)", "Map-only bounds")
    # Round stage: soft edge damping (asymptotic approach to rails) when band + anchor available; else hard clip fallback
    p_hat_round_soft_damped = None
    p_hat_round_soft_edge_factor = None
    p_hat_round_soft_room_frac = None
    p_hat_round_soft_delta_raw = None
    p_hat_round_soft_delta_applied = None
    p_hat_round_soft_direction = None
    p_hat_round_soft_hit = None
    if not apply_round_clamp:
        p_hat_post_round_bounds = float(p_hat_pre_bounds)
    elif frozen_lo is not None and frozen_hi is not None and frozen_hi > frozen_lo:
        soft_result = _soft_damp_into_round_band(
            p_hat_pre_bounds, p_hat_round_anchor_before_mid, float(frozen_lo), float(frozen_hi),
        )
        p_hat_post_round_bounds = float(soft_result["p_soft"])
        p_hat_round_soft_damped = p_hat_post_round_bounds
        p_hat_round_soft_edge_factor = soft_result["edge_damp_factor"]
        p_hat_round_soft_room_frac = soft_result["edge_room_frac"]
        p_hat_round_soft_delta_raw = soft_result["delta_raw"]
        p_hat_round_soft_delta_applied = soft_result["delta_soft"]
        p_hat_round_soft_direction = soft_result["edge_direction"]
        p_hat_round_soft_hit = soft_result["edge_soft_hit"]
    elif frozen_lo is not None and frozen_hi is not None:
        p_hat_post_round_bounds = float(np.clip(p_hat_pre_bounds, float(frozen_lo), float(frozen_hi)))
        p_hat_round_soft_damped = None
        p_hat_round_soft_edge_factor = None
        p_hat_round_soft_room_frac = None
        p_hat_round_soft_delta_raw = None
        p_hat_round_soft_delta_applied = None
        p_hat_round_soft_direction = None
        p_hat_round_soft_hit = None
    else:
        p_hat_post_round_bounds = float(p_hat_pre_bounds)
        p_hat_round_soft_damped = None
        p_hat_round_soft_edge_factor = None
        p_hat_round_soft_room_frac = None
        p_hat_round_soft_delta_raw = None
        p_hat_round_soft_delta_applied = None
        p_hat_round_soft_direction = None
        p_hat_round_soft_hit = None
    p_hat_round_clip_delta = p_hat_post_round_bounds - p_hat_pre_bounds
    p_hat_clipped_by_round = abs(p_hat_round_clip_delta) > 1e-9

    # Map clamp stage (state_bound_lower .. state_bound_upper)
    if apply_map_clamp and state_bound_lower is not None and state_bound_upper is not None:
        p_hat_post_map_bounds = float(np.clip(p_hat_post_round_bounds, float(state_bound_lower), float(state_bound_upper)))
    else:
        p_hat_post_map_bounds = float(p_hat_post_round_bounds)
    p_hat_map_clip_delta = p_hat_post_map_bounds - p_hat_post_round_bounds
    p_hat_clipped_by_map = abs(p_hat_map_clip_delta) > 1e-9

    p_hat = float(p_hat_post_map_bounds)  # intra-round final; terminal override may apply below

    st.session_state["cs2_p_hat_pre_bounds"] = p_hat_pre_bounds
    st.session_state["cs2_p_hat_post_round_bounds"] = p_hat_post_round_bounds
    st.session_state["cs2_p_hat_post_map_bounds"] = p_hat_post_map_bounds
    st.session_state["cs2_p_hat_clipped_by_round"] = p_hat_clipped_by_round
    st.session_state["cs2_p_hat_clipped_by_map"] = p_hat_clipped_by_map
    st.session_state["cs2_p_hat_round_clip_delta"] = p_hat_round_clip_delta
    st.session_state["cs2_p_hat_map_clip_delta"] = p_hat_map_clip_delta
    st.session_state["cs2_p_hat_round_anchor_before_mid"] = p_hat_round_anchor_before_mid
    st.session_state["cs2_p_series_if_a_wins_next_round"] = p_series_if_a_wins_next_round
    st.session_state["cs2_p_series_if_b_wins_next_round"] = p_series_if_b_wins_next_round

    # Debug plumbing: populate q_intra_round_win_a from main path or helper so debug/chart show correct source
    _q_main = midround_result.get("q") if isinstance(midround_result, dict) else None
    _q_helper = st.session_state.get("cs2_phat_intra_q_intra_round_win_a_dbg")
    if _q_main is not None:
        try:
            _qf = float(_q_main)
            if np.isfinite(_qf):
                q_intra_round_win_a = _qf
                q_intra_round_win_a_source = "main"
                q_intra_round_win_a_reason = "midround_v2_q"
            else:
                q_intra_round_win_a_source = "none"
                q_intra_round_win_a_reason = "not_computed"
        except (TypeError, ValueError):
            q_intra_round_win_a_source = "none"
            q_intra_round_win_a_reason = "not_computed"
    elif _q_helper is not None:
        try:
            _qf = float(_q_helper)
            if np.isfinite(_qf):
                q_intra_round_win_a = _qf
                q_intra_round_win_a_source = "helper_debug_fallback"
                q_intra_round_win_a_reason = "phat_intra_debug_valid"
            else:
                q_intra_round_win_a_source = "none"
                q_intra_round_win_a_reason = "intraround_helper_invalid"
        except (TypeError, ValueError):
            q_intra_round_win_a_source = "none"
            q_intra_round_win_a_reason = "intraround_helper_invalid"
    else:
        q_intra_round_win_a_source = "none"
        if not (frozen_lo is not None and frozen_hi is not None):
            q_intra_round_win_a_reason = "no_roundstate"
        elif st.session_state.get("cs2_phat_intra_valid"):
            q_intra_round_win_a_reason = "dynamic_mode_enabled_but_q_helper_missing"
        else:
            q_intra_round_win_a_reason = "intraround_helper_invalid"

    # Debug plumbing: set branch_endpoint_source_used and reason from mode + third_driver + env
    _branch_mode = st.session_state.get("cs2_branch_endpoint_source_mode", "Frozen endpoints (current)")
    _third_driver = st.session_state.get("cs2_endpoint_pos_third_driver_source", "legacy_non_dynamic")
    _env_valid = bool(st.session_state.get("cs2_rail_endpoint_env_valid", False))
    _dyn_avail_pre = (
        bool(st.session_state.get("cs2_endpoint_pos_valid", False))
        and st.session_state.get("cs2_endpoint_pos_a_active_dbg") is not None
        and st.session_state.get("cs2_endpoint_pos_b_active_dbg") is not None
        and np.isfinite(float(st.session_state.get("cs2_endpoint_pos_a_active_dbg") or 0))
        and np.isfinite(float(st.session_state.get("cs2_endpoint_pos_b_active_dbg") or 0))
    )
    if _branch_mode == "Dynamic endpoints (debug)":
        if _third_driver == "econ_debug":
            branch_endpoint_source_used = "dynamic_envelope_econ_debug"
            branch_endpoint_source_reason = "dynamic_mode_enabled_env_valid"
        elif _third_driver == "econ_missing_invalid":
            branch_endpoint_source_used = "dynamic_envelope_fallback_bounds_missing"
            branch_endpoint_source_reason = "dynamic_mode_enabled_but_econ_invalid"
        elif not _env_valid:
            branch_endpoint_source_used = "dynamic_envelope_fallback_bounds_missing"
            branch_endpoint_source_reason = "dynamic_mode_enabled_but_bounds_missing"
        else:
            branch_endpoint_source_used = "dynamic_envelope_econ_debug" if _dyn_avail_pre else "dynamic_envelope_fallback_bounds_missing"
            branch_endpoint_source_reason = "dynamic_mode_enabled_env_valid" if _dyn_avail_pre else "dynamic_mode_enabled_but_bounds_missing"
    else:
        branch_endpoint_source_used = "frozen_current_mode"
        branch_endpoint_source_reason = "frozen_fallback_dynamic_unavailable" if not _dyn_avail_pre else "frozen_mode_selected"

    st.session_state["cs2_q_intra_round_win_a"] = q_intra_round_win_a
    st.session_state["cs2_q_intra_round_win_a_source"] = q_intra_round_win_a_source
    st.session_state["cs2_q_intra_round_win_a_reason"] = q_intra_round_win_a_reason
    st.session_state["cs2_intra_score_raw"] = intra_score_raw
    st.session_state["cs2_intra_score_alive"] = intra_score_alive
    st.session_state["cs2_intra_score_hp"] = intra_score_hp
    st.session_state["cs2_intra_score_bomb"] = intra_score_bomb
    st.session_state["cs2_intra_score_armor"] = intra_score_armor
    st.session_state["cs2_intra_score_loadout"] = intra_score_loadout
    st.session_state["cs2_intra_score_post_reliability"] = intra_score_post_reliability
    st.session_state["cs2_intra_score_time_scale"] = intra_score_time_scale
    st.session_state["cs2_p_hat_structural_mixture"] = p_hat_structural_mixture
    st.session_state["cs2_p_hat_between_endpoints"] = p_hat_between_endpoints
    st.session_state["cs2_endpoint_span_abs"] = endpoint_span_abs
    # Do NOT assign to cs2_branch_endpoint_source_mode (widget-owned; assign only in selectbox key).
    st.session_state["cs2_branch_endpoint_source_used"] = branch_endpoint_source_used
    st.session_state["cs2_branch_endpoint_source_reason"] = branch_endpoint_source_reason
    st.session_state["cs2_branch_endpoint_selected_a"] = float(branch_endpoint_selected_a) if branch_endpoint_selected_a is not None else None
    st.session_state["cs2_branch_endpoint_selected_b"] = float(branch_endpoint_selected_b) if branch_endpoint_selected_b is not None else None
    st.session_state["cs2_branch_endpoint_selected_span_abs"] = float(branch_endpoint_selected_span_abs) if branch_endpoint_selected_span_abs is not None else None
    _dyn_avail = (
        bool(st.session_state.get("cs2_endpoint_pos_valid", False))
        and st.session_state.get("cs2_endpoint_pos_a_active_dbg") is not None
        and st.session_state.get("cs2_endpoint_pos_b_active_dbg") is not None
        and np.isfinite(float(st.session_state.get("cs2_endpoint_pos_a_active_dbg") or 0))
        and np.isfinite(float(st.session_state.get("cs2_endpoint_pos_b_active_dbg") or 0))
    )
    st.session_state["cs2_branch_endpoint_dynamic_available"] = _dyn_avail
    st.session_state["cs2_p_hat_round_soft_damped"] = p_hat_round_soft_damped
    st.session_state["cs2_p_hat_round_soft_edge_factor"] = p_hat_round_soft_edge_factor
    st.session_state["cs2_p_hat_round_soft_room_frac"] = p_hat_round_soft_room_frac
    st.session_state["cs2_p_hat_round_soft_delta_raw"] = p_hat_round_soft_delta_raw
    st.session_state["cs2_p_hat_round_soft_delta_applied"] = p_hat_round_soft_delta_applied
    st.session_state["cs2_p_hat_round_soft_direction"] = p_hat_round_soft_direction
    st.session_state["cs2_p_hat_round_soft_hit"] = p_hat_round_soft_hit

    # Terminal resolution priority: after round-end snap and mid-round adjustment, force p_hat to terminal when CONTRACT is terminal.
    # Contract-aware: Series winner + n_maps>1 => override only when series is over (one side reached series target), not on map end alone.
    p_hat_pre_terminal_override = float(p_hat)
    terminal_override_applied = False
    terminal_override_reason = ""
    terminal_team_a_won = None
    terminal_override_blocked_reason = None
    rounds_a_now = int(rounds_a)
    rounds_b_now = int(rounds_b)
    win_target = cs2_current_win_target(rounds_a_now, rounds_b_now)
    map_over_by_score_a = (rounds_a_now >= win_target and (rounds_a_now - rounds_b_now) >= 2)
    map_over_by_score_b = (rounds_b_now >= win_target and (rounds_b_now - rounds_a_now) >= 2)
    game_ended_flag = bool(st.session_state.get("cs2_live_game_ended", False))
    match_ended_flag = bool(st.session_state.get("cs2_live_match_ended", False))
    match_status_flag = str(st.session_state.get("cs2_live_match_status", "") or "").strip().lower() in ("ended", "finished", "completed")
    feed_terminal = game_ended_flag or match_ended_flag or match_status_flag
    _map_over_a = map_over_by_score_a or (feed_terminal and rounds_a_now > rounds_b_now)
    _map_over_b = map_over_by_score_b or (feed_terminal and rounds_b_now > rounds_a_now)
    _map_over_detected = _map_over_a or _map_over_b

    _scope = str(contract_scope) if contract_scope else "Map winner (this map)"
    _n_maps = int(n_maps) if n_maps is not None else 1
    _series_target = None
    _series_over = False
    if _scope == "Series winner" and _n_maps > 1:
        _series_target = _bestof_target(_n_maps)
        _maps_a = int(maps_a_won) if maps_a_won is not None else 0
        _maps_b = int(maps_b_won) if maps_b_won is not None else 0
        _series_over = (_maps_a >= _series_target) or (_maps_b >= _series_target)
        if _map_over_detected and not _series_over:
            terminal_override_blocked_reason = "series_not_terminal_after_map_end"
        elif _series_over:
            if _maps_a >= _series_target:
                p_hat = 0.99
                terminal_override_applied = True
                terminal_override_reason = "series_over_a"
                terminal_team_a_won = True
            else:
                p_hat = 0.01
                terminal_override_applied = True
                terminal_override_reason = "series_over_b"
                terminal_team_a_won = False
    else:
        # Map winner (or single-map): map over => terminal override
        if _map_over_a:
            p_hat = 0.99
            terminal_override_applied = True
            terminal_override_reason = "map_over_a"
            terminal_team_a_won = True
        elif _map_over_b:
            p_hat = 0.01
            terminal_override_applied = True
            terminal_override_reason = "map_over_b"
            terminal_team_a_won = False

    st.session_state["cs2_terminal_contract_scope"] = _scope
    st.session_state["cs2_terminal_series_target"] = _series_target
    st.session_state["cs2_terminal_series_over"] = _series_over
    st.session_state["cs2_terminal_map_over_detected"] = _map_over_detected
    st.session_state["cs2_terminal_override_blocked_reason"] = terminal_override_blocked_reason
    st.session_state["cs2_terminal_override_applied"] = terminal_override_applied
    st.session_state["cs2_terminal_override_reason"] = terminal_override_reason
    st.session_state["cs2_terminal_team_a_won"] = terminal_team_a_won
    st.session_state["cs2_p_hat_pre_terminal_override"] = p_hat_pre_terminal_override
    st.session_state["cs2_p_hat_post_terminal_override"] = float(p_hat)
    st.session_state["cs2_bo3_game_ended_flag"] = game_ended_flag
    st.session_state["cs2_bo3_match_ended_flag"] = match_ended_flag
    st.session_state["cs2_bo3_match_status_flag"] = match_status_flag
    st.session_state["cs2_feed_terminal_used"] = feed_terminal
    # Final p_hat (used for chart + downstream display; may be terminal override)

    # (State bounds already computed above for bounds-mode; used for display here.)

    # Instrument panel (trader summary): persist for next-run display at top
    st.session_state["cs2_instrument_team_a"] = str(team_a) if team_a else ""
    st.session_state["cs2_instrument_team_b"] = str(team_b) if team_b else ""
    st.session_state["cs2_instrument_rounds_a"] = int(rounds_a)
    st.session_state["cs2_instrument_rounds_b"] = int(rounds_b)
    st.session_state["cs2_instrument_map_name"] = str(map_sel) if map_sel else "—"
    st.session_state["cs2_instrument_market_mid"] = float(market_mid)
    st.session_state["cs2_instrument_p_hat"] = float(p_hat)
    st.session_state["cs2_instrument_band_lo"] = float(lo)
    st.session_state["cs2_instrument_band_hi"] = float(hi)
    st.session_state["cs2_instrument_band_state_lo"] = float(band_state_lo) if band_state_lo is not None else None
    st.session_state["cs2_instrument_band_state_hi"] = float(band_state_hi) if band_state_hi is not None else None
    st.session_state["cs2_instrument_state_bound_lower"] = float(state_bound_lower) if state_bound_lower is not None else None
    st.session_state["cs2_instrument_state_bound_upper"] = float(state_bound_upper) if state_bound_upper is not None else None
    st.session_state["cs2_instrument_edge_pp"] = float((market_mid - p_hat) * 100.0)
    st.session_state["cs2_instrument_spread_pp"] = float(spread * 100.0)
    st.session_state["cs2_instrument_live_source"] = st.session_state.get("cs2_live_source", "—")
    st.session_state["cs2_instrument_rail_mode"] = st.session_state.get("cs2_rail_mode", "—")
    st.session_state["cs2_instrument_q_intra_source"] = st.session_state.get("cs2_q_intra_round_win_a_source") or "—"

    colM1, colM2, colM3, colM4 = st.columns(4)
    with colM1:
        st.metric(line_label, f"{p_hat*100:.1f}%")
    with colM2:
        st.metric("Certainty band (K CI)", f"[{lo*100:.1f}%, {hi*100:.1f}%]")
    with colM3:
        st.metric("Market bid / ask", f"{bid*100:.1f}% / {ask*100:.1f}%")
    with colM4:
        st.metric("Spread (abs / rel)", f"{spread*100:.1f} pp / {rel_spread*100:.1f}%")

    if state_bound_lower is not None and state_bound_upper is not None:
        st.caption(f"**State bounds (map resolution):** [{state_bound_lower*100:.1f}%, {state_bound_upper*100:.1f}%] — series prob if B wins / if A wins this map. Prices or P hat outside = over-extended or error.")
    # BO3 MIDROUND V1 — optional compact debug line
    _mf = st.session_state.get("cs2_midround_last_features") or {}
    _mr = st.session_state.get("cs2_midround_last_result")
    if _mr is not None and st.session_state.get("cs2_midround_p_hat_final_source") in ("midround_v1", "midround_v2", "midround_v2_mixture"):
        st.caption(f"Mid-round: alive_diff={_mf.get('alive_diff', '—')} hp_diff={_mf.get('hp_diff_alive', 0):.0f} bomb={_mf.get('bomb_planted', 0)} adj={_mr.get('mid_adj_total', 0):+.3f} clamped={_mr.get('mid_clamped_hit', False)}")

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
        debug_data = {
            "p0_map": float(p0_map),
            "p_hat_map": float(p_hat_map),
            "band_map": [float(lo_map), float(hi_map)],
            "series_fmt": series_fmt,
            "maps_a_won": int(maps_a_won),
            "maps_b_won": int(maps_b_won),
            "note": "Series conversion assumes future maps i.i.d with p_future = p0_map (MVP)."
        }
        if state_bound_lower is not None and state_bound_upper is not None:
            debug_data["state_bounds_map_resolution"] = [float(state_bound_lower), float(state_bound_upper)]
            debug_data["state_note"] = "Series prob if B wins this map (min) / if A wins this map (max). Prices and P hat outside = over-extended or error."
        st.write(debug_data)

    # Force-add a chart snapshot when series score changes so state bounds shift shows as a step at that timestamp
    cur_a = int(maps_a_won) if maps_a_won is not None else 0
    cur_b = int(maps_b_won) if maps_b_won is not None else 0
    prev_a = st.session_state.get("cs2_prev_maps_a_won")
    prev_b = st.session_state.get("cs2_prev_maps_b_won")
    if prev_a is not None and prev_b is not None:
        if cur_a != int(prev_a) or cur_b != int(prev_b):
            st.session_state["cs2_auto_add_snapshot_this_run"] = True
            st.session_state["cs2_force_snapshot_reason"] = "series_score_change"
    st.session_state["cs2_prev_maps_a_won"] = cur_a
    st.session_state["cs2_prev_maps_b_won"] = cur_b

    # Force snapshot when critical config changes while locked (do not fire on first init)
    _config_locked_now = bool(st.session_state.get("cs2_live_config_locked", False))
    _prev_p0_map = st.session_state.get("cs2_prev_p0_map")
    _prev_p0_series = st.session_state.get("cs2_prev_p0_series")
    _prev_team_lock = st.session_state.get("cs2_prev_team_lock")
    _prev_team_a_is_team_one = st.session_state.get("cs2_prev_team_a_is_team_one")
    _cur_p0_map = float(p0_map) if p0_map is not None else None
    _cur_p0_series = float(p0_series) if p0_series is not None else None
    _cur_team_lock = bool(st.session_state.get("cs2_live_team_lock", False))
    _cur_team_a_is_team_one = bool(st.session_state.get("cs2_bo3_live_team_a_is_team_one", True))
    _any_prev = (_prev_p0_map is not None or _prev_p0_series is not None or _prev_team_lock is not None or _prev_team_a_is_team_one is not None)
    _changed = False
    if _any_prev:
        if _prev_p0_map is not None and _cur_p0_map is not None and abs(_prev_p0_map - _cur_p0_map) > 1e-9:
            _changed = True
        if _prev_p0_series is not None and _cur_p0_series is not None and abs(_prev_p0_series - _cur_p0_series) > 1e-9:
            _changed = True
        if _prev_team_lock is not None and _prev_team_lock != _cur_team_lock:
            _changed = True
        if _prev_team_a_is_team_one is not None and _prev_team_a_is_team_one != _cur_team_a_is_team_one:
            _changed = True
    if _config_locked_now and _any_prev and _changed:
        st.session_state["cs2_auto_add_snapshot_this_run"] = True
        st.session_state["cs2_force_snapshot_reason"] = "config_change"
    st.session_state["cs2_prev_p0_map"] = _cur_p0_map
    st.session_state["cs2_prev_p0_series"] = _cur_p0_series
    st.session_state["cs2_prev_team_lock"] = _cur_team_lock
    st.session_state["cs2_prev_team_a_is_team_one"] = _cur_team_a_is_team_one

    manual_add = st.button("Add snapshot")
    _config_locked = bool(st.session_state.get("cs2_live_config_locked", False))
    auto_add = (
        _config_locked
        and bool(st.session_state.get("cs2_auto_add_snapshot_this_run", False))
        and (
            bool(st.session_state.get("cs2_bo3_auto_active"))
            or st.session_state.get("cs2_live_source") == "GRID"
            or (st.session_state.get("cs2_live_source") == CS2_LIVE_SOURCE_BO3 and st.session_state.get("cs2_bo3_source_mode") == "REPLAY (from bo3_pulls.jsonl)")
        )
    )
    _do_add_snapshot = manual_add or auto_add
    if _do_add_snapshot:
        st.session_state["cs2_auto_add_snapshot_this_run"] = False
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
        map_index = int(st.session_state.get("cs2_inplay_map_index", 0))

        _snap_ts = datetime.now().isoformat(timespec="seconds")
        # BO3 MARKET DELAY ALIGN: use delayed market for logging (align with delayed BO3 feed)
        _delay_sec = float(st.session_state.get("cs2_market_logging_delay_sec", 120))
        _bid_d, _ask_d, _mid_d, _delayed_ts, _buffer_hit = _cs2_market_delayed_snapshot(_delay_sec, bid, ask)
        _market_live_ts = time.time()
        _feed_ts = _market_live_ts  # BO3 feed ts not in scope here; use local now
        # BO3 MARKET DELAY ALIGN
        snapshot_row = {
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
            "econ_a": float(econ_a),
            "econ_b": float(econ_b),
            # BO3 ECON LATCH — optional debug fields for snapshot row
            "econ_source_used": econ_source_used,
            "team_a_money_total": float(team_a_money_total),
            "team_b_money_total": float(team_b_money_total),
            "econ_feed_a_raw": float(econ_feed_a_raw) if econ_feed_a_raw is not None else None,
            "econ_feed_b_raw": float(econ_feed_b_raw) if econ_feed_b_raw is not None else None,
            "econ_latched_a": float(econ_latched_a) if econ_latched_a is not None else None,
            "econ_latched_b": float(econ_latched_b) if econ_latched_b is not None else None,
            "econ_latched_round_number": int(econ_latched_round_number) if econ_latched_round_number is not None else None,
            "p0_series": float(p0_series) if 'p0_series' in locals() else float(p0_map),
            "p0_map": float(p0_map),
            "p_hat_map": float(p_hat_map),
            "p_hat": float(p_hat),
            "band_lo": float(lo),
            "band_hi": float(hi),
            # BO3 STATE BANDS: round-state endpoint bands (if A/B wins current round)
            "band_if_a_round": float(band_if_a_round) if band_if_a_round is not None else None,
            "band_if_b_round": float(band_if_b_round) if band_if_b_round is not None else None,
            "band_state_lo": float(band_state_lo) if band_state_lo is not None else None,
            "band_state_hi": float(band_state_hi) if band_state_hi is not None else None,
            # BO3 STATE BANDS
            "state_bound_lower": float(state_bound_lower) if state_bound_lower is not None else None,
            "state_bound_upper": float(state_bound_upper) if state_bound_upper is not None else None,
            # BO3 STATE BANDS CONSISTENCY: debug validation (in-memory only)
            "p_hat_current_helper_check": float(p_hat),
            "state_band_contains_phat": bool(
                band_state_lo is not None and band_state_hi is not None
                and (float(band_state_lo) - 1e-6 <= float(p_hat) <= float(band_state_hi) + 1e-6)
            ),
            # BO3 STATE BANDS CONSISTENCY
            "band_lo_map": float(lo_map),
            "band_hi_map": float(hi_map),
            # BO3 MARKET DELAY ALIGN: log delayed market for alignment with BO3 feed
            "market_bid": float(_bid_d),
            "market_ask": float(_ask_d),
            "market_mid": float(_mid_d),
            "spread": float(_ask_d - _bid_d),
            "rel_spread": float((_ask_d - _bid_d) / _mid_d) if _mid_d > 0 else 0.0,
            "dev_mid_pp": float((_mid_d - p_hat) * 100.0),
            "buy_edge_pp": float((p_hat - _ask_d) * 100.0),
            "sell_edge_pp": float((_bid_d - p_hat) * 100.0),
            "feed_ts_epoch": float(_feed_ts) if _feed_ts is not None else None,
            "market_live_ts_epoch": float(_market_live_ts),
            "market_delayed_ts_epoch": float(_delayed_ts) if _delayed_ts is not None else None,
            "market_delay_sec_used": float(_delay_sec),
            "market_delay_buffer_hit": bool(_buffer_hit),
            # BO3 MARKET DELAY ALIGN
            # BO3 LOADOUT VALUE FIX — in-memory debug fields (do not break CSV schema)
            "team_a_cash_total": float(st.session_state.get("cs2_live_team_a_cash_total")) if st.session_state.get("cs2_live_team_a_cash_total") is not None else None,
            "team_b_cash_total": float(st.session_state.get("cs2_live_team_b_cash_total")) if st.session_state.get("cs2_live_team_b_cash_total") is not None else None,
            "team_a_loadout_est_total": float(st.session_state.get("cs2_live_team_a_loadout_est_total")) if st.session_state.get("cs2_live_team_a_loadout_est_total") is not None else None,
            "team_b_loadout_est_total": float(st.session_state.get("cs2_live_team_b_loadout_est_total")) if st.session_state.get("cs2_live_team_b_loadout_est_total") is not None else None,
            "team_a_total_resources": float(st.session_state.get("cs2_live_team_a_total_resources")) if st.session_state.get("cs2_live_team_a_total_resources") is not None else None,
            "team_b_total_resources": float(st.session_state.get("cs2_live_team_b_total_resources")) if st.session_state.get("cs2_live_team_b_total_resources") is not None else None,
            "team_a_alive_cash_total": float(st.session_state.get("cs2_live_team_a_alive_cash_total")) if st.session_state.get("cs2_live_team_a_alive_cash_total") is not None else None,
            "team_b_alive_cash_total": float(st.session_state.get("cs2_live_team_b_alive_cash_total")) if st.session_state.get("cs2_live_team_b_alive_cash_total") is not None else None,
            "team_a_alive_loadout_est_total": float(st.session_state.get("cs2_live_team_a_alive_loadout_est_total")) if st.session_state.get("cs2_live_team_a_alive_loadout_est_total") is not None else None,
            "team_b_alive_loadout_est_total": float(st.session_state.get("cs2_live_team_b_alive_loadout_est_total")) if st.session_state.get("cs2_live_team_b_alive_loadout_est_total") is not None else None,
            "team_a_alive_total_resources": float(st.session_state.get("cs2_live_team_a_alive_total_resources")) if st.session_state.get("cs2_live_team_a_alive_total_resources") is not None else None,
            "team_b_alive_total_resources": float(st.session_state.get("cs2_live_team_b_alive_total_resources")) if st.session_state.get("cs2_live_team_b_alive_total_resources") is not None else None,
            "team_a_alive_count": int(st.session_state.get("cs2_live_team_a_alive_count")) if st.session_state.get("cs2_live_team_a_alive_count") is not None else None,
            "team_b_alive_count": int(st.session_state.get("cs2_live_team_b_alive_count")) if st.session_state.get("cs2_live_team_b_alive_count") is not None else None,
            "team_a_hp_alive_total": float(st.session_state.get("cs2_live_team_a_hp_alive_total")) if st.session_state.get("cs2_live_team_a_hp_alive_total") is not None else None,
            "team_b_hp_alive_total": float(st.session_state.get("cs2_live_team_b_hp_alive_total")) if st.session_state.get("cs2_live_team_b_hp_alive_total") is not None else None,
            "loadout_est_source": "derived_from_weapons" if st.session_state.get("cs2_live_team_a_loadout_est_total") is not None else None,
            # BO3 LOADOUT VALUE FIX
            # BO3 MIDROUND V1 — in-memory debug (do not break CSV schema)
            "p_base_frozen": float(st.session_state.get("cs2_midround_p_base_frozen")) if st.session_state.get("cs2_midround_p_base_frozen") is not None else None,
            "round_context_offset": float(st.session_state.get("cs2_round_context_offset")) if st.session_state.get("cs2_round_context_offset") is not None else None,
            "p_round_context": float(st.session_state.get("cs2_p_round_context")) if st.session_state.get("cs2_p_round_context") is not None else None,
            "round_ctx_econ_component": float(_ctx.get("round_ctx_econ_component")) if (_ctx := st.session_state.get("cs2_round_context_debug")) and "round_ctx_econ_component" in _ctx else None,
            "round_ctx_loadout_component": float(_ctx.get("round_ctx_loadout_component")) if (_ctx := st.session_state.get("cs2_round_context_debug")) and "round_ctx_loadout_component" in _ctx else None,
            "round_ctx_side_component": float(_ctx.get("round_ctx_side_component")) if (_ctx := st.session_state.get("cs2_round_context_debug")) and "round_ctx_side_component" in _ctx else None,
            "round_ctx_armor_component": float(_ctx.get("round_ctx_armor_component")) if (_ctx := st.session_state.get("cs2_round_context_debug")) and "round_ctx_armor_component" in _ctx else None,
            "p_mid_raw": float(st.session_state["cs2_midround_last_result"]["p_mid_raw"]) if (st.session_state.get("cs2_midround_last_result") and "p_mid_raw" in st.session_state["cs2_midround_last_result"]) else None,
            "p_mid_clamped": float(st.session_state["cs2_midround_last_result"]["p_mid_clamped"]) if (st.session_state.get("cs2_midround_last_result") and "p_mid_clamped" in st.session_state["cs2_midround_last_result"]) else None,
            "p_hat_final_source": st.session_state.get("cs2_midround_p_hat_final_source"),
            "mid_adj_total": float(st.session_state["cs2_midround_last_result"]["mid_adj_total"]) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_total" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_adj_alive": float(st.session_state["cs2_midround_last_result"]["mid_adj_alive"]) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_alive" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_adj_bomb": float(st.session_state["cs2_midround_last_result"]["mid_adj_bomb"]) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_bomb" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_adj_hp": float(st.session_state["cs2_midround_last_result"]["mid_adj_hp"]) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_hp" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_time_scale": float(st.session_state["cs2_midround_last_result"]["mid_time_scale"]) if (st.session_state.get("cs2_midround_last_result") and "mid_time_scale" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_clamped_hit": bool(st.session_state["cs2_midround_last_result"]["mid_clamped_hit"]) if (st.session_state.get("cs2_midround_last_result") and "mid_clamped_hit" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_clamp_distance": float(st.session_state["cs2_midround_last_result"]["mid_clamp_distance"]) if (st.session_state.get("cs2_midround_last_result") and "mid_clamp_distance" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_adj_armor": float(st.session_state["cs2_midround_last_result"].get("mid_adj_armor")) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_armor" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_adj_loadout": float(st.session_state["cs2_midround_last_result"].get("mid_adj_loadout")) if (st.session_state.get("cs2_midround_last_result") and "mid_adj_loadout" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_reliability_mult": float(st.session_state["cs2_midround_last_result"].get("mid_reliability_mult")) if (st.session_state.get("cs2_midround_last_result") and "mid_reliability_mult" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_feature_has_armor": bool(st.session_state["cs2_midround_last_result"].get("mid_feature_has_armor")) if (st.session_state.get("cs2_midround_last_result") and "mid_feature_has_armor" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_feature_has_alive_loadout": bool(st.session_state["cs2_midround_last_result"].get("mid_feature_has_alive_loadout")) if (st.session_state.get("cs2_midround_last_result") and "mid_feature_has_alive_loadout" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_raw_total_pre_reliability": float(st.session_state["cs2_midround_last_result"].get("mid_raw_total_pre_reliability")) if (st.session_state.get("cs2_midround_last_result") and "mid_raw_total_pre_reliability" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_raw_total_post_reliability": float(st.session_state["cs2_midround_last_result"].get("mid_raw_total_post_reliability")) if (st.session_state.get("cs2_midround_last_result") and "mid_raw_total_post_reliability" in st.session_state["cs2_midround_last_result"]) else None,
            "mid_alive_diff": _mf.get("alive_diff") if (_mf := st.session_state.get("cs2_midround_last_features")) else None,
            "mid_hp_diff_alive": float(_mf["hp_diff_alive"]) if (_mf := st.session_state.get("cs2_midround_last_features")) and "hp_diff_alive" in _mf else None,
            "mid_bomb_planted": _mf.get("bomb_planted") if (_mf := st.session_state.get("cs2_midround_last_features")) else None,
            "mid_time_remaining_s": float(_mf["time_remaining_s"]) if (_mf := st.session_state.get("cs2_midround_last_features")) and "time_remaining_s" in _mf else None,
            "round_state_frozen_key": st.session_state.get("cs2_midround_round_key"),
            "round_state_frozen_lo": float(st.session_state["cs2_midround_band_state_lo_frozen"]) if st.session_state.get("cs2_midround_band_state_lo_frozen") is not None else None,
            "round_state_frozen_hi": float(st.session_state["cs2_midround_band_state_hi_frozen"]) if st.session_state.get("cs2_midround_band_state_hi_frozen") is not None else None,
            "round_state_band_if_a": float(st.session_state["cs2_midround_band_if_a_round_frozen"]) if st.session_state.get("cs2_midround_band_if_a_round_frozen") is not None else None,
            "round_state_band_if_b": float(st.session_state["cs2_midround_band_if_b_round_frozen"]) if st.session_state.get("cs2_midround_band_if_b_round_frozen") is not None else None,
            "rail_p_state_center": float(_rd["rail_p_state_center"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "rail_p_state_center" in _rd else None,
            "rail_p_if_next_round_win": float(_rd["rail_p_if_next_round_win"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "rail_p_if_next_round_win" in _rd else None,
            "rail_p_if_next_round_loss": float(_rd["rail_p_if_next_round_loss"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "rail_p_if_next_round_loss" in _rd else None,
            "rail_upper_width": float(_rd["rail_upper_width"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "rail_upper_width" in _rd else None,
            "rail_lower_width": float(_rd["rail_lower_width"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "rail_lower_width" in _rd else None,
            "rail_asymmetry_ratio": float(_rd["rail_asymmetry_ratio"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_asymmetry_ratio") is not None else None,
            "rail_current_rounds_a": int(_rd["current_rounds_a"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "current_rounds_a" in _rd else None,
            "rail_current_rounds_b": int(_rd["current_rounds_b"]) if (_rd := st.session_state.get("cs2_rail_debug")) and "current_rounds_b" in _rd else None,
            # V2 rail model debug
            "rail_model_version": str(_rd["rail_model_version"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_model_version") else None,
            "rail_anchor": float(_rd["anchor"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("anchor") is not None else None,
            "rail_if_a_wins": float(_rd["p_if_a_wins"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("p_if_a_wins") is not None else None,
            "rail_if_b_wins": float(_rd["p_if_b_wins"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("p_if_b_wins") is not None else None,
            "rail_base_rounds_a": int(st.session_state["cs2_rail_base_rounds_a"]) if st.session_state.get("cs2_rail_base_rounds_a") is not None else None,
            "rail_base_rounds_b": int(st.session_state["cs2_rail_base_rounds_b"]) if st.session_state.get("cs2_rail_base_rounds_b") is not None else None,
            "rail_cf_a_rounds_a": int(st.session_state["cs2_rail_cf_a_rounds_a"]) if st.session_state.get("cs2_rail_cf_a_rounds_a") is not None else None,
            "rail_cf_a_rounds_b": int(st.session_state["cs2_rail_cf_a_rounds_b"]) if st.session_state.get("cs2_rail_cf_a_rounds_b") is not None else None,
            "rail_cf_b_rounds_a": int(st.session_state["cs2_rail_cf_b_rounds_a"]) if st.session_state.get("cs2_rail_cf_b_rounds_a") is not None else None,
            "rail_cf_b_rounds_b": int(st.session_state["cs2_rail_cf_b_rounds_b"]) if st.session_state.get("cs2_rail_cf_b_rounds_b") is not None else None,
            "rail_endpoint_if_a_round": float(st.session_state["cs2_rail_endpoint_if_a_round"]) if st.session_state.get("cs2_rail_endpoint_if_a_round") is not None else None,
            "rail_endpoint_if_b_round": float(st.session_state["cs2_rail_endpoint_if_b_round"]) if st.session_state.get("cs2_rail_endpoint_if_b_round") is not None else None,
            "rail_endpoint_span_abs": float(st.session_state["cs2_rail_endpoint_span_abs"]) if st.session_state.get("cs2_rail_endpoint_span_abs") is not None else None,
            "rail_endpoint_degenerate": bool(st.session_state.get("cs2_rail_endpoint_degenerate", False)),
            "rail_endpoint_env_valid": bool(st.session_state.get("cs2_rail_endpoint_env_valid", False)),
            "rail_endpoint_env_base_span_abs": float(st.session_state["cs2_rail_endpoint_env_base_span_abs"]) if st.session_state.get("cs2_rail_endpoint_env_base_span_abs") is not None else None,
            "rail_endpoint_env_fraction_k": float(st.session_state["cs2_rail_endpoint_env_fraction_k"]) if st.session_state.get("cs2_rail_endpoint_env_fraction_k") is not None else None,
            "rail_endpoint_env_round_leverage": float(st.session_state["cs2_rail_endpoint_env_round_leverage"]) if st.session_state.get("cs2_rail_endpoint_env_round_leverage") is not None else None,
            "rail_endpoint_env_series_leverage": float(st.session_state["cs2_rail_endpoint_env_series_leverage"]) if st.session_state.get("cs2_rail_endpoint_env_series_leverage") is not None else None,
            "rail_endpoint_env_half_width": float(st.session_state["cs2_rail_endpoint_env_half_width"]) if st.session_state.get("cs2_rail_endpoint_env_half_width") is not None else None,
            "rail_endpoint_a_env_low": float(st.session_state["cs2_rail_endpoint_a_env_low"]) if st.session_state.get("cs2_rail_endpoint_a_env_low") is not None else None,
            "rail_endpoint_a_env_high": float(st.session_state["cs2_rail_endpoint_a_env_high"]) if st.session_state.get("cs2_rail_endpoint_a_env_high") is not None else None,
            "rail_endpoint_b_env_low": float(st.session_state["cs2_rail_endpoint_b_env_low"]) if st.session_state.get("cs2_rail_endpoint_b_env_low") is not None else None,
            "rail_endpoint_b_env_high": float(st.session_state["cs2_rail_endpoint_b_env_high"]) if st.session_state.get("cs2_rail_endpoint_b_env_high") is not None else None,
            "rail_endpoint_a_env_center": float(st.session_state["cs2_rail_endpoint_a_env_center"]) if st.session_state.get("cs2_rail_endpoint_a_env_center") is not None else None,
            "rail_endpoint_b_env_center": float(st.session_state["cs2_rail_endpoint_b_env_center"]) if st.session_state.get("cs2_rail_endpoint_b_env_center") is not None else None,
            "endpoint_pos_valid": bool(st.session_state.get("cs2_endpoint_pos_valid", False)),
            "endpoint_pos_d_alive": float(st.session_state["cs2_endpoint_pos_d_alive"]) if st.session_state.get("cs2_endpoint_pos_d_alive") is not None else None,
            "endpoint_pos_d_loadout": float(st.session_state["cs2_endpoint_pos_d_loadout"]) if st.session_state.get("cs2_endpoint_pos_d_loadout") is not None else None,
            "endpoint_pos_d_armor": float(st.session_state["cs2_endpoint_pos_d_armor"]) if st.session_state.get("cs2_endpoint_pos_d_armor") is not None else None,
            "endpoint_pos_score_a": float(st.session_state["cs2_endpoint_pos_score_a"]) if st.session_state.get("cs2_endpoint_pos_score_a") is not None else None,
            "endpoint_pos_score_b": float(st.session_state["cs2_endpoint_pos_score_b"]) if st.session_state.get("cs2_endpoint_pos_score_b") is not None else None,
            "endpoint_pos_a_quality_pos": float(st.session_state["cs2_endpoint_pos_a_quality_pos"]) if st.session_state.get("cs2_endpoint_pos_a_quality_pos") is not None else None,
            "endpoint_pos_b_quality_pos": float(st.session_state["cs2_endpoint_pos_b_quality_pos"]) if st.session_state.get("cs2_endpoint_pos_b_quality_pos") is not None else None,
            "endpoint_pos_a_active_dbg": float(st.session_state["cs2_endpoint_pos_a_active_dbg"]) if st.session_state.get("cs2_endpoint_pos_a_active_dbg") is not None else None,
            "endpoint_pos_b_active_dbg": float(st.session_state["cs2_endpoint_pos_b_active_dbg"]) if st.session_state.get("cs2_endpoint_pos_b_active_dbg") is not None else None,
            "endpoint_pos_a_active_within_env": bool(st.session_state.get("cs2_endpoint_pos_a_active_within_env", False)),
            "endpoint_pos_b_active_within_env": bool(st.session_state.get("cs2_endpoint_pos_b_active_within_env", False)),
            "endpoint_econ_valid": bool(st.session_state.get("cs2_endpoint_econ_valid", False)),
            "endpoint_econ_d_alive_survival": float(st.session_state.get("cs2_endpoint_econ_d_alive_survival")) if st.session_state.get("cs2_endpoint_econ_d_alive_survival") is not None else None,
            "endpoint_econ_d_alive_loadout": float(st.session_state.get("cs2_endpoint_econ_d_alive_loadout")) if st.session_state.get("cs2_endpoint_econ_d_alive_loadout") is not None else None,
            "endpoint_econ_d_alive_cash": float(st.session_state.get("cs2_endpoint_econ_d_alive_cash")) if st.session_state.get("cs2_endpoint_econ_d_alive_cash") is not None else None,
            "endpoint_econ_score_a": float(st.session_state.get("cs2_endpoint_econ_score_a")) if st.session_state.get("cs2_endpoint_econ_score_a") is not None else None,
            "endpoint_econ_score_b": float(st.session_state.get("cs2_endpoint_econ_score_b")) if st.session_state.get("cs2_endpoint_econ_score_b") is not None else None,
            "endpoint_pos_third_driver_source": str(st.session_state.get("cs2_endpoint_pos_third_driver_source", "legacy_non_dynamic")),
            "drv_valid_microstate": bool(st.session_state.get("cs2_drv_valid_microstate", False)),
            "drv_valid_roundstate": bool(st.session_state.get("cs2_drv_valid_roundstate", False)),
            "drv_team_a_side": st.session_state.get("cs2_drv_team_a_side"),
            "drv_bomb_planted": st.session_state.get("cs2_drv_bomb_planted"),
            "drv_round_phase": st.session_state.get("cs2_drv_round_phase"),
            "drv_round_time_remaining_s": float(st.session_state.get("cs2_drv_round_time_remaining_s")) if st.session_state.get("cs2_drv_round_time_remaining_s") is not None else None,
            "drv_alive_count_a": st.session_state.get("cs2_drv_alive_count_a"),
            "drv_alive_count_b": st.session_state.get("cs2_drv_alive_count_b"),
            "drv_hp_alive_total_a": float(st.session_state.get("cs2_drv_hp_alive_total_a")) if st.session_state.get("cs2_drv_hp_alive_total_a") is not None else None,
            "drv_hp_alive_total_b": float(st.session_state.get("cs2_drv_hp_alive_total_b")) if st.session_state.get("cs2_drv_hp_alive_total_b") is not None else None,
            "drv_alive_loadout_est_total_a": float(st.session_state.get("cs2_drv_alive_loadout_est_total_a")) if st.session_state.get("cs2_drv_alive_loadout_est_total_a") is not None else None,
            "drv_alive_loadout_est_total_b": float(st.session_state.get("cs2_drv_alive_loadout_est_total_b")) if st.session_state.get("cs2_drv_alive_loadout_est_total_b") is not None else None,
            "drv_alive_cash_total_a": float(st.session_state.get("cs2_drv_alive_cash_total_a")) if st.session_state.get("cs2_drv_alive_cash_total_a") is not None else None,
            "drv_alive_cash_total_b": float(st.session_state.get("cs2_drv_alive_cash_total_b")) if st.session_state.get("cs2_drv_alive_cash_total_b") is not None else None,
            "drv_source_contract": st.session_state.get("cs2_drv_source_contract"),
            "phat_intra_valid": bool(st.session_state.get("cs2_phat_intra_valid", False)),
            "phat_intra_side_known": st.session_state.get("cs2_phat_intra_side_known"),
            "phat_intra_clock_known": st.session_state.get("cs2_phat_intra_clock_known"),
            "phat_intra_phase_mult": float(st.session_state.get("cs2_phat_intra_phase_mult")) if st.session_state.get("cs2_phat_intra_phase_mult") is not None else None,
            "phat_intra_urgency": float(st.session_state.get("cs2_phat_intra_urgency")) if st.session_state.get("cs2_phat_intra_urgency") is not None else None,
            "phat_intra_d_alive": float(st.session_state.get("cs2_phat_intra_d_alive")) if st.session_state.get("cs2_phat_intra_d_alive") is not None else None,
            "phat_intra_d_hp": float(st.session_state.get("cs2_phat_intra_d_hp")) if st.session_state.get("cs2_phat_intra_d_hp") is not None else None,
            "phat_intra_d_loadout": float(st.session_state.get("cs2_phat_intra_d_loadout")) if st.session_state.get("cs2_phat_intra_d_loadout") is not None else None,
            "phat_intra_bomb_term_signed": float(st.session_state.get("cs2_phat_intra_bomb_term_signed")) if st.session_state.get("cs2_phat_intra_bomb_term_signed") is not None else None,
            "phat_intra_clock_term_signed": float(st.session_state.get("cs2_phat_intra_clock_term_signed")) if st.session_state.get("cs2_phat_intra_clock_term_signed") is not None else None,
            "phat_intra_bomb_clock_combo_signed": float(st.session_state.get("cs2_phat_intra_bomb_clock_combo_signed")) if st.session_state.get("cs2_phat_intra_bomb_clock_combo_signed") is not None else None,
            "phat_intra_loadout_presence": float(st.session_state.get("cs2_phat_intra_loadout_presence")) if st.session_state.get("cs2_phat_intra_loadout_presence") is not None else None,
            "phat_intra_w_alive_eff": float(st.session_state.get("cs2_phat_intra_w_alive_eff")) if st.session_state.get("cs2_phat_intra_w_alive_eff") is not None else None,
            "phat_intra_score_alive": float(st.session_state.get("cs2_phat_intra_score_alive")) if st.session_state.get("cs2_phat_intra_score_alive") is not None else None,
            "phat_intra_score_hp": float(st.session_state.get("cs2_phat_intra_score_hp")) if st.session_state.get("cs2_phat_intra_score_hp") is not None else None,
            "phat_intra_score_loadout": float(st.session_state.get("cs2_phat_intra_score_loadout")) if st.session_state.get("cs2_phat_intra_score_loadout") is not None else None,
            "phat_intra_score_bomb": float(st.session_state.get("cs2_phat_intra_score_bomb")) if st.session_state.get("cs2_phat_intra_score_bomb") is not None else None,
            "phat_intra_score_clock": float(st.session_state.get("cs2_phat_intra_score_clock")) if st.session_state.get("cs2_phat_intra_score_clock") is not None else None,
            "phat_intra_score_bomb_clock_combo": float(st.session_state.get("cs2_phat_intra_score_bomb_clock_combo")) if st.session_state.get("cs2_phat_intra_score_bomb_clock_combo") is not None else None,
            "phat_intra_score_raw_pre_reliability": float(st.session_state.get("cs2_phat_intra_score_raw_pre_reliability")) if st.session_state.get("cs2_phat_intra_score_raw_pre_reliability") is not None else None,
            "phat_intra_score_raw_post_reliability": float(st.session_state.get("cs2_phat_intra_score_raw_post_reliability")) if st.session_state.get("cs2_phat_intra_score_raw_post_reliability") is not None else None,
            "phat_intra_q_intra_round_win_a_dbg": float(st.session_state.get("cs2_phat_intra_q_intra_round_win_a_dbg")) if st.session_state.get("cs2_phat_intra_q_intra_round_win_a_dbg") is not None else None,
            "rail_score_term": float(_rd["rail_score_term"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_score_term") is not None else None,
            "rail_side_term": float(_rd["rail_side_term"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_side_term") is not None else None,
            "rail_econ_term": float(_rd["rail_econ_term"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_econ_term") is not None else None,
            "rail_loadout_term": float(_rd["rail_loadout_term"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_loadout_term") is not None else None,
            "rail_round_end_latched": bool(st.session_state.get("cs2_rail_round_end_latched", False)),
            "rail_resolved_to_prev_endpoint": bool(st.session_state.get("cs2_rail_resolved_to_prev_endpoint", False)),
            "rail_inputs_latched": bool(_rd.get("rail_inputs_latched", False)) if (_rd := st.session_state.get("cs2_rail_debug")) else False,
            "rail_used_smoothing": bool(_rd.get("rail_used_smoothing", False)) if (_rd := st.session_state.get("cs2_rail_debug")) else False,
            "rail_smoothing_alpha": float(_rd["rail_smoothing_alpha"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_smoothing_alpha") is not None else None,
            "rail_input_econ_a": float(_rd["rail_input_econ_a"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_input_econ_a") is not None else None,
            "rail_input_econ_b": float(_rd["rail_input_econ_b"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_input_econ_b") is not None else None,
            "rail_input_loadout_a": float(_rd["rail_input_loadout_a"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_input_loadout_a") is not None else None,
            "rail_input_loadout_b": float(_rd["rail_input_loadout_b"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_input_loadout_b") is not None else None,
            "rail_series_maps_won_a": int(_rd["rail_series_maps_won_a"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_series_maps_won_a") is not None else None,
            "rail_series_maps_won_b": int(_rd["rail_series_maps_won_b"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_series_maps_won_b") is not None else None,
            "rail_series_term": float(_rd["rail_series_term"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_series_term") is not None else None,
            "rail_series_asymmetry": float(_rd["rail_series_asymmetry"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_series_asymmetry") is not None else None,
            "rail_anchor_canonical": float(_rd["rail_anchor_canonical"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_anchor_canonical") is not None else None,
            "rail_if_a_canonical": float(_rd["rail_if_a_canonical"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_if_a_canonical") is not None else None,
            "rail_if_b_canonical": float(_rd["rail_if_b_canonical"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_if_b_canonical") is not None else None,
            "rail_if_a_adjusted": float(_rd["rail_if_a_adjusted"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_if_a_adjusted") is not None else None,
            "rail_if_b_adjusted": float(_rd["rail_if_b_adjusted"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_if_b_adjusted") is not None else None,
            "rail_asymmetry_bias": float(_rd["rail_asymmetry_bias"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_asymmetry_bias") is not None else None,
            "rail_asymmetry_mult_a": float(_rd["rail_asymmetry_mult_a"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_asymmetry_mult_a") is not None else None,
            "rail_asymmetry_mult_b": float(_rd["rail_asymmetry_mult_b"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_asymmetry_mult_b") is not None else None,
            "rail_series_context_used": bool(_rd.get("rail_series_context_used", False)) if (_rd := st.session_state.get("cs2_rail_debug")) else False,
            "rail_anchor_source": str(_rd["rail_anchor_source"]) if (_rd := st.session_state.get("cs2_rail_debug")) and _rd.get("rail_anchor_source") else None,
            "round_transition_detected": bool(st.session_state.get("round_transition_detected", False)),
            "round_transition_prev_key": st.session_state.get("round_transition_prev_key"),
            "round_transition_new_key": st.session_state.get("round_transition_new_key"),
            "round_transition_winner": st.session_state.get("round_transition_winner"),
            "round_transition_resolved_p": float(st.session_state["round_transition_resolved_p"]) if st.session_state.get("round_transition_resolved_p") is not None else None,
            "round_transition_resolution_source": st.session_state.get("round_transition_resolution_source", "none"),
            "round_finalization_snap_occurred": bool(st.session_state.get("cs2_live_round_finalization_snap_occurred", False)),
            "round_snap_applied_this_tick": bool(st.session_state.get("cs2_round_snap_applied_this_tick", False)),
            "terminal_override_applied": bool(st.session_state.get("cs2_terminal_override_applied", False)),
            "terminal_override_reason": str(st.session_state.get("cs2_terminal_override_reason", "")),
            "terminal_team_a_won": st.session_state.get("cs2_terminal_team_a_won"),
            "p_hat_pre_terminal_override": float(st.session_state["cs2_p_hat_pre_terminal_override"]) if st.session_state.get("cs2_p_hat_pre_terminal_override") is not None else None,
            "p_hat_post_terminal_override": float(st.session_state["cs2_p_hat_post_terminal_override"]) if st.session_state.get("cs2_p_hat_post_terminal_override") is not None else None,
            # CS2 terminal override (contract-aware): debug fields
            "terminal_contract_scope": str(st.session_state.get("cs2_terminal_contract_scope", "")),
            "terminal_series_target": int(st.session_state["cs2_terminal_series_target"]) if st.session_state.get("cs2_terminal_series_target") is not None else None,
            "terminal_series_over": bool(st.session_state.get("cs2_terminal_series_over", False)),
            "terminal_map_over_detected": bool(st.session_state.get("cs2_terminal_map_over_detected", False)),
            "terminal_override_blocked_reason": str(st.session_state.get("cs2_terminal_override_blocked_reason", "")) or None,
            # CS2 bounds mode (diagnostic): pre/post stage values and clip flags
            "p_hat_pre_bounds": float(st.session_state["cs2_p_hat_pre_bounds"]) if st.session_state.get("cs2_p_hat_pre_bounds") is not None else None,
            "p_hat_post_round_bounds": float(st.session_state["cs2_p_hat_post_round_bounds"]) if st.session_state.get("cs2_p_hat_post_round_bounds") is not None else None,
            "p_hat_post_map_bounds": float(st.session_state["cs2_p_hat_post_map_bounds"]) if st.session_state.get("cs2_p_hat_post_map_bounds") is not None else None,
            "p_hat_bounds_mode": str(st.session_state.get("cs2_bounds_mode", "Normal (bounded)")),
            "p_hat_clipped_by_round": bool(st.session_state.get("cs2_p_hat_clipped_by_round", False)),
            "p_hat_clipped_by_map": bool(st.session_state.get("cs2_p_hat_clipped_by_map", False)),
            "p_hat_round_clip_delta": float(st.session_state["cs2_p_hat_round_clip_delta"]) if st.session_state.get("cs2_p_hat_round_clip_delta") is not None else None,
            "p_hat_map_clip_delta": float(st.session_state["cs2_p_hat_map_clip_delta"]) if st.session_state.get("cs2_p_hat_map_clip_delta") is not None else None,
            # CS2 soft round damping (diagnostic)
            "p_hat_round_anchor_before_mid": float(st.session_state["cs2_p_hat_round_anchor_before_mid"]) if st.session_state.get("cs2_p_hat_round_anchor_before_mid") is not None else None,
            "p_hat_round_soft_damped": float(st.session_state["cs2_p_hat_round_soft_damped"]) if st.session_state.get("cs2_p_hat_round_soft_damped") is not None else None,
            "p_hat_round_soft_edge_factor": float(st.session_state["cs2_p_hat_round_soft_edge_factor"]) if st.session_state.get("cs2_p_hat_round_soft_edge_factor") is not None else None,
            "p_hat_round_soft_room_frac": float(st.session_state["cs2_p_hat_round_soft_room_frac"]) if st.session_state.get("cs2_p_hat_round_soft_room_frac") is not None else None,
            "p_hat_round_soft_delta_raw": float(st.session_state["cs2_p_hat_round_soft_delta_raw"]) if st.session_state.get("cs2_p_hat_round_soft_delta_raw") is not None else None,
            "p_hat_round_soft_delta_applied": float(st.session_state["cs2_p_hat_round_soft_delta_applied"]) if st.session_state.get("cs2_p_hat_round_soft_delta_applied") is not None else None,
            "p_hat_round_soft_direction": str(st.session_state.get("cs2_p_hat_round_soft_direction")) if st.session_state.get("cs2_p_hat_round_soft_direction") is not None else None,
            "p_hat_round_soft_hit": bool(st.session_state.get("cs2_p_hat_round_soft_hit", False)) if st.session_state.get("cs2_p_hat_round_soft_hit") is not None else None,
            # Series-endpoint mixture (structural p_hat)
            "p_series_if_a_wins_next_round": float(st.session_state["cs2_p_series_if_a_wins_next_round"]) if st.session_state.get("cs2_p_series_if_a_wins_next_round") is not None else None,
            "p_series_if_b_wins_next_round": float(st.session_state["cs2_p_series_if_b_wins_next_round"]) if st.session_state.get("cs2_p_series_if_b_wins_next_round") is not None else None,
            "q_intra_round_win_a": float(st.session_state["cs2_q_intra_round_win_a"]) if st.session_state.get("cs2_q_intra_round_win_a") is not None else None,
            "q_intra_round_win_a_source": str(st.session_state.get("cs2_q_intra_round_win_a_source", "none")),
            "q_intra_round_win_a_reason": str(st.session_state.get("cs2_q_intra_round_win_a_reason", "not_computed")),
            "intra_score_raw": float(st.session_state["cs2_intra_score_raw"]) if st.session_state.get("cs2_intra_score_raw") is not None else None,
            "intra_score_alive": float(st.session_state["cs2_intra_score_alive"]) if st.session_state.get("cs2_intra_score_alive") is not None else None,
            "intra_score_hp": float(st.session_state["cs2_intra_score_hp"]) if st.session_state.get("cs2_intra_score_hp") is not None else None,
            "intra_score_bomb": float(st.session_state["cs2_intra_score_bomb"]) if st.session_state.get("cs2_intra_score_bomb") is not None else None,
            "intra_score_armor": float(st.session_state["cs2_intra_score_armor"]) if st.session_state.get("cs2_intra_score_armor") is not None else None,
            "intra_score_loadout": float(st.session_state["cs2_intra_score_loadout"]) if st.session_state.get("cs2_intra_score_loadout") is not None else None,
            "intra_score_post_reliability": float(st.session_state["cs2_intra_score_post_reliability"]) if st.session_state.get("cs2_intra_score_post_reliability") is not None else None,
            "intra_score_time_scale": float(st.session_state["cs2_intra_score_time_scale"]) if st.session_state.get("cs2_intra_score_time_scale") is not None else None,
            "p_hat_structural_mixture": float(st.session_state["cs2_p_hat_structural_mixture"]) if st.session_state.get("cs2_p_hat_structural_mixture") is not None else None,
            "p_hat_between_endpoints": bool(st.session_state.get("cs2_p_hat_between_endpoints", False)) if st.session_state.get("cs2_p_hat_between_endpoints") is not None else None,
            "endpoint_span_abs": float(st.session_state["cs2_endpoint_span_abs"]) if st.session_state.get("cs2_endpoint_span_abs") is not None else None,
            "branch_endpoint_source_mode": str(st.session_state.get("cs2_branch_endpoint_source_mode", "Frozen endpoints (current)")),
            "branch_endpoint_source_used": str(st.session_state.get("cs2_branch_endpoint_source_used", "n/a")),
            "branch_endpoint_source_reason": str(st.session_state.get("cs2_branch_endpoint_source_reason", "unknown")),
            "branch_endpoint_selected_a": float(st.session_state["cs2_branch_endpoint_selected_a"]) if st.session_state.get("cs2_branch_endpoint_selected_a") is not None else None,
            "branch_endpoint_selected_b": float(st.session_state["cs2_branch_endpoint_selected_b"]) if st.session_state.get("cs2_branch_endpoint_selected_b") is not None else None,
            "branch_endpoint_selected_span_abs": float(st.session_state["cs2_branch_endpoint_selected_span_abs"]) if st.session_state.get("cs2_branch_endpoint_selected_span_abs") is not None else None,
            "branch_endpoint_dynamic_available": bool(st.session_state.get("cs2_branch_endpoint_dynamic_available", False)),
            "bo3_game_ended_flag": bool(st.session_state.get("cs2_bo3_game_ended_flag", False)),
            "bo3_match_ended_flag": bool(st.session_state.get("cs2_bo3_match_ended_flag", False)),
            "bo3_match_status_flag": bool(st.session_state.get("cs2_bo3_match_status_flag", False)),
            "feed_terminal_used": bool(st.session_state.get("cs2_feed_terminal_used", False)),
            "round_key_current": st.session_state.get("cs2_midround_round_key"),
            "round_key_frozen": st.session_state.get("cs2_midround_round_key"),
            # Round-endpoint freeze timing diagnostics
            "round_freeze_active": bool(st.session_state.get("cs2_round_freeze_active", False)),
            "round_freeze_key": str(st.session_state["cs2_round_freeze_key"]) if st.session_state.get("cs2_round_freeze_key") is not None else None,
            "round_freeze_game_number": int(st.session_state["cs2_round_freeze_game_number"]) if st.session_state.get("cs2_round_freeze_game_number") is not None else None,
            "round_freeze_rounds_a": int(st.session_state["cs2_round_freeze_rounds_a"]) if st.session_state.get("cs2_round_freeze_rounds_a") is not None else None,
            "round_freeze_rounds_b": int(st.session_state["cs2_round_freeze_rounds_b"]) if st.session_state.get("cs2_round_freeze_rounds_b") is not None else None,
            "round_freeze_phase_at_capture": str(st.session_state.get("cs2_round_freeze_phase_at_capture")) if st.session_state.get("cs2_round_freeze_phase_at_capture") is not None else None,
            "round_freeze_band_if_a": float(st.session_state["cs2_round_freeze_band_if_a"]) if st.session_state.get("cs2_round_freeze_band_if_a") is not None else None,
            "round_freeze_band_if_b": float(st.session_state["cs2_round_freeze_band_if_b"]) if st.session_state.get("cs2_round_freeze_band_if_b") is not None else None,
            "round_freeze_tick_ts": float(st.session_state["cs2_round_freeze_tick_ts"]) if st.session_state.get("cs2_round_freeze_tick_ts") is not None else None,
            # Round-end snap diagnostics
            "round_snap_used": bool(st.session_state.get("cs2_round_snap_used", False)),
            "round_snap_target_side": str(st.session_state.get("cs2_round_snap_target_side")) if st.session_state.get("cs2_round_snap_target_side") is not None else None,
            "round_snap_value": float(st.session_state["cs2_round_snap_value"]) if st.session_state.get("cs2_round_snap_value") is not None else None,
            "round_snap_prev_freeze_key": str(st.session_state.get("cs2_round_snap_prev_freeze_key")) if st.session_state.get("cs2_round_snap_prev_freeze_key") is not None else None,
            "round_transition_tick_detected": bool(st.session_state.get("cs2_round_transition_tick_detected", False)),
            "round_transition_used_prev_frozen_for_snap": bool(st.session_state.get("cs2_round_transition_used_prev_frozen_for_snap", False)),
            "round_transition_latched_new_round_context": bool(st.session_state.get("cs2_round_transition_latched_new_round_context", False)),
            "round_transition_rail_computed_after_latch": bool(st.session_state.get("cs2_round_transition_rail_computed_after_latch", False)),
            "round_transition_game_number": int(st.session_state["cs2_round_transition_game_number"]) if st.session_state.get("cs2_round_transition_game_number") is not None else None,
            "round_transition_prev_round_key": str(st.session_state.get("cs2_round_transition_prev_round_key")) if st.session_state.get("cs2_round_transition_prev_round_key") is not None else None,
            "round_transition_new_round_key": str(st.session_state.get("cs2_round_transition_new_round_key")) if st.session_state.get("cs2_round_transition_new_round_key") is not None else None,
            "latched_band_lo": float(st.session_state["cs2_live_latched_band_lo"]) if st.session_state.get("cs2_live_latched_band_lo") is not None else None,
            "latched_p_base": float(st.session_state["cs2_live_latched_p_base"]) if st.session_state.get("cs2_live_latched_p_base") is not None else None,
            "latched_round_key": st.session_state.get("cs2_live_latched_round_key"),
            # BO3 MIDROUND V1
            # GRID / live source debug (in-memory row)
            "live_source_selected": st.session_state.get("cs2_live_source"),
            "grid_valid": st.session_state.get("cs2_grid_valid"),
            "grid_completeness_score": float(st.session_state.get("cs2_grid_completeness_score")) if st.session_state.get("cs2_grid_completeness_score") is not None else None,
            "grid_staleness_seconds": int(st.session_state.get("cs2_grid_staleness_seconds")) if st.session_state.get("cs2_grid_staleness_seconds") is not None else None,
            "grid_has_players": st.session_state.get("cs2_grid_has_players"),
            "grid_has_clock": st.session_state.get("cs2_grid_has_clock"),
            "grid_used_reduced_features": st.session_state.get("cs2_grid_used_reduced_features"),
        }
        snapshot_row["snapshot_ts_iso"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        snapshot_row["snapshot_ts_epoch_ms"] = int(time.time() * 1000)
        snapshot_row["snapshot_schema_version"] = 1
        snapshot_row["app_build_version"] = "dev"
        snapshot_row["match_id"] = str(st.session_state.get("cs2_inplay_match_id", "")).strip() or None
        snapshot_row["force_reason"] = st.session_state.get("cs2_force_snapshot_reason")
        st.session_state["cs2_live_rows"].append(snapshot_row)
        st.session_state.pop("cs2_force_snapshot_reason", None)
        if bool(st.session_state.get("cs2_inplay_persist", False)) and snapshot_row.get("match_id"):
            try:
                persist_cs2_replay_snapshot(snapshot_row)
                ml_row = derive_cs2_ml_feature_row(snapshot_row)
                persist_cs2_ml_feature_snapshot(ml_row)
            except Exception as _e_parquet:
                st.warning(f"CS2 parquet persist failed: {_e_parquet}")

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
                    "bid": float(_bid_d),
                    "ask": float(_ask_d),
                    "mid": float(_mid_d),
                    "spread_abs": float(_ask_d - _bid_d),
                    "spread_rel": float((_ask_d - _bid_d) / _mid_d) if _mid_d > 0 else 0.0,
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
        # BO3 MARKET DELAY ALIGN: use same delayed market as snapshot row
        _row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "match_id": str(st.session_state.get("cs2_inplay_match_id", "")).strip(),
            "contract_scope": str(contract_scope),
            "bid": float(_bid_d),
            "ask": float(_ask_d),
            "mid": float(_mid_d),
            "spread_abs": float(_ask_d - _bid_d),
            "p_fair": float(p_hat),
            "band_lo": float(lo),
            "band_hi": float(hi),
        }
        # BO3 MARKET DELAY ALIGN
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

    # --- Chart section (full-width, no parent columns) ---
    # Restore chart settings after GRID-triggered rerun so show p_cal, round state bands, chart window etc. stay
    _CS2_CHART_PRESERVE_KEYS = (
        "cs2_show_pcal", "cs2_show_round_state_bands", "cs2_show_state_bounds", "cs2_show_kappa_bands", "cs2_chart_window",
        "cs2_show_raw_phat_debug_line", "cs2_branch_endpoint_source_mode", "cs2_show_endpoint_numeric_debug",
    )
    if "cs2_chart_preserve" in st.session_state:
        _pres = st.session_state.pop("cs2_chart_preserve", None)
        if isinstance(_pres, dict):
            for _k in _CS2_CHART_PRESERVE_KEYS:
                if _k in _pres:
                    st.session_state[_k] = _pres[_k]

    section_header("Chart")

    # Main CS2 chart panel (full-width, wide-friendly)
    st.markdown('<div class="cs2-chart-panel">', unsafe_allow_html=True)
    if len(st.session_state["cs2_live_rows"]) > 0:
        chart_df = pd.DataFrame(st.session_state["cs2_live_rows"])

        # BO3 CHART WINDOW: rolling window for display only (do not change stored data or logging)
        _window_options = {"Last 100 snapshots": 100, "Last 250 snapshots": 250, "Last 500 snapshots": 500, "Full session": None}
        _window_opts_list = list(_window_options.keys())
        # Init key so we don't rely on index= (prevents Streamlit rerun snapback)
        if "cs2_chart_window" not in st.session_state or st.session_state.get("cs2_chart_window") not in _window_opts_list:
            st.session_state["cs2_chart_window"] = "Last 250 snapshots"

        # Row 1: chart display controls (window + overlay toggles)
        r1a, r1b, r1c, r1d, r1e, r1f, r1g = st.columns(7)
        with r1a:
            _window_label = st.selectbox(
                "Chart window",
                options=_window_opts_list,
                key="cs2_chart_window",
                help="Show a rolling window of recent snapshots to keep the chart readable during live polling.",
            )
        with r1b:
            show_pcal = st.checkbox("Show p_calibrated overlay", value=False, key="cs2_show_pcal")
        with r1c:
            show_round_state_bands = st.checkbox("Show Round State Bands", value=True, key="cs2_show_round_state_bands")
        with r1d:
            show_state_bounds = st.checkbox("Show state bounds (map resolution)", value=True, key="cs2_show_state_bounds",
                help="Upper = series prob if A wins this map; lower = if B wins.")
        with r1e:
            show_kappa_bands = st.checkbox("Show Kappa bands", value=True, key="cs2_show_kappa_bands",
                help="Show the K-band (band_lo / band_hi) fair-odds interval on the chart.")
        with r1f:
            show_raw_phat_debug = st.checkbox("Show raw p_hat (pre-bounds) debug line", value=False, key="cs2_show_raw_phat_debug_line",
                help="Faint line for p_hat before round/map clamps (diagnostic).")
        with r1g:
            show_dynamic_phat_debug = st.checkbox("Show dynamic structural p_hat debug line", value=False, key="cs2_show_dynamic_structural_phat_debug_line",
                help="Faint dotted line for diagnostic p_hat when row has data.")

        _window_size = _window_options[_window_label]
        if _window_size is not None and len(chart_df) > _window_size:
            chart_display_df = chart_df.tail(_window_size).copy()
        else:
            chart_display_df = chart_df
        _display_t_set = set(chart_display_df["t"].values)  # BO3 CHART WINDOW: filter markers to window
        pcal = load_p_calibration_json(APP_DIR)

        # TEMP DEBUG (wiring / clipping audit): show last 5 snapshot rows near the chart
        _dbg_cols = [
            "market_mid",
            "p0_map",
            "p_hat_map",
            "q_intra_round_win_a",
            "p_hat_pre_bounds",
            "p_hat",
            "state_bound_lower",
            "state_bound_upper",
        ]
        try:
            _dbg_tail = chart_display_df.reindex(columns=_dbg_cols).tail(5).copy()
            st.dataframe(_dbg_tail, use_container_width=True, height=170)
        except Exception:
            pass

        # Row 2: endpoint source / rail mode / rail pull scale & cap / debug toggle
        r2a, r2b, r2c, r2d, r2e = st.columns(5)
        with r2a:
            _branch_endpoint_opts = ["Frozen endpoints (current)", "Dynamic endpoints (debug)"]
            _branch_endpoint_current = st.session_state.get("cs2_branch_endpoint_source_mode", "Frozen endpoints (current)")
            if _branch_endpoint_current not in _branch_endpoint_opts:
                _branch_endpoint_current = "Frozen endpoints (current)"
            _branch_endpoint_index = _branch_endpoint_opts.index(_branch_endpoint_current)
            st.selectbox(
                "CS2 Branch Endpoint Source",
                options=_branch_endpoint_opts,
                index=_branch_endpoint_index,
                key="cs2_branch_endpoint_source_mode",
                help="Structural mixture endpoint source.",
            )
        with r2b:
            _rail_mode_opts = ["ANCHOR_PULLED", "RAW_COUNTERFACTUAL"]
            _rail_mode_current = st.session_state.get("cs2_rail_mode", "ANCHOR_PULLED")
            if _rail_mode_current not in _rail_mode_opts:
                _rail_mode_current = "ANCHOR_PULLED"
            st.selectbox(
                "CS2 rail mode",
                options=_rail_mode_opts,
                index=_rail_mode_opts.index(_rail_mode_current),
                key="cs2_rail_mode",
                help="ANCHOR_PULLED = pull round-state bands toward 1/0; RAW_COUNTERFACTUAL = use p_if_a/p_if_b directly.",
            )
        with r2c:
            if st.session_state.get("cs2_rail_mode", "ANCHOR_PULLED") == "ANCHOR_PULLED":
                st.number_input(
                    "Rail pull scale",
                    min_value=0.0, max_value=2.0, value=float(st.session_state.get("cs2_rail_pull_scale", 0.7)),
                    step=0.05, format="%.2f", key="cs2_rail_pull_scale",
                    help="Scale applied to steps-to-win pull before capping (default 0.7).",
                )
        with r2d:
            if st.session_state.get("cs2_rail_mode", "ANCHOR_PULLED") == "ANCHOR_PULLED":
                st.number_input(
                    "Rail pull cap",
                    min_value=0.0, max_value=1.0, value=float(st.session_state.get("cs2_rail_pull_cap", 0.6)),
                    step=0.05, format="%.2f", key="cs2_rail_pull_cap",
                    help="Max pull magnitude toward 1/0 (default 0.6).",
                )
        with r2e:
            show_endpoint_numeric_debug = st.checkbox("Show CS2 endpoint numeric debug", value=False, key="cs2_show_endpoint_numeric_debug",
                help="Display latest endpoint envelope and position diagnostics from session state (display-only).")

        # Row 3: action bar (Clear chart / Start next half / Start next map / Export snapshots)
        r3a, r3b, r3c, r3d = st.columns(4)
        with r3a:
            if st.button("Clear chart"):
                st.session_state["cs2_live_rows"] = []
        with r3b:
            st.button("Start next half", key="cs2_next_half", on_click=_cs2_start_next_half)
        with r3c:
            st.button("Start next map", key="cs2_next_map", on_click=_cs2_start_next_map)
        with r3d:
            if st.button("Export snapshots to Diagnostics"):
                st.session_state["cs2_live_export_df"] = pd.DataFrame(st.session_state["cs2_live_rows"])
    
        if show_endpoint_numeric_debug:
            def _fmt4(v):
                if v is None:
                    return None
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return round(float(v), 4)
                return v
            ss = st.session_state
            _a_center = ss.get("cs2_rail_endpoint_a_env_center")
            _b_center = ss.get("cs2_rail_endpoint_b_env_center")
            _a_active = ss.get("cs2_endpoint_pos_a_active_dbg")
            _b_active = ss.get("cs2_endpoint_pos_b_active_dbg")
            a_env_offset = (_a_active - _a_center) if _a_active is not None and _a_center is not None and np.isfinite(_a_active) and np.isfinite(_a_center) else None
            b_env_offset = (_b_active - _b_center) if _b_active is not None and _b_center is not None and np.isfinite(_b_active) and np.isfinite(_b_center) else None
            with st.expander("CS2 endpoint numeric debug (latest)", expanded=False):
                st.write("**Core**")
                st.write("branch_endpoint_source_mode:", ss.get("cs2_branch_endpoint_source_mode"))
                st.write("branch_endpoint_source_used:", ss.get("cs2_branch_endpoint_source_used"), "| branch_endpoint_source_reason:", ss.get("cs2_branch_endpoint_source_reason"))
                st.write("branch_endpoint_dynamic_available:", ss.get("cs2_branch_endpoint_dynamic_available"))
                st.write("q_intra_round_win_a:", _fmt4(ss.get("cs2_q_intra_round_win_a")), "| q_intra_round_win_a_source:", ss.get("cs2_q_intra_round_win_a_source"), "| q_intra_round_win_a_reason:", ss.get("cs2_q_intra_round_win_a_reason"))
                st.write("**Canonical debug driver bundle**")
                st.write("drv_valid_microstate:", ss.get("cs2_drv_valid_microstate"), "| drv_valid_roundstate:", ss.get("cs2_drv_valid_roundstate"), "| drv_team_a_side:", ss.get("cs2_drv_team_a_side"))
                st.write("drv_bomb_planted:", ss.get("cs2_drv_bomb_planted"), "| drv_round_phase:", ss.get("cs2_drv_round_phase"), "| drv_round_time_remaining_s:", _fmt4(ss.get("cs2_drv_round_time_remaining_s")))
                st.write("drv_alive_count_a:", ss.get("cs2_drv_alive_count_a"), "| drv_alive_count_b:", ss.get("cs2_drv_alive_count_b"))
                st.write("drv_hp_alive_total_a:", _fmt4(ss.get("cs2_drv_hp_alive_total_a")), "| drv_hp_alive_total_b:", _fmt4(ss.get("cs2_drv_hp_alive_total_b")))
                st.write("drv_alive_loadout_est_total_a:", _fmt4(ss.get("cs2_drv_alive_loadout_est_total_a")), "| drv_alive_loadout_est_total_b:", _fmt4(ss.get("cs2_drv_alive_loadout_est_total_b")))
                st.write("drv_alive_cash_total_a:", _fmt4(ss.get("cs2_drv_alive_cash_total_a")), "| drv_alive_cash_total_b:", _fmt4(ss.get("cs2_drv_alive_cash_total_b")))
                st.write("drv_source_contract:", ss.get("cs2_drv_source_contract"))
                st.write("**A/B branch envelope + active position**")
                st.write("a_env_low:", _fmt4(ss.get("cs2_rail_endpoint_a_env_low")), "| a_env_high:", _fmt4(ss.get("cs2_rail_endpoint_a_env_high")), "| a_env_center:", _fmt4(_a_center))
                st.write("a_active_dbg:", _fmt4(_a_active), "| A offset from center:", _fmt4(a_env_offset))
                st.write("b_env_low:", _fmt4(ss.get("cs2_rail_endpoint_b_env_low")), "| b_env_high:", _fmt4(ss.get("cs2_rail_endpoint_b_env_high")), "| b_env_center:", _fmt4(_b_center))
                st.write("b_active_dbg:", _fmt4(_b_active), "| B offset from center:", _fmt4(b_env_offset))
                st.write("endpoint_position_raw_diffs d_alive:", _fmt4(ss.get("cs2_endpoint_pos_d_alive")), "| d_loadout:", _fmt4(ss.get("cs2_endpoint_pos_d_loadout")))
                st.write("endpoint_pos_third_driver_source:", ss.get("cs2_endpoint_pos_third_driver_source"))
                st.write("**Branch econ consequence (debug)**")
                st.write("endpoint_econ_valid:", ss.get("cs2_endpoint_econ_valid"))
                st.write("endpoint_econ_d_alive_survival:", _fmt4(ss.get("cs2_endpoint_econ_d_alive_survival")), "| endpoint_econ_d_alive_loadout:", _fmt4(ss.get("cs2_endpoint_econ_d_alive_loadout")), "| endpoint_econ_d_alive_cash:", _fmt4(ss.get("cs2_endpoint_econ_d_alive_cash")))
                st.write("endpoint_econ_score_a:", _fmt4(ss.get("cs2_endpoint_econ_score_a")), "| endpoint_econ_score_b:", _fmt4(ss.get("cs2_endpoint_econ_score_b")))
                st.write("**PHat intraround (debug helper)**")
                st.write("phat_intra_valid:", ss.get("cs2_phat_intra_valid"), "| phat_intra_side_known:", ss.get("cs2_phat_intra_side_known"), "| phat_intra_clock_known:", ss.get("cs2_phat_intra_clock_known"))
                st.write("phat_intra_phase_mult:", _fmt4(ss.get("cs2_phat_intra_phase_mult")), "| phat_intra_urgency:", _fmt4(ss.get("cs2_phat_intra_urgency")))
                st.write("phat_intra_d_alive:", _fmt4(ss.get("cs2_phat_intra_d_alive")), "| phat_intra_d_hp:", _fmt4(ss.get("cs2_phat_intra_d_hp")), "| phat_intra_d_loadout:", _fmt4(ss.get("cs2_phat_intra_d_loadout")))
                st.write("phat_intra_loadout_presence:", _fmt4(ss.get("cs2_phat_intra_loadout_presence")), "| phat_intra_w_alive_eff:", _fmt4(ss.get("cs2_phat_intra_w_alive_eff")))
                st.write("phat_intra_bomb_term_signed:", _fmt4(ss.get("cs2_phat_intra_bomb_term_signed")), "| phat_intra_clock_term_signed:", _fmt4(ss.get("cs2_phat_intra_clock_term_signed")), "| phat_intra_bomb_clock_combo_signed:", _fmt4(ss.get("cs2_phat_intra_bomb_clock_combo_signed")))
                st.write("phat_intra_score_alive:", _fmt4(ss.get("cs2_phat_intra_score_alive")), "| score_hp:", _fmt4(ss.get("cs2_phat_intra_score_hp")), "| score_loadout:", _fmt4(ss.get("cs2_phat_intra_score_loadout")), "| score_bomb:", _fmt4(ss.get("cs2_phat_intra_score_bomb")), "| score_clock:", _fmt4(ss.get("cs2_phat_intra_score_clock")), "| score_bomb_clock_combo:", _fmt4(ss.get("cs2_phat_intra_score_bomb_clock_combo")))
                st.write("phat_intra_score_raw_post_reliability:", _fmt4(ss.get("cs2_phat_intra_score_raw_post_reliability")), "| phat_intra_q_intra_round_win_a_dbg:", _fmt4(ss.get("cs2_phat_intra_q_intra_round_win_a_dbg")))
                _ed = ss.get("cs2_rail_debug") or {}
                st.write("**Rail mode / pull (fallback)**")
                st.write("rail_mode_used:", _ed.get("rail_mode_used"), "| rail_pull_win_used:", _fmt4(_ed.get("rail_pull_win_used")), "| rail_pull_loss_used:", _fmt4(_ed.get("rail_pull_loss_used")))
                st.write("rail_steps_to_a_win:", _ed.get("rail_steps_to_a_win"), "| rail_steps_to_b_win:", _ed.get("rail_steps_to_b_win"))
                st.write("band_if_a_round_raw:", _fmt4(_ed.get("band_if_a_round_raw")), "| band_if_b_round_raw:", _fmt4(_ed.get("band_if_b_round_raw")))
                st.write("rail_terminal_override_a_used:", _ed.get("rail_terminal_override_a_used"), "| rail_terminal_override_b_used:", _ed.get("rail_terminal_override_b_used"), "| rail_terminal_if_a:", _fmt4(_ed.get("rail_terminal_if_a")), "| rail_terminal_if_b:", _fmt4(_ed.get("rail_terminal_if_b")))
                st.write("rail_terminal_smoothing_bypass_a:", _ed.get("rail_terminal_smoothing_bypass_a"), "| rail_terminal_smoothing_bypass_b:", _ed.get("rail_terminal_smoothing_bypass_b"), "| rail_terminal_endpoint_exact_preserved:", _ed.get("rail_terminal_endpoint_exact_preserved"))
                st.write("rail_band_if_a_round pre/post:", _fmt4(_ed.get("rail_band_if_a_round_pre_smooth")), "/", _fmt4(_ed.get("rail_band_if_a_round_post_smooth")), "| rail_band_if_b_round pre/post:", _fmt4(_ed.get("rail_band_if_b_round_pre_smooth")), "/", _fmt4(_ed.get("rail_band_if_b_round_post_smooth")))
    
        plot_df = chart_display_df[["t","p_hat","band_lo","band_hi","market_mid"]].copy()
        if show_pcal and pcal:
            plot_df["p_hat_cal"] = plot_df["p_hat"].apply(lambda x: apply_p_calibration(x, pcal, "cs2"))
        # BO3 STATE BANDS: add round-state band columns when available (e.g. from newer snapshots)
        if show_round_state_bands and "band_state_lo" in chart_display_df.columns and "band_state_hi" in chart_display_df.columns:
            plot_df["band_state_lo"] = chart_display_df["band_state_lo"]
            plot_df["band_state_hi"] = chart_display_df["band_state_hi"]
        # BO3 STATE BANDS
        if show_state_bounds and "state_bound_lower" in chart_display_df.columns and "state_bound_upper" in chart_display_df.columns:
            plot_df["state_bound_lower"] = chart_display_df["state_bound_lower"].astype(float).ffill().bfill()
            plot_df["state_bound_upper"] = chart_display_df["state_bound_upper"].astype(float).ffill().bfill()
        # Apply current instrument bounds only to the latest point (preserve historical bounds; new ticks store new bounds)
        _chart_slo = st.session_state.get("cs2_instrument_state_bound_lower")
        _chart_shi = st.session_state.get("cs2_instrument_state_bound_upper")
        if show_state_bounds and _chart_slo is not None and _chart_shi is not None:
            try:
                _slo_f = float(_chart_slo)
                _shi_f = float(_chart_shi)
                if np.isfinite(_slo_f) and np.isfinite(_shi_f) and len(plot_df) > 0:
                    plot_df.loc[plot_df.index[-1], "state_bound_lower"] = _slo_f
                    plot_df.loc[plot_df.index[-1], "state_bound_upper"] = _shi_f
            except (TypeError, ValueError):
                pass
        if show_raw_phat_debug and "p_hat_pre_bounds" in chart_display_df.columns:
            plot_df["p_hat_pre_bounds"] = chart_display_df["p_hat_pre_bounds"]
        if show_dynamic_phat_debug and "q_intra_round_win_a" in chart_display_df.columns and "endpoint_pos_a_active_dbg" in chart_display_df.columns and "endpoint_pos_b_active_dbg" in chart_display_df.columns:
            q = chart_display_df["q_intra_round_win_a"].fillna(np.nan)
            a = chart_display_df["endpoint_pos_a_active_dbg"].fillna(np.nan)
            b = chart_display_df["endpoint_pos_b_active_dbg"].fillna(np.nan)
            plot_df["p_hat_structural_dynamic_dbg"] = np.where(np.isfinite(q) & np.isfinite(a) & np.isfinite(b), q * a + (1.0 - q) * b, np.nan)
        plot_df = plot_df.set_index("t")
        # Use Plotly so we can overlay backtest entry/exit markers on the same chart
        try:
            import plotly.graph_objects as go
            fig_cs2 = go.Figure()
            t_vals = plot_df.index.astype(int).tolist()
            _chart_cols = ["market_mid", "p_hat"]
            if show_kappa_bands:
                _chart_cols.extend(["band_lo", "band_hi"])
            _chart_cols.extend(["band_state_lo", "band_state_hi", "state_bound_lower", "state_bound_upper"])
            for col in _chart_cols:
                if col in plot_df.columns:
                    y_list = plot_df[col].tolist()
                    # Only add state_bound traces if they have at least one finite value (avoid invisible lines)
                    if col in ("state_bound_lower", "state_bound_upper"):
                        try:
                            vals = [float(y) if y is not None else np.nan for y in y_list]
                            if not any(np.isfinite(v) for v in vals):
                                continue
                        except (TypeError, ValueError):
                            continue
                    fig_cs2.add_trace(go.Scatter(x=t_vals, y=y_list, name=col, mode="lines"))
            # BO3 STATE BANDS: band_state_lo/hi added above when show_round_state_bands and columns exist; state_bound_* = map-resolution bounds
            if show_pcal and pcal and "p_hat_cal" in plot_df.columns:
                fig_cs2.add_trace(go.Scatter(x=t_vals, y=plot_df["p_hat_cal"].tolist(), name="p_hat_cal", mode="lines"))
            if show_raw_phat_debug and "p_hat_pre_bounds" in plot_df.columns:
                fig_cs2.add_trace(go.Scatter(x=t_vals, y=plot_df["p_hat_pre_bounds"].tolist(), name="p_hat_pre_bounds (raw)", mode="lines",
                    line=dict(dash="dot", width=1, color="rgba(128,128,128,0.6)")))
            if show_dynamic_phat_debug and "p_hat_structural_dynamic_dbg" in plot_df.columns:
                fig_cs2.add_trace(go.Scatter(x=t_vals, y=plot_df["p_hat_structural_dynamic_dbg"].tolist(), name="p_hat_structural_dynamic_dbg", mode="lines",
                    line=dict(dash="dot", width=1, color="rgba(180,100,200,0.5)")))
            # Overlay backtest entry/exit for current match when we have trades and snapshot_ts for alignment
            cs2_match_id = str(st.session_state.get("cs2_inplay_match_id", "")).strip()
            bt_trades = st.session_state.get("inplay_bt_trades") or {}
            if not bt_trades and cs2_match_id:
                bt_trades = _load_inplay_trades_from_disk(PROJECT_ROOT / "logs")
                if bt_trades:
                    st.session_state["inplay_bt_trades"] = bt_trades
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
                        # BO3 CHART WINDOW: only show markers inside chart window
                        _entry_keep = [i for i in range(len(t_entry)) if t_entry[i] in _display_t_set]
                        t_entry = [t_entry[i] for i in _entry_keep]
                        y_entry = [y_entry[i] for i in _entry_keep]
                        _exit_keep = [i for i in range(len(t_exit)) if t_exit[i] in _display_t_set]
                        t_exit = [t_exit[i] for i in _exit_keep]
                        y_exit = [y_exit[i] for i in _exit_keep]
                        exit_reasons = [exit_reasons[i] for i in _exit_keep]
                        exit_pnls = [exit_pnls[i] for i in _exit_keep]
                        # BO3 CHART WINDOW
                        if t_entry:
                            sides = match_trades["side"]
                            long_m = (sides == "LONG").tolist()
                            short_m = (sides == "SHORT").tolist()
                            long_m = [long_m[i] for i in _entry_keep if i < len(long_m)]
                            short_m = [short_m[i] for i in _entry_keep if i < len(short_m)]
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
                            # BO3 CHART WINDOW: only show open marker if inside chart window
                            if t_open in _display_t_set:
                                if side == "LONG":
                                    fig_cs2.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry LONG (open)", mode="markers", marker=dict(symbol=style["sym_l"], size=12, color=style["color"])))
                                elif side == "SHORT":
                                    fig_cs2.add_trace(go.Scatter(x=[t_open], y=[y_open], name=f"{sid} Entry SHORT (open)", mode="markers", marker=dict(symbol=style["sym_s"], size=12, color=style["color"])))
                            # BO3 CHART WINDOW
            _rev = (str(st.session_state.get("cs2_inplay_match_id", "")).strip() or "cs2_live_chart_v1")
            fig_cs2.update_layout(
                title="Chart",
                xaxis_title="t (snapshot index)",
                yaxis_title="Price / Fair",
                height=600,
                autosize=True,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0.0,
                ),
                uirevision=_rev,
            )
            st.plotly_chart(
                fig_cs2,
                use_container_width=True,
                key="cs2_chart_plotly",
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "showTips": True,
                },
            )
        except ImportError:
            st.line_chart(plot_df, use_container_width=True, height=840)
        except Exception:
            st.line_chart(plot_df, use_container_width=True, height=840)
    
        # Snapshot table (keep, but collapse by default under the chart)
        with st.expander("Snapshots (table)", expanded=False):
            st.dataframe(chart_df, use_container_width=True)
    
        st.markdown("---")
        st.caption("CS2 Raw Live Payload Debug")
        cs2_show_raw_payload_debug_stream = st.checkbox("Show CS2 raw payload debug stream", value=False, key="cs2_show_raw_payload_debug_stream",
            help="Show rolling stream of full BO3 live payload snapshots and focused summary per tick.")
        if cs2_show_raw_payload_debug_stream:
            _debug_rows = st.session_state.get("cs2_live_raw_payload_debug_rows") or []
            with st.expander("CS2 Raw Live Payload Debug Stream", expanded=False):
                if not _debug_rows:
                    st.caption("No debug rows yet. Enable BO3 auto and ensure feed is updating.")
                else:
                    # Compact table: summary columns, newest first
                    _summary_keys = [
                        "debug_index", "ts_local",
                        "cs2_live_game_number", "cs2_live_round_number", "cs2_live_round_phase",
                        "cs2_live_round_time_remaining_s", "cs2_live_bomb_planted",
                        "cs2_live_team_a_alive_count", "cs2_live_team_b_alive_count",
                        "cs2_live_team_a_alive_loadout_total", "cs2_live_team_b_alive_loadout_total",
                        "cs2_live_team_a_armor_alive_total", "cs2_live_team_b_armor_alive_total",
                        "cs2_live_intraround_state_valid", "cs2_live_intraround_state_source",
                    ]
                    _table_data = []
                    for _r in reversed(_debug_rows):
                        _table_data.append({k: _r.get(k) for k in _summary_keys})
                    _table_df = pd.DataFrame(_table_data)
                    st.dataframe(_table_df, use_container_width=True)
                    # Selectbox: choose row by debug index / timestamp
                    _options = [f"{_r.get('debug_index')} — {_r.get('ts_local', '')}" for _r in reversed(_debug_rows)]
                    _sel_idx = st.selectbox("Select row (debug index — timestamp)", range(len(_options)), format_func=lambda i: _options[i], key="cs2_raw_payload_debug_sel")
                    _chosen = _debug_rows[-(_sel_idx + 1)] if 0 <= _sel_idx < len(_debug_rows) else None
                    if _chosen is not None:
                        st.json(_chosen)
                        st.code(json.dumps(_chosen, indent=2, default=str), language="json")
    else:
        st.info("Add at least one snapshot to see the chart.")

    st.markdown("</div>", unsafe_allow_html=True)

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

    section_header("Chart")
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
            if not bt_trades and val_match_id:
                bt_trades = _load_inplay_trades_from_disk(PROJECT_ROOT / "logs")
                if bt_trades:
                    st.session_state["inplay_bt_trades"] = bt_trades
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
            fig_val.update_layout(
                title="Chart", xaxis_title="t (snapshot index)", yaxis_title="Price / Fair",
                height=840, width=1400, autosize=False,
            )
            st.plotly_chart(
                fig_val,
                use_container_width=False,
                key="val_chart_plotly",
                config={
                    "responsive": False,
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "showTips": True,
                },
            )
        except ImportError:
            st.line_chart(plot_df.set_index("t"), use_container_width=True, height=840)
        except Exception:
            st.line_chart(plot_df.set_index("t"), use_container_width=True, height=840)
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
            fig.update_layout(
                title=f"Match {sel_match} — {sel_strategy}", xaxis_title="Time", yaxis_title="Price / Fair",
                height=840, width=1400, autosize=False,
            )
            st.plotly_chart(
                fig,
                use_container_width=False,
                key="inplay_bt_chart_plotly",
                config={
                    "responsive": False,
                    "displayModeBar": True,
                    "scrollZoom": True,
                    "doubleClick": "reset",
                    "showTips": True,
                },
            )
        except ImportError:
            st.warning("Install plotly to see the chart: pip install plotly")
        except Exception as e:
            st.warning(f"Chart error: {e}")
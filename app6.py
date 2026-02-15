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

import streamlit as st
import pandas as pd
import requests

# ---- Windows asyncio fix: ensure subprocess support for Playwright ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ==========================
# Helpers (strong normalization + fuzzy match)
# ==========================
_ZWS = "\u200b\u200c\u200d"

def _clean_spaces(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").replace(_ZWS, "")
    return re.sub(r"\s+", " ", s).strip()

def normalize_name(s: str) -> str:
    """Robust canonicalization: NFKC, lowercase, drop 'team ', strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace("team ", " ").replace("_", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    return _clean_spaces(s)

def looks_like(a: str, b: str) -> bool:
    """Loose match: token containment OR ~0.72 similarity."""
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta.issubset(tb) or tb.issubset(ta)):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.72

def gosu_name_from_slug(slug: str) -> str:
    """
    Convert a Gosu team slug to the display-ish name used on match rows.
    e.g., '14009-team-spirit' -> 'team spirit'
    """
    s = re.sub(r"^\d+-", "", str(slug or ""))
    return s.replace("-", " ").strip()

# ==========================
# Recency weighting (piecewise, sum-preserving)
# ==========================
def piecewise_recent_weights(n: int, K: int = 6, decay: float = 0.85, floor: float = 0.6, newest_first: bool = True):
    """
    Last K matches get weight=1.0. Older matches decay by 'decay' per step beyond K,
    floored at 'floor'. Then rescale so the sum of weights ~= n (mean weight ≈ 1),
    preserving the overall score scale while emphasizing recency.
    """
    if n <= 0:
        return []
    idx = range(n) if newest_first else range(n - 1, -1, -1)
    raw = []
    for i in idx:
        if i < K:
            w = 1.0
        else:
            steps = i - K + 1
            w = max(floor, decay ** steps)
        raw.append(w)
    s = sum(raw) or 1.0
    factor = n / s
    return [w * factor for w in raw]

# ==========================
# Tier lookup (Dota-only extras)
# ==========================
def get_team_tier(opp: str, df: pd.DataFrame) -> float:
    """
    Works for both CS2 and Dota:
    - CS2: matches on norm_team only (unchanged behavior).
    - Dota: if norm_gosu exists, we also match on that.
    """
    target = normalize_name(opp)
    df_local = df.copy()

    # Always ensure norm_team exists
    if "norm_team" not in df_local.columns:
        df_local["norm_team"] = df_local["team"].apply(normalize_name)

    # Candidate columns depend on what's available (Dota adds norm_gosu)
    candidate_cols = ["norm_team"]
    if "norm_gosu" in df_local.columns:
        candidate_cols.append("norm_gosu")

    # Exact first
    try:
        exact_mask = (df_local[candidate_cols] == target).any(axis=1)
        exact = df_local.loc[exact_mask]
        if not exact.empty:
            return float(exact.iloc[0]["tier"])
    except Exception:
        pass

    # Fuzzy match over whichever columns we have
    best_idx, best_score = None, 0.0
    for idx, row in df_local[candidate_cols].iterrows():
        for cand in row.values:
            cand = str(cand)
            if looks_like(target, cand):
                score = SequenceMatcher(None, target, cand).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = idx

    if best_idx is not None:
        return float(df_local.loc[best_idx, "tier"])

    st.warning(f"No match for '{opp}', defaulting to Tier 5")
    return 5.0

# ==========================
# Load CSVs
# ==========================
@st.cache_data
def load_cs2_teams() -> pd.DataFrame:
    # Must include: team, tier, rank, hltv_id, slug
    df = pd.read_csv("cs2_rankings_merged.csv")
    df["norm_team"] = df["team"].apply(normalize_name)
    return df

@st.cache_data
def load_dota_teams() -> pd.DataFrame:
    # Must include: team, tier, rank, slug, opendota_id
    df = pd.read_csv("dota2_gosu_rankings.csv")
    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: x.strip().replace(" ", "_"))
    df["gosu_display"] = df["slug"].apply(gosu_name_from_slug)
    df["norm_team"] = df["team"].apply(normalize_name)
    df["norm_gosu"] = df["gosu_display"].apply(normalize_name)
    return df

# ==========================
# External Scrapers
# ==========================
def scrape_cs2_matches(team_id: str, team_slug: str):
    """Run scraper.py as a subprocess and parse the last JSON line (working CS2 path)."""
    try:
        python_exe = sys.executable
        app_dir = Path(__file__).resolve().parent
        script_path = str(app_dir / "scraper.py")

        result = subprocess.run(
            [python_exe, script_path, str(team_id), team_slug],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(app_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[-1] if result.stdout else "[]"
            return json.loads(line)
        else:
            st.error("CS2 scraper failed")
            with st.expander("CS2 scraper stderr"):
                st.code(result.stderr or "(no stderr)")
            with st.expander("CS2 scraper stdout"):
                st.code(result.stdout or "(no stdout)")
            return []
    except Exception as e:
        st.error(f"CS2 scraper error: {e}")
        return []

def scrape_dota_matches_gosu_subprocess(team_slug: str, team_name: str, target: int = 14,
                                        headed: bool = True, browser_channel: str = "bundled",
                                        zoom: int = 80):
    """
    Run gosu_dota_scraper.py as a subprocess and parse the last JSON array from stdout.
    """
    try:
        python_exe = sys.executable
        app_dir = Path(__file__).resolve().parent
        script_path = str(app_dir / "gosu_dota_scraper.py")

        cmd = [
            python_exe, script_path,
            "--team-slug", team_slug,
            "--team-name", team_name,
            "--target", str(target),
            "--zoom", str(zoom),
        ]
        if headed:
            cmd.append("--headed")
        if browser_channel in ("chrome", "msedge"):
            cmd += ["--channel", browser_channel]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(app_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            st.error(f"Gosu scraper failed (exit {result.returncode}).")
            with st.expander("Gosu scraper stderr"):
                st.code(result.stderr or "(no stderr)")
            with st.expander("Gosu scraper stdout (tail)"):
                st.code((result.stdout or "")[-2000:])
            return []

        stdout = result.stdout or ""

        # Extract last JSON array from stdout (robust to extra prints)
        start = stdout.rfind("[")
        end = stdout.rfind("]")
        if start == -1 or end == -1 or end < start:
            st.warning("Gosu scraper returned no parseable JSON.")
            with st.expander("Gosu scraper raw stdout"):
                st.code(stdout[-4000:] if stdout else "(empty)")
            return []

        data = json.loads(stdout[start:end+1])

        out = []
        for row in data:
            opp = row.get("opponent", "Unknown")
            win = bool(row.get("win", False))
            out.append({"opponent": opp, "win": win})
        return out

    except subprocess.TimeoutExpired:
        st.error("Gosu scraper timed out. Try again or reduce match count.")
        return []
    except Exception as e:
        st.error(f"Gosu scraper error: {e}")
        return []

# OpenDota (optional)
def fetch_dota_matches(team_id: int, limit: int = 15):
    url = f"https://api.opendota.com/api/teams/{team_id}/matches"
    try:
        response = requests.get(url, timeout=20)
        if response.status_code != 200:
            st.error(f"OpenDota API error for team {team_id}: {response.status_code}")
            return []
        data = response.json()
        matches = []
        for m in data[:limit]:
            opp_name = m.get("opposing_team_name", "Unknown")
            radiant = m.get("radiant", None)
            radiant_win = m.get("radiant_win", None)
            if radiant is None or radiant_win is None:
                continue
            win = (radiant and radiant_win) or (not radiant and not radiant_win)
            matches.append({"opponent": opp_name, "win": win})
        return matches
    except Exception as e:
        st.error(f"Error fetching OpenDota data: {e}")
        return []

# ==========================
# Scoring Logic with Tier-Based + Recency Adjustments
# ==========================
def calculate_score(matches,
                    df: pd.DataFrame,
                    current_opponent_tier=None,
                    weight_scheme: str = "piecewise",
                    K: int = 6,
                    decay: float = 0.85,
                    floor: float = 0.6,
                    newest_first: bool = True):
    """
    weight_scheme:
      - "piecewise": last K at full weight, older decay to 'floor', sum-preserving
      - anything else -> flat weights (no recency)
    """
    raw_score = 0.0
    adjusted_score = 0.0
    breakdown = []

    n = len(matches)
    if weight_scheme == "piecewise":
        weights = piecewise_recent_weights(n, K=K, decay=decay, floor=floor, newest_first=newest_first)
    else:
        weights = [1.0] * n

    for i, match in enumerate(matches):
        opp = match["opponent"]
        win = match["win"]
        tier = get_team_tier(opp, df)

        # Base tier points (unchanged)
        if win:
            if tier == 1: points = 4
            elif tier == 1.5: points = 3
            elif tier == 2: points = 2.5
            elif tier == 3: points = 2
            elif tier == 4: points = 1.5
            else: points = 1
        else:
            if tier == 1: points = -1
            elif tier == 1.5: points = -1.5
            elif tier == 2: points = -2
            elif tier == 3: points = -2.5
            elif tier == 4: points = -3
            else: points = -4

        raw_score += points  # flat sum for audit

        # ---- Tier-gap weighting (COMPRESSED to ~0.85–1.15) ----
        # Compute the old asymmetric weight (same logic as before)
        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier
            if win:
                if tier_gap < 0:
                    weight_tier_old = 1 + min(0.4, abs(tier_gap) * 0.2)
                else:
                    weight_tier_old = 1 - min(0.3, tier_gap * 0.15)
            else:
                if tier_gap < 0:
                    weight_tier_old = 1 - min(0.2, abs(tier_gap) * 0.1)
                else:
                    weight_tier_old = 1 + min(0.5, tier_gap * 0.25)
            weight_tier_old = max(0.5, min(weight_tier_old, 1.5))
        else:
            tier_gap = 0.0
            weight_tier_old = 1.0

        # Compress toward 1 so the final band is ~0.85–1.15
        # 0.5 away from 1 becomes 0.15 (scale=0.3). Then hard-clip to [0.85, 1.15].
        weight_tier = 1 + (weight_tier_old - 1) * 0.3
        weight_tier = max(0.85, min(weight_tier, 1.15))

        # Apply recency weight on top of tier-gap weight
        w_match = weights[i]
        adj_points = points * weight_tier * w_match
        adjusted_score += adj_points

        # Per-match formula line (explicit factors)
        breakdown.append(
            f"{'Win' if win else 'Loss'} vs {opp} (OppTier {tier}, Gap {tier_gap:.1f}) "
            f"Pts={points:+.2f} × TierW={weight_tier:.2f} × RecW={w_match:.3f} = {adj_points:+.3f}"
        )

    return raw_score, adjusted_score, breakdown

# ==========================
# Odds helpers + mapping
# ==========================
def american_to_decimal(odds: int) -> float:
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

def implied_prob_from_american(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_fair_odds_curve(gap: float,
                              base_C=30, alpha=0.03,
                              tail_cutoff=14,
                              L=0.97, k=0.09, x0=0, v=1.0):
    """Original hand-tuned mapping from gap -> p."""
    if abs(gap) <= tail_cutoff:
        C_dynamic = base_C / (1 + alpha * abs(gap))
        p_a = 1 / (1 + 10 ** (-gap / C_dynamic))
    else:
        p_a = L / ((1 + math.exp(-k * (gap - x0))) ** v)
        p_b = L - p_a
        total = p_a + p_b
        p_a /= total
    return p_a

def calculate_fair_odds_from_p(p_a: float):
    p_b = 1 - p_a
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)
    return fair_a, fair_b

def logistic_mapping(adj_gap: float, a: float, b: float) -> float:
    """p = 1 / (1 + exp(-(a + b*gap)))"""
    return 1.0 / (1.0 + math.exp(-(a + b * adj_gap)))

def color_ev(ev: float) -> str:
    if ev >= 40:
        return f"✅ **{round(ev, 2)}%**"
    elif 11 <= ev < 40:
        return f"🟡 {round(ev, 2)}%"
    else:
        return f"🔴 {round(ev, 2)}%"

# ==========================
# Session diagnostics & logging
# ==========================
def init_metrics():
    st.session_state.setdefault("metrics", {"total": 0, "fav_value": 0, "dog_value": 0, "no_bet": 0})
    st.session_state.setdefault("logs", [])

def update_metrics(ev_a: float, ev_b: float, odds_a: int, odds_b: int):
    m = st.session_state["metrics"]
    m["total"] += 1
    imp_a = implied_prob_from_american(odds_a)
    imp_b = implied_prob_from_american(odds_b)
    market_fav = "A" if imp_a >= imp_b else "B"
    if ev_a > 0 or ev_b > 0:
        if ev_a >= ev_b:
            pick = "A"; ev_pick = ev_a
        else:
            pick = "B"; ev_pick = ev_b
        if ev_pick <= 0:
            m["no_bet"] += 1
        else:
            if pick == market_fav: m["fav_value"] += 1
            else: m["dog_value"] += 1
    else:
        m["no_bet"] += 1
    st.session_state["metrics"] = m

def log_row(entry: dict):
    st.session_state["logs"].append(entry)

def export_logs_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("logs", []))

# ==========================
# Streamlit App
# ==========================
st.title("Esports Fair Odds Calculator (CS2 + Dota2)")
init_metrics()

# Global calibration controls (applies to both tabs)
with st.expander("Calibration & Mapping (optional)"):
    use_clip = st.checkbox("Clip extreme adjusted gaps before mapping", value=False,
                           help="Prevents absurd probabilities on outlier gaps.")
    clip_limit = st.slider("Clip limit (|gap|)", 10, 40, 25, 1)
    use_logistic = st.checkbox("Use calibrated logistic mapping (gap → p)", value=False,
                               help="OFF = use the original curve. ON = use p = 1/(1+exp(-(a+b*gap))).")
    colA, colB = st.columns(2)
    with colA:
        a_param = st.number_input("Logistic a (intercept)", value=0.0, step=0.01, format="%.4f")
    with colB:
        b_param = st.number_input("Logistic b (slope)", value=0.18, step=0.01, format="%.4f",
                                  help="Fit this from your logged data later. Placeholder default.")

tabs = st.tabs(["CS2", "Dota2", "Diagnostics / Export"])

# --------------------------
# CS2 TAB
# --------------------------
with tabs[0]:
    st.header("CS2 Fair Odds")
    df_cs2 = load_cs2_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (CS2)", df_cs2["team"].tolist(), key="cs2_a")
        team_a_id = str(int(df_cs2.loc[df_cs2["team"] == team_a, "hltv_id"].values[0]))
        team_a_slug = df_cs2.loc[df_cs2["team"] == team_a, "slug"].values[0]
        if st.button("Scrape Team A (CS2)"):
            matches_a = scrape_cs2_matches(team_a_id, team_a_slug)
            st.session_state["cs2_matches_a"] = matches_a
        matches_a = st.session_state.get("cs2_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (CS2)", df_cs2["team"].tolist(), key="cs2_b")
        team_b_id = str(int(df_cs2.loc[df_cs2["team"] == team_b, "hltv_id"].values[0]))
        team_b_slug = df_cs2.loc[df_cs2["team"] == team_b, "slug"].values[0]
        if st.button("Scrape Team B (CS2)"):
            matches_b = scrape_cs2_matches(team_b_id, team_b_slug)
            st.session_state["cs2_matches_b"] = matches_b
        matches_b = st.session_state.get("cs2_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (CS2)", value=-140, key="cs2_odds_a")
    odds_b = st.number_input("Team B Market Odds (CS2)", value=+120, key="cs2_odds_b")

    # Recency controls (piecewise)
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

            team_a_tier = float(df_cs2.loc[df_cs2["team"] == team_a, "tier"].values[0])
            team_b_tier = float(df_cs2.loc[df_cs2["team"] == team_b, "tier"].values[0])

            raw_a, adj_a, breakdown_a = calculate_score(
                matches_a, df_cs2, current_opponent_tier=team_b_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True
            )
            raw_b, adj_b, breakdown_b = calculate_score(
                matches_b, df_cs2, current_opponent_tier=team_a_tier,
                weight_scheme="piecewise", K=K_cs2, decay=decay_cs2, floor=floor_cs2, newest_first=True
            )

            raw_gap = raw_a - raw_b
            adj_gap = adj_a - adj_b
            if use_clip:
                adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

            # Probability mapping
            if use_logistic:
                p = logistic_mapping(adj_gap, a_param, b_param)
            else:
                p = calculate_fair_odds_curve(adj_gap)

            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100
            fair_a, fair_b = calculate_fair_odds_from_p(p)

            st.subheader("Summary (CS2)")
            st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
            st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a:
                st.text(line)

            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b:
                st.text(line)

            # Diagnostics & logging
            update_metrics(ev_a, ev_b, odds_a, odds_b)
            log_row({
                "timestamp": datetime.utcnow().isoformat(),
                "game": "CS2",
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
                "ev_a_pct": ev_a, "ev_b_pct": ev_b
            })

        else:
            st.warning("Scrape both teams first.")

# --------------------------
# Dota TAB
# --------------------------
with tabs[1]:
    st.header("Dota 2 Fair Odds")
    df_dota = load_dota_teams()

    src = st.radio("Data source", ["OpenDota (API)", "GosuGamers (Scrape)"], horizontal=True, index=1)
    headed_toggle = st.toggle("Show browser during Gosu scrape", value=True,
                              help="Use a visible browser window for the scrape.")
    browser_channel = st.selectbox(
        "Browser for Gosu scrape",
        options=["bundled", "chrome", "msedge"],
        index=0,
        key="dota_browser_channel",
        help="Use 'bundled' to mimic CLI (Playwright Chromium). Or force Chrome/Edge if installed.",
    )
    zoom_pct = st.slider("Gosu page zoom (%)", min_value=60, max_value=110, value=80, step=5,
                         help="Zoom out a bit to keep the bottom paginator in view.")
    target_matches = st.slider("Matches to use (last N)", min_value=8, max_value=30, value=14, step=1)

    # Recency controls (piecewise)
    st.subheader("Recency Weighting (Dota2)")
    K_dota = st.slider("Full-weight recent matches (K)", 3, 12, 6, key="K_dota")
    decay_dota = st.slider("Decay per step beyond K", 0.75, 0.95, 0.85, 0.01, key="decay_dota")
    floor_dota = st.slider("Minimum weight floor", 0.40, 0.90, 0.60, 0.01, key="floor_dota")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Dota2)", df_dota["team"].tolist(), key="dota_a")
        team_a_id = int(df_dota.loc[df_dota["team"] == team_a, "opendota_id"].values[0])
        team_a_slug = df_dota.loc[df_dota["team"] == team_a, "slug"].values[0]
        if st.button("Scrape Team A (Dota2)"):
            if src.startswith("OpenDota"):
                matches_a = fetch_dota_matches(team_a_id, limit=target_matches)
            else:
                gosu_name_a = gosu_name_from_slug(team_a_slug)
                matches_a = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_a_slug,
                    team_name=gosu_name_a,
                    target=target_matches,
                    headed=headed_toggle,
                    browser_channel=browser_channel,
                    zoom=zoom_pct,
                )
            st.session_state["dota_matches_a"] = matches_a
        matches_a = st.session_state.get("dota_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (Dota2)", df_dota["team"].tolist(), key="dota_b")
        team_b_id = int(df_dota.loc[df_dota["team"] == team_b, "opendota_id"].values[0])
        team_b_slug = df_dota.loc[df_dota["team"] == team_b, "slug"].values[0]
        if st.button("Scrape Team B (Dota2)"):
            if src.startswith("OpenDota"):
                matches_b = fetch_dota_matches(team_b_id, limit=target_matches)
            else:
                gosu_name_b = gosu_name_from_slug(team_b_slug)
                matches_b = scrape_dota_matches_gosu_subprocess(
                    team_slug=team_b_slug,
                    team_name=gosu_name_b,
                    target=target_matches,
                    headed=headed_toggle,
                    browser_channel=browser_channel,
                    zoom=zoom_pct,
                )
            st.session_state["dota_matches_b"] = matches_b
        matches_b = st.session_state.get("dota_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (Dota2)", value=-140, key="dota_odds_a")
    odds_b = st.number_input("Team B Market Odds (Dota2)", value=+120, key="dota_odds_b")

    if st.button("Calculate (Dota2)"):
        if "dota_matches_a" in st.session_state and "dota_matches_b" in st.session_state:
            matches_a = st.session_state["dota_matches_a"][:target_matches]
            matches_b = st.session_state["dota_matches_b"][:target_matches]
            if not matches_a or not matches_b:
                st.warning("Scrape both teams first (Dota2).")
            else:
                st.text(f"Using last {min(len(matches_a), len(matches_b))} matches for both teams.")

                team_a_tier = float(df_dota.loc[df_dota["team"] == team_a, "tier"].values[0])
                team_b_tier = float(df_dota.loc[df_dota["team"] == team_b, "tier"].values[0])

                raw_a, adj_a, breakdown_a = calculate_score(
                    matches_a, df_dota, current_opponent_tier=team_b_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True
                )
                raw_b, adj_b, breakdown_b = calculate_score(
                    matches_b, df_dota, current_opponent_tier=team_a_tier,
                    weight_scheme="piecewise", K=K_dota, decay=decay_dota, floor=floor_dota, newest_first=True
                )

                raw_gap = raw_a - raw_b
                adj_gap = adj_a - adj_b
                if use_clip:
                    adj_gap = max(min(adj_gap, clip_limit), -clip_limit)

                if use_logistic:
                    p = logistic_mapping(adj_gap, a_param, b_param)
                else:
                    p = calculate_fair_odds_curve(adj_gap)

                dec_a = american_to_decimal(odds_a)
                dec_b = american_to_decimal(odds_b)
                ev_a = ((p * dec_a) - 1) * 100
                ev_b = (((1 - p) * dec_b) - 1) * 100
                fair_a, fair_b = calculate_fair_odds_from_p(p)

                st.subheader("Summary (Dota2)")
                st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
                st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
                st.text(f"Market Odds: {odds_a} / {odds_b}")
                st.text(f"Fair Odds:   {fair_a} / {fair_b}")
                st.text(f"Win Probability: {round(p * 100, 2)}%")
                st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
                st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.3f}/{adj_b:.3f}")

                st.subheader(f"{team_a} Breakdown")
                for line in breakdown_a:
                    st.text(line)

                st.subheader(f"{team_b} Breakdown")
                for line in breakdown_b:
                    st.text(line)

                update_metrics(ev_a, ev_b, odds_a, odds_b)
                log_row({
                    "timestamp": datetime.utcnow().isoformat(),
                    "game": "Dota2",
                    "team_a": team_a, "team_b": team_b,
                    "tier_a": team_a_tier, "tier_b": team_b_tier,
                    "K": K_dota, "decay": decay_dota, "floor": floor_dota,
                    "raw_a": raw_a, "raw_b": raw_b,
                    "adj_a": adj_a, "adj_b": adj_b, "adj_gap": adj_gap,
                    "use_logistic": use_logistic, "a": a_param, "b": b_param,
                    "use_clip": use_clip, "clip_limit": clip_limit,
                    "p_model": p,
                    "odds_a": odds_a, "odds_b": odds_b,
                    "fair_a": fair_a, "fair_b": fair_b,
                    "ev_a_pct": ev_a, "ev_b_pct": ev_b
                })
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
    csv = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "fair_odds_logs.csv", "text/csv")

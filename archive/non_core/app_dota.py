import math
import pandas as pd
import streamlit as st
from difflib import get_close_matches

from liquipedia_scrape import fetch_recent_matches_liquipedia

# ---------- Helpers ----------
def normalize_name(s: str) -> str:
    s = s.lower().strip().replace("_", " ")
    s = " ".join(s.split())
    return s

def load_dota_teams():
    df = pd.read_csv("dota2_full_team_list_with_slugs.csv")
    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: x.strip().replace(" ", "_"))
    if "norm_team" not in df.columns:
        df["norm_team"] = df["team"].apply(normalize_name)
    return df

def get_team_tier(opp, df, seen_defaults=None):
    norm_opp = normalize_name(opp)
    row = df.loc[df["norm_team"] == norm_opp]
    if not row.empty:
        return float(row["tier"].values[0])
    match = get_close_matches(norm_opp, df["norm_team"].values, n=1, cutoff=0.90)
    if match:
        return float(df.loc[df["norm_team"] == match[0], "tier"].values[0])
    if seen_defaults is not None and norm_opp not in seen_defaults:
        st.warning(f"No tier match for '{opp}' — defaulting to Tier 5")
        seen_defaults.add(norm_opp)
    return 5.0

# ---------- EMA Scoring ----------
def calculate_score(matches, df, current_opponent_tier=None, tier_defaults=None, ema_span=None):
    alpha = 2 / (ema_span + 1) if (ema_span and ema_span > 1) else None
    weights = []
    ema_coef_prev = 0.0
    for _ in matches:  # newest first
        if alpha:
            ema_coef_prev = alpha * 1.0 + (1 - alpha) * ema_coef_prev
            weights.append(ema_coef_prev)
        else:
            weights.append(1.0)
    W = sum(weights) if weights else 1.0
    weights = [w / W for w in weights]

    raw_score = 0.0
    adjusted_score = 0.0
    breakdown = []

    for (match, w) in zip(matches, weights):
        opp = match["opponent"]
        win = match["win"]
        tier = get_team_tier(opp, df, seen_defaults=tier_defaults)

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

        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier
            if win:
                weight_gap = 1 + min(0.4, abs(tier_gap) * 0.2) if tier_gap < 0 else 1 - min(0.3, tier_gap * 0.15)
            else:
                weight_gap = 1 - min(0.2, abs(tier_gap) * 0.1) if tier_gap < 0 else 1 + min(0.5, tier_gap * 0.25)
            weight_gap = max(0.5, min(weight_gap, 1.5))
        else:
            tier_gap = 0.0
            weight_gap = 1.0

        raw_score += points * w
        adj_pts = points * weight_gap * w
        adjusted_score += adj_pts

        breakdown.append(
            f"{'Win' if win else 'Loss'} vs {opp} "
            f"(Tier {tier}, Gap {tier_gap:.1f}, Wgap {weight_gap:.2f}, EMAw {w:.3f}) "
            f"→ {points:+} → Adj {adj_pts:+.2f}"
        )

    return raw_score, adjusted_score, breakdown

# ---------- Odds ----------
def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

def color_ev(ev):
    if ev >= 40:
        return f"✅ {round(ev, 2)}%"
    elif 11 <= ev < 40:
        return f"🟡 {round(ev, 2)}%"
    else:
        return f"🔴 {round(ev, 2)}%"

def calculate_fair_odds(gap,
                        base_C=30, alpha=0.03,
                        tail_cutoff=14,
                        L=0.97, k=0.09, x0=0, v=1.0):
    if abs(gap) <= tail_cutoff:
        C_dynamic = base_C / (1 + alpha * abs(gap))
        p_a = 1 / (1 + 10 ** (-gap / C_dynamic))
    else:
        p_a = L / ((1 + math.exp(-k * (gap - x0))) ** v)
        p_b = L - p_a
        total = p_a + p_b
        p_a /= total
    p_b = 1 - p_a

    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)

    return p_a, fair_a, fair_b

# ---------- App ----------
st.title("Dota 2 Fair Odds — Liquipedia Results Scraper (No Series Clicks) + EMA")

df_dota = load_dota_teams()

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Team A", df_dota["team"].tolist(), key="dota_a")
    limit_a = st.number_input("Matches to fetch (A)", value=15, min_value=3, max_value=50, step=1)
    debug_a = st.checkbox("Debug A (prints to terminal)", value=False)
    if st.button("Scrape Team A"):
        try:
            matches_a = fetch_recent_matches_liquipedia(team_a, limit=int(limit_a), debug=debug_a)
            st.session_state["dota_matches_a"] = matches_a
        except Exception as e:
            st.error(f"Scrape A failed: {e}")
    matches_a = st.session_state.get("dota_matches_a", [])
    st.write(matches_a)

with col2:
    team_b = st.selectbox("Team B", df_dota["team"].tolist(), key="dota_b")
    limit_b = st.number_input("Matches to fetch (B)", value=15, min_value=3, max_value=50, step=1)
    debug_b = st.checkbox("Debug B (prints to terminal)", value=False)
    if st.button("Scrape Team B"):
        try:
            matches_b = fetch_recent_matches_liquipedia(team_b, limit=int(limit_b), debug=debug_b)
            st.session_state["dota_matches_b"] = matches_b
        except Exception as e:
            st.error(f"Scrape B failed: {e}")
    matches_b = st.session_state.get("dota_matches_b", [])
    st.write(matches_b)

st.divider()
ema_on = st.toggle("Use EMA (recency weighting)", value=True)
ema_span = st.number_input("EMA span (matches)", value=12, min_value=5, max_value=30, step=1) if ema_on else None

odds_a = st.number_input("Team A Market Odds", value=-140, key="odds_a")
odds_b = st.number_input("Team B Market Odds", value=120, key="odds_b")

if st.button("Calculate"):
    if matches_a and matches_b:
        n = min(len(matches_a), len(matches_b))
        matches_a = matches_a[:n]
        matches_b = matches_b[:n]
        st.text(f"Using last {n} matches for both teams (most recent first).")

        team_a_tier = float(df_dota[df_dota["team"] == team_a]["tier"].values[0])
        team_b_tier = float(df_dota[df_dota["team"] == team_b]["tier"].values[0])

        tier_defaults = st.session_state.setdefault("dota_tier_defaults", set())

        raw_a, adj_a, breakdown_a = calculate_score(
            matches_a, df_dota, current_opponent_tier=team_b_tier,
            tier_defaults=tier_defaults, ema_span=(ema_span if ema_on else None)
        )
        raw_b, adj_b, breakdown_b = calculate_score(
            matches_b, df_dota, current_opponent_tier=team_a_tier,
            tier_defaults=tier_defaults, ema_span=(ema_span if ema_on else None)
        )

        raw_gap = raw_a - raw_b
        adj_gap = adj_a - adj_b

        p, fairA, fairB = calculate_fair_odds(adj_gap)
        dec_a = american_to_decimal(odds_a)
        dec_b = american_to_decimal(odds_b)
        ev_a = ((p * dec_a) - 1) * 100
        ev_b = (((1 - p) * dec_b) - 1) * 100

        st.subheader("Summary")
        st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
        st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
        st.text(f"Market Odds: {odds_a} / {odds_b}")
        st.text(f"Fair Odds:   {fairA} / {fairB}")
        st.text(f"Win Probability: {round(p * 100, 2)}%")
        st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")

        if st.checkbox("Show breakdowns", value=True):
            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a: st.text(line)
            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b: st.text(line)

        if tier_defaults:
            st.info("Defaulted to Tier 5 for: " + ", ".join(sorted(tier_defaults)))
    else:
        st.warning("Scrape both teams first.")

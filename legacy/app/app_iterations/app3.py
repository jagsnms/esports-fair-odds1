import streamlit as st
import pandas as pd
import subprocess
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from difflib import get_close_matches

def normalize_name(name):
    return name.lower().replace("team ", "").replace("_", "").strip()

def get_team_tier(opp, df):
    norm_opp = normalize_name(opp)
    df["norm_team"] = df["team"].apply(normalize_name)

    if norm_opp in df["norm_team"].values:
        return float(df.loc[df["norm_team"] == norm_opp, "tier"].values[0])
    else:
        matches = get_close_matches(norm_opp, df["norm_team"].values, n=1, cutoff=0.75)
        if matches:
            matched_name = matches[0]
            return float(df.loc[df["norm_team"] == matched_name, "tier"].values[0])
        else:
            st.warning(f"No match for '{opp}', defaulting to Tier 5")
            return 5.0


# ==========================
# Load CSVs
# ==========================
def load_cs2_teams():
    return pd.read_csv("cs2_rankings_merged.csv")  # Must include: team, tier, rank, hltv_id, slug

def load_dota_teams():
    df = pd.read_csv("dota2_full_team_list_with_slugs.csv")  # Must include: team, tier, rank, slug
    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: x.strip().replace(" ", "_"))
    return df

# ==========================
# Scrapers
# ==========================
def scrape_cs2_matches(team_id, team_slug):
    try:
        result = subprocess.run(
            ["python", "scraper.py", str(team_id), team_slug],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            return json.loads(result.stdout.splitlines()[-1])
        else:
            st.error("CS2 scraper failed")
            return []
    except Exception as e:
        st.error(f"CS2 scraper error: {e}")
        return []

def scrape_dota_matches(slug, limit=10):
    BASE_URL = "https://liquipedia.net/dota2/"
    url = f"{BASE_URL}{slug}/Played_Matches"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Failed to fetch Liquipedia page for {slug}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    matches = []
    rows = soup.select("table.wikitable tbody tr")

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        date_span = row.select_one("span.timer-object")
        if not date_span or not date_span.get("data-timestamp"):
            continue
        timestamp = int(date_span["data-timestamp"])
        match_date = datetime.fromtimestamp(timestamp)

        score_td = row.select_one(".match-table-score")
        if not score_td:
            continue
        score_text = score_td.get_text(strip=True)
        if ":" not in score_text:
            continue
        try:
            left_score, right_score = [int(x.strip()) for x in score_text.split(":")]
        except ValueError:
            continue
        win = left_score > right_score

        teams = row.select(".block-team a[title]")
        if len(teams) < 2:
            continue
        opponent_name = teams[1].get("title")

        matches.append({"opponent": opponent_name, "win": win, "date": match_date})

    matches.sort(key=lambda x: x["date"], reverse=True)
    for m in matches:
        del m["date"]
    return matches[:limit]

# ==========================
# Scoring Logic with Tier-Based Adjustments
# ==========================
def calculate_score(matches, df, current_opponent_tier=None):
    raw_score = 0
    adjusted_score = 0
    breakdown = []

    for match in matches:
        opp = match["opponent"]
        win = match["win"]

        # Get tier using fuzzy matching
        tier = get_team_tier(opp, df)

        # Base asymmetric points
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

        raw_score += points

        # Tier-gap adjustment
        if current_opponent_tier is not None:
            tier_gap = tier - current_opponent_tier

            if win:
                if tier_gap < 0:  # beat stronger team
                    weight = 1 + min(0.4, abs(tier_gap) * 0.2)  # max +40%
                else:  # beat weaker team
                    weight = 1 - min(0.3, tier_gap * 0.15)  # max -30%
            else:
                if tier_gap < 0:  # lost to stronger team
                    weight = 1 - min(0.2, abs(tier_gap) * 0.1)  # mild discount
                else:  # lost to weaker team
                    weight = 1 + min(0.5, tier_gap * 0.25)  # max +50% penalty

            # Apply soft caps
            weight = max(0.5, min(weight, 1.5))
        else:
            weight = 1
            tier_gap = 0

        adjusted_points = points * weight
        adjusted_score += adjusted_points

        breakdown.append(
            f"{'Win' if win else 'Loss'} vs {opp} "
            f"(Tier {tier}, Gap {tier_gap:.1f}, Weight {weight:.2f}) "
            f"→ {points:+} → Adj {adjusted_points:+.2f}"
        )

    return raw_score, adjusted_score, breakdown

# ==========================
# Fair Odds with Dynamic Scaling
# ==========================
def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

def color_ev(ev):
    if ev >= 40: return f"✅ **{round(ev,2)}%**"
    elif 11 <= ev < 40: return f"🟡 {round(ev,2)}%"
    else: return f"🔴 {round(ev,2)}%"

def calculate_fair_odds(gap, base_C=30, alpha=0.03):
    """
    Dynamic scaling for fair odds calculation.
    base_C = default scaler for midrange gaps
    alpha = sensitivity for adjusting C dynamically
    """
    # Adjust C based on magnitude of gap: bigger gaps = smaller C = steeper curve
    C_dynamic = base_C / (1 + alpha * abs(gap))

    # Logistic curve using dynamic C
    p_a = 1 / (1 + 10 ** (-gap / C_dynamic))
    p_b = 1 - p_a

    # Convert to American odds
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)

    return p_a, fair_a, fair_b

# ==========================
# Streamlit App
# ==========================
st.title("Esports Fair Odds Calculator (CS2 + Dota2)")
tabs = st.tabs(["CS2", "Dota2"])

# --------------------------
# CS2 TAB
# --------------------------
with tabs[0]:
    st.header("CS2 Fair Odds")
    df_cs2 = load_cs2_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (CS2)", df_cs2["team"].tolist(), key="cs2_a")
        team_a_id = str(int(df_cs2[df_cs2["team"] == team_a]["hltv_id"].values[0]))
        team_a_slug = df_cs2[df_cs2["team"] == team_a]["slug"].values[0]
        if st.button("Scrape Team A (CS2)"):
            matches_a = scrape_cs2_matches(team_a_id, team_a_slug)
            st.session_state["cs2_matches_a"] = matches_a
        matches_a = st.session_state.get("cs2_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (CS2)", df_cs2["team"].tolist(), key="cs2_b")
        team_b_id = str(int(df_cs2[df_cs2["team"] == team_b]["hltv_id"].values[0]))
        team_b_slug = df_cs2[df_cs2["team"] == team_b]["slug"].values[0]
        if st.button("Scrape Team B (CS2)"):
            matches_b = scrape_cs2_matches(team_b_id, team_b_slug)
            st.session_state["cs2_matches_b"] = matches_b
        matches_b = st.session_state.get("cs2_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (CS2)", value=-140, key="cs2_odds_a")
    odds_b = st.number_input("Team B Market Odds (CS2)", value=+120, key="cs2_odds_b")

    if st.button("Calculate (CS2)"):
        if matches_a and matches_b:
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            team_a_tier = float(df_cs2[df_cs2["team"] == team_a]["tier"].values[0])
            team_b_tier = float(df_cs2[df_cs2["team"] == team_b]["tier"].values[0])

            raw_a, adj_a, breakdown_a = calculate_score(matches_a, df_cs2, current_opponent_tier=team_b_tier)
            raw_b, adj_b, breakdown_b = calculate_score(matches_b, df_cs2, current_opponent_tier=team_a_tier)

            raw_gap = raw_a - raw_b
            adj_gap = adj_a - adj_b

            p, fair_a, fair_b = calculate_fair_odds(adj_gap)
            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100

            st.subheader("Summary (CS2)")
            st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
            st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.2f}/{adj_b:.2f}")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a:
                st.text(line)

            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b:
                st.text(line)
        else:
            st.warning("Scrape both teams first.")

# --------------------------
# Dota TAB
# --------------------------
with tabs[1]:
    st.header("Dota 2 Fair Odds")
    df_dota = load_dota_teams()

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Select Team A (Dota2)", df_dota["team"].tolist(), key="dota_a")
        team_a_slug = df_dota[df_dota["team"] == team_a]["slug"].values[0]
        if st.button("Scrape Team A (Dota2)"):
            matches_a = scrape_dota_matches(team_a_slug)
            st.session_state["dota_matches_a"] = matches_a
        matches_a = st.session_state.get("dota_matches_a", [])
        st.write(matches_a)

    with col2:
        team_b = st.selectbox("Select Team B (Dota2)", df_dota["team"].tolist(), key="dota_b")
        team_b_slug = df_dota[df_dota["team"] == team_b]["slug"].values[0]
        if st.button("Scrape Team B (Dota2)"):
            matches_b = scrape_dota_matches(team_b_slug)
            st.session_state["dota_matches_b"] = matches_b
        matches_b = st.session_state.get("dota_matches_b", [])
        st.write(matches_b)

    odds_a = st.number_input("Team A Market Odds (Dota2)", value=-140, key="dota_odds_a")
    odds_b = st.number_input("Team B Market Odds (Dota2)", value=+120, key="dota_odds_b")

    if st.button("Calculate (Dota2)"):
        if matches_a and matches_b:
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            team_a_tier = float(df_dota[df_dota["team"] == team_a]["tier"].values[0])
            team_b_tier = float(df_dota[df_dota["team"] == team_b]["tier"].values[0])

            raw_a, adj_a, breakdown_a = calculate_score(matches_a, df_dota, current_opponent_tier=team_b_tier)
            raw_b, adj_b, breakdown_b = calculate_score(matches_b, df_dota, current_opponent_tier=team_a_tier)

            raw_gap = raw_a - raw_b
            adj_gap = adj_a - adj_b

            p, fair_a, fair_b = calculate_fair_odds(adj_gap)
            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100

            st.subheader("Summary (Dota2)")
            st.text(f"{team_a} (Tier {team_a_tier}) vs {team_b} (Tier {team_b_tier})")
            st.text(f"Raw Gap: {raw_gap:.2f}, Adjusted Gap: {adj_gap:.2f}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Score Breakdown: Raw {raw_a:.2f}/{raw_b:.2f} | Adj {adj_a:.2f}/{adj_b:.2f}")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a:
                st.text(line)

            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b:
                st.text(line)
        else:
            st.warning("Scrape both teams first.")

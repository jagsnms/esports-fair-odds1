import streamlit as st
import pandas as pd
import subprocess
import json

# Load CSV dynamically based on game selection
def load_teams(game):
    if game == "CS2":
        return pd.read_csv("cs2_tiers_clean.csv")
    else:
        return pd.read_csv("dota2_tiers_clean.csv")

# Run scraper
def scrape_matches(team_id, team_slug, game):
    try:
        if game == "CS2":
            result = subprocess.run(
                ["python", "scraper.py", str(team_id), team_slug],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                return json.loads(result.stdout.splitlines()[-1])
            else:
                st.error("Scraper failed")
                return []
        else:
            st.warning("Dota2 scraping not implemented. Enter matches manually or skip.")
            return []
    except Exception as e:
        st.error(f"Scraper error: {e}")
        return []

# Calculate tier score with breakdown
def calculate_score(matches, df):
    score = 0
    breakdown = []
    for match in matches:
        opp = match["opponent"]
        win = match["win"]
        tier = 5  # Default Tier 5 if not found
        if opp in df["team"].values:
            tier = float(df[df["team"] == opp]["tier"].values[0])  # Support 1.5
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
        score += points
        breakdown.append(f"{'Win' if win else 'Loss'} vs {opp} (Tier {tier}) → {points:+}")
    return score, breakdown

# Convert score gap to win probability and fair odds
def calculate_fair_odds(gap, C=35):
    p_a = 1 / (1 + 10 ** (-gap / C))
    p_b = 1 - p_a
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)
    return p_a, fair_a, fair_b

# Convert American odds to decimal
def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

# Streamlit UI
st.title("Esports Fair Odds Calculator")

game = st.selectbox("Select Game", ["CS2", "Dota2"])
df = load_teams(game)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Team A")
    team_a = st.selectbox("Select Team A", df["team"].tolist())
    team_a_id = str(int(df[df["team"] == team_a]["hltv_id"].values[0]))
    team_a_slug = df[df["team"] == team_a]["slug"].values[0]
    if st.button("Scrape Team A"):
        matches_a = scrape_matches(team_a_id, team_a_slug, game)
        st.session_state["matches_a"] = matches_a
    matches_a = st.session_state.get("matches_a", [])
    st.write(matches_a)

with col2:
    st.subheader("Team B")
    team_b = st.selectbox("Select Team B", df["team"].tolist())
    team_b_id = str(int(df[df["team"] == team_b]["hltv_id"].values[0]))
    team_b_slug = df[df["team"] == team_b]["slug"].values[0]
    if st.button("Scrape Team B"):
        matches_b = scrape_matches(team_b_id, team_b_slug, game)
        st.session_state["matches_b"] = matches_b
    matches_b = st.session_state.get("matches_b", [])
    st.write(matches_b)

st.subheader("Market Odds")
odds_a = st.number_input("Team A Market Odds (American)", value=-140)
odds_b = st.number_input("Team B Market Odds (American)", value=+120)

if st.button("Calculate"):
    if matches_a and matches_b:
        # Calculate scores and breakdowns
        score_a, breakdown_a = calculate_score(matches_a, df)
        score_b, breakdown_b = calculate_score(matches_b, df)
        gap = score_a - score_b

        # Fair odds
        p, fair_a, fair_b = calculate_fair_odds(gap)

        # EV calculation
        dec_a = american_to_decimal(odds_a)
        dec_b = american_to_decimal(odds_b)
        ev_a = ((p * dec_a) - 1) * 100
        ev_b = (((1 - p) * dec_b) - 1) * 100

        st.subheader("Summary")
        st.text(f"{team_a} vs {team_b}")
        st.text(f"Market Odds: {odds_a} / {odds_b}")
        st.text(f"Fair Odds:   {fair_a} / {fair_b}")
        st.text(f"Win Probability: {round(p * 100, 2)}%")
        st.text(f"EV: {round(ev_a, 2)}% / {round(ev_b, 2)}%")
        st.text(f"Tier Score Difference: {gap} ( {team_a}: {score_a} | {team_b}: {score_b} )")

        st.subheader(f"{team_a} Score Breakdown")
        for line in breakdown_a:
            st.text(line)

        st.subheader(f"{team_b} Score Breakdown")
        for line in breakdown_b:
            st.text(line)
    else:
        st.warning("Please scrape matches for both teams before calculating.")


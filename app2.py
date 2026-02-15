import streamlit as st
import pandas as pd
import subprocess
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ==========================
# Load CSVs
# ==========================
def load_cs2_teams():
    return pd.read_csv("cs2_tiers_clean.csv")

def load_dota_teams():
    df = pd.read_csv("dota2_full_team_list_with_slugs.csv")

    if "slug" not in df.columns:
        df["slug"] = df["team"].apply(lambda x: x.strip().replace(" ", "_"))
    return df

# ==========================
# Scrapers
# ==========================
# CS2: HLTV Scraper via subprocess
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

# Dota2: Liquipedia Scraper
def scrape_dota_matches(slug, limit=10):
    BASE_URL = "https://liquipedia.net/dota2/"
    url = f"{BASE_URL}{slug}/Played_Matches"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
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
# Calculation Functions
# ==========================
def calculate_score(matches, df):
    score = 0
    breakdown = []
    for match in matches:
        opp = match["opponent"]
        win = match["win"]
        tier = 5
        if opp in df["team"].values:
            tier = float(df[df["team"] == opp]["tier"].values[0])
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

def calculate_fair_odds(gap, C=30):
    p_a = 1 / (1 + 10 ** (-gap / C))
    p_b = 1 - p_a
    fair_a = -round(100 * p_a / (1 - p_a)) if p_a >= 0.5 else round(100 * (1 - p_a) / p_a)
    fair_b = -round(100 * p_b / (1 - p_b)) if p_b >= 0.5 else round(100 * (1 - p_b) / p_b)
    return p_a, fair_a, fair_b

def american_to_decimal(odds):
    return (odds / 100 + 1) if odds > 0 else (100 / abs(odds)) + 1

def color_ev(ev):
    if ev >= 40: return f"✅ **{round(ev,2)}%**"
    elif 11 <= ev < 40: return f"🟡 {round(ev,2)}%"
    else: return f"🔴 {round(ev,2)}%"

# ==========================
# Streamlit Tabs
# ==========================
st.title("Esports Fair Odds Calculator (CS2 + Dota2)")
tabs = st.tabs(["CS2", "Dota2"])

## --------------------------
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
            # Force both teams to use the same number of recent matches
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            # Calculate scores and breakdowns
            score_a, breakdown_a = calculate_score(matches_a, df_cs2)
            score_b, breakdown_b = calculate_score(matches_b, df_cs2)
            gap = score_a - score_b

            # Probability & odds
            p, fair_a, fair_b = calculate_fair_odds(gap)
            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100

            # Output
            st.subheader("Summary (CS2)")
            st.text(f"{team_a} vs {team_b}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Tier Score Difference: {gap} ( {team_a}: {score_a} | {team_b}: {score_b} )")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a:
                st.text(line)

            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b:
                st.text(line)
        else:
            st.warning("Scrape both teams first.")

# # --------------------------
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
            # Force both teams to use the same number of recent matches
            min_matches = min(len(matches_a), len(matches_b))
            matches_a = matches_a[:min_matches]
            matches_b = matches_b[:min_matches]
            st.text(f"Using last {min_matches} matches for both teams.")

            # Calculate scores and breakdowns
            score_a, breakdown_a = calculate_score(matches_a, df_dota)
            score_b, breakdown_b = calculate_score(matches_b, df_dota)
            gap = score_a - score_b

            # Probability & odds
            p, fair_a, fair_b = calculate_fair_odds(gap)
            dec_a = american_to_decimal(odds_a)
            dec_b = american_to_decimal(odds_b)
            ev_a = ((p * dec_a) - 1) * 100
            ev_b = (((1 - p) * dec_b) - 1) * 100

            # Output
            st.subheader("Summary (Dota2)")
            st.text(f"{team_a} vs {team_b}")
            st.text(f"Market Odds: {odds_a} / {odds_b}")
            st.text(f"Fair Odds:   {fair_a} / {fair_b}")
            st.text(f"Win Probability: {round(p * 100, 2)}%")
            st.markdown(f"EV: {color_ev(ev_a)} / {color_ev(ev_b)}")
            st.text(f"Tier Score Difference: {gap} ( {team_a}: {score_a} | {team_b}: {score_b} )")

            st.subheader(f"{team_a} Breakdown")
            for line in breakdown_a:
                st.text(line)

            st.subheader(f"{team_b} Breakdown")
            for line in breakdown_b:
                st.text(line)
        else:
            st.warning("Scrape both teams first.")

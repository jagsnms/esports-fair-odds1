import streamlit as st
import pandas as pd
import math

# ----------------------
# Utility Functions
# ----------------------
def load_tier_list(game):
    if game == "CS2":
        return pd.read_csv("cs2_tiers_clean.csv")
    else:
        return pd.read_csv("dota2_tiers_clean.csv")

def get_team_tier(team_name, tier_df):
    if team_name in tier_df["Name"].values:
        return float(tier_df.loc[tier_df["Name"] == team_name, "Tier"].values[0])
    return 5.0  # Default Tier 5 for unknown teams

def calc_match_score(win, opp_tier):
    if win:
        return {1: 4.0, 1.5: 3.0, 2: 2.5, 3: 2.0, 4: 1.5}.get(opp_tier, 1.0)
    else:
        return {1: -1.0, 1.5: -1.5, 2: -2.0, 3: -2.5, 4: -3.0}.get(opp_tier, -4.0)

def logistic_probability(gap):
    return 1 / (1 + math.pow(10, -gap / 40))

def moneyline_from_prob(p):
    return -100 * (p / (1 - p)) if p > 0.5 else 100 * ((1 - p) / p)

def calculate_ev(prob, odds):
    try:
        odds = int(odds)
    except:
        return None
    if odds > 0:  # Underdog
        payout = odds / 100
    else:  # Favorite
        payout = 100 / abs(odds)
    ev = (prob * payout) - (1 - prob)
    return ev * 100  # in %

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Dual-Team Fair Odds Calculator", layout="wide")
st.title("Dual-Team Esports Fair Odds Calculator")

menu = st.sidebar.radio("Navigation", ["Betting Calculator", "Manage Tier Lists"])

if menu == "Betting Calculator":
    # Select Game
    game = st.selectbox("Select Game", ["CS2", "Dota2"])
    tier_df = load_tier_list(game)

    # Team and Odds
    st.subheader("Match Setup")
    col_match = st.columns([2, 1, 1])
    with col_match[0]:
        team_a = st.text_input("Team A Name", "")
    with col_match[1]:
        team_a_odds = st.text_input("Team A Odds (e.g., -140)", "-140")
    with col_match[2]:
        team_b_odds = st.text_input("Team B Odds (e.g., +120)", "+120")

    team_b = st.text_input("Team B Name", "")

    st.markdown("---")
    st.subheader(f"Enter Last Matches for {team_a}")
    team_a_matches = []
    for i in range(10):
        cols = st.columns([1, 2, 2])
        with cols[0]:
            win_a = st.checkbox("Win", key=f"a_win_{i}")
        with cols[1]:
            opponent_a = st.selectbox("Opponent", tier_df["Name"].tolist() + ["Other (Tier 5)"], key=f"a_opp_{i}")
        with cols[2]:
            alias_a = ""
            if opponent_a == "Other (Tier 5)":
                alias_a = st.text_input("Alias", key=f"a_alias_{i}")
        team_a_matches.append((win_a, opponent_a))

    st.markdown("---")
    st.subheader(f"Enter Last Matches for {team_b}")
    team_b_matches = []
    for i in range(10):
        cols = st.columns([1, 2, 2])
        with cols[0]:
            win_b = st.checkbox("Win", key=f"b_win_{i}")
        with cols[1]:
            opponent_b = st.selectbox("Opponent", tier_df["Name"].tolist() + ["Other (Tier 5)"], key=f"b_opp_{i}")
        with cols[2]:
            alias_b = ""
            if opponent_b == "Other (Tier 5)":
                alias_b = st.text_input("Alias", key=f"b_alias_{i}")
        team_b_matches.append((win_b, opponent_b))

    if st.button("Calculate"):
        # Calculate Team A score
        team_a_score = 0
        for win, opp in team_a_matches:
            if opp == "" or opp is None:
                continue
            opp_tier = get_team_tier(opp, tier_df) if opp != "Other (Tier 5)" else 5
            team_a_score += calc_match_score(win, opp_tier)

        # Calculate Team B score
        team_b_score = 0
        for win, opp in team_b_matches:
            if opp == "" or opp is None:
                continue
            opp_tier = get_team_tier(opp, tier_df) if opp != "Other (Tier 5)" else 5
            team_b_score += calc_match_score(win, opp_tier)

        gap = team_a_score - team_b_score
        prob_a = logistic_probability(gap)
        prob_b = 1 - prob_a
        fair_a = moneyline_from_prob(prob_a)
        fair_b = moneyline_from_prob(prob_b)
        ev_a = calculate_ev(prob_a, team_a_odds)
        ev_b = calculate_ev(prob_b, team_b_odds)

        st.markdown("### Results")
        st.markdown(f"**Match:** {team_a} vs {team_b}")
        st.text(f"""
Market Odds:   {team_a_odds:<8} {team_b_odds}
Fair Odds:     {int(fair_a):<8} {int(fair_b)}
Win %:         {prob_a*100:.1f}%    {prob_b*100:.1f}%
EV:            {ev_a:.2f}%    {ev_b:.2f}%
Tier Score:    {team_a_score:<8.1f} {team_b_score}
Gap: {gap:.1f}
""")

elif menu == "Manage Tier Lists":
    st.header("Current Tier Lists")
    game = st.selectbox("Select Game", ["CS2", "Dota2"])
    tier_df = load_tier_list(game)
    st.dataframe(tier_df)
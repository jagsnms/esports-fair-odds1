import streamlit as st
import pandas as pd
import math

# ----------------------
# Utility Functions
# ----------------------

def load_tier_list(game):
    if game == "CS2":
        return pd.read_csv("cs2_tiers.csv")
    else:
        return pd.read_csv("dota2_tiers.csv")

def calc_match_score(win, opp_tier):
    # Asymmetric scoring logic
    if win:
        return {1: 4.0, 1.5: 3.0, 2: 2.5, 3: 2.0, 4: 1.5}.get(opp_tier, 1.0)
    else:
        return {1: -1.0, 1.5: -1.5, 2: -2.0, 3: -2.5, 4: -3.0}.get(opp_tier, -4.0)

def logistic_probability(gap):
    return 1 / (1 + math.pow(10, -gap / 45))

def moneyline_from_prob(p):
    return -100 * (p / (1 - p)) if p > 0.5 else 100 * ((1 - p) / p)

def calculate_ev(prob, odds):
    # Odds expected in American format: e.g., -120 or +150
    try:
        odds = int(odds)
    except:
        return None

    if odds > 0:  # Underdog
        payout = odds / 100
    else:  # Favorite
        payout = 100 / abs(odds)

    ev = (prob * payout) - (1 - prob)
    return ev * 100  # Convert to %
    

# ----------------------
# Streamlit UI
# ----------------------

st.set_page_config(page_title="Esports Fair Odds Calculator", layout="wide")
st.title("Esports Fair Odds Calculator (CS2 & Dota2)")

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ["Betting Calculator", "Manage Tier Lists"])

if menu == "Betting Calculator":
    # Select Game
    game = st.selectbox("Select Game", ["CS2", "Dota2"])
    tier_df = load_tier_list(game)

    # Team Selection
    main_team = st.selectbox("Select Analyzing Team", tier_df["Name"].tolist())

    # Market Odds
    st.subheader("Enter Market Odds")
    col1, col2 = st.columns(2)
    with col1:
        market_odds_team = st.text_input(f"{main_team} Odds (e.g., -120)", "-120")
    with col2:
        market_odds_opp = st.text_input("Opponent Odds (e.g., +100)", "+100")

    # Match History Input
    st.subheader("Enter Last 10 Matches for Analyzing Team")
    match_data = []
    for i in range(10):
        cols = st.columns([1, 1, 2])
        with cols[0]:
            score_main = st.number_input(f"Match {i+1} - {main_team}", min_value=0, max_value=3, key=f"main_score_{i}")
        with cols[1]:
            score_opp = st.number_input("Opp Score", min_value=0, max_value=3, key=f"opp_score_{i}")
        with cols[2]:
            opponent = st.selectbox("Opponent", tier_df["Name"].tolist() + ["NA"], key=f"opp_{i}")
        match_data.append({"main_score": score_main, "opp_score": score_opp, "opponent": opponent})

    # Opponent Total Score Input
    st.subheader("Opponent's Total Score (enter manually for now)")
    opp_total = st.number_input("Opponent Total Tier Score", value=0.0)

    if st.button("Calculate"):
        # Calculate main team score
        main_total = 0
        for row in match_data:
            if row["opponent"] == "NA":
                opp_tier = 5
            else:
                opp_tier = tier_df.loc[tier_df["Name"] == row["opponent"], "Tier"].values[0]
            win = row["main_score"] > row["opp_score"]
            main_total += calc_match_score(win, opp_tier)

        gap = main_total - opp_total
        p = logistic_probability(gap)
        fair_ml = moneyline_from_prob(p)
        ev = calculate_ev(p, market_odds_team)

        st.success(f"Results for {main_team}:")
        st.write(f"Tier Score: {main_total}")
        st.write(f"Gap: {gap}")
        st.write(f"Win Probability: {p*100:.2f}%")
        st.write(f"Fair Moneyline: {fair_ml:.0f}")
        if ev is not None:
            st.write(f"EV vs Market Odds: {ev:.2f}%")
        else:
            st.write("Invalid odds format for EV calculation.")

        # Export results
        result_df = pd.DataFrame({
            "Team": [main_team],
            "Tier Score": [main_total],
            "Gap": [gap],
            "Win Probability": [f"{p*100:.2f}%"],
            "Fair Moneyline": [round(fair_ml)],
            "Market Odds": [market_odds_team],
            "EV": [f"{ev:.2f}%" if ev is not None else "N/A"]
        })
        st.download_button("Download Results as CSV", result_df.to_csv(index=False), "results.csv", "text/csv")

elif menu == "Manage Tier Lists":
    st.header("View Tier Lists")
    game = st.selectbox("Select Game for Tier List", ["CS2", "Dota2"])
    tier_df = load_tier_list(game)
    st.dataframe(tier_df)

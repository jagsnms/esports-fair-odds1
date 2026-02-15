import pandas as pd
import requests
from difflib import get_close_matches

# Path to your Dota CSV
csv_path = "dota2_tiers_clean.csv"
teams = pd.read_csv(csv_path)

# Add team_id column if missing
if "team_id" not in teams.columns:
    teams["team_id"] = ""

# Fetch all OpenDota teams
print("Fetching OpenDota team list...")
response = requests.get("https://api.opendota.com/api/teams")
all_teams = response.json()

# Create dict for fast lookup
od_names = {t["name"]: t["team_id"] for t in all_teams if t.get("name")}
od_tags = {t["tag"]: t["team_id"] for t in all_teams if t.get("tag")}

# Match teams from CSV
for i, name in enumerate(teams["team"]):
    # Try exact name match first
    if name in od_names:
        teams.at[i, "team_id"] = od_names[name]
        print(f"Exact match: {name} → {od_names[name]}")
    else:
        # Try fuzzy match
        matches = get_close_matches(name, od_names.keys(), n=1, cutoff=0.7)
        if matches:
            match_name = matches[0]
            teams.at[i, "team_id"] = od_names[match_name]
            print(f"Fuzzy match: {name} → {match_name} ({od_names[match_name]})")
        else:
            print(f"⚠ No match found for {name}")

# Save updated CSV
teams.to_csv(csv_path, index=False)
print("\n✅ Team IDs updated and saved to", csv_path)

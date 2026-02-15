import pandas as pd

# Load original and new files
original = pd.read_csv("cs2_tiers_clean.csv")
blast = pd.read_csv("blast_rankings.csv")

# Normalize team names
original["team_norm"] = original["team"].str.lower().str.strip()
blast["team_norm"] = blast["team"].str.lower().str.strip()

# Merge
merged = blast.merge(original[["team_norm", "hltv_id", "slug", "tier"]],
                     on="team_norm",
                     how="left")

# Drop the helper column
merged.drop(columns=["team_norm"], inplace=True)

# Fill missing tiers with default (e.g., Tier 5)
merged["tier"] = merged["tier"].fillna(5)

# Save the final merged file
merged.to_csv("cs2_rankings_merged.csv", index=False)

print(f"✅ cs2_rankings_merged.csv created successfully with {len(merged)} teams.")
print(f"Matched teams: {merged['hltv_id'].notna().sum()}, Missing: {merged['hltv_id'].isna().sum()}")

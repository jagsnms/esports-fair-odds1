import requests
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://liquipedia.net/dota2/"

def scrape_liquipedia_matches(slug, limit=10):
    url = f"{BASE_URL}{slug}/Played_Matches"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching {slug}: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    matches = []

    rows = soup.select("table.wikitable tbody tr")

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 6:
            continue

        # Extract timestamp
        date_span = row.select_one("span.timer-object")
        if not date_span or not date_span.get("data-timestamp"):
            continue
        try:
            timestamp = int(date_span["data-timestamp"])
            match_date = datetime.fromtimestamp(timestamp)
        except:
            continue

        # Extract score
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

        # Determine win/loss (your team always left)
        win = left_score > right_score

        # Extract opponent (second .block-team element)
        teams = row.select(".block-team a[title]")
        if len(teams) < 2:
            continue
        opponent_name = teams[1].get("title")  # Second team is opponent

        matches.append({
            "opponent": opponent_name,
            "win": win,
            "date": match_date
        })

    matches.sort(key=lambda x: x["date"], reverse=True)

    for m in matches:
        del m["date"]

    return matches[:limit]

# Test
slug = "Team_Spirit"
recent_matches = scrape_liquipedia_matches(slug)
print(f"Recent matches for {slug}:")
for match in recent_matches:
    print(match)

from playwright.sync_api import sync_playwright
import time

def fetch_last_matches(team_id, team_slug, limit=10):
    url = f"https://www.hltv.org/team/{team_id}/{team_slug}"
    matches = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)

        # Click Matches tab
        page.wait_for_selector('div.tab[data-content-id="matchesBox"]', timeout=30000)
        page.evaluate("document.querySelector('div.tab[data-content-id=\"matchesBox\"]').click()")

        # Wait for matches container
        page.wait_for_selector('#matchesBox', timeout=15000)

        # Scroll twice for lazy load
        page.mouse.wheel(0, 500)
        page.wait_for_timeout(3000)
        page.mouse.wheel(0, 800)
        page.wait_for_timeout(5000)

        # ✅ Find all match tables under matchesBox
        tables = page.query_selector_all('#matchesBox table.match-table')
        if len(tables) < 2:
            print("DEBUG: Could not find Recent results table.")
            browser.close()
            return matches

        # ✅ Get the second table (Recent results)
        recent_results_table = tables[1]
        rows = recent_results_table.query_selector_all('tr.team-row')

        if not rows:
            print("DEBUG: No rows found in Recent results table.")
            browser.close()
            return matches

        for row in rows[:limit]:
            teams = row.query_selector_all('div.team-flex')
            score_elem = row.query_selector('div.score-cell')
            if len(teams) == 2 and score_elem:
                score_text = score_elem.inner_text().strip()
                if ':' not in score_text:
                    continue
                scores = score_text.split(':')
                try:
                    score1, score2 = int(scores[0].strip()), int(scores[1].strip())
                except ValueError:
                    continue
                team1 = teams[0].inner_text().strip()
                team2 = teams[1].inner_text().strip()
                opponent = team2 if team1.lower() == team_slug.lower() else team1
                win = (score1 > score2) if team1.lower() == team_slug.lower() else (score2 > score1)
                matches.append({'opponent': opponent, 'win': win})

        browser.close()
    return matches

if __name__ == "__main__":
    print(fetch_last_matches(9565, "vitality"))

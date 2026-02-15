import requests

def find_team_id(search_name):
    url = "https://api.opendota.com/api/teams"
    response = requests.get(url)
    if response.status_code == 200:
        teams = response.json()
        for team in teams:
            if team.get("name") and search_name.lower() in team["name"].lower():
                print(f"Found: {team['name']} | ID: {team['team_id']}")
    else:
        print("Error fetching teams:", response.status_code)

find_team_id("BetBoom")

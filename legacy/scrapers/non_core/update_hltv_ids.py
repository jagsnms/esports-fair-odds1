import requests

def find_opendota_team_id(name):
    url = f"https://api.opendota.com/api/search?q={name}"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    results = response.json()
    teams = [item for item in results if item.get('team_id')]
    if not teams:
        print("No teams found.")
        return

    print("\nMatches found:")
    for t in teams[:10]:  # top 10 matches
        print(f"Name: {t.get('name')}, Team ID: {t.get('team_id')}")

find_opendota_team_id("Kopite")

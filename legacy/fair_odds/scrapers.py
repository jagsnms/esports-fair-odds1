"""Scraper subprocess wrappers for CS2 and Dota match data."""
import json
import os
import subprocess
import sys

import streamlit as st

from .paths import PROJECT_ROOT


def scrape_cs2_matches(team_id: str, team_slug: str):
    """Run scraper.py as a subprocess and parse the last JSON line (CS2)."""
    try:
        script_path = str(PROJECT_ROOT / "scraper.py")
        result = subprocess.run(
            [sys.executable, script_path, str(team_id), team_slug],
            capture_output=True, text=True, timeout=180, cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[-1] if result.stdout else "[]"
            return json.loads(line)
        else:
            st.error("CS2 scraper failed")
            with st.expander("CS2 scraper stderr"):
                st.code(result.stderr or "(no stderr)")
            with st.expander("CS2 scraper stdout"):
                st.code(result.stdout or "(no stdout)")
            return []
    except Exception as e:
        st.error(f"CS2 scraper error: {e}")
        return []


def scrape_dota_matches_gosu_subprocess(
    team_slug: str, team_name: str, target: int = 14,
    headed: bool = True, browser_channel: str = "bundled",
    zoom: int = 80,
):
    """Run gosu_dota_scraper.py and parse the last JSON array from stdout."""
    try:
        script_path = str(PROJECT_ROOT / "gosu_dota_scraper.py")
        cmd = [sys.executable, script_path, "--team-slug", team_slug, "--team-name", team_name,
               "--target", str(target), "--zoom", str(zoom)]
        if headed:
            cmd.append("--headed")
        if browser_channel in ("chrome", "msedge"):
            cmd += ["--channel", browser_channel]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            st.error(f"Gosu scraper failed (exit {result.returncode}).")
            with st.expander("Gosu scraper stderr"):
                st.code(result.stderr or "(no stderr)")
            with st.expander("Gosu scraper stdout (tail)"):
                st.code((result.stdout or "")[-2000:])
            return []

        stdout = result.stdout or ""
        start = stdout.rfind("[")
        end = stdout.rfind("]")
        if start == -1 or end == -1 or end < start:
            st.warning("Gosu scraper returned no parseable JSON.")
            with st.expander("Gosu scraper raw stdout"):
                st.code(stdout[-4000:] if stdout else "(empty)")
            return []
        data = json.loads(stdout[start : end + 1])
        out = []
        for row in data:
            opp = row.get("opponent", "Unknown")
            win = bool(row.get("win", False))
            res_txt = str(row.get("result", "") or row.get("score", "") or row.get("status", "")).lower()
            is_draw = ("1-1" in res_txt) or ("draw" in res_txt) or ("tie" in res_txt)
            out.append({"opponent": opp, "win": win, "draw": is_draw})
        return out
    except subprocess.TimeoutExpired:
        st.error("Gosu scraper timed out. Try again or reduce match count.")
        return []
    except Exception as e:
        st.error(f"Gosu scraper error: {e}")
        return []


def fetch_dota_matches(team_id: int, limit: int = 15):
    """Fetch Dota matches from OpenDota API. Returns list of {opponent, win, draw}."""
    import requests
    url = f"https://api.opendota.com/api/teams/{team_id}/matches"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            st.error(f"OpenDota API error for team {team_id}: {r.status_code}")
            return []
        data = r.json()
        matches = []
        for m in (data or [])[:limit]:
            opp_name = m.get("opposing_team_name", "Unknown")
            radiant = m.get("radiant", None)
            radiant_win = m.get("radiant_win", None)
            if radiant is None or radiant_win is None:
                continue
            win = (radiant and radiant_win) or (not radiant and not radiant_win)
            matches.append({"opponent": opp_name, "win": win, "draw": False})
        return matches
    except Exception as e:
        st.error(f"Error fetching OpenDota data: {e}")
        return []

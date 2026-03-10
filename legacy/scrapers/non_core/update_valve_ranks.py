
import argparse
import csv
import datetime as dt
import os
import re
from collections import OrderedDict

import pandas as pd
import requests
from bs4 import BeautifulSoup

TEAM_HREF_RE = re.compile(r"/team/(\d+)(?:/|$)")

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def fetch_valve_ranking(url: str, timeout: int = 20) -> list[dict]:
    \"\"\"Return a list of dicts [{hltv_id:int, team_name:str, rank:int}] in page order.
    We dedupe team links by first appearance to preserve rank order.
    \"\"\"
    headers = {"User-Agent": DEFAULT_UA, "Accept-Language": "en-US,en;q=0.9"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "lxml")  # fallback to html.parser if lxml missing

    seen = set()
    ordered = []
    # Prefer tighter selectors if available (rows with rank), else generic anchor scan
    # Try to find ranking rows that contain a team link
    rows = soup.select("div.ranking > div, div.col-box, li, tr")
    anchors = soup.select('a[href^="/team/"]')
    candidates = rows if rows else anchors

    for node in candidates:
        # Prefer to parse anchors inside rows
        a_tags = node.select('a[href^="/team/"]') if hasattr(node, "select") else []
        if not a_tags and isinstance(node, (str,)):
            continue
        for a in (a_tags or [node] if hasattr(node, "get") else []):
            href = a.get("href") if hasattr(a, "get") else None
            if not href:
                continue
            m = TEAM_HREF_RE.search(href)
            if not m:
                continue
            tid = int(m.group(1))
            # Try to get a reasonably clean display name
            name = (a.get_text(strip=True) or "").strip()
            if tid not in seen:
                seen.add(tid)
                ordered.append((tid, name))

    # If nothing was found via rows, fallback to scanning all anchors
    if not ordered and anchors:
        for a in anchors:
            href = a.get("href", "")
            m = TEAM_HREF_RE.search(href)
            if m:
                tid = int(m.group(1))
                name = (a.get_text(strip=True) or "").strip()
                if tid not in seen:
                    seen.add(tid)
                    ordered.append((tid, name))

    # Build output list with ranks
    data = []
    for i, (tid, name) in enumerate(ordered, start=1):
        data.append({"hltv_id": tid, "team_name": name, "rank_valve": i})
    return data

def infer_rank_date_from_url(url: str) -> str | None:
    # expect .../valve-ranking/teams/YYYY/month/day
    m = re.search(r"/valve-ranking/teams/(\d{4})/([a-z]+)/(\d{1,2})", url)
    if not m:
        return None
    year, month_str, day = m.groups()
    try:
        # Map month name to month number
        month = dt.datetime.strptime(month_str[:3], "%b").month
    except ValueError:
        try:
            month = dt.datetime.strptime(month_str, "%B").month
        except ValueError:
            return None
    y = int(year)
    d = int(day)
    return dt.date(y, month, d).isoformat()

def compute_tier_from_rank(rank: int | float | None, mode: str) -> str:
    \"\"\"Return tier label given rank using configured boundaries.
    Modes:
      - legacy_cutoffs: T1 (1-6), T1.5 (7-12), T2 (13-28), T3 (29-54), T4 (55+), T5 if no rank
      - six_tier_top61:   T1 (1-6), T1.5 (7-12), T2 (13-24), T3 (25-41), T4 (42-61), T5 (>61 or no rank)
        (Adjust as needed; boundaries are centralized here.)
    \"\"\"
    if rank is None or pd.isna(rank):
        return "T5"
    r = int(rank)
    if mode == "legacy_cutoffs":
        if 1 <= r <= 6:
            return "T1"
        if 7 <= r <= 12:
            return "T1.5"
        if 13 <= r <= 28:
            return "T2"
        if 29 <= r <= 54:
            return "T3"
        if r >= 55:
            return "T4"
        return "T5"
    elif mode == "six_tier_top61":
        if 1 <= r <= 6:
            return "T1"
        if 7 <= r <= 12:
            return "T1.5"
        if 13 <= r <= 24:
            return "T2"
        if 25 <= r <= 41:
            return "T3"
        if 42 <= r <= 61:
            return "T4"
        return "T5"
    else:
        raise ValueError(f"Unknown tier mode: {mode}")

def update_csv(csv_path: str, out_path: str, url: str, tier_mode: str, overwrite_tier: bool = False) -> dict:
    # Fetch rankings
    data = fetch_valve_ranking(url)
    if not data:
        raise RuntimeError("No team entries parsed from the HLTV page; markup might have changed.")
    rank_map = {row["hltv_id"]: row["rank_valve"] for row in data}
    name_map = {row["hltv_id"]: row["team_name"] for row in data}  # informational

    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure we have an hltv_id column
    if "hltv_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'hltv_id' column.")

    # Inject/Update rank_valve + rank_valve_date
    df["rank_valve"] = df["hltv_id"].map(rank_map)
    rank_date = infer_rank_date_from_url(url) or dt.date.today().isoformat()
    df["rank_valve_date"] = rank_date

    # Compute new tier column WITHOUT touching existing 'tier' unless requested
    df["tier_auto"] = df["rank_valve"].apply(lambda r: compute_tier_from_rank(r, tier_mode))
    df["tier_src"] = f"valve_{rank_date}_{tier_mode}"

    if overwrite_tier and "tier" in df.columns:
        df.loc[:, "tier"] = df["tier_auto"]

    # Summary stats
    n_total = len(df)
    n_ranked = df["rank_valve"].notna().sum()
    n_unranked = n_total - n_ranked
    missing_ids = df.loc[df["rank_valve"].isna(), "hltv_id"].tolist()

    # Write out
    # Avoid clobbering by default
    if os.path.abspath(csv_path) == os.path.abspath(out_path):
        backup = csv_path + ".bak"
        df.to_csv(backup, index=False, quoting=csv.QUOTE_MINIMAL)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    return {
        "total_rows": n_total,
        "updated_ranks": int(n_ranked),
        "unmatched_ids": missing_ids[:50],  # sample first 50 for brevity
        "out_csv": out_path,
        "rank_date": rank_date,
        "tier_mode": tier_mode,
        "overwrite_tier": overwrite_tier,
        "source_url": url,
    }

def main():
    ap = argparse.ArgumentParser(description="Update CS2 CSV with Valve ranks from HLTV without touching other fields.")
    ap.add_argument("--csv", required=True, help="Path to existing teams CSV (must include hltv_id column).")
    ap.add_argument("--out", required=True, help="Path to write updated CSV.")
    ap.add_argument("--url", default="https://www.hltv.org/valve-ranking/teams/2025/september/9", help="Valve ranking page URL to scrape.")
    ap.add_argument("--tier-mode", default="legacy_cutoffs", choices=["legacy_cutoffs", "six_tier_top61"], help="Tier boundary scheme.")
    ap.add_argument("--overwrite-tier", action="store_true", help="If set, overwrite existing 'tier' column using computed tiers.")
    args = ap.parse_args()

    info = update_csv(args.csv, args.out, args.url, args.tier_mode, args.overwrite_tier)
    print("Done.")
    print(info)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Print top GRID CS2 series candidates (Central Data, titleId=28).
Usage: from repo root, python scripts/grid_list_cs2_series.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.ingest.grid_central_data import GRID_CS2_TITLE_ID, fetch_cs2_series_candidates

TOP_N = 10


def main() -> int:
    print(f"GRID CS2 series candidates (titleId={GRID_CS2_TITLE_ID}, Central Data)")
    print("Fetching (no cache)...")
    try:
        candidates = fetch_cs2_series_candidates(limit=TOP_N)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    print(f"Total: {len(candidates)}")
    for i, c in enumerate(candidates[:TOP_N], 1):
        sid = c.get("series_id") or "?"
        name = c.get("name") or c.get("tournament_name") or "-"
        tournament = c.get("tournament_name") or "-"
        start = c.get("start_time") or "-"
        updated = c.get("updated_at") or "-"
        stype = c.get("type") or "-"
        level = c.get("live_data_feed_level") or "-"
        rank = c.get("live_data_feed_rank")
        rank_str = str(rank) if rank is not None else "-"
        psl = c.get("product_service_levels")
        psl_str = ",".join(str(x) for x in psl) if psl else "-"
        print(f"  {i}. id={sid}  name={name!r}  tournament={tournament!r}")
        print(f"      start_time={start}  updated_at={updated}  type={stype}")
        print(f"      live_data_feed_level={level}  live_data_feed_rank={rank_str}  product_service_levels={psl_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

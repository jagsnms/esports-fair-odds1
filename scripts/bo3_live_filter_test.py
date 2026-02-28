"""
Test whether BO3's "current" (live) filter misses matches that are actually live by our criteria.

1. Pull matches with status=current (what we use now).
2. Pull matches with status=upcoming (matches "around" live).
3. For each match in the combined set, run our live test: get_live_match_snapshot(mid).
4. Report: in current vs upcoming, and which have snapshot OK (our "live"). Find any that
   are NOT in current but snapshot works = BO3's filter may be inaccurate.

Run from project root: python scripts/bo3_live_filter_test.py
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


async def fetch_matches_with_status(status_filter: str):
    """Fetch /matches with filter[matches.status][in]=status_filter. Returns list of match dicts."""
    import aiohttp
    url = "https://api.bo3.gg/api/v1/matches"
    params = {
        "scope": "widget-matches",
        "page[offset]": 0,
        "page[limit]": 100,
        "sort": "tier_rank,-start_date",
        "filter[matches.status][in]": status_filter,
        "filter[matches.discipline_id][eq]": 1,
        "with": "teams,tournament,ai_predictions,games,streams",
    }
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://bo3.gg",
        "referer": "https://bo3.gg/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("results", data.get("matches", data.get("data", []))) or []
    return []


def extract_match_ids_and_live_coverage(items: list) -> list[tuple[int, bool]]:
    """From API match items, return [(id, live_coverage), ...]. live_coverage from API."""
    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        if mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        lc = bool(m.get("live_coverage", False))
        out.append((mid, lc))
    return out


async def our_live_test(match_id: int) -> bool:
    """Our criterion: snapshot API returns 200 and non-empty dict."""
    try:
        from engine.ingest.bo3_client import get_snapshot
        snap = await get_snapshot(match_id)
        return isinstance(snap, dict) and bool(snap)
    except Exception:
        return False


async def main():
    print("=== 1) Pull BO3 'current' (their live filter) ===\n")
    try:
        current_items = await fetch_matches_with_status("current")
    except Exception as e:
        print(f"Current pull failed: {e}", file=sys.stderr)
        return
    current_pairs = extract_match_ids_and_live_coverage(current_items)
    current_ids = {mid for mid, _ in current_pairs}
    current_lc = {mid: lc for mid, lc in current_pairs}
    print(f"Current count: {len(current_ids)}")
    print(f"Current match ids: {sorted(current_ids)}")
    for mid, lc in current_pairs:
        print(f"  {mid}  API live_coverage={lc}")
    print()

    print("=== 2) Pull BO3 'upcoming' (around live) ===\n")
    try:
        upcoming_items = await fetch_matches_with_status("upcoming")
    except Exception as e:
        print(f"Upcoming pull failed: {e}", file=sys.stderr)
        return
    upcoming_pairs = extract_match_ids_and_live_coverage(upcoming_items)
    upcoming_ids = {mid for mid, _ in upcoming_pairs}
    print(f"Upcoming count: {len(upcoming_ids)}")
    print(f"Upcoming match ids (first 20): {sorted(upcoming_ids)[:20]}")
    print()

    # Union: all match IDs we will test with our snapshot
    all_ids = current_ids | upcoming_ids
    only_current = current_ids - upcoming_ids
    only_upcoming = upcoming_ids - current_ids
    in_both = current_ids & upcoming_ids
    print("=== 3) Overlap ===\n")
    print(f"Only in current: {len(only_current)}  {sorted(only_current)}")
    print(f"Only in upcoming: {len(only_upcoming)}  (first 15) {sorted(only_upcoming)[:15]}")
    print(f"In both: {len(in_both)}")
    print()

    print("=== 4) Our live test (get_snapshot per match) ===\n")
    print("Testing each match in current + upcoming with get_snapshot(mid)...")
    our_live: dict[int, bool] = {}
    for i, mid in enumerate(sorted(all_ids)):
        ok = await our_live_test(mid)
        our_live[mid] = ok
        if (i + 1) % 10 == 0 or i == 0 or i == len(all_ids) - 1:
            print(f"  tested {i + 1}/{len(all_ids)} ...")
    print()

    print("=== 5) Results: BO3 'current' vs our snapshot ===\n")
    in_current_our_live = [mid for mid in current_ids if our_live.get(mid)]
    only_upcoming_our_live = [mid for mid in only_upcoming if our_live.get(mid)]
    in_current_not_our_live = [mid for mid in current_ids if not our_live.get(mid)]

    print(f"In BO3 'current' and our snapshot OK: {len(in_current_our_live)}  {in_current_our_live}")
    print(f"In BO3 'current' but our snapshot FAIL: {len(in_current_not_our_live)}  {in_current_not_our_live}")
    print(f"Only in 'upcoming' but our snapshot OK (BO3 filter may have missed these): {len(only_upcoming_our_live)}  {only_upcoming_our_live}")

    if only_upcoming_our_live:
        print("\n  -> At least one match is in 'upcoming' only yet has a working snapshot (our live).")
        print("  -> Consider pulling current+upcoming and filtering by our snapshot/live_coverage.")
    else:
        print("\n  -> No matches in 'upcoming' had a working snapshot; BO3 'current' aligns with our test.")


if __name__ == "__main__":
    asyncio.run(main())

"""
Compare raw BO3 live-matches API response vs app list_live_matches() output.
Run from project root: python scripts/bo3_live_matches_compare.py

Shows whether the app drops any matches and why (missing id, bet_updates structure, etc.).
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


async def raw_pull():
    """Same request as cs2api get_live_matches(): GET /matches with widget-matches params."""
    import aiohttp
    url = "https://api.bo3.gg/api/v1/matches"
    params = {
        "scope": "widget-matches",
        "page[offset]": 0,
        "page[limit]": 100,
        "sort": "tier_rank,-start_date",
        "filter[matches.status][in]": "current",
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
            return await resp.json()


async def main():
    print("=== 1) Raw API pull (same URL/params as cs2api get_live_matches) ===\n")
    try:
        raw = await raw_pull()
    except Exception as e:
        print(f"Raw pull failed: {e}", file=sys.stderr)
        return
    if isinstance(raw, list):
        raw_items = raw
    elif isinstance(raw, dict):
        raw_items = raw.get("results", raw.get("matches", raw.get("data", [])))
    else:
        raw_items = []
    if not isinstance(raw_items, list):
        raw_items = []
    raw_total = len(raw_items)
    if isinstance(raw, dict) and "total" in raw:
        tot = raw["total"]
        if isinstance(tot, dict) and "count" in tot:
            raw_total = tot["count"]
        elif isinstance(tot, (int, float)):
            raw_total = int(tot)
    print(f"Raw response keys: {list(raw.keys()) if isinstance(raw, dict) else 'list'}")
    print(f"Raw items (results) count: {len(raw_items)} (API total.count: {raw_total})")
    if raw_total > len(raw_items):
        print(f"  -> API reports {raw_total} total but only {len(raw_items)} in this page (page[limit]=100; pagination not implemented).")
    if raw_items and isinstance(raw_items[0], dict):
        print(f"First raw item top-level keys: {list(raw_items[0].keys())}")
    raw_ids = []
    for i, m in enumerate(raw_items):
        if not isinstance(m, dict):
            raw_ids.append(None)
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        raw_ids.append(mid)
    print(f"Raw match ids: {raw_ids}\n")

    print("=== 2) App path (engine.ingest.bo3_client.list_live_matches) ===\n")
    try:
        from engine.ingest.bo3_client import list_live_matches
        app_list = await list_live_matches()
    except Exception as e:
        print(f"App pull failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return
    app_ids = [m.get("id") for m in app_list if isinstance(m, dict)]
    print(f"App list count: {len(app_list)}")
    print(f"App match ids: {app_ids}\n")

    print("=== 3) Compare ===\n")
    raw_id_set = {x for x in raw_ids if x is not None}
    app_id_set = set(app_ids)
    only_raw = raw_id_set - app_id_set
    only_app = app_id_set - raw_id_set
    if only_app:
        print(f"Only in app (unexpected): {only_app}")
    if only_raw:
        print(f"Only in raw (dropped by app): {only_raw}")
        for mid in only_raw:
            idx = next((i for i, x in enumerate(raw_ids) if x == mid), None)
            if idx is None:
                continue
            m = raw_items[idx]
            reason = []
            if m.get("id") is None and m.get("match_id") is None and m.get("matchId") is None:
                reason.append("missing id/match_id/matchId")
            bet = m.get("bet_updates")
            if not isinstance(bet, dict):
                reason.append("bet_updates missing or not dict")
            else:
                t1 = bet.get("team_1")
                t2 = bet.get("team_2")
                if not isinstance(t1, dict):
                    reason.append("bet_updates.team_1 missing or not dict")
                if not isinstance(t2, dict):
                    reason.append("bet_updates.team_2 missing or not dict")
            print(f"  match id {mid}: reason(s) dropped = {reason or 'unknown'}")
    if not only_raw and not only_app and raw_id_set == app_id_set:
        print("Raw and app lists match (same ids, same count).")
    elif len(raw_items) != len(app_list):
        print(f"Count mismatch: raw={len(raw_items)} app={len(app_list)}")


if __name__ == "__main__":
    asyncio.run(main())

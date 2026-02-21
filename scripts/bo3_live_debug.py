"""
One-off debug: fetch live matches, then try snapshot for each. Print full errors and responses.
Run from project root: python scripts/bo3_live_debug.py
"""
import asyncio
import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


async def main():
    try:
        from cs2api import CS2
    except ImportError as e:
        print(f"ImportError: {e}", file=sys.stderr)
        print("pip install cs2api>=0.1.3", file=sys.stderr)
        return

    print("=== Fetching live matches ===")
    try:
        async with CS2() as cs2:
            raw = await cs2.get_live_matches()
        print(f"Raw type: {type(raw)}")
        if isinstance(raw, dict):
            print(f"Keys: {list(raw.keys())}")
            items = raw.get("results", raw.get("matches", raw.get("data", [])))
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        print(f"Number of matches: {len(items) if isinstance(items, list) else 'N/A'}")
        if isinstance(items, list) and items:
            for i, m in enumerate(items[:10]):
                if isinstance(m, dict):
                    mid = m.get("id") or m.get("match_id") or m.get("matchId")
                    bet = m.get("bet_updates") or {}
                    t1 = bet.get("team_1") or {}
                    t2 = bet.get("team_2") or {}
                    n1 = (t1.get("name") if isinstance(t1, dict) else None) or "?"
                    n2 = (t2.get("name") if isinstance(t2, dict) else None) or "?"
                    print(f"  [{i}] id={mid}  {n1} vs {n2}")
        else:
            print("  (no matches or unexpected structure)")
            if isinstance(raw, dict):
                print(f"  Raw (truncated): {json.dumps(raw, default=str)[:500]}")
    except Exception as e:
        print(f"ERROR get_live_matches: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    print("\n=== Trying snapshot for each match ===")
    match_ids = []
    if isinstance(items, list):
        for m in items:
            if isinstance(m, dict):
                mid = m.get("id") or m.get("match_id") or m.get("matchId")
                if mid is not None:
                    match_ids.append(int(mid))

    if not match_ids:
        print("  No match IDs to try.")
        return

    for mid in match_ids[:15]:  # cap at 15
        try:
            async with CS2() as cs2:
                snap = await cs2.get_live_match_snapshot(mid)
            if snap and isinstance(snap, dict):
                t1 = snap.get("team_one") or {}
                t2 = snap.get("team_two") or {}
                n1 = (t1.get("name") if isinstance(t1, dict) else None) or "?"
                n2 = (t2.get("name") if isinstance(t2, dict) else None) or "?"
                s1 = t1.get("score") if isinstance(t1, dict) else None
                s2 = t2.get("score") if isinstance(t2, dict) else None
                map_name = snap.get("map_name") or "?"
                print(f"  match_id={mid}  OK  {n1} vs {n2}  score {s1}-{s2}  map={map_name}")
            else:
                print(f"  match_id={mid}  EMPTY  (snap type={type(snap)}, truthy={bool(snap)})")
        except Exception as e:
            print(f"  match_id={mid}  ERROR  {type(e).__name__}: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())

"""
BO3.gg full API pull: fetch live matches, find one by team name (e.g. ex-FUT vs FOKUS REALITY),
then get_live_match_snapshot(match_id) and save the complete raw JSON + print all keys for mid-round inventory.

Usage (from project root):
  python scripts/bo3_full_pull.py
  python scripts/bo3_full_pull.py --match-id 12345
  python scripts/bo3_full_pull.py --search "ex-fut" "fokus"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOGS_DIR = ROOT / "logs"

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(str(s).lower().split())


def _match_contains(m: dict, *terms: str) -> bool:
    """True if match has any of the terms in team names (from bet_updates or snapshot)."""
    bet = m.get("bet_updates") or {}
    t1 = bet.get("team_1") or {}
    t2 = bet.get("team_2") or {}
    for d in (t1, t2):
        if isinstance(d, dict):
            n = (d.get("name") or d.get("team_name") or "").strip()
            if any(t.lower() in _normalize(n) for t in terms):
                return True
    for key in ("team_one", "team_two", "team_1", "team_2"):
        obj = m.get(key)
        if isinstance(obj, dict):
            n = (obj.get("name") or obj.get("team_name") or "").strip()
            if any(t.lower() in _normalize(n) for t in terms):
                return True
    return False


def deep_keys(obj, prefix: str = "") -> list[str]:
    """Recursively collect key paths (e.g. team_one.players[0].health)."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            out.append(p)
            out.extend(deep_keys(v, p))
    elif isinstance(obj, list) and obj:
        # Sample first 2 elements for structure
        for i, item in enumerate(obj[:2]):
            out.extend(deep_keys(item, f"{prefix}[{i}]"))
    return out


async def main():
    parser = argparse.ArgumentParser(description="BO3.gg full API pull: live matches + full snapshot for one match")
    parser.add_argument("--match-id", type=int, help="Use this match ID directly (skip search)")
    parser.add_argument("--search", nargs="+", default=["ex-fut", "fokus", "reality"],
                        help="Terms to search for in team names (default: ex-fut fokus reality)")
    parser.add_argument("--out-dir", type=Path, default=LOGS_DIR, help="Directory to write raw JSON")
    args = parser.parse_args()

    try:
        from cs2api import CS2
    except ImportError:
        print("pip install cs2api>=0.1.3", file=sys.stderr)
        return 1

    # 1) Fetch live matches
    print("Fetching BO3.gg live matches (get_live_matches)...")
    async with CS2() as cs2:
        raw_matches = await cs2.get_live_matches()

    if isinstance(raw_matches, list):
        items = raw_matches
    elif isinstance(raw_matches, dict):
        items = raw_matches.get("results", raw_matches.get("matches", raw_matches.get("data", [])))
        if not isinstance(items, list):
            items = []
    else:
        items = []

    print(f"  Returned {len(items)} match(es)")
    if not items:
        print("  No live matches. Run again when a match is live, or use --match-id if you have an ID.")
        # Save raw response for inspection
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / "bo3_full_pull_live_matches_raw.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(raw_matches, f, indent=2, default=str)
        print(f"  Saved raw get_live_matches response -> {out_path}")
        return 0

    # List all matches
    for i, m in enumerate(items):
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        bet = m.get("bet_updates") or {}
        t1 = bet.get("team_1") or {}
        t2 = bet.get("team_2") or {}
        n1 = (t1.get("name") if isinstance(t1, dict) else None) or "?"
        n2 = (t2.get("name") if isinstance(t2, dict) else None) or "?"
        print(f"  [{i}] id={mid}  {n1} vs {n2}")

    # 2) Resolve match_id
    match_id = args.match_id
    if match_id is None:
        terms = [t.strip() for t in args.search if t.strip()]
        found = None
        for m in items:
            if isinstance(m, dict) and _match_contains(m, *terms):
                found = m
                break
        if found is not None:
            match_id = found.get("id") or found.get("match_id") or found.get("matchId")
            print(f"\nFound match by search {terms!r}: id={match_id}")
        else:
            # Use first match so we still get a full pull for inspection
            first = items[0] if isinstance(items[0], dict) else {}
            match_id = first.get("id") or first.get("match_id") or first.get("matchId")
            print(f"\nNo match matching {terms!r}; using first match id={match_id}")

    if match_id is None:
        print("No match_id available.")
        return 1

    # Save raw live matches for inspection
    args.out_dir.mkdir(parents=True, exist_ok=True)
    live_path = args.out_dir / "bo3_full_pull_live_matches_raw.json"
    with open(live_path, "w", encoding="utf-8") as f:
        json.dump(raw_matches, f, indent=2, default=str)
    print(f"\nSaved raw get_live_matches -> {live_path}")

    # 3) Full snapshot pull (try chosen match_id first, then others on 404)
    all_ids = [m.get("id") or m.get("match_id") or m.get("matchId") for m in items if isinstance(m, dict)]
    all_ids = [int(x) for x in all_ids if x is not None]
    if match_id not in all_ids:
        all_ids.insert(0, match_id)
    snapshot = None
    used_id = None
    for mid in all_ids:
        print(f"  Trying snapshot match_id={mid}...")
        try:
            async with CS2() as cs2:
                snap = await cs2.get_live_match_snapshot(int(mid))
            if snap and isinstance(snap, dict):
                snapshot = snap
                used_id = mid
                print(f"  OK -> using match_id={mid}")
                break
        except Exception as e:
            print(f"  {mid}: {e}")
    if not snapshot or not isinstance(snapshot, dict):
        print("  No snapshot available for any live match (404 or empty). Check raw live_matches JSON.")
        return 0
    match_id = used_id or match_id

    # 4) Save full raw snapshot
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"bo3_full_pull_snapshot_{match_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"  Saved full raw snapshot -> {out_file}")

    # 5) Print key inventory (all paths for mid-round mechanics)
    print("\n--- Full snapshot key inventory (for mid-round mechanics) ---")
    print("Top-level keys:", list(snapshot.keys()))
    paths = sorted(set(deep_keys(snapshot)))
    print(f"All key paths ({len(paths)} total):")
    for p in paths:
        print(f"  {p}")
    if len(paths) > 100:
        print("  ... (showing all)")

    # 6) Highlight fields relevant to mid-round
    print("\n--- Snapshot fields relevant to mid-round ---")
    t1 = snapshot.get("team_one") or {}
    t2 = snapshot.get("team_two") or {}
    for label, team in (("team_one", t1), ("team_two", t2)):
        if not isinstance(team, dict):
            continue
        print(f"  {label}: name={team.get('name')} score={team.get('score')} match_score={team.get('match_score')} side={team.get('side')}")
        players = team.get("player_states") or team.get("players") or []
        if isinstance(players, list) and players:
            p0 = players[0] if isinstance(players[0], dict) else {}
            print(f"    player_states[0] keys: {list(p0.keys()) if p0 else []}")
    print(f"  map_name: {snapshot.get('map_name')}")
    print(f"  round_number: {snapshot.get('round_number')}")
    print(f"  game_number: {snapshot.get('game_number')}")
    print(f"  bomb_planted: {snapshot.get('bomb_planted')}")
    print(f"  round_time_remaining: {snapshot.get('round_time_remaining')} round_phase: {snapshot.get('round_phase')}")
    print(f"  game_state: {snapshot.get('game_state')} phase: {snapshot.get('phase')}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)

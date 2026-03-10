"""
Sandbox probe: BO3.gg live CS2 data via cs2api.
- Pull live matches, one snapshot; print keys + trimmed preview; save raw JSON.
- Report whether snapshot includes: map score, series score, current map,
  team sides, round state, alive/dead, economy/money.
Do NOT modify files outside this folder.
"""
from pathlib import Path
import asyncio
import json
from cs2api import CS2

SANDBOX_DIR = Path(__file__).resolve().parent
OUT_LIVE_MATCHES = SANDBOX_DIR / "raw_live_matches.json"
OUT_SNAPSHOT = SANDBOX_DIR / "raw_live_snapshot.json"

# Fields we want to confirm in the snapshot
FIELDS_TO_CHECK = [
    "map score",
    "series score",
    "current map",
    "team sides",
    "round state",
    "alive/dead",
    "economy/money",
]


def print_keys_and_preview(obj, label: str, max_str_len: int = 120):
    """Print top-level keys and a trimmed string preview."""
    print(f"\n--- {label} ---")
    if isinstance(obj, dict):
        print("Keys:", list(obj.keys()))
        preview = json.dumps(obj, indent=2, default=str)[:max_str_len]
    elif isinstance(obj, list):
        print("Length:", len(obj))
        if obj:
            print("First item keys:", list(obj[0].keys()) if isinstance(obj[0], dict) else type(obj[0]))
        preview = json.dumps(obj, indent=2, default=str)[:max_str_len]
    else:
        preview = str(obj)[:max_str_len]
    print("Preview:", preview + ("..." if len(str(obj)) > max_str_len else ""))


def deep_keys(obj, prefix=""):
    """Recursively collect key paths (e.g. 'match.teams[0].name')."""
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            keys.append(path)
            keys.extend(deep_keys(v, path))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, item in enumerate(obj[:2]):  # sample first 2
            keys.extend(deep_keys(item, f"{prefix}[{i}]"))
    return keys


def check_snapshot_fields(snapshot: dict) -> None:
    """Check snapshot for desired fields and print report."""
    all_paths = set(deep_keys(snapshot))
    flat_str = json.dumps(snapshot).lower()

    print("\n--- Snapshot field check ---")
    for name in FIELDS_TO_CHECK:
        # Look for key-like or value-like hints
        tokens = name.replace("/", " ").split()
        found_path = any(
            t in p.lower() for p in all_paths for t in tokens
        )
        found_anywhere = any(t in flat_str for t in tokens)
        status = "likely" if (found_path or found_anywhere) else "not found"
        print(f"  {name}: {status}")
    print("\nAll snapshot key paths (sample):")
    for p in sorted(all_paths)[:60]:
        print("  ", p)
    if len(all_paths) > 60:
        print("  ... and more.")


async def main():
    async with CS2() as cs2:
        # 1) Pull current live matches
        live_matches = await cs2.get_live_matches()
        print_keys_and_preview(live_matches, "Live matches")
        with open(OUT_LIVE_MATCHES, "w", encoding="utf-8") as f:
            json.dump(live_matches, f, indent=2, default=str)
        print(f"\nSaved raw live matches -> {OUT_LIVE_MATCHES}")

        # 2) Pull one live match snapshot
        match_id = None
        if isinstance(live_matches, list) and live_matches:
            first = live_matches[0]
            match_id = first.get("id") or first.get("match_id") or first.get("matchId")
        if not match_id and isinstance(live_matches, dict):
            lst = live_matches.get("results", live_matches.get("matches", live_matches.get("data", [])))
            if lst:
                first = lst[0] if isinstance(lst[0], dict) else lst
                match_id = first.get("id") or first.get("match_id") or first.get("matchId")

        if not match_id:
            print("\nNo match id found in live_matches; cannot fetch snapshot.")
            print("Live structure:", type(live_matches), json.dumps(live_matches, indent=2, default=str)[:800])
            return

        snapshot = await cs2.get_live_match_snapshot(match_id)
        print_keys_and_preview(snapshot, "Live match snapshot")
        with open(OUT_SNAPSHOT, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"\nSaved raw snapshot -> {OUT_SNAPSHOT}")

        # 4 & 5) Confirm fields
        if isinstance(snapshot, dict):
            check_snapshot_fields(snapshot)
        else:
            print("Snapshot is not a dict; field check skipped.")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

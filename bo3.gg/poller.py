"""
BO3.gg live data poller: reads control file, fetches snapshot every 5s, writes feed file.
Run from project root: python -m bo3.gg.poller
Or: python bo3.gg/poller.py (with PYTHONPATH=.)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Project root: parent of bo3.gg
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CONTROL_FILE = LOGS_DIR / "bo3_live_control.json"
FEED_FILE = LOGS_DIR / "bo3_live_feed.json"
INTERVAL_SEC = 5
STALE_SEC = 15


def _ensure_logs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def read_control() -> dict:
    if not CONTROL_FILE.exists():
        return {}
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_feed(payload: dict, error: str | None = None, team_a_is_team_one: bool = True):
    """Write feed file. payload is the full raw snapshot from the API (so the feed provides everything)."""
    _ensure_logs()
    out = {
        "timestamp": time.time(),
        "payload": payload,
        "team_a_is_team_one": team_a_is_team_one,
    }
    if error:
        out["error"] = error
        out["snapshot_status"] = "empty"
    else:
        out["snapshot_status"] = "live" if payload else "empty"
    try:
        with open(FEED_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


def main_loop():
    # Write feed immediately so app sees file exists (even if we crash on import or first fetch)
    write_feed({}, error="poller_starting")
    print("poller: status=empty error=poller_starting (first write)", file=sys.stderr, flush=True)

    # Import here so poller can run without app deps if needed
    try:
        from fair_odds.bo3_adapter import fetch_bo3_snapshot
    except ImportError:
        # Run from bo3.gg folder: add parent to path
        sys.path.insert(0, str(PROJECT_ROOT))
        from fair_odds.bo3_adapter import fetch_bo3_snapshot

    last_snapshot = {}
    while True:
        control = read_control()
        if not control.get("active"):
            # Exit when deactivated so app can restart when user activates again
            write_feed(last_snapshot, error="inactive", team_a_is_team_one=control.get("team_a_is_team_one", True))
            print("poller: exiting (active=false)", file=sys.stderr, flush=True)
            break
        match_id = control.get("match_id")
        team_a_is_team_one = control.get("team_a_is_team_one", True)
        if not match_id:
            write_feed(last_snapshot, error="no match_id in control", team_a_is_team_one=team_a_is_team_one)
            print("poller: status=empty error=no match_id in control", file=sys.stderr, flush=True)
            time.sleep(INTERVAL_SEC)
            continue
        try:
            snapshot = fetch_bo3_snapshot(int(match_id))
            if not snapshot:
                write_feed(last_snapshot, error="snapshot fetch returned empty", team_a_is_team_one=team_a_is_team_one)
                print("poller: status=empty error=snapshot fetch returned empty", file=sys.stderr, flush=True)
            else:
                last_snapshot = snapshot
                write_feed(snapshot, team_a_is_team_one=team_a_is_team_one)
                keys = ",".join(sorted(snapshot.keys())) if isinstance(snapshot, dict) else "(none)"
                print(f"poller: status=live payload_keys={keys}", file=sys.stderr, flush=True)
        except Exception as e:
            write_feed(last_snapshot, error=str(e), team_a_is_team_one=team_a_is_team_one)
            print(f"poller: status=empty error={e!r}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    main_loop()

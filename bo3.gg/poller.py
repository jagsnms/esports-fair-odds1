"""
BO3.gg live data poller: reads control file, fetches snapshot every 5s, writes feed file.
Run from project root: python -m bo3.gg.poller
Or: python bo3.gg/poller.py (with PYTHONPATH=.)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Project root: parent of bo3.gg
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
CONTROL_FILE = LOGS_DIR / "bo3_live_control.json"
FEED_FILE = LOGS_DIR / "bo3_live_feed.json"
BO3_RECORD_DEFAULT = LOGS_DIR / "bo3_pulls.jsonl"
INTERVAL_SEC = 5
STALE_SEC = 15
ENV_BO3_RECORD_JSONL = "BO3_RECORD_JSONL"


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


def write_feed(
    payload: dict,
    error: str | None = None,
    team_a_is_team_one: bool = True,
    raw_ts_utc: str | None = None,
    raw_record_path: str | None = None,
):
    """Write feed file. payload is the full raw snapshot from the API (so the feed provides everything)."""
    _ensure_logs()
    out = {
        "timestamp": time.time(),
        "payload": payload,
        "team_a_is_team_one": team_a_is_team_one,
    }
    if raw_ts_utc:
        out["raw_ts_utc"] = raw_ts_utc
    if raw_record_path:
        out["raw_record_path"] = raw_record_path
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

    try:
        from fair_odds.bo3_adapter import fetch_bo3_snapshot
        from fair_odds.record_jsonl import append_jsonl, utc_iso_z
    except ImportError:
        sys.path.insert(0, str(PROJECT_ROOT))
        from fair_odds.bo3_adapter import fetch_bo3_snapshot
        from fair_odds.record_jsonl import append_jsonl, utc_iso_z

    last_snapshot = {}
    while True:
        control = read_control()
        if not control.get("active"):
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

        record_enabled = control.get("record_enabled", True)
        record_path = None
        if record_enabled:
            env_path = os.environ.get(ENV_BO3_RECORD_JSONL)
            if env_path and str(env_path).strip():
                record_path = Path(str(env_path).strip())
            elif control.get("record_path"):
                record_path = Path(str(control["record_path"]).strip())
            else:
                record_path = BO3_RECORD_DEFAULT

        snapshot = None
        error_msg = None
        try:
            snapshot = fetch_bo3_snapshot(int(match_id), record_jsonl_path=None)
            if not snapshot:
                error_msg = "snapshot fetch returned empty"
        except Exception as e:
            error_msg = str(e)

        if record_path:
            try:
                rec = {
                    "ts_utc": utc_iso_z(),
                    "source": "BO3",
                    "label": "live_snapshot",
                    "match_id": int(match_id),
                    "team_a_is_team_one": bool(team_a_is_team_one),
                    "ok": bool(snapshot),
                    "error": error_msg,
                    "payload": snapshot if isinstance(snapshot, dict) else {},
                }
                append_jsonl(record_path, rec)
            except Exception:
                pass

        if not snapshot:
            write_feed(last_snapshot, error=error_msg or "empty", team_a_is_team_one=team_a_is_team_one)
            print(f"poller: status=empty error={error_msg or 'snapshot fetch returned empty'}", file=sys.stderr, flush=True)
        else:
            last_snapshot = snapshot
            write_feed(
                snapshot,
                team_a_is_team_one=team_a_is_team_one,
                raw_ts_utc=rec.get("ts_utc") if record_path else None,
                raw_record_path=str(record_path) if record_path else None,
            )
            keys = ",".join(sorted(snapshot.keys())) if isinstance(snapshot, dict) else "(none)"
            print(f"poller: status=live payload_keys={keys}", file=sys.stderr, flush=True)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    main_loop()


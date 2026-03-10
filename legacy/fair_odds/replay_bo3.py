"""
BO3 replay: load JSONL recordings and write replay frames into the feed file contract.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List


def load_bo3_jsonl(path: Path) -> List[dict]:
    """
    Read JSONL file; return list of dicts.
    Filter to entries where source == "BO3" and label in ("live_snapshot", "snapshot").
    Never throw; return [] on error.
    """
    out: List[dict] = []
    try:
        path = Path(path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("source") != "BO3":
                    continue
                if obj.get("label") not in ("live_snapshot", "snapshot"):
                    continue
                out.append(obj)
    except Exception:
        pass
    return out


def group_bo3_by_match(entries: List[dict]) -> Dict[int, List[dict]]:
    """
    Build {match_id: [entries...]} sorted by ts_utc (ISO string) if present.
    Only include entries with ok is True and payload is dict.
    """
    groups: Dict[int, List[dict]] = {}
    for e in entries:
        if e.get("ok") is not True:
            continue
        payload = e.get("payload")
        if not isinstance(payload, dict):
            continue
        mid = e.get("match_id")
        if mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        if mid not in groups:
            groups[mid] = []
        groups[mid].append(e)
    for mid in groups:
        lst = groups[mid]
        lst.sort(key=lambda x: (x.get("ts_utc") or ""))
    return groups


def write_replay_frame_to_feed(
    feed_path: Path,
    payload: dict,
    team_a_is_team_one: bool,
    error: str | None = None,
) -> None:
    """
    Write one replay frame to feed_path in the same shape as the poller.
    snapshot_status is set to "replay".
    """
    try:
        feed_path = Path(feed_path)
        feed_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "timestamp": time.time(),
            "payload": payload,
            "team_a_is_team_one": team_a_is_team_one,
            "snapshot_status": "replay",
        }
        if error is not None:
            out["error"] = error
        with open(feed_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass

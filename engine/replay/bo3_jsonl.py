"""
Load and iterate BO3 JSONL recorder format (e.g. logs/bo3_pulls.jsonl).
Entries: source==BO3, ok is True, payload is dict, match_id present.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional


def _coerce_match_id(obj: dict) -> Optional[int]:
    mid = obj.get("match_id")
    if mid is None:
        return None
    try:
        return int(mid)
    except (TypeError, ValueError):
        return None


def load_bo3_jsonl_entries(path: str) -> list[dict]:
    """
    Read JSONL file; keep lines where source==BO3, ok is True, payload is dict, match_id exists (int-coercible).
    Sort by obj.get("ts_utc") if present, else keep file order.
    """
    entries: list[dict] = []
    try:
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
                if obj.get("ok") is not True:
                    continue
                if not isinstance(obj.get("payload"), dict):
                    continue
                if _coerce_match_id(obj) is None:
                    continue
                entries.append(obj)
    except FileNotFoundError:
        return []
    except OSError:
        return []

    if entries and any("ts_utc" in e for e in entries):
        try:
            entries.sort(key=lambda e: (e.get("ts_utc") or ""))
        except (TypeError, ValueError):
            pass
    return entries


def group_by_match(entries: list[dict]) -> dict[int, list[dict]]:
    """Group entries by match_id. Optional helper."""
    out: dict[int, list[dict]] = {}
    for e in entries:
        mid = _coerce_match_id(e)
        if mid is not None:
            out.setdefault(mid, []).append(e)
    return out


def iter_payloads(
    entries: list[dict],
    match_id: Optional[int] = None,
) -> Iterator[tuple[int, dict]]:
    """
    Yield (match_id, payload_dict) in order.
    If match_id is provided, yield only that match's entries; else yield all in order.
    """
    for e in entries:
        mid = _coerce_match_id(e)
        if mid is None:
            continue
        if match_id is not None and mid != match_id:
            continue
        payload = e.get("payload")
        if isinstance(payload, dict):
            yield (mid, payload)

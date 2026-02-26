"""
Async BO3 client using cs2api. No asyncio.run; caller is already async.
"""
from __future__ import annotations

from typing import Any

try:
    from cs2api import CS2
except ImportError:
    CS2 = None  # type: ignore


async def list_live_matches() -> list[dict[str, Any]]:
    """Return list of live matches: id, team1_name, team2_name, bo_type."""
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")
    async with CS2() as cs2:
        raw = await cs2.get_live_matches()
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("results", raw.get("matches", raw.get("data", [])))
        if not isinstance(items, list):
            items = []
    else:
        items = []
    out: list[dict[str, Any]] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id") or m.get("match_id") or m.get("matchId")
        if mid is None:
            continue
        bet = m.get("bet_updates") or {}
        t1 = bet.get("team_1") if isinstance(bet, dict) else {}
        t2 = bet.get("team_2") if isinstance(bet, dict) else {}
        t1 = t1 if isinstance(t1, dict) else {}
        t2 = t2 if isinstance(t2, dict) else {}
        name1 = t1.get("name", "Team 1") if isinstance(t1.get("name"), str) else "Team 1"
        name2 = t2.get("name", "Team 2") if isinstance(t2.get("name"), str) else "Team 2"
        bo_type = m.get("bo_type", 3)
        out.append({
            "id": mid,
            "team1_name": str(name1),
            "team2_name": str(name2),
            "bo_type": int(bo_type) if bo_type is not None else 3,
        })
    return out


async def get_snapshot(match_id: int) -> dict[str, Any] | None:
    """Fetch live match snapshot for match_id. Returns raw snapshot dict or None."""
    if CS2 is None:
        raise ImportError("cs2api is not installed. pip install cs2api>=0.1.3")
    async with CS2() as cs2:
        snap = await cs2.get_live_match_snapshot(match_id)
    return snap if isinstance(snap, dict) else None

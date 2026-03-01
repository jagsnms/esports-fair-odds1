"""
BO3 readiness cache: TTL-based cache for probe results; filter who needs probing; select top K telemetry-ready.
Pure helpers; no FastAPI. Used by runner for BO3 auto-track.
"""
from __future__ import annotations

from typing import Any

DEFAULT_TTL_S = 60.0

# Deterministic tier ranking: higher int = better tier. Unknown/None -> 0.
TIER_RANK_MAP = {"s": 5, "a": 4, "b": 3, "c": 2, "d": 1}


def tier_rank(tier: str | int | None) -> int:
    """
    Map tier to a numeric rank for sorting. Accepts str (e.g. 's', 'a', 'S', 'A') or None.
    s=5, a=4, b=3, c=2, d=1, unknown/None=0.
    """
    if tier is None:
        return 0
    if isinstance(tier, int):
        if tier in (1, 2, 3, 4, 5):
            return tier
        return 0
    key = str(tier).strip().lower() if tier else ""
    return TIER_RANK_MAP.get(key, 0)


def filter_needs_probe(
    match_ids: list[int],
    cache: dict[int, dict[str, Any]],
    now_ts: float,
    ttl_s: float = DEFAULT_TTL_S,
    budget: int = 40,
) -> list[int]:
    """
    Return match_ids that need probing: not in cache or entry older than ttl_s.
    Capped at budget. Order preserved from match_ids (first N that need probe).
    """
    needs: list[int] = []
    for mid in match_ids:
        if len(needs) >= budget:
            break
        entry = cache.get(mid)
        if entry is None:
            needs.append(mid)
            continue
        last_unix = entry.get("last_probe_ts_unix") or 0.0
        if now_ts - last_unix > ttl_s:
            needs.append(mid)
    return needs


def update_cache_from_results(
    cache: dict[int, dict[str, Any]],
    results: list[dict[str, Any]],
    now_ts: float,
) -> None:
    """Merge probe results into cache. Adds last_probe_ts_unix for TTL."""
    for r in results:
        mid = r.get("match_id")
        if mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        entry = {
            "telemetry_ready": bool(r.get("telemetry_ready", False)),
            "status_code": int(r.get("status_code", 0)) if r.get("status_code") is not None else 0,
            "reason": str(r.get("reason", "") or ""),
            "last_probe_ts": r.get("last_probe_ts"),
            "last_probe_ts_unix": now_ts,
        }
        cache[mid] = entry


def _candidate_rank_key(c: dict[str, Any], cache: dict[int, dict[str, Any]]) -> tuple:
    """
    Sort key for selection. Lower tuple = better.
    1) telemetry_ready True first (not ready -> True, so ready comes first)
    2) last_probe_ts_unix DESC -> use -ts so more recent = smaller key
    3) live_coverage True (not live -> True)
    4) tier_rank DESC -> use -tier_rank so higher tier = smaller key
    5) mid for stability
    """
    mid = c.get("id") or c.get("match_id")
    if mid is not None:
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            mid = 0
    else:
        mid = 0
    entry = cache.get(mid) if isinstance(mid, int) else None
    ready = bool(entry and entry.get("telemetry_ready", False))
    last_unix = float(entry.get("last_probe_ts_unix") or 0.0) if entry else 0.0
    live = bool(c.get("live_coverage", False))
    tr = tier_rank(c.get("tier"))
    return (not ready, -last_unix, not live, -tr, mid)


def select_telemetry_ready_match_ids(
    candidates: list[dict[str, Any]],
    cache: dict[int, dict[str, Any]],
    limit: int,
) -> list[int]:
    """
    From candidates, select up to limit match_ids that are telemetry_ready in cache.
    Sort: 1) telemetry_ready, 2) last_probe_ts_unix DESC (recent first), 3) live_coverage, 4) tier_rank DESC.
    Returns list of match_id ints.
    """
    ready_candidates = []
    candidate_ids = set()
    for c in candidates:
        mid = c.get("id") or c.get("match_id")
        if mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        if mid in candidate_ids:
            continue
        entry = cache.get(mid)
        if not entry or not entry.get("telemetry_ready", False):
            continue
        candidate_ids.add(mid)
        ready_candidates.append(c)
    ready_candidates.sort(key=lambda c: _candidate_rank_key(c, cache))
    out: list[int] = []
    seen: set[int] = set()
    for c in ready_candidates:
        mid = c.get("id") or c.get("match_id")
        if mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue
        if mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
        if len(out) >= limit:
            break
    return out

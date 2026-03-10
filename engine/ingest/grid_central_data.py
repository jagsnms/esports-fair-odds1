"""
GRID Central Data: CS2 series discovery (titleId=28). Cached, rate-limited.
No FastAPI; reuse GRID_API_KEY from env. Uses allSeries with titleIds filter.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# CS2 titleId from GRID (Counter Strike 2)
GRID_CS2_TITLE_ID = "28"

CENTRAL_DATA_GRAPHQL_URL = "https://api-op.grid.gg/central-data/graphql"
CACHE_MIN_REFRESH_S = 30.0
DEFAULT_ORDER_DIRECTION = "ASC"  # soonest first (StartTimeScheduled ASC)

# Live data feed level ranking (higher = better for telemetry)
LEVEL_RANK = {"FULL": 3, "PARTIAL": 2, "UNKNOWN": 1, "NONE": 0}

# In-memory cache (keyed by limit + order_direction so ordering is correct)
_cache: dict[str, Any] = {"result": [], "ts": 0.0, "limit": 25, "order_direction": DEFAULT_ORDER_DIRECTION}

QUERY_CS2_SERIES = """
query CS2SeriesCandidates($orderBy: SeriesOrderBy!, $orderDirection: OrderDirection!, $first: Int!, $filter: SeriesFilter) {
  allSeries(
    orderBy: $orderBy,
    orderDirection: $orderDirection,
    first: $first,
    filter: $filter
  ) {
    totalCount
    edges {
      cursor
      node {
        id
        title { name nameShortened }
        tournament { id name }
        startTimeScheduled
        updatedAt
        type
        productServiceLevels
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def _load_api_key(env_path: Path | None = None) -> str:
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        if env_path and env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()
    key = os.environ.get("GRID_API_KEY")
    if not key or not str(key).strip():
        raise ValueError("GRID_API_KEY not set in environment or .env")
    return str(key).strip()


def normalize_service_levels(product_service_levels: Any) -> dict[str, str]:
    """
    Normalize GRID productServiceLevels to a dict of product -> level.
    e.g. {"liveDataFeed": "FULL", "centralData": "FULL"}.
    Returns {} if missing/unknown structure.
    """
    if not product_service_levels:
        return {}
    if not isinstance(product_service_levels, list):
        return {}
    out: dict[str, str] = {}
    for item in product_service_levels:
        if isinstance(item, dict):
            # e.g. {"product": "liveDataFeed", "level": "FULL"} or {"name": "...", "level": "..."}
            key = item.get("product") or item.get("name") or item.get("service")
            level = item.get("level")
            if key is not None and level is not None:
                out[str(key)] = str(level)
        elif isinstance(item, str):
            # Single string or list of strings: no product key, skip (unknown which product)
            continue
    return out


def live_data_feed_level(normalized: dict[str, str]) -> str:
    """Return liveDataFeed level: FULL, PARTIAL, NONE, or UNKNOWN."""
    level = normalized.get("liveDataFeed") or normalized.get("live_data_feed")
    if level in ("FULL", "PARTIAL", "NONE"):
        return level
    return "UNKNOWN"


def service_level_rank(level: str) -> int:
    """FULL=3, PARTIAL=2, UNKNOWN=1, NONE=0."""
    return LEVEL_RANK.get(level, LEVEL_RANK["UNKNOWN"])


def _post_graphql(url: str, query: str, variables: dict[str, Any], api_key: str) -> dict[str, Any]:
    try:
        import requests
    except ImportError:
        raise ImportError("requests required for GRID Central Data")
    r = requests.post(
        url,
        json={"query": query, "variables": variables},
        headers={"Content-Type": "application/json", "x-api-key": api_key},
        timeout=20,
    )
    if r.status_code != 200:
        return {"errors": [{"message": f"HTTP {r.status_code}"}]}
    try:
        return r.json()
    except Exception as e:
        return {"errors": [{"message": str(e)}]}


def _normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    """Build a normalized candidate dict; handle missing fields gracefully."""
    out: dict[str, Any] = {
        "series_id": None,
        "name": None,
        "tournament_name": None,
        "start_time": None,
        "updated_at": None,
        "status": None,
        "finished": None,
        "product_service_levels": None,
        "live_data_feed_level": "UNKNOWN",
        "live_data_feed_rank": service_level_rank("UNKNOWN"),
    }
    sid = node.get("id")
    if sid is not None:
        out["series_id"] = str(sid)
    title = node.get("title")
    if isinstance(title, dict):
        out["name"] = title.get("name") or title.get("nameShortened")
    elif title is not None:
        out["name"] = str(title)
    tournament = node.get("tournament")
    if isinstance(tournament, dict):
        out["tournament_name"] = tournament.get("name")
    out["start_time"] = node.get("startTimeScheduled")
    out["updated_at"] = node.get("updatedAt")
    out["type"] = node.get("type")
    psl = node.get("productServiceLevels")
    if isinstance(psl, list):
        out["product_service_levels"] = [x for x in psl if x is not None]
    elif psl is not None:
        out["product_service_levels"] = [psl]
    normalized_sl = normalize_service_levels(psl)
    level = live_data_feed_level(normalized_sl)
    out["live_data_feed_level"] = level
    out["live_data_feed_rank"] = service_level_rank(level)
    return out


def fetch_cs2_series_candidates(
    limit: int = 25,
    order_direction: str = DEFAULT_ORDER_DIRECTION,
    api_key: str | None = None,
    env_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch CS2 series candidates from Central Data (titleId=28).
    Default order: StartTimeScheduled ASC (soonest first).
    Returns normalized list including series_id, name, tournament_name, start_time,
    live_data_feed_level, live_data_feed_rank, product_service_levels, etc.
    """
    key = api_key or _load_api_key(env_path)
    direction = "ASC" if str(order_direction).upper() == "ASC" else "DESC"
    variables = {
        "orderBy": "StartTimeScheduled",
        "orderDirection": direction,
        "first": min(100, max(1, limit)),
        "filter": {"titleIds": {"in": [GRID_CS2_TITLE_ID]}},
    }
    data = _post_graphql(CENTRAL_DATA_GRAPHQL_URL, QUERY_CS2_SERIES.strip(), variables, key)
    if data.get("errors"):
        return []
    conn = (data.get("data") or {}).get("allSeries")
    if not isinstance(conn, dict):
        return []
    edges = conn.get("edges") or []
    result = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node = edge.get("node")
        if isinstance(node, dict):
            result.append(_normalize_node(node))
    return _sort_candidates_soonest_relevant(result)


def _sort_candidates_soonest_relevant(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by soonest start_time first, then by live_data_feed_rank descending (FULL best)."""
    def key(c: dict[str, Any]) -> tuple[str, int]:
        st = c.get("start_time")
        start_val = st if st is not None else "9999-99-99"
        rank = -(c.get("live_data_feed_rank") or 0)
        return (str(start_val), rank)
    return sorted(candidates, key=key)


def _auto_track_sort_key(c: dict[str, Any]) -> tuple[int, str, str]:
    """Sort key for auto-track: rank DESC, start_time ASC (None last), updated_at DESC."""
    rank = -(c.get("live_data_feed_rank") or 0)
    st = c.get("start_time")
    start_val = st if st is not None else "9999-99-99"
    updated = c.get("updated_at")
    updated_val = (updated if isinstance(updated, str) else str(updated)) if updated is not None else ""
    return (rank, str(start_val), updated_val)


def select_best_series_ids(
    candidates: list[dict[str, Any]],
    limit: int,
    min_rank: int = 2,
    allow_unknown_fallback: bool = True,
) -> list[str]:
    """
    Select best series IDs for auto-track: primary rank DESC, secondary start_time ASC, tertiary updated_at DESC.
    Only include candidates with live_data_feed_rank >= min_rank (default 2 = FULL or PARTIAL).
    If none match and allow_unknown_fallback, include up to limit with rank >= 1 (UNKNOWN).
    Skips candidates with missing series_id.
    """
    sorted_candidates = sorted(candidates, key=_auto_track_sort_key)
    selected: list[str] = []
    for c in sorted_candidates:
        sid = c.get("series_id")
        if sid is None or not str(sid).strip():
            continue
        rank = c.get("live_data_feed_rank") or 0
        if rank >= min_rank:
            selected.append(str(sid).strip())
            if len(selected) >= limit:
                break
    if not selected and allow_unknown_fallback:
        for c in sorted_candidates:
            sid = c.get("series_id")
            if sid is None or not str(sid).strip():
                continue
            rank = c.get("live_data_feed_rank") or 0
            if rank >= 1:
                selected.append(str(sid).strip())
                if len(selected) >= limit:
                    break
    return selected


def get_cs2_series_candidates(
    limit: int = 25,
    order_direction: str = DEFAULT_ORDER_DIRECTION,
    cache_interval_s: float = CACHE_MIN_REFRESH_S,
    env_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return CS2 series candidates; use in-memory cache for cache_interval_s.
    Default order ASC (soonest first).
    """
    direction = "ASC" if str(order_direction).upper() == "ASC" else "DESC"
    now = time.time()
    if (
        now - _cache["ts"] >= cache_interval_s
        or _cache.get("limit") != limit
        or _cache.get("order_direction") != direction
    ):
        try:
            _cache["result"] = fetch_cs2_series_candidates(
                limit=limit, order_direction=direction, env_path=env_path
            )
            _cache["ts"] = now
            _cache["limit"] = limit
            _cache["order_direction"] = direction
        except Exception:
            pass
    return list(_cache.get("result", []))


def clear_cs2_candidates_cache() -> None:
    """Reset cache (e.g. for tests)."""
    global _cache
    _cache = {
        "result": [],
        "ts": 0.0,
        "limit": 25,
        "order_direction": DEFAULT_ORDER_DIRECTION,
    }

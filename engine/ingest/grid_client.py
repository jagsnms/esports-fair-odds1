"""
GRID ingest: fetch series state via legacy GraphQL client (auth, connection).
Reuses adapters/grid_probe transport; no FastAPI deps.
"""
from __future__ import annotations

from typing import Any


def fetch_series_state(series_id: str, api_key: str | None = None) -> dict[str, Any]:
    """
    Fetch live series state for one series. Calls legacy GRID GraphQL client.
    Returns dict with "data" -> "seriesState" on success, or "errors" on failure.
    """
    try:
        from adapters.grid_probe.grid_graphql_client import (
            SERIES_STATE_GRAPHQL_URL,
            load_api_key,
            post_graphql,
        )
        from adapters.grid_probe.grid_queries import QUERY_SERIES_STATE_RICH
    except ImportError:
        return {"errors": [{"message": "GRID adapter not available (adapters.grid_probe)"}]}
    if not series_id or not str(series_id).strip():
        return {"errors": [{"message": "series_id required"}]}
    key = api_key
    if not key:
        try:
            key = load_api_key()
        except Exception as e:
            return {"errors": [{"message": f"GRID API key: {e}"}]}
    out = post_graphql(
        SERIES_STATE_GRAPHQL_URL,
        QUERY_SERIES_STATE_RICH,
        variables={"id": str(series_id).strip()},
        api_key=key,
        timeout=15,
    )
    return out

# GRID PROBE V1 — Standalone probe for GRID Open Access CS2 (Central Data + Live Series State).
# New files only. Does NOT modify app35_ml.py or BO3 pipeline.
"""
Run from project root: python -m adapters.grid_probe.probe_grid_cs2
Or: python adapters/grid_probe/probe_grid_cs2.py (with PYTHONPATH=.)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# GRID PROBE V1 — config (edit as needed)
MANUAL_SERIES_ID: Optional[str] = None
USE_MANUAL_SERIES_ID_FIRST: bool = False
SAVE_PRETTY_JSON: bool = True

# GRID PROBE V1 — Central Data allSeries (required: orderBy, orderDirection)
CENTRAL_ORDER_BY: str = "StartTimeScheduled"
CENTRAL_ORDER_DIRECTION: str = "DESC"
CENTRAL_FIRST: int = 10
CENTRAL_MINIMAL_MODE: bool = True  # True = use QUERY_FIND_CS2_SERIES_MINIMAL (node { id } only)
# GRID PROBE V1 — filter mode: live_only | live_games_empty | recent_only
CENTRAL_FILTER_MODE: str = "live_games_empty"

# Output paths (under this module dir)
PROBE_DIR = Path(__file__).resolve().parent
RAW_CENTRAL_PATH = PROBE_DIR / "raw_grid_central_data.json"
RAW_SERIES_STATE_PATH = PROBE_DIR / "raw_grid_series_state.json"

# Project root for .env (one level up from adapters)
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if SAVE_PRETTY_JSON:
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            json.dump(data, f, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def _get_central_filter() -> dict[str, Any]:
    # GRID PROBE V1 — build filter from CENTRAL_FILTER_MODE (live_only | live_games_empty | recent_only)
    mode = (CENTRAL_FILTER_MODE or "live_only").strip().lower()
    if mode == "live_games_empty":
        return {"live": {"games": {}}}
    if mode == "recent_only":
        gte = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return {"updatedAt": {"gte": gte}}
    # default: live_only
    return {"live": {}}


def _extract_series_ids(payload: dict) -> list[str]:
    # GRID PROBE V1 — extract series IDs from allSeries connection: data.allSeries.edges[].node.id
    ids: list[str] = []
    data = payload.get("data") or {}
    all_series = data.get("allSeries")
    if not isinstance(all_series, dict):
        return ids
    edges = all_series.get("edges")
    if not isinstance(edges, list):
        return ids
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node = edge.get("node")
        if isinstance(node, dict) and node.get("id"):
            ids.append(str(node["id"]))
    return ids


def main() -> None:
    # GRID PROBE V1 — ensure probe dir on path so imports work when run as script or -m
    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import (
        CENTRAL_DATA_GRAPHQL_URL,
        SERIES_STATE_GRAPHQL_URL,
        load_api_key,
        post_graphql,
    )
    from grid_queries import (
        QUERY_FIND_CS2_SERIES,
        QUERY_FIND_CS2_SERIES_MINIMAL,
        QUERY_SERIES_STATE,
    )

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)
    print("[GRID PROBE V1] API key loaded (not printed).")

    series_id: Optional[str] = MANUAL_SERIES_ID if USE_MANUAL_SERIES_ID_FIRST else None

    # Step 1: Central Data query (allSeries with orderBy, orderDirection + filter)
    if not series_id:
        print("[GRID PROBE V1] Step 1: Central Data query (allSeries) ...")
        central_query = QUERY_FIND_CS2_SERIES_MINIMAL if CENTRAL_MINIMAL_MODE else QUERY_FIND_CS2_SERIES
        central_filter = _get_central_filter()
        central_vars = {
            "orderBy": CENTRAL_ORDER_BY,
            "orderDirection": CENTRAL_ORDER_DIRECTION,
            "first": CENTRAL_FIRST,
            "filter": central_filter,
        }
        # GRID PROBE V1 — print exact filter variables before request
        print("  Central variables (exact):", json.dumps(central_vars, indent=2))
        try:
            central = post_graphql(
                CENTRAL_DATA_GRAPHQL_URL,
                central_query,
                variables=central_vars,
                api_key=api_key,
            )
        except Exception as e:
            central = {"errors": [{"message": str(e)}]}
        _save_json(RAW_CENTRAL_PATH, central)
        if central.get("errors"):
            print("  GraphQL errors (exact):")
            for err in central.get("errors", []):
                print("   ", err.get("message", err))
            print("  Enum values may differ; inspect GraphQL error and adjust CENTRAL_ORDER_BY / CENTRAL_ORDER_DIRECTION or CENTRAL_FILTER_MODE.")
        else:
            data = central.get("data") or {}
            all_series = data.get("allSeries") or {}
            edges = all_series.get("edges") or []
            total = all_series.get("totalCount")
            if total is not None:
                print(f"  totalCount: {total}")
            for i, edge in enumerate(edges[:10]):
                node = edge.get("node") or {}
                nid = node.get("id", "?")
                title = node.get("title")
                started_at = node.get("startedAt")
                parts = [f"id={nid}"]
                if title is not None:
                    parts.append(f"title={title!r}")
                if started_at is not None:
                    parts.append(f"startedAt={started_at}")
                print("  node:", " ".join(parts))
            if len(edges) > 10:
                print(f"  ... and {len(edges) - 10} more edges")
        ids = _extract_series_ids(central)
        if ids:
            series_id = ids[0]
            print(f"  Series IDs found (using first): {ids[:5]}")
        else:
            print("  No series IDs extracted from response.")

    if not series_id and MANUAL_SERIES_ID:
        series_id = MANUAL_SERIES_ID
        print(f"[GRID PROBE V1] Using MANUAL_SERIES_ID: {series_id}")

    # Step 2: Series State query (seriesState(id: ...))
    if series_id:
        print("[GRID PROBE V1] Step 2: Series State query (seriesState) ...")
        try:
            state = post_graphql(
                SERIES_STATE_GRAPHQL_URL,
                QUERY_SERIES_STATE,
                variables={"id": series_id},
                api_key=api_key,
            )
        except Exception as e:
            state = {"errors": [{"message": str(e)}]}
        _save_json(RAW_SERIES_STATE_PATH, state)
        if state.get("errors"):
            print("  GraphQL errors (see raw file):")
            for err in state.get("errors", []):
                print("   ", err.get("message", err))
        else:
            data = state.get("data") or {}
            ss = data.get("seriesState") or {}
            if isinstance(ss, dict):
                print("  id:", ss.get("id"))
                print("  title:", ss.get("title"))
                print("  valid:", ss.get("valid"))
                print("  updatedAt:", ss.get("updatedAt"))
                teams = ss.get("teams")
                games = ss.get("games")
                if isinstance(teams, list):
                    print("  teams count:", len(teams))
                if isinstance(games, list):
                    print("  games count:", len(games))
    else:
        print("[GRID PROBE V1] No series ID available. Set MANUAL_SERIES_ID or fix Central Data query.")
        print("[GRID PROBE V1] Stopping gracefully. Inspect raw_grid_central_data.json and adjust query if needed.")

    print("\n[GRID PROBE V1] Done. If GraphQL errors occurred, adjust queries or set CENTRAL_MINIMAL_MODE = True.")


if __name__ == "__main__":
    main()

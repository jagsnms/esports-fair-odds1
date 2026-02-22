# One-off: fetch GRID series for 2026-02-22 and search for BetBoom vs Gentle Mates.
# Run from project root: python -m adapters.grid_probe.fetch_feb22_find_betboom
from __future__ import annotations

import json
import sys
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Query with optional after for pagination
QUERY_WITH_AFTER = """
query FindCS2Series($orderBy: SeriesOrderBy!, $orderDirection: OrderDirection!, $first: Int, $filter: SeriesFilter, $after: String) {
  allSeries(orderBy: $orderBy, orderDirection: $orderDirection, first: $first, filter: $filter, after: $after) {
    totalCount
    edges { cursor node { id title { name nameShortened } type updatedAt startTimeScheduled teams { baseInfo { id name nameShortened } } } }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    return " ".join(str(s).lower().split())


def _matches(s: str, *terms: str) -> bool:
    n = _normalize(s)
    return any(t.lower() in n for t in terms)


def main() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from adapters.grid_probe import grid_graphql_client
    from adapters.grid_probe import grid_queries

    api_key = grid_graphql_client.load_api_key(ENV_PATH)
    filter_feb22 = {
        "startTimeScheduled": {
            "gte": "2026-02-22T00:00:00.000Z",
            "lte": "2026-02-22T23:59:59.000Z",
        }
    }

    # 1) Fetch all Feb 22 series (paginate to get full 74)
    print("Fetching allSeries for 2026-02-22 (full day UTC), paginating ...")
    all_edges = []
    after = None
    total_count = None
    while True:
        vars = {
            "orderBy": "StartTimeScheduled",
            "orderDirection": "ASC",
            "first": 50,
            "filter": filter_feb22,
        }
        if after:
            vars["after"] = after
        r = grid_graphql_client.post_graphql(
            grid_graphql_client.CENTRAL_DATA_GRAPHQL_URL,
            QUERY_WITH_AFTER.strip(),
            variables=vars,
            api_key=api_key,
        )
        if r.get("errors"):
            print("Errors:", json.dumps(r["errors"], indent=2))
            break
        data = r.get("data") or {}
        conn = data.get("allSeries") or {}
        total_count = conn.get("totalCount")
        edges = conn.get("edges") or []
        all_edges.extend(edges)
        page_info = conn.get("pageInfo") or {}
        if not page_info.get("hasNextPage") or not edges:
            break
        after = page_info.get("endCursor")
        if not after:
            break
    print(f"  totalCount (Feb 22): {total_count}, fetched: {len(all_edges)}")

    # 2) No filter (first 50 only)
    print("\nFetching allSeries with no filter (first=50) ...")
    r2 = grid_graphql_client.post_graphql(
        grid_graphql_client.CENTRAL_DATA_GRAPHQL_URL,
        grid_queries.QUERY_FIND_CS2_SERIES.strip(),
        variables={
            "orderBy": "StartTimeScheduled",
            "orderDirection": "DESC",
            "first": 50,
            "filter": None,
        },
        api_key=api_key,
    )
    edges2 = []
    if not r2.get("errors"):
        data2 = r2.get("data") or {}
        edges2 = (data2.get("allSeries") or {}).get("edges") or []
        print(f"  returned: {len(edges2)}")
    else:
        print("  Errors (no filter)")

    def collect_series(edges: list) -> list[dict]:
        out = []
        for edge in edges:
            node = edge.get("node") if isinstance(edge, dict) else {}
            if not isinstance(node, dict):
                continue
            sid = node.get("id")
            if not sid:
                continue
            title = node.get("title")
            name = None
            if isinstance(title, dict):
                name = title.get("name") or title.get("nameShortened") or ""
            elif title is not None:
                name = str(title)
            teams_raw = node.get("teams") or []
            team_names = []
            for t in teams_raw:
                if isinstance(t, dict):
                    info = t.get("baseInfo") or t.get("team") or t
                    if isinstance(info, dict):
                        n = (info.get("name") or info.get("nameShortened") or "").strip()
                        if n:
                            team_names.append(n)
            out.append({
                "id": sid,
                "title": name,
                "startTimeScheduled": node.get("startTimeScheduled"),
                "type": node.get("type"),
                "team_names": team_names,
            })
        return out

    series_feb22 = collect_series(all_edges)
    series_no_filter = collect_series(edges2) if not r2.get("errors") else []

    # Search for BetBoom vs Gentle Mates
    print("\n--- Search: BetBoom vs Gentle Mates (2/22) ---")
    targets = ("betboom", "gentle", "mates")
    found_feb22 = []
    for s in series_feb22:
        title_ok = _matches(s.get("title"), "betboom", "gentle", "mates")
        teams_ok = any(_matches(t, "betboom", "gentle", "mates") for t in s.get("team_names") or [])
        if title_ok or teams_ok:
            found_feb22.append(s)
    found_no_filter = []
    for s in series_no_filter:
        title_ok = _matches(s.get("title"), "betboom", "gentle", "mates")
        teams_ok = any(_matches(t, "betboom", "gentle", "mates") for t in s.get("team_names") or [])
        if title_ok or teams_ok:
            found_no_filter.append(s)

    if found_feb22:
        print("In Feb 22 list:")
        for s in found_feb22:
            print(f"  id={s['id']}  start={s['startTimeScheduled']}  type={s['type']}")
            print(f"    title: {s['title']}")
            print(f"    teams: {s['team_names']}")
    else:
        print("Not found in Feb 22 filtered list.")

    if found_no_filter:
        print("\nIn no-filter list (first 50):")
        for s in found_no_filter:
            print(f"  id={s['id']}  start={s['startTimeScheduled']}  type={s['type']}")
            print(f"    title: {s['title']}")
            print(f"    teams: {s['team_names']}")
    else:
        print("\nNot found in no-filter list (first 50).")

    # Search for MOUZ vs The Mongolz (CS2, ~4am PCT 2/22 = 12:00 UTC)
    print("\n--- Search: MOUZ vs The Mongolz (CS2, ~4am PCT = 12:00 UTC 2/22) ---")
    found_mouz = [s for s in series_feb22 if any(_matches(t, "mouz", "mongolz") for t in s.get("team_names") or []) or _matches(s.get("title"), "mouz", "mongolz")]
    if found_mouz:
        for s in found_mouz:
            print(f"  id={s['id']}  start={s['startTimeScheduled']}  type={s['type']}")
            print(f"    title: {s['title']}")
            print(f"    teams: {s['team_names']}")
    else:
        print("Not found in Feb 22 list.")
        # Show series around 12:00 UTC (4am PCT) in case timezone differs
        print("  Series near 12:00 UTC on 2/22:")
        for s in series_feb22:
            start = s.get("startTimeScheduled") or ""
            if "2026-02-22T11:" in start or "2026-02-22T12:" in start or "2026-02-22T13:" in start:
                print(f"    {start}  {s['team_names'] or s['title']}  (type={s['type']})")

    # Dump all Feb 22 series so we can see what we do have
    print("\n--- All series on 2026-02-22 (first 20) ---")
    for s in series_feb22[:20]:
        print(f"  {s['startTimeScheduled']}  {s['team_names'] or s['title'] or s['id']}")
    if len(series_feb22) > 20:
        print(f"  ... and {len(series_feb22) - 20} more.")


if __name__ == "__main__":
    main()

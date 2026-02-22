# One-off: run the same "Fetch GRID Live Series" call as the app, write raw response to file.
# Run from project root: python -m adapters.grid_probe.fetch_live_series_to_file
from __future__ import annotations

import json
from pathlib import Path

PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
OUTPUT_RAW = PROBE_DIR / "raw_grid_fetch_live_series_response.json"
OUTPUT_PARSED = PROBE_DIR / "raw_grid_fetch_live_series_parsed.json"


def main() -> None:
    from adapters.grid_probe import grid_graphql_client
    from adapters.grid_probe import grid_queries

    api_key = grid_graphql_client.load_api_key(PROJECT_ROOT / ".env")
    central_vars = {
        "orderBy": "StartTimeScheduled",
        "orderDirection": "DESC",
        "first": 50,
        "filter": {"live": {"games": {}}},
    }
    central = grid_graphql_client.post_graphql(
        grid_graphql_client.CENTRAL_DATA_GRAPHQL_URL,
        grid_queries.QUERY_FIND_CS2_SERIES.strip(),
        variables=central_vars,
        api_key=api_key,
    )

    # Full raw response
    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        json.dump(central, f, indent=2, ensure_ascii=False)
    print(f"Wrote raw response: {OUTPUT_RAW}")

    # Parsed rows (same logic as app35_ml.py Fetch button)
    rows = []
    if not central.get("errors"):
        data = central.get("data") or {}
        all_series = data.get("allSeries") or {}
        edges = all_series.get("edges") or []
        for edge in edges:
            node = edge.get("node") if isinstance(edge, dict) else {}
            if not isinstance(node, dict):
                continue
            sid = node.get("id")
            if sid is None:
                continue
            title = node.get("title")
            name = None
            if isinstance(title, dict):
                name = title.get("name") or title.get("nameShortened") or str(title)
            elif title is not None:
                name = str(title)
            teams_raw = node.get("teams")
            team_names = []
            if isinstance(teams_raw, list):
                for t in teams_raw:
                    if isinstance(t, dict):
                        team_obj = t.get("baseInfo") if isinstance(t.get("baseInfo"), dict) else t.get("team") or t
                        if isinstance(team_obj, dict):
                            team_name = (team_obj.get("name") or team_obj.get("nameShortened") or "").strip()
                            if team_name:
                                team_names.append(team_name)
            teams_display = " vs ".join(team_names) if team_names else None
            series_type = node.get("type")
            title_short = None
            if isinstance(title, dict):
                title_short = title.get("nameShortened") or title.get("name")
            if title_short is None and name:
                title_short = name
            rows.append({
                "series_id": str(sid),
                "title": title,
                "title_short": title_short,
                "name": name,
                "teams": team_names or None,
                "teams_display": teams_display,
                "map": None,
                "updated_at": node.get("updatedAt"),
                "startTimeScheduled": node.get("startTimeScheduled"),
                "valid": None,
                "series_type": series_type,
            })

    with open(OUTPUT_PARSED, "w", encoding="utf-8") as f:
        json.dump({"count": len(rows), "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"Wrote parsed rows ({len(rows)}): {OUTPUT_PARSED}")


if __name__ == "__main__":
    main()

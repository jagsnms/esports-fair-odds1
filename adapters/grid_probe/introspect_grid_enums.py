# GRID PROBE V1 — One-shot introspection for enum values (Central Data).
"""
Run from project root. Two requests (one per enum). No retries, no loops.
Central Data endpoint only.

  python -m adapters.grid_probe.introspect_grid_enums

Fetches SeriesOrderBy and OrderDirection enum values.
Saves: raw_grid_enum_SeriesOrderBy.json, raw_grid_enum_OrderDirection.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# GRID PROBE V1 — one __type per request (API rejects multiple __type in one query)
INTROSPECT_TYPE_QUERY = """
query IntrospectType($typeName: String!) {
  __type(name: $typeName) {
    kind
    name
    enumValues { name description }
  }
}
"""

# GRID PROBE V1 — output paths
RAW_ENUM_SERIES_ORDER_BY_PATH = PROBE_DIR / "raw_grid_enum_SeriesOrderBy.json"
RAW_ENUM_ORDER_DIRECTION_PATH = PROBE_DIR / "raw_grid_enum_OrderDirection.json"


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def _print_enum(type_obj: dict | None, label: str) -> list[str]:
    # GRID PROBE V1 — print enum values; return list of names
    names: list[str] = []
    if not type_obj or type_obj.get("kind") != "ENUM":
        print(f"[GRID PROBE V1] {label}: (not an enum or missing)")
        return names
    ev = type_obj.get("enumValues") or []
    for v in ev:
        if isinstance(v, dict) and v.get("name"):
            names.append(v["name"])
    print(f"[GRID PROBE V1] {label} values:")
    for n in names:
        print(f"  {n}")
    return names


def main() -> None:
    # GRID PROBE V1 — Central Data only, one request
    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import CENTRAL_DATA_GRAPHQL_URL, load_api_key, post_graphql

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    # GRID PROBE V1 — request 1: SeriesOrderBy
    print("[GRID PROBE V1] Central Data — introspecting SeriesOrderBy ...")
    data1 = post_graphql(
        CENTRAL_DATA_GRAPHQL_URL,
        INTROSPECT_TYPE_QUERY.strip(),
        variables={"typeName": "SeriesOrderBy"},
        api_key=api_key,
    )
    series_order_by = None
    if not data1.get("errors"):
        series_order_by = (data1.get("data") or {}).get("__type")
        if isinstance(series_order_by, dict):
            _save_json(RAW_ENUM_SERIES_ORDER_BY_PATH, series_order_by)
    else:
        print("[GRID PROBE V1] SeriesOrderBy errors:", data1.get("errors"))

    # GRID PROBE V1 — request 2: OrderDirection
    print("[GRID PROBE V1] Central Data — introspecting OrderDirection ...")
    data2 = post_graphql(
        CENTRAL_DATA_GRAPHQL_URL,
        INTROSPECT_TYPE_QUERY.strip(),
        variables={"typeName": "OrderDirection"},
        api_key=api_key,
    )
    order_direction = None
    if not data2.get("errors"):
        order_direction = (data2.get("data") or {}).get("__type")
        if isinstance(order_direction, dict):
            _save_json(RAW_ENUM_ORDER_DIRECTION_PATH, order_direction)
    else:
        print("[GRID PROBE V1] OrderDirection errors:", data2.get("errors"))

    print("")
    _print_enum(series_order_by, "SeriesOrderBy")
    print("")
    _print_enum(order_direction, "OrderDirection")


if __name__ == "__main__":
    main()

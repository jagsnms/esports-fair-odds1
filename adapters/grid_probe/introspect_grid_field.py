# GRID PROBE V1 — Introspect a root Query field: args, return type, and return-type fields.
"""
Run from project root. One request (or two when return type is OBJECT). No retries, no loops.

  python -m adapters.grid_probe.introspect_grid_field --central --field allSeries
  python -m adapters.grid_probe.introspect_grid_field --series-state --field seriesState

Saves:
  central + allSeries   -> raw_grid_central_allSeries_introspection.json
  series_state + seriesState -> raw_grid_seriesState_introspection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# GRID PROBE V1 — introspection: Query type fields with full args and return type
INTROSPECT_QUERY_TYPE = """
query IntrospectQueryType {
  __type(name: "Query") {
    name
    fields {
      name
      args {
        name
        type {
          kind
          name
          ofType { kind name ofType { kind name ofType { kind name } } }
        }
      }
      type {
        kind
        name
        ofType { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
      }
    }
  }
}
"""

# GRID PROBE V1 — introspection: concrete type fields (one level)
INTROSPECT_TYPE_FIELDS = """
query IntrospectType($typeName: String!) {
  __type(name: $typeName) {
    name
    kind
    fields {
      name
      type {
        kind
        name
        ofType { kind name }
      }
    }
  }
}
"""


def _resolve_type_name(t: Optional[dict]) -> Optional[str]:
    # GRID PROBE V1 — follow ofType until we get a concrete type name
    while t:
        if t.get("kind") in ("OBJECT", "INTERFACE", "SCALAR", "ENUM", "UNION"):
            return t.get("name")
        t = t.get("ofType") if isinstance(t.get("ofType"), dict) else None
    return None


def _format_type_chain(t: Optional[dict]) -> str:
    # GRID PROBE V1 — human-readable type chain (e.g. "[Series]!" or "SeriesState")
    if not t:
        return "?"
    kind = t.get("kind", "")
    name = t.get("name") or ""
    of = t.get("ofType") if isinstance(t.get("ofType"), dict) else None
    if kind == "NON_NULL":
        return _format_type_chain(of) + "!"
    if kind == "LIST":
        return "[" + _format_type_chain(of) + "]"
    return name or "?"


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def _print_summary(
    field_name: str,
    field_data: dict,
    return_type_fields: Optional[list[dict]] = None,
) -> None:
    # GRID PROBE V1 — clean summary: args, return type, likely query shape
    print("[GRID PROBE V1] --- Field summary ---")
    print(f"  field: {field_name}")

    args = field_data.get("args") or []
    if args:
        print("  arguments:")
        for a in args:
            name = a.get("name", "?")
            t = a.get("type")
            required = t and t.get("kind") == "NON_NULL"
            type_str = _format_type_chain(t)
            print(f"    - {name}: {type_str}  {'(required)' if required else ''}")
    else:
        print("  arguments: (none)")

    ret = field_data.get("type")
    type_name = _resolve_type_name(ret)
    print(f"  return type: {_format_type_chain(ret)}")

    if return_type_fields:
        print("  return type fields (one level):")
        for f in return_type_fields:
            name = f.get("name", "?")
            t = f.get("type")
            print(f"    - {name}: {_format_type_chain(t)}")

    # Likely query shape
    print("[GRID PROBE V1] --- Likely query shape ---")
    arg_part = ""
    if args:
        arg_part = "(" + ", ".join(f'{a.get("name")}: $var' for a in args) + ")"
    print(f"  query {{ {field_name}{arg_part} {{ ... }} }}")


def main() -> None:
    # GRID PROBE V1 — endpoint + field target
    parser = argparse.ArgumentParser(
        description="GRID PROBE V1: Introspect a root Query field (args, return type, return-type fields)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--central", action="store_true", help="Central Data GraphQL")
    group.add_argument("--series-state", action="store_true", help="Live Data Feed Series State GraphQL")
    parser.add_argument(
        "--field",
        required=True,
        choices=["allSeries", "seriesState"],
        help="Root field to introspect",
    )
    args = parser.parse_args()

    # GRID PROBE V1 — output paths
    if args.central and args.field == "allSeries":
        out_path = PROBE_DIR / "raw_grid_central_allSeries_introspection.json"
    elif args.series_state and args.field == "seriesState":
        out_path = PROBE_DIR / "raw_grid_seriesState_introspection.json"
    else:
        print("[GRID PROBE V1] Invalid combo: use --central --field allSeries or --series-state --field seriesState")
        sys.exit(1)

    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import (
        CENTRAL_DATA_GRAPHQL_URL,
        SERIES_STATE_GRAPHQL_URL,
        load_api_key,
        post_graphql,
    )

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    url = CENTRAL_DATA_GRAPHQL_URL if args.central else SERIES_STATE_GRAPHQL_URL
    endpoint_label = "Central Data" if args.central else "Series State"
    print(f"[GRID PROBE V1] {endpoint_label} — introspecting root field '{args.field}' ...")

    # GRID PROBE V1 — one request: Query type fields
    data: dict[str, Any] = post_graphql(url, INTROSPECT_QUERY_TYPE.strip(), api_key=api_key)
    if data.get("errors"):
        _save_json(out_path, data)
        print("[GRID PROBE V1] GraphQL returned errors; see saved file.")
        return

    query_type = (data.get("data") or {}).get("__type") or {}
    fields = query_type.get("fields") or []
    field_data = next((f for f in fields if isinstance(f, dict) and f.get("name") == args.field), None)
    if not field_data:
        _save_json(out_path, data)
        print(f"[GRID PROBE V1] Root field '{args.field}' not found. See saved file for all Query fields.")
        return

    result: dict[str, Any] = {
        "endpoint": "central" if args.central else "series_state",
        "field": args.field,
        "queryField": field_data,
        "returnTypeFields": None,
    }

    # GRID PROBE V1 — optional second request: return type fields (one level) when OBJECT
    type_name = _resolve_type_name(field_data.get("type"))
    if type_name:
        data2 = post_graphql(
            url,
            INTROSPECT_TYPE_FIELDS.strip(),
            variables={"typeName": type_name},
            api_key=api_key,
        )
        if not data2.get("errors"):
            type_info = (data2.get("data") or {}).get("__type") or {}
            result["returnTypeFields"] = type_info.get("fields") or []

    _save_json(out_path, result)
    _print_summary(
        args.field,
        field_data,
        result.get("returnTypeFields"),
    )


if __name__ == "__main__":
    main()

# GRID PROBE V1 — Introspect a type on the Central Data endpoint (e.g. SeriesFilter, Series).
"""
Run from project root. One request per type. No retries, no loops.
Central Data endpoint only.

  python -m adapters.grid_probe.introspect_grid_central_type SeriesFilter
  python -m adapters.grid_probe.introspect_grid_central_type Series

Saves: raw_grid_central_type_<TypeName>.json
Requests: kind, name, inputFields (input types), fields (object types) with names/types.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# GRID PROBE V1 — inputFields for INPUT_OBJECT, fields for OBJECT
INTROSPECT_TYPE_QUERY = """
query IntrospectType($typeName: String!) {
  __type(name: $typeName) {
    kind
    name
    inputFields {
      name
      type {
        kind
        name
        ofType { kind name }
      }
    }
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

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def _type_str(t: dict | None) -> str:
    # GRID PROBE V1 — simple type display
    if not t:
        return "?"
    kind = t.get("kind", "")
    name = t.get("name", "")
    if name:
        return name
    of = t.get("ofType") if isinstance(t.get("ofType"), dict) else None
    if kind == "NON_NULL":
        return _type_str(of) + "!"
    if kind == "LIST":
        return "[" + _type_str(of) + "]"
    return kind or "?"


def main() -> None:
    # GRID PROBE V1
    type_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not type_name:
        print("Usage: python -m adapters.grid_probe.introspect_grid_central_type <TypeName>")
        print("  e.g. SeriesFilter, Series")
        sys.exit(1)

    out_path = PROBE_DIR / f"raw_grid_central_type_{type_name}.json"

    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import CENTRAL_DATA_GRAPHQL_URL, load_api_key, post_graphql

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    print(f"[GRID PROBE V1] Central Data — introspecting type '{type_name}' ...")
    data = post_graphql(
        CENTRAL_DATA_GRAPHQL_URL,
        INTROSPECT_TYPE_QUERY.strip(),
        variables={"typeName": type_name},
        api_key=api_key,
    )

    if data.get("errors"):
        print("[GRID PROBE V1] GraphQL errors:", data.get("errors"))
        sys.exit(1)

    type_obj = (data.get("data") or {}).get("__type")
    if not type_obj:
        print("[GRID PROBE V1] No __type in response.")
        sys.exit(1)

    _save_json(out_path, type_obj)

    kind = type_obj.get("kind", "")
    name = type_obj.get("name", "")

    input_fields = type_obj.get("inputFields") or []
    if input_fields:
        print(f"[GRID PROBE V1] Type {name} (kind={kind}) inputFields:")
        for f in input_fields:
            if isinstance(f, dict) and f.get("name"):
                print(f"  {f['name']}: {_type_str(f.get('type'))}")

    fields = type_obj.get("fields") or []
    if fields:
        print(f"[GRID PROBE V1] Type {name} (kind={kind}) fields:")
        for f in fields:
            if isinstance(f, dict) and f.get("name"):
                print(f"  {f['name']}: {_type_str(f.get('type'))}")


if __name__ == "__main__":
    main()

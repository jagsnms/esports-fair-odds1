# GRID PROBE V1 — Introspect SeriesState and nested OBJECT types on Series State endpoint.
"""
Run from project root. Single run, no retries, no polling.
Series State endpoint only.

  python -m adapters.grid_probe.introspect_grid_series_state_nested_types

Introspects SeriesState and recursively nested OBJECT types up to MAX_DEPTH.
One __type request per type. Saves raw_grid_seriesState_nested_types.json.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

# GRID PROBE V1 — max depth of nested types to introspect (0 = SeriesState only)
MAX_DEPTH = 3

# GRID PROBE V1
PROBE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_DIR.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUT_PATH = PROBE_DIR / "raw_grid_seriesState_nested_types.json"

# GRID PROBE V1 — full type chain to resolve LIST/NON_NULL to concrete type name
INTROSPECT_TYPE_QUERY = """
query IntrospectType($typeName: String!) {
  __type(name: $typeName) {
    name
    kind
    fields {
      name
      type {
        kind
        name
        ofType { kind name ofType { kind name ofType { kind name } } }
      }
    }
  }
}
"""


def _resolve_object_type_name(t: Optional[dict]) -> Optional[str]:
    # GRID PROBE V1 — follow ofType until OBJECT (or SCALAR/INTERFACE/UNION/ENUM); return name for OBJECT
    while t and isinstance(t, dict):
        kind = t.get("kind")
        name = t.get("name")
        if kind == "OBJECT":
            return name
        t = t.get("ofType") if isinstance(t.get("ofType"), dict) else None
    return None


def _get_field_type_names(type_obj: dict) -> list[str]:
    # GRID PROBE V1 — from __type result, collect concrete OBJECT type names from fields
    names: list[str] = []
    for f in type_obj.get("fields") or []:
        if not isinstance(f, dict):
            continue
        t = f.get("type")
        name = _resolve_object_type_name(t if isinstance(t, dict) else None)
        if name and name not in names:
            names.append(name)
    return names


def _save_json(path: Path, data: dict) -> None:
    # GRID PROBE V1
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[GRID PROBE V1] Saved: {path}")


def main() -> None:
    # GRID PROBE V1
    if str(PROBE_DIR) not in sys.path:
        sys.path.insert(0, str(PROBE_DIR))
    from grid_graphql_client import SERIES_STATE_GRAPHQL_URL, load_api_key, post_graphql

    print("[GRID PROBE V1] Loading API key from .env ...")
    try:
        api_key = load_api_key(ENV_PATH)
    except Exception as e:
        print(f"[GRID PROBE V1] Failed to load API key: {e}")
        sys.exit(1)

    results: dict[str, dict] = {}
    visited: set[str] = set()
    # (type_name, depth)
    queue: deque[tuple[str, int]] = deque([("SeriesState", 0)])

    while queue:
        type_name, depth = queue.popleft()
        if type_name in visited or depth > MAX_DEPTH:
            continue
        visited.add(type_name)
        print(f"[GRID PROBE V1] Introspecting type '{type_name}' (depth={depth}) ...")
        data = post_graphql(
            SERIES_STATE_GRAPHQL_URL,
            INTROSPECT_TYPE_QUERY.strip(),
            variables={"typeName": type_name},
            api_key=api_key,
        )
        if data.get("errors"):
            print(f"[GRID PROBE V1] GraphQL errors for {type_name}:", data.get("errors"))
            continue
        type_obj = (data.get("data") or {}).get("__type")
        if not type_obj:
            continue
        results[type_name] = type_obj
        nested = _get_field_type_names(type_obj)
        for n in nested:
            if n not in visited and depth + 1 <= MAX_DEPTH:
                queue.append((n, depth + 1))

    _save_json(OUT_PATH, results)

    # GRID PROBE V1 — readable summary
    print("\n[GRID PROBE V1] --- SeriesState fields and types ---")
    ss = results.get("SeriesState")
    if ss:
        for f in ss.get("fields") or []:
            if isinstance(f, dict) and f.get("name"):
                t = f.get("type")
                kind = t.get("kind") if isinstance(t, dict) else "?"
                name = t.get("name") if isinstance(t, dict) else None
                of = t.get("ofType") if isinstance(t, dict) else None
                elem = _resolve_object_type_name(of if isinstance(of, dict) else None)
                if elem:
                    print(f"  {f['name']}: -> {elem}")
                else:
                    print(f"  {f['name']}: {name or kind}")
    print("\n[GRID PROBE V1] --- Nested object types discovered ---")
    for name in sorted(results.keys()):
        if name == "SeriesState":
            continue
        print(f"  {name}")
    print("\n[GRID PROBE V1] --- Fields per nested type ---")
    for name in sorted(results.keys()):
        if name == "SeriesState":
            continue
        type_obj = results[name]
        fields = type_obj.get("fields") or []
        print(f"  {name}:")
        for f in fields:
            if isinstance(f, dict) and f.get("name"):
                t = f.get("type")
                kind = t.get("kind") if isinstance(t, dict) else "?"
                tname = t.get("name") if isinstance(t, dict) else None
                elem = _resolve_object_type_name(t if isinstance(t, dict) else None)
                if elem:
                    print(f"    {f['name']}: -> {elem}")
                else:
                    print(f"    {f['name']}: {tname or kind}")


if __name__ == "__main__":
    main()
